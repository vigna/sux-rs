//! Block reseeding experiments: buckets are grouped in blocks of G
//! consecutive buckets; each block has a pattern seed t in [0, T) that
//! re-randomizes the in-slice offsets of all keys of the block. Blocks are
//! processed sequentially; within a block, buckets are processed by PHast
//! priority. For each block we try the T patterns from a snapshot of the state
//! and keep the best.

use crate::plus::{Conf, Geom, PATTERN_KEYS};
use crate::{Cyclic, ef_bits, mul_hi};

#[inline(always)]
pub fn offset(c: u64, t: u32, g: &Geom) -> usize {
    if t == 0 {
        (c & g.l_mask) as usize
    } else {
        (mul_hi(
            c,
            PATTERN_KEYS[(t & 63) as usize].wrapping_add((t as u64 >> 6) << 1),
        ) & g.l_mask) as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockConf {
    pub g: usize,
    pub t: u32,
    /// stop at the first trial with zero bumped keys
    pub early_exit: bool,
}

pub struct BlockLevel {
    pub geom: Geom,
    pub seeds: Vec<u16>,
    pub block_seeds: Vec<u32>,
    pub bucket_begin: Vec<usize>,
    pub bumped_keys: usize,
    pub trials: usize,
}

/// Places a bucket with PHast+ shift search using pattern t. Returns seed.
#[inline]
fn place_shift(
    used: &mut Cyclic<512>,
    keys: &[u64],
    g: &Geom,
    t: u32,
    last_shift: u16,
    bases: &mut Vec<usize>,
) -> u16 {
    bases.clear();
    bases.extend(keys.iter().map(|&c| g.slice_begin(c) + offset(c, t, g)));
    let mut shift = 0u16;
    while shift < last_shift {
        let mut u = 0u64;
        for &b in bases.iter() {
            u |= used.get64(b + shift as usize);
        }
        if u != u64::MAX {
            let total = shift + u.trailing_ones() as u16;
            if total >= last_shift {
                return 0;
            }
            bases.sort_unstable();
            if bases.windows(2).any(|w| w[0] == w[1]) {
                return 0;
            }
            for &b in bases.iter() {
                used.set(b + total as usize);
            }
            return total + 1;
        }
        shift += 64;
    }
    0
}

pub fn build_level_blocks(hashes: &[u64], m: usize, conf: &Conf, bc: &BlockConf) -> BlockLevel {
    let n = hashes.len();
    let geom = Geom::new(n, m, conf);
    let nb = geom.buckets;
    let mut bucket_begin = vec![0usize; nb + 1];
    for &c in hashes {
        bucket_begin[geom.bucket(c) + 1] += 1;
    }
    for i in 0..nb {
        bucket_begin[i + 1] += bucket_begin[i];
    }
    let last_shift = (1u16 << conf.s) - 1;
    let mut seeds = vec![0u16; nb];
    let nblocks = nb.div_ceil(bc.g);
    let mut block_seeds = vec![0u32; nblocks];
    let mut used = Cyclic::<512>::default();
    let mut snapshot = Cyclic::<512>::default();
    let mut best_used = Cyclic::<512>::default();
    let mut order: Vec<(i64, usize)> = Vec::with_capacity(bc.g);
    let mut bases = Vec::with_capacity(32);
    let mut trial_seeds = vec![0u16; bc.g];
    let mut best_seeds = vec![0u16; bc.g];
    let mut bumped_total = 0;
    let mut value_to_clear = 0usize;
    let mut trials = 0;
    for blk in 0..nblocks {
        let b0 = blk * bc.g;
        let b1 = (b0 + bc.g).min(nb);
        // clear values below the first base of this block
        if bucket_begin[b0] < n {
            let end = geom.slice_begin(hashes[bucket_begin[b0]]);
            while value_to_clear < end {
                used.clear(value_to_clear);
                value_to_clear += 1;
            }
        }
        // priority order (fixed across trials)
        order.clear();
        for b in b0..b1 {
            let sz = bucket_begin[b + 1] - bucket_begin[b];
            if sz > 0 {
                order.push((conf.eval(b, sz), b));
            }
        }
        order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        snapshot.w.copy_from_slice(&*used.w);
        let mut best = (usize::MAX, 0u32);
        for t in 0..bc.t {
            trials += 1;
            used.w.copy_from_slice(&*snapshot.w);
            let mut bumped = 0;
            for &(_, b) in &order {
                let keys = &hashes[bucket_begin[b]..bucket_begin[b + 1]];
                let s = place_shift(&mut used, keys, &geom, t, last_shift, &mut bases);
                trial_seeds[b - b0] = s;
                if s == 0 {
                    bumped += keys.len();
                }
            }
            if bumped < best.0 {
                best = (bumped, t);
                best_seeds[..b1 - b0].copy_from_slice(&trial_seeds[..b1 - b0]);
                best_used.w.copy_from_slice(&*used.w);
                if bumped == 0 && bc.early_exit {
                    break;
                }
            }
        }
        used.w.copy_from_slice(&*best_used.w);
        seeds[b0..b1].copy_from_slice(&best_seeds[..b1 - b0]);
        block_seeds[blk] = best.1;
        bumped_total += best.0;
    }
    BlockLevel {
        geom,
        seeds,
        block_seeds,
        bucket_begin,
        bumped_keys: bumped_total,
        trials,
    }
}

pub struct Report {
    pub n: usize,
    pub levels: Vec<(usize, usize, usize, usize)>, // keys, buckets, bumped, trials
    pub seed_bits: f64,
    pub block_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.block_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

pub fn mphf_blocks(n: usize, conf: &Conf, bc: &BlockConf, seed: u64) -> Report {
    use crate::plus::level_hash;
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let mut seed_bits = 0.0;
    let mut block_bits = 0.0;
    let mut total_range = 0usize;
    let mut level = 0u64;
    let tbits = (bc.t as f64).log2().ceil();
    while ids.len() > 8192 {
        let mut h: Vec<(u64, u64)> = ids
            .iter()
            .map(|&id| (level_hash(id, level, seed), id))
            .collect();
        h.sort_unstable_by_key(|x| x.0);
        let hashes: Vec<u64> = h.iter().map(|x| x.0).collect();
        let k = hashes.len();
        let lv = build_level_blocks(&hashes, k, conf, bc);
        seed_bits += (lv.geom.buckets * conf.s as usize) as f64;
        block_bits += lv.block_seeds.len() as f64 * tbits;
        total_range += k;
        let mut next = Vec::with_capacity(lv.bumped_keys);
        for b in 0..lv.geom.buckets {
            if lv.seeds[b] == 0 {
                for j in lv.bucket_begin[b]..lv.bucket_begin[b + 1] {
                    next.push(h[j].1);
                }
            }
        }
        levels.push((k, lv.geom.buckets, lv.bumped_keys, lv.trials));
        ids = next;
        level += 1;
    }
    let last_n = ids.len();
    let last_range = (last_n + 10) * 120 / 100;
    let last_bits = (last_n as f64 / 4.0).ceil() * 8.0;
    total_range += last_range;
    let ef_entries = total_range.saturating_sub(n);
    Report {
        n,
        levels,
        seed_bits,
        block_bits,
        ef_bits: ef_bits(ef_entries, n),
        last_bits,
    }
}
