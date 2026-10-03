//! Row-permutation placement: slots are organized in 64-slot rows; a key has
//! a base row and a per-key permutation of columns pi(s) = ((s ^ x) + r) mod 64.
//! The seed is (a, s): row offset a in [0, A), column selector s in [0, 64).
//! Feasibility of all 64 values of s for a key is computed with one rotation
//! and a bit-index XOR permutation of the row occupancy word.

use crate::plus::level_hash;
use crate::{ef_bits, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct RowConf {
    /// seed bits
    pub s: u32,
    pub lambda: f64,
    /// rows per slice (power of two)
    pub lr: usize,
    pub weights: [i64; 7],
    pub window: usize,
    /// 0 = first feasible s; 1 = min sum of columns among feasible s
    pub choice: u8,
    /// 0 = row + a; 1 = random row in slice per block (min row-sum block wins)
    pub rowmode: u8,
}

impl RowConf {
    /// number of row offsets
    pub fn a(&self) -> usize {
        ((1usize << self.s) - 1).div_ceil(64)
    }
    pub fn eval(&self, bucket: usize, size: usize) -> i64 {
        let w = if size <= 7 {
            self.weights[size - 1]
        } else {
            let l = self.weights[6];
            let p = self.weights[5];
            l + (l - p) * (size - 7) as i64
        };
        w - 1024 * bucket as i64
    }
}

#[derive(Clone, Copy)]
pub struct RG {
    pub buckets: usize,
    pub rows: usize,
    pub num_rslices: usize,
    pub lr_mask: u64,
    pub rowmode: u8,
}

impl RG {
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        mul_hi(c, self.buckets as u64) as usize
    }
    #[inline(always)]
    pub fn row(&self, c: u64) -> usize {
        mul_hi(c, self.num_rslices as u64) as usize + ((c >> 12) & self.lr_mask) as usize
    }
    #[inline(always)]
    pub fn xr(c: u64) -> (u32, u32) {
        ((c & 63) as u32, ((c >> 6) & 63) as u32)
    }
    /// row and (x, r) of key c in block a
    #[inline(always)]
    pub fn block(&self, c: u64, a: usize) -> (usize, u32, u32) {
        if self.rowmode == 0 {
            let (x, r) = Self::xr(c);
            (self.row(c) + a, x, r)
        } else {
            let h = mul_hi(c, crate::plus::PATTERN_KEYS[a & 63])
                ^ (c.wrapping_mul(crate::plus::PATTERN_KEYS[(a + 17) & 63]));
            let rsb = mul_hi(c, self.num_rslices as u64) as usize;
            (
                rsb + (h & self.lr_mask) as usize,
                ((h >> 20) & 63) as u32,
                ((h >> 26) & 63) as u32,
            )
        }
    }
    #[inline(always)]
    pub fn pos(&self, c: u64, seed: u16) -> usize {
        let v = (seed - 1) as usize;
        let (a, s) = (v / 64, (v % 64) as u32);
        let (row, x, r) = self.block(c, a);
        row * 64 + (((s ^ x) + r) & 63) as usize
    }
}

/// new[s] = w[s ^ x]
#[inline(always)]
pub fn xor_perm(mut w: u64, x: u32) -> u64 {
    const M: [u64; 6] = [
        0x5555_5555_5555_5555,
        0x3333_3333_3333_3333,
        0x0F0F_0F0F_0F0F_0F0F,
        0x00FF_00FF_00FF_00FF,
        0x0000_FFFF_0000_FFFF,
        0x0000_0000_FFFF_FFFF,
    ];
    for k in 0..6 {
        if x >> k & 1 != 0 {
            let sh = 1 << k;
            w = ((w & M[k]) << sh) | ((w >> sh) & M[k]);
        }
    }
    w
}

pub struct Level {
    pub g: RG,
    pub seeds: Vec<u16>,
    pub bb: Vec<usize>,
    pub bumped: usize,
    pub by_size: Vec<(usize, usize)>,
}

pub fn build_level(hashes: &[u64], m: usize, rc: &RowConf) -> Level {
    let n = hashes.len();
    let a_max = rc.a();
    let rows = m / 64;
    let lr = rc.lr.min((rows / 2).max(1).next_power_of_two());
    let num_rslices = if rc.rowmode == 0 {
        rows + 1 - lr - (a_max - 1)
    } else {
        rows + 1 - lr
    };
    let g = RG {
        buckets: 1.max((n as f64 / rc.lambda).round() as usize),
        rows: m.div_ceil(64),
        num_rslices,
        lr_mask: lr as u64 - 1,
        rowmode: rc.rowmode,
    };
    let nb = g.buckets;
    let mut bb = vec![0usize; nb + 1];
    for &c in hashes {
        bb[g.bucket(c) + 1] += 1;
    }
    for i in 0..nb {
        bb[i + 1] += bb[i];
    }
    // occupancy words; positions >= m pre-marked as used
    let mut used = vec![0u64; g.rows + a_max + 2];
    if m % 64 != 0 {
        used[m / 64] = !0u64 << (m % 64);
    }
    for w in used.iter_mut().skip(m.div_ceil(64)) {
        *w = !0;
    }
    let mut seeds = vec![0u16; nb];
    let size = |b: usize| bb[b + 1] - bb[b];
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
    let mut in_heap = vec![false; nb];
    let mut span_begin = 0usize;
    while span_begin < nb && size(span_begin) == 0 {
        span_begin += 1;
    }
    let span_end = |sb: usize| (sb + rc.window).min(nb);
    if span_begin < nb {
        for b in span_begin..span_end(span_begin) {
            if size(b) != 0 {
                heap.push((rc.eval(b, size(b)), Reverse(b)));
                in_heap[b] = true;
            }
        }
    }
    let mut bumped = 0;
    let mut by_size = vec![(0usize, 0usize); 32];
    let max_seed = (1u32 << rc.s) - 1;
    let mut info: Vec<(usize, u32, u32)> = Vec::new();
    let mut posv: Vec<usize> = Vec::new();
    while let Some((_, Reverse(b))) = heap.pop() {
        in_heap[b] = false;
        let keys = &hashes[bb[b]..bb[b + 1]];
        let mut seed = 0u16;
        let mut best_block: Option<(usize, u16)> = None; // (row sum, seed)
        'outer: for a in 0..a_max {
            info.clear();
            info.extend(keys.iter().map(|&c| g.block(c, a)));
            if rc.rowmode == 1 {
                let rs: usize = info.iter().map(|x| x.0).sum();
                if let Some((bs, _)) = best_block {
                    if rs >= bs {
                        continue;
                    }
                }
            }
            let mut t = 0u64;
            for &(row, x, r) in &info {
                t |= xor_perm(used[row].rotate_right(r), x);
                if t == u64::MAX {
                    break;
                }
            }
            // seeds beyond max_seed are invalid
            let first = (a * 64) as u32;
            if first + 64 > max_seed {
                let valid = max_seed - first; // number of valid s
                if valid < 64 {
                    t |= !0u64 << valid;
                }
            }
            let mut best: Option<(usize, u32)> = None;
            let mut free = !t;
            while free != 0 {
                let s = free.trailing_zeros();
                free &= free - 1;
                // self-collision check
                posv.clear();
                let mut sum = 0;
                for &(row, x, r) in &info {
                    let p = row * 64 + (((s ^ x) + r) & 63) as usize;
                    posv.push(p);
                    sum += p;
                }
                posv.sort_unstable();
                if posv.windows(2).any(|w| w[0] == w[1]) {
                    continue;
                }
                if rc.choice == 0 {
                    best = Some((sum, s));
                    break;
                }
                if best.map_or(true, |(bs, _)| sum < bs) {
                    best = Some((sum, s));
                }
            }
            if let Some((_, s)) = best {
                let sd = (a as u32 * 64 + s + 1) as u16;
                if rc.rowmode == 0 {
                    best_block = Some((0, sd));
                    break 'outer;
                }
                let rs: usize = info.iter().map(|x| x.0).sum();
                best_block = Some((rs, sd));
            }
        }
        if let Some((_, sd)) = best_block {
            seed = sd;
            for &c in keys {
                let p = g.pos(c, sd);
                used[p / 64] |= 1 << (p % 64);
            }
        }
        seeds[b] = seed;
        let sz = keys.len().min(31);
        by_size[sz].0 += 1;
        if seed == 0 {
            bumped += keys.len();
            by_size[sz].1 += 1;
        }
        if b == span_begin {
            let old_end = span_end(span_begin);
            span_begin += 1;
            while span_begin < old_end && !in_heap[span_begin] {
                span_begin += 1;
            }
            if span_begin == old_end {
                while span_begin < nb && size(span_begin) == 0 {
                    span_begin += 1;
                }
                if span_begin == nb {
                    break;
                }
            }
            for b2 in old_end..span_end(span_begin) {
                if size(b2) != 0 {
                    heap.push((rc.eval(b2, size(b2)), Reverse(b2)));
                    in_heap[b2] = true;
                }
            }
        }
    }
    Level {
        g,
        seeds,
        bb,
        bumped,
        by_size,
    }
}

pub struct Report {
    pub n: usize,
    pub levels: Vec<(usize, usize, usize)>,
    pub seed_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
    pub by_size0: Vec<(usize, usize)>,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

pub fn mphf(n: usize, rc: &RowConf, seed: u64, verify: bool) -> Report {
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let mut seed_bits = 0.0;
    let mut total_range = 0usize;
    let mut level = 0u64;
    let mut by_size0 = None;
    while ids.len() > 8192 {
        let mut h: Vec<(u64, u64)> = ids
            .iter()
            .map(|&id| (level_hash(id, level, seed), id))
            .collect();
        h.sort_unstable_by_key(|x| x.0);
        let hs: Vec<u64> = h.iter().map(|x| x.0).collect();
        let k = hs.len();
        let lv = build_level(&hs, k, rc);
        let mut next = Vec::with_capacity(lv.bumped);
        let mut seen = if verify {
            Some(vec![0u64; k.div_ceil(64) + 1])
        } else {
            None
        };
        for b in 0..lv.g.buckets {
            let v = lv.seeds[b];
            for j in lv.bb[b]..lv.bb[b + 1] {
                if v == 0 {
                    next.push(h[j].1);
                } else if let Some(s) = seen.as_mut() {
                    let p = lv.g.pos(hs[j], v);
                    assert!(p < k, "out of range {p} {k}");
                    assert!(s[p / 64] >> (p % 64) & 1 == 0, "collision");
                    s[p / 64] |= 1 << (p % 64);
                }
            }
        }
        seed_bits += lv.g.buckets as f64 * rc.s as f64;
        total_range += k;
        levels.push((k, lv.g.buckets, lv.bumped));
        if by_size0.is_none() {
            by_size0 = Some(lv.by_size);
        }
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
        ef_bits: ef_bits(ef_entries, n),
        last_bits,
        by_size0: by_size0.unwrap_or_default(),
    }
}
