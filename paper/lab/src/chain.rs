//! Chained seeds: the offset pattern of bucket b depends on seed[b - 1];
//! seed[b] is a PHast+ shift. Buckets are processed in index order with
//! bounded backtracking (a failing bucket asks its predecessor for its next
//! feasible shift, which changes the failing bucket's pattern).

use crate::plus::{PATTERN_KEYS, level_hash};
use crate::twoclass::Bits;
use crate::{ef_bits, mix64, mul_hi};

#[derive(Clone, Debug)]
pub struct ChainConf {
    pub s: u32,
    pub lambda: f64,
    pub l: usize,
    /// max number of alternative predecessor shifts tried
    pub tries: usize,
    /// if false, pattern does not depend on predecessor (plain index-order PHast+)
    pub chained: bool,
}

#[derive(Clone, Copy)]
pub struct CG {
    pub buckets: usize,
    pub num_slices: usize,
    pub l_mask: u64,
    pub chained: bool,
}

impl CG {
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        mul_hi(c, self.buckets as u64) as usize
    }
    #[inline(always)]
    pub fn base(&self, c: u64, prev: u16) -> usize {
        let off = if !self.chained || prev == 0 {
            c & self.l_mask
        } else {
            mix64(c ^ PATTERN_KEYS[(prev & 63) as usize].wrapping_mul(prev as u64)) & self.l_mask
        };
        mul_hi(c, self.num_slices as u64) as usize + off as usize
    }
}

/// First feasible shift >= from, < dmax.
fn next_shift(used: &Bits, bases: &[usize], from: u16, dmax: u16) -> Option<u16> {
    let mut shift = from;
    while shift < dmax {
        let mut u = 0u64;
        for &x in bases {
            u |= used.get64(x + shift as usize);
        }
        if shift + 64 > dmax {
            u |= !0u64 << (dmax - shift);
        }
        if u != u64::MAX {
            return Some(shift + u.trailing_ones() as u16);
        }
        shift += 64;
    }
    None
}

fn self_collide(bases: &[usize]) -> bool {
    let mut s = bases.to_vec();
    s.sort_unstable();
    s.windows(2).any(|w| w[0] == w[1])
}

pub struct Level {
    pub g: CG,
    pub seeds: Vec<u16>,
    pub bb: Vec<usize>,
    pub bumped: usize,
    pub backtracks: usize,
}

pub fn build_level(hashes: &[u64], m: usize, cc: &ChainConf) -> Level {
    let n = hashes.len();
    let dmax = ((1u32 << cc.s) - 1) as u16;
    let l = cc.l.min((m / 2 + 1).next_power_of_two());
    let g = CG {
        buckets: 1.max((n as f64 / cc.lambda).round() as usize),
        num_slices: m + 1 - l - (dmax as usize - 1),
        l_mask: l as u64 - 1,
        chained: cc.chained,
    };
    let nb = g.buckets;
    let mut bb = vec![0usize; nb + 1];
    for &c in hashes {
        bb[g.bucket(c) + 1] += 1;
    }
    for i in 0..nb {
        bb[i + 1] += bb[i];
    }
    let mut used = Bits::new(m + 64);
    let mut seeds = vec![0u16; nb];
    let mut bumped = 0;
    let mut backtracks = 0;
    let mut bases: Vec<usize> = Vec::new();
    let mut pbases: Vec<usize> = Vec::new();
    let keys_of = |b: usize| &hashes[bb[b]..bb[b + 1]];
    for b in 0..nb {
        let keys = keys_of(b);
        if keys.is_empty() {
            seeds[b] = 0;
            continue;
        }
        let prev = if b > 0 { seeds[b - 1] } else { 0 };
        bases.clear();
        bases.extend(keys.iter().map(|&c| g.base(c, prev)));
        let mut ok = None;
        if !self_collide(&bases) {
            ok = next_shift(&used, &bases, 0, dmax);
        }
        if ok.is_none()
            && cc.chained
            && b > 0
            && cc.tries > 0
            && seeds[b - 1] != 0
            && !keys_of(b - 1).is_empty()
        {
            // backtrack predecessor: try its next feasible shifts
            let pkeys = keys_of(b - 1);
            let pprev = if b > 1 { seeds[b - 2] } else { 0 };
            pbases.clear();
            pbases.extend(pkeys.iter().map(|&c| g.base(c, pprev)));
            let orig = seeds[b - 1];
            // unmark predecessor
            for &x in &pbases {
                used.w[(x + orig as usize - 1) / 64] &= !(1 << ((x + orig as usize - 1) % 64));
            }
            let mut from = orig; // next shift after orig - 1
            let mut done = false;
            for _ in 0..cc.tries {
                let Some(d) = next_shift(&used, &pbases, from, dmax) else {
                    break;
                };
                from = d + 1;
                let ps = d + 1;
                backtracks += 1;
                // tentatively place predecessor
                for &x in &pbases {
                    used.set(x + d as usize);
                }
                bases.clear();
                bases.extend(keys.iter().map(|&c| g.base(c, ps)));
                if !self_collide(&bases) {
                    if let Some(dd) = next_shift(&used, &bases, 0, dmax) {
                        seeds[b - 1] = ps;
                        ok = Some(dd);
                        done = true;
                        break;
                    }
                }
                for &x in &pbases {
                    used.w[(x + d as usize) / 64] &= !(1 << ((x + d as usize) % 64));
                }
            }
            if !done {
                // restore predecessor
                for &x in &pbases {
                    used.set(x + orig as usize - 1);
                }
                bases.clear();
                bases.extend(keys.iter().map(|&c| g.base(c, orig)));
            }
        }
        match ok {
            Some(d) => {
                seeds[b] = d + 1;
                for &x in &bases {
                    used.set(x + d as usize);
                }
            }
            None => {
                seeds[b] = 0;
                bumped += keys.len();
            }
        }
    }
    Level {
        g,
        seeds,
        bb,
        bumped,
        backtracks,
    }
}

pub struct Report {
    pub n: usize,
    pub levels: Vec<(usize, usize, usize, usize)>,
    pub seed_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

pub fn mphf(n: usize, cc: &ChainConf, seed: u64, verify: bool) -> Report {
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let mut seed_bits = 0.0;
    let mut total_range = 0usize;
    let mut level = 0u64;
    while ids.len() > 8192 {
        let mut h: Vec<(u64, u64)> = ids
            .iter()
            .map(|&id| (level_hash(id, level, seed), id))
            .collect();
        h.sort_unstable_by_key(|x| x.0);
        let hs: Vec<u64> = h.iter().map(|x| x.0).collect();
        let k = hs.len();
        let lv = build_level(&hs, k, cc);
        let mut next = Vec::new();
        let mut seen = if verify {
            Some(Bits::new(k + 64))
        } else {
            None
        };
        for b in 0..lv.g.buckets {
            let v = lv.seeds[b];
            let prev = if b > 0 { lv.seeds[b - 1] } else { 0 };
            for j in lv.bb[b]..lv.bb[b + 1] {
                if v == 0 {
                    next.push(h[j].1);
                } else if let Some(s) = seen.as_mut() {
                    let p = lv.g.base(hs[j], prev) + v as usize - 1;
                    assert!(p < k);
                    assert!(!s.get(p), "collision");
                    s.set(p);
                }
            }
        }
        seed_bits += lv.g.buckets as f64 * cc.s as f64;
        total_range += k;
        levels.push((k, lv.g.buckets, lv.bumped, lv.backtracks));
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
    }
}
