//! Lattice-steered PHast+: slots at positions p with p % q == q - 1 are
//! "lattice" slots; placement minimizes k * shift + penalty * (#lattice slots
//! used) among feasible shifts (R patterns x D shifts). Holes are then
//! counted on/off lattice and encoded separately.

use crate::plus::{PATTERN_KEYS, level_hash};
use crate::twoclass::Bits;
use crate::{ef_bits, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct LatConf {
    pub lambda: f64,
    pub l: usize,
    pub r: u16,
    pub d: u16,
    /// lattice period (power of two)
    pub q: usize,
    /// penalty per lattice slot used (in units of shift positions)
    pub penalty: usize,
    pub weights: [i64; 7],
    pub window: usize,
}

impl LatConf {
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
pub struct LG {
    pub buckets: usize,
    pub num_slices: usize,
    pub l_mask: u64,
    pub d: u16,
}

impl LG {
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        mul_hi(c, self.buckets as u64) as usize
    }
    #[inline(always)]
    pub fn base(&self, c: u64, r: u16) -> usize {
        let off = if r == 0 {
            c & self.l_mask
        } else {
            mul_hi(c, PATTERN_KEYS[r as usize]) & self.l_mask
        };
        mul_hi(c, self.num_slices as u64) as usize + off as usize
    }
    #[inline(always)]
    pub fn pos(&self, c: u64, seed: u16) -> usize {
        let s = seed - 1;
        self.base(c, s / self.d) + (s % self.d) as usize
    }
}

/// Lattice mask for 64 consecutive positions starting at x: bit j set iff (x + j) % q == q - 1.
#[inline(always)]
fn lat_mask(x: usize, q: usize) -> u64 {
    // pattern with bit at q-1, 2q-1, ...
    let base: u64 = match q {
        2 => 0xAAAA_AAAA_AAAA_AAAA,
        4 => 0x8888_8888_8888_8888,
        8 => 0x8080_8080_8080_8080,
        16 => 0x8000_8000_8000_8000,
        32 => 0x8000_0000_8000_0000,
        _ => 0,
    };
    // shift so that bit j corresponds to position x + j
    base.rotate_right((x % q) as u32)
}

pub struct Level {
    pub g: LG,
    pub seeds: Vec<u16>,
    pub bb: Vec<usize>,
    pub bumped: usize,
    pub used: Bits,
}

fn search(
    used: &Bits,
    keys: &[u64],
    g: &LG,
    lc: &LatConf,
    bases: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
) -> u16 {
    let k = keys.len();
    let mut best: Option<(usize, u16)> = None;
    for pr in 0..lc.r {
        bases.clear();
        bases.extend(keys.iter().map(|&c| g.base(c, pr)));
        let base_sum: usize = bases.iter().sum();
        if let Some((s, _)) = best {
            if base_sum * 1 >= s {
                // objective lower bound: base_sum (shift 0, no penalty)
                if base_sum >= s {
                    continue;
                }
            }
        }
        sorted.clear();
        sorted.extend_from_slice(bases);
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            continue;
        }
        let mut shift = 0u16;
        'scan: while shift < lc.d {
            let mut u = 0u64;
            for &x in bases.iter() {
                u |= used.get64(x + shift as usize);
            }
            if shift + 64 > lc.d {
                u |= !0u64 << (lc.d - shift);
            }
            let mut free = !u;
            while free != 0 {
                let j = free.trailing_zeros() as u16;
                free &= free - 1;
                let dd = shift + j;
                let lower = base_sum + dd as usize * k;
                if let Some((s, _)) = best {
                    if lower >= s {
                        break 'scan;
                    }
                }
                // count lattice slots used
                let mut cnt = 0usize;
                if lc.penalty > 0 {
                    for &x in bases.iter() {
                        if (x + dd as usize) % lc.q == lc.q - 1 {
                            cnt += 1;
                        }
                    }
                }
                let obj = lower + cnt * lc.penalty;
                if best.map_or(true, |(s, _)| obj < s) {
                    best = Some((obj, pr * lc.d + dd + 1));
                }
                if lc.penalty == 0 {
                    break 'scan;
                }
            }
            shift += 64;
        }
    }
    let _ = lat_mask;
    best.map_or(0, |x| x.1)
}

pub fn build_level(hashes: &[u64], m: usize, lc: &LatConf) -> Level {
    let n = hashes.len();
    let l = lc.l.min((m / 2 + 1).next_power_of_two());
    let g = LG {
        buckets: 1.max((n as f64 / lc.lambda).round() as usize),
        num_slices: m + 1 - l - (lc.d as usize - 1),
        l_mask: l as u64 - 1,
        d: lc.d,
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
    let size = |b: usize| bb[b + 1] - bb[b];
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
    let mut in_heap = vec![false; nb];
    let mut span_begin = 0usize;
    while span_begin < nb && size(span_begin) == 0 {
        span_begin += 1;
    }
    let span_end = |sb: usize| (sb + lc.window).min(nb);
    if span_begin < nb {
        for b in span_begin..span_end(span_begin) {
            if size(b) != 0 {
                heap.push((lc.eval(b, size(b)), Reverse(b)));
                in_heap[b] = true;
            }
        }
    }
    let mut bumped = 0;
    let mut bases = Vec::new();
    let mut sorted = Vec::new();
    while let Some((_, Reverse(b))) = heap.pop() {
        in_heap[b] = false;
        let keys = &hashes[bb[b]..bb[b + 1]];
        let s = search(&used, keys, &g, lc, &mut bases, &mut sorted);
        seeds[b] = s;
        if s == 0 {
            bumped += keys.len();
        } else {
            for &c in keys {
                used.set(g.pos(c, s));
            }
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
                    heap.push((lc.eval(b2, size(b2)), Reverse(b2)));
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
        used,
    }
}

pub struct Report {
    pub n: usize,
    pub levels: Vec<(usize, usize, usize, usize, usize)>, // keys, buckets, bumped, holes on lattice, holes off lattice
    pub seed_bits: f64,
    pub ef_plain_bits: f64,
    pub hole_bits: f64,
    pub last_bits: f64,
}

/// Elias-Fano style estimate for holes split into lattice/off-lattice; the
/// lattice holes can also be stored as a bitmap over lattice positions.
pub fn mphf(n: usize, lc: &LatConf, seed: u64) -> Report {
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let mut seed_bits = 0.0;
    let mut total_range = 0usize;
    let mut level = 0u64;
    let mut l0_holes = (0usize, 0usize);
    let mut l0_range = 0usize;
    while ids.len() > 8192 {
        let mut h: Vec<(u64, u64)> = ids
            .iter()
            .map(|&id| (level_hash(id, level, seed), id))
            .collect();
        h.sort_unstable_by_key(|x| x.0);
        let hs: Vec<u64> = h.iter().map(|x| x.0).collect();
        let k = hs.len();
        // lattice only at level 0
        let mut lc2 = lc.clone();
        if level > 0 {
            lc2.penalty = 0;
        }
        let lv = build_level(&hs, k, &lc2);
        let (mut on, mut off) = (0, 0);
        for p in 0..k {
            if !lv.used.get(p) {
                if p % lc.q == lc.q - 1 {
                    on += 1;
                } else {
                    off += 1;
                }
            }
        }
        if level == 0 {
            l0_holes = (on, off);
            l0_range = k;
        }
        let mut next = Vec::new();
        for b in 0..lv.g.buckets {
            if lv.seeds[b] == 0 {
                for j in lv.bb[b]..lv.bb[b + 1] {
                    next.push(h[j].1);
                }
            }
        }
        seed_bits += lv.g.buckets as f64 * ((lc.r as u32 * lc.d as u32 + 1) as f64).log2().ceil();
        total_range += k;
        levels.push((k, lv.g.buckets, lv.bumped, on, off));
        ids = next;
        level += 1;
    }
    let last_n = ids.len();
    let last_range = (last_n + 10) * 120 / 100;
    let last_bits = (last_n as f64 / 4.0).ceil() * 8.0;
    total_range += last_range;
    let ef_entries = total_range.saturating_sub(n);
    // Plain: EF over all entries (as PHast).
    let ef_plain = ef_bits(ef_entries, n);
    // Split: level-0 holes on lattice as EF over lattice index space,
    // off-lattice level-0 holes as EF over full space, plus the remaining
    // entries (filler entries for levels >= 1 beyond level-0 holes) as EF.
    let (on, off) = l0_holes;
    let lattice_pts = l0_range / lc.q;
    let on_bits = ef_bits(on, lattice_pts).min(lattice_pts as f64 * 1.0625);
    let off_bits = ef_bits(off, n);
    // entries beyond level-0 holes (levels>=1 holes repeat) ~ handled by plain EF over the excess
    let extra = ef_entries.saturating_sub(on + off);
    let extra_bits = if extra > 0 { ef_bits(extra, n) } else { 0.0 };
    Report {
        n,
        levels,
        seed_bits,
        ef_plain_bits: ef_plain,
        hole_bits: on_bits + off_bits + extra_bits,
        last_bits,
    }
}
