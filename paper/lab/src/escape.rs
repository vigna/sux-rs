//! Two-tier seeds: primary S1-bit seeds (0 = escape) with R1 patterns x D1
//! shifts; escaped buckets get an S2-bit secondary seed with R2 patterns x
//! D2 shifts (0 = bump). Space accounting includes line counters (62 seeds
//! per 64-byte line + 16-bit counter).

use crate::plus::{PATTERN_KEYS, level_hash};
use crate::twoclass::Bits;
use crate::{ef_bits, mix64, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct EscConf {
    pub lambda: f64,
    pub l: usize,
    pub r1: u16,
    pub d1: u16,
    pub s2: u32,
    pub r2: u16,
    pub d2: u16,
    pub weights: [i64; 7],
    pub window: usize,
    /// seeds per line for primary accounting (62 = 64 bytes - 16-bit counter)
    pub per_line: usize,
    /// secondary: 0 = min sum over all patterns, 1 = first pattern with a feasible shift
    pub sec_choice: u8,
}

impl EscConf {
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
    pub fn max_shift(&self) -> usize {
        (self.d1.max(self.d2) - 1) as usize
    }
}

#[derive(Clone, Copy)]
pub struct EG {
    pub buckets: usize,
    pub num_slices: usize,
    pub l_mask: u64,
}

impl EG {
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        mul_hi(c, self.buckets as u64) as usize
    }
    #[inline(always)]
    pub fn base(&self, c: u64, pattern: u32) -> usize {
        let off = if pattern == 0 {
            c & self.l_mask
        } else {
            mix64(c ^ PATTERN_KEYS[(pattern & 63) as usize].wrapping_mul(pattern as u64 + 1))
                & self.l_mask
        };
        mul_hi(c, self.num_slices as u64) as usize + off as usize
    }
}

/// Best (min-sum) seed with r patterns x d shifts, patterns numbered from
/// `pbase`. Returns seed in 1..=r*d or 0.
fn search(
    used: &Bits,
    keys: &[u64],
    g: &EG,
    pbase: u32,
    r: u16,
    d: u16,
    first: bool,
    bases: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
) -> u32 {
    let k = keys.len();
    let mut best: Option<(usize, u32)> = None;
    for pr in 0..r as u32 {
        bases.clear();
        bases.extend(keys.iter().map(|&c| g.base(c, pbase + pr)));
        let base_sum: usize = bases.iter().sum();
        if let Some((s, _)) = best {
            if base_sum >= s {
                continue;
            }
        }
        sorted.clear();
        sorted.extend_from_slice(bases);
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            continue;
        }
        let mut shift = 0u16;
        while shift < d {
            let mut u = 0u64;
            for &x in bases.iter() {
                u |= used.get64(x + shift as usize);
            }
            if shift + 64 > d {
                u |= !0u64 << (d - shift);
            }
            if u != u64::MAX {
                let dd = shift + u.trailing_ones() as u16;
                let sum = base_sum + dd as usize * k;
                if best.map_or(true, |(s, _)| sum < s) {
                    best = Some((sum, pr * d as u32 + dd as u32 + 1));
                }
                break;
            }
            shift += 64;
        }
        if first && best.is_some() {
            break;
        }
    }
    best.map_or(0, |x| x.1)
}

#[inline(always)]
pub fn pos(c: u64, seed: u32, g: &EG, pbase: u32, d: u16) -> usize {
    let v = seed - 1;
    g.base(c, pbase + v / d as u32) + (v % d as u32) as usize
}

pub struct Level {
    pub g: EG,
    pub p: Vec<u16>,
    pub sec: Vec<u32>,
    pub bb: Vec<usize>,
    pub escaped: usize,
    pub escaped_keys: usize,
    pub bumped: usize,
}

/// secondary patterns start at this index (independent from primary)
pub const PBASE2: u32 = 1000;

pub fn build_level(hashes: &[u64], m: usize, ec: &EscConf) -> Level {
    let n = hashes.len();
    let l = ec.l.min((m / 2 + 1).next_power_of_two());
    let g = EG {
        buckets: 1.max((n as f64 / ec.lambda).round() as usize),
        num_slices: m + 1 - l - ec.max_shift(),
        l_mask: l as u64 - 1,
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
    let mut p = vec![0u16; nb];
    let mut sec = vec![0u32; nb];
    let size = |b: usize| bb[b + 1] - bb[b];
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
    let mut in_heap = vec![false; nb];
    let mut span_begin = 0usize;
    while span_begin < nb && size(span_begin) == 0 {
        span_begin += 1;
    }
    let span_end = |sb: usize| (sb + ec.window).min(nb);
    if span_begin < nb {
        for b in span_begin..span_end(span_begin) {
            if size(b) != 0 {
                heap.push((ec.eval(b, size(b)), Reverse(b)));
                in_heap[b] = true;
            }
        }
    }
    let (mut escaped, mut escaped_keys, mut bumped) = (0, 0, 0);
    let mut bases = Vec::new();
    let mut sorted = Vec::new();
    while let Some((_, Reverse(b))) = heap.pop() {
        in_heap[b] = false;
        let keys = &hashes[bb[b]..bb[b + 1]];
        let s1 = search(
            &used,
            keys,
            &g,
            0,
            ec.r1,
            ec.d1,
            false,
            &mut bases,
            &mut sorted,
        );
        if s1 != 0 {
            p[b] = s1 as u16;
            for &c in keys {
                used.set(pos(c, s1, &g, 0, ec.d1));
            }
        } else {
            escaped += 1;
            escaped_keys += keys.len();
            let s2 = if ec.s2 == 0 {
                0
            } else {
                search(
                    &used,
                    keys,
                    &g,
                    PBASE2,
                    ec.r2,
                    ec.d2,
                    ec.sec_choice == 1,
                    &mut bases,
                    &mut sorted,
                )
            };
            sec[b] = s2;
            if s2 != 0 {
                for &c in keys {
                    used.set(pos(c, s2, &g, PBASE2, ec.d2));
                }
            } else {
                bumped += keys.len();
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
                    heap.push((ec.eval(b2, size(b2)), Reverse(b2)));
                    in_heap[b2] = true;
                }
            }
        }
    }
    Level {
        g,
        p,
        sec,
        bb,
        escaped,
        escaped_keys,
        bumped,
    }
}

pub struct Report {
    pub n: usize,
    /// keys, buckets, escaped buckets, escaped keys, bumped keys
    pub levels: Vec<(usize, usize, usize, usize, usize)>,
    pub p_bits: f64,
    pub s_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.p_bits + self.s_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

pub fn mphf(n: usize, ec: &EscConf, seed: u64, verify: bool) -> Report {
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let (mut p_bits, mut s_bits) = (0.0, 0.0);
    let mut total_range = 0usize;
    let mut level = 0u64;
    let s1 = ((ec.r1 as u32 * ec.d1 as u32 + 1) as f64).log2().ceil();
    while ids.len() > 8192 {
        let mut h: Vec<(u64, u64)> = ids
            .iter()
            .map(|&id| (level_hash(id, level, seed), id))
            .collect();
        h.sort_unstable_by_key(|x| x.0);
        let hs: Vec<u64> = h.iter().map(|x| x.0).collect();
        let k = hs.len();
        let lv = build_level(&hs, k, ec);
        let mut next = Vec::with_capacity(lv.bumped);
        let mut seen = if verify {
            Some(Bits::new(k + 64))
        } else {
            None
        };
        for b in 0..lv.g.buckets {
            for j in lv.bb[b]..lv.bb[b + 1] {
                let c = hs[j];
                let pp = if lv.p[b] != 0 {
                    Some(pos(c, lv.p[b] as u32, &lv.g, 0, ec.d1))
                } else if lv.sec[b] != 0 {
                    Some(pos(c, lv.sec[b], &lv.g, PBASE2, ec.d2))
                } else {
                    None
                };
                match pp {
                    None => next.push(h[j].1),
                    Some(x) => {
                        if let Some(s) = seen.as_mut() {
                            assert!(x < k);
                            assert!(!s.get(x), "collision");
                            s.set(x);
                        }
                    }
                }
            }
        }
        p_bits += lv.g.buckets as f64 * s1 * 64.0 / ec.per_line as f64;
        s_bits += lv.escaped as f64 * ec.s2 as f64;
        total_range += k;
        levels.push((k, lv.g.buckets, lv.escaped, lv.escaped_keys, lv.bumped));
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
        p_bits,
        s_bits,
        ef_bits: ef_bits(ef_entries, n),
        last_bits,
    }
}
