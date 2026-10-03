//! PHast+ with threshold (fingerprint) bumping encoded in the seed.
//!
//! Seed value v (S bits, 0 = bump all):
//! - 1 ..= a: all keys placed with shift v - 1;
//! - a + 1 ..= 2^S - 1: u = v - a - 1, class t = u / dt + 1, shift u % dt;
//!   keys with fp(c) >= theta_t are bumped, others placed.

use crate::plus::{PATTERN_KEYS, level_hash};
use crate::twoclass::Bits;
use crate::{ef_bits, mix64, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct ThrConf {
    pub s: u32,
    pub lambda: f64,
    pub l: usize,
    /// number of full-keep shifts
    pub a: u16,
    /// number of threshold classes
    pub t: u16,
    /// shifts per threshold class
    pub dt: u16,
    /// kept fraction for class i (1-based): thetas[i-1]
    pub thetas: Vec<f64>,
    pub weights: [i64; 7],
    pub window: usize,
    /// choose among threshold classes the one maximizing kept keys (else first that works)
    pub best_kept: bool,
    /// if true, thetas are offset cutoffs in [0, 1) of L: keep keys with offset >= theta * L
    pub offset_mode: bool,
}

impl ThrConf {
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
        (self.a.max(self.dt) - 1) as usize
    }
}

#[inline(always)]
pub fn fp(c: u64) -> u64 {
    mix64(c ^ 0x2545f4914f6cdd1d)
}

#[derive(Clone, Copy)]
pub struct G {
    pub buckets: usize,
    pub num_slices: usize,
    pub l_mask: u64,
}

impl G {
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        mul_hi(c, self.buckets as u64) as usize
    }
    #[inline(always)]
    pub fn base(&self, c: u64) -> usize {
        mul_hi(c, self.num_slices as u64) as usize + (c & self.l_mask) as usize
    }
}

#[inline(always)]
pub fn kept(c: u64, thr: u64, g: &G, offset_mode: bool) -> bool {
    if thr == u64::MAX {
        return true;
    }
    if offset_mode {
        (c & g.l_mask) >= thr
    } else {
        fp(c) < thr
    }
}

/// Decodes a seed: returns (threshold, shift); threshold u64::MAX = all.
#[inline(always)]
pub fn decode(v: u16, tc: &ThrConf, thr: &[u64]) -> (u64, usize) {
    if v <= tc.a {
        (u64::MAX, (v - 1) as usize)
    } else {
        let u = v - tc.a - 1;
        (thr[(u / tc.dt) as usize], (u % tc.dt) as usize)
    }
}

/// First feasible shift in [0, dmax) for given bases (no self-collision check).
#[inline]
fn first_shift(used: &Bits, bases: &[usize], dmax: u16) -> Option<u16> {
    let mut shift = 0u16;
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
    let mut s: Vec<usize> = bases.to_vec();
    s.sort_unstable();
    s.windows(2).any(|w| w[0] == w[1])
}

pub struct Level {
    pub g: G,
    pub seeds: Vec<u16>,
    pub bb: Vec<usize>,
    pub bumped: usize,
    pub thresholded: usize,
    pub full_bumps: usize,
}

pub fn build_level(hashes: &[u64], m: usize, tc: &ThrConf) -> Level {
    let n = hashes.len();
    let l = tc.l.min((m / 2 + 1).next_power_of_two());
    let g = G {
        buckets: 1.max((n as f64 / tc.lambda).round() as usize),
        num_slices: m + 1 - l - tc.max_shift(),
        l_mask: l as u64 - 1,
    };
    let thr: Vec<u64> = thr_values(tc, l);
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
    let (mut bumped, mut thresholded, mut full_bumps) = (0, 0, 0);
    let span_end = |sb: usize| (sb + tc.window).min(nb);
    if span_begin < nb {
        for b in span_begin..span_end(span_begin) {
            if size(b) != 0 {
                heap.push((tc.eval(b, size(b)), Reverse(b)));
                in_heap[b] = true;
            }
        }
    }
    let mut bases: Vec<usize> = Vec::new();
    let mut sub: Vec<usize> = Vec::new();
    while let Some((_, Reverse(b))) = heap.pop() {
        in_heap[b] = false;
        let keys = &hashes[bb[b]..bb[b + 1]];
        bases.clear();
        bases.extend(keys.iter().map(|&c| g.base(c)));
        let mut seed = 0u16;
        if !self_collide(&bases) {
            if let Some(d) = first_shift(&used, &bases, tc.a) {
                seed = d + 1;
                for &x in &bases {
                    used.set(x + d as usize);
                }
            }
        }
        if seed == 0 {
            // threshold classes
            let mut best: Option<(usize, usize, u16, u16)> = None; // (kept, -?, class, shift)
            for t in 0..tc.t {
                sub.clear();
                for (i, &c) in keys.iter().enumerate() {
                    if kept(c, thr[t as usize], &g, tc.offset_mode) {
                        sub.push(bases[i]);
                    }
                }
                if sub.is_empty() || self_collide(&sub) {
                    continue;
                }
                if let Some(d) = first_shift(&used, &sub, tc.dt) {
                    let nk = sub.len();
                    if best.map_or(true, |(k, _, _, _)| nk > k) {
                        best = Some((nk, 0, t, d));
                    }
                    if !tc.best_kept {
                        break;
                    }
                }
            }
            if let Some((nkept, _, t, d)) = best {
                seed = tc.a + 1 + t * tc.dt + d;
                for (i, &c) in keys.iter().enumerate() {
                    if kept(c, thr[t as usize], &g, tc.offset_mode) {
                        used.set(bases[i] + d as usize);
                    }
                }
                bumped += keys.len() - nkept;
                thresholded += 1;
            } else {
                bumped += keys.len();
                full_bumps += 1;
            }
        }
        seeds[b] = seed;
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
                    heap.push((tc.eval(b2, size(b2)), Reverse(b2)));
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
        thresholded,
        full_bumps,
    }
}

pub struct Report {
    pub n: usize,
    /// keys, buckets, bumped, thresholded buckets, fully bumped buckets
    pub levels: Vec<(usize, usize, usize, usize, usize)>,
    pub seed_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

pub fn thr_values(tc: &ThrConf, l: usize) -> Vec<u64> {
    if tc.offset_mode {
        tc.thetas.iter().map(|&x| (x * l as f64) as u64).collect()
    } else {
        tc.thetas
            .iter()
            .map(|&x| (x * u64::MAX as f64) as u64)
            .collect()
    }
}

pub fn mphf(n: usize, tc: &ThrConf, seed: u64, verify: bool) -> Report {
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
        let lv = build_level(&hs, k, tc);
        let thr = thr_values(tc, lv.g.l_mask as usize + 1);
        let mut next = Vec::with_capacity(lv.bumped);
        let mut seen = if verify {
            Some(Bits::new(k + 64))
        } else {
            None
        };
        for b in 0..lv.g.buckets {
            let v = lv.seeds[b];
            for j in lv.bb[b]..lv.bb[b + 1] {
                let c = hs[j];
                let placed = if v == 0 {
                    None
                } else {
                    let (t, d) = decode(v, tc, &thr);
                    if kept(c, t, &lv.g, tc.offset_mode) {
                        Some(lv.g.base(c) + d)
                    } else {
                        None
                    }
                };
                match placed {
                    None => next.push(h[j].1),
                    Some(p) => {
                        if let Some(s) = seen.as_mut() {
                            assert!(p < k);
                            assert!(!s.get(p), "collision");
                            s.set(p);
                        }
                    }
                }
            }
        }
        assert_eq!(next.len(), lv.bumped);
        seed_bits += lv.g.buckets as f64 * tc.s as f64;
        total_range += k;
        levels.push((k, lv.g.buckets, lv.bumped, lv.thresholded, lv.full_bumps));
        ids = next;
        level += 1;
    }
    let _ = PATTERN_KEYS;
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
