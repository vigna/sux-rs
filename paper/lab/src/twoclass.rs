//! Multi-sweep PHast+: keys are split by a hash bit into classes; class 0
//! (bulk) uses large buckets and is placed first over the whole range; class
//! 1 (fillers) uses tiny buckets with its own seed array and is placed in a
//! second sweep into the free slots left by class 0. Each class uses R
//! patterns x D shifts (R = 1 is plain PHast+).

use crate::plus::{Conf, PATTERN_KEYS, level_hash};
use crate::{ef_bits, mix64, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Full occupancy bitmap.
pub struct Bits {
    pub w: Vec<u64>,
}

impl Bits {
    pub fn new(m: usize) -> Self {
        Self {
            w: vec![0; m.div_ceil(64) + 2],
        }
    }
    #[inline(always)]
    pub fn get(&self, v: usize) -> bool {
        self.w[v / 64] >> (v % 64) & 1 != 0
    }
    #[inline(always)]
    pub fn set(&mut self, v: usize) {
        self.w[v / 64] |= 1 << (v % 64);
    }
    #[inline(always)]
    pub fn get64(&self, v: usize) -> u64 {
        let c = v / 64;
        let b = v % 64;
        let lo = self.w[c];
        if b == 0 {
            return lo;
        }
        (lo >> b) | (self.w[c + 1] << (64 - b))
    }
}

#[derive(Clone, Debug)]
pub struct ClassConf {
    pub lambda: f64,
    pub l: usize,
    pub r: u16,
    pub d: u16,
    /// seed bits (for space accounting): ceil(log2(r*d+1))
    pub weights: [i64; 7],
    pub window: usize,
}

impl ClassConf {
    pub fn seed_bits(&self) -> f64 {
        ((self.r as u32 * self.d as u32 + 1) as f64).log2().ceil()
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
pub struct CGeom {
    pub buckets: usize,
    pub num_slices: usize,
    pub l_mask: u64,
    pub d: u16,
}

impl CGeom {
    pub fn new(n: usize, m: usize, cc: &ClassConf) -> Self {
        let l = cc.l.min((m / 2 + 1).next_power_of_two());
        let buckets = 1.max((n as f64 / cc.lambda).round() as usize);
        Self {
            buckets,
            num_slices: m + 1 - l - (cc.d as usize - 1),
            l_mask: l as u64 - 1,
            d: cc.d,
        }
    }
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        mul_hi(c, self.buckets as u64) as usize
    }
    #[inline(always)]
    pub fn slice_begin(&self, c: u64) -> usize {
        mul_hi(c, self.num_slices as u64) as usize
    }
    #[inline(always)]
    pub fn base(&self, c: u64, r: u16) -> usize {
        let off = if r == 0 {
            c & self.l_mask
        } else {
            mul_hi(c, PATTERN_KEYS[r as usize]) & self.l_mask
        };
        self.slice_begin(c) + off as usize
    }
    #[inline(always)]
    pub fn pos(&self, c: u64, seed: u16) -> usize {
        let s = seed - 1;
        self.base(c, s / self.d) + (s % self.d) as usize
    }
}

fn search(used: &Bits, keys: &[u64], g: &CGeom, r: u16) -> u16 {
    let k = keys.len();
    let mut best: Option<(usize, u16)> = None;
    let mut bases = [0usize; 64];
    if k > 64 {
        return 0;
    }
    for pr in 0..r {
        for (i, &c) in keys.iter().enumerate() {
            bases[i] = g.base(c, pr);
        }
        let bs = &bases[..k];
        let base_sum: usize = bs.iter().sum();
        if let Some((s, _)) = best {
            if base_sum >= s {
                continue;
            }
        }
        let mut sorted = [0usize; 64];
        sorted[..k].copy_from_slice(bs);
        sorted[..k].sort_unstable();
        if sorted[..k].windows(2).any(|w| w[0] == w[1]) {
            continue;
        }
        let mut shift = 0u16;
        while shift < g.d {
            let mut u = 0u64;
            for &x in bs {
                u |= used.get64(x + shift as usize);
            }
            if shift + 64 > g.d {
                u |= !0u64 << (g.d - shift);
            }
            if u != u64::MAX {
                let d = shift + u.trailing_ones() as u16;
                let sum = base_sum + d as usize * k;
                if best.map_or(true, |(s, _)| sum < s) {
                    best = Some((sum, pr * g.d + d + 1));
                }
                break;
            }
            shift += 64;
        }
    }
    best.map_or(0, |x| x.1)
}

/// One sweep over sorted hashes of a class. Returns (geom, seeds, bucket_begin, bumped keys).
pub static THRESHOLDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[inline(always)]
pub fn fp(c: u64) -> u64 {
    crate::mix64(c ^ 0x2545f4914f6cdd1d)
}

pub fn sweep(
    hashes: &[u64],
    m: usize,
    cc: &ClassConf,
    used: &mut Bits,
) -> (CGeom, Vec<u16>, Vec<usize>, usize) {
    let (g, s, bb, b, _) = sweep_t(hashes, m, cc, used);
    (g, s, bb, b)
}

/// Like sweep, but also returns per-bucket thresholds (u64::MAX = all keys kept).
pub fn sweep_t(
    hashes: &[u64],
    m: usize,
    cc: &ClassConf,
    used: &mut Bits,
) -> (CGeom, Vec<u16>, Vec<usize>, usize, Vec<u64>) {
    let nthr = THRESHOLDS.load(std::sync::atomic::Ordering::Relaxed);
    let n = hashes.len();
    let g = CGeom::new(n, m, cc);
    let nb = g.buckets;
    let mut bb = vec![0usize; nb + 1];
    for &c in hashes {
        bb[g.bucket(c) + 1] += 1;
    }
    for i in 0..nb {
        bb[i + 1] += bb[i];
    }
    let mut seeds = vec![0u16; nb];
    let mut thrs = vec![u64::MAX; nb];
    let size = |b: usize| bb[b + 1] - bb[b];
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
    let mut in_heap = vec![false; nb];
    let mut span_begin = 0usize;
    while span_begin < nb && size(span_begin) == 0 {
        span_begin += 1;
    }
    let mut bumped = 0;
    if span_begin == nb {
        return (g, seeds, bb, 0, thrs);
    }
    let span_end = |sb: usize| (sb + cc.window).min(nb);
    for b in span_begin..span_end(span_begin) {
        if size(b) != 0 {
            heap.push((cc.eval(b, size(b)), Reverse(b)));
            in_heap[b] = true;
        }
    }
    while let Some((_, Reverse(b))) = heap.pop() {
        in_heap[b] = false;
        let keys = &hashes[bb[b]..bb[b + 1]];
        let s = search(used, keys, &g, cc.r);
        seeds[b] = s;
        if s == 0 {
            let mut done = false;
            for t in 1..=nthr {
                // keep keys with fp < (nthr + 1 - t) / (nthr + 1)
                let thr = (u64::MAX / (nthr as u64 + 1)) * (nthr + 1 - t) as u64;
                let sub: Vec<u64> = keys.iter().copied().filter(|&c| fp(c) < thr).collect();
                if sub.is_empty() || sub.len() == keys.len() {
                    continue;
                }
                let s2 = search(used, &sub, &g, cc.r);
                if s2 != 0 {
                    for &c in &sub {
                        used.set(g.pos(c, s2));
                    }
                    bumped += keys.len() - sub.len();
                    seeds[b] = s2; // NOTE: threshold class not encoded (oracle)
                    thrs[b] = thr;
                    done = true;
                    break;
                }
            }
            if !done {
                bumped += keys.len();
            }
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
                    heap.push((cc.eval(b2, size(b2)), Reverse(b2)));
                    in_heap[b2] = true;
                }
            }
        }
    }
    (g, seeds, bb, bumped, thrs)
}

pub struct Report {
    pub n: usize,
    /// per level: keys, class-0 keys, class-0 bumped, class-1 keys, class-1 bumped
    pub levels: Vec<(usize, usize, usize, usize, usize)>,
    pub seed_bits0: f64,
    pub seed_bits1: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits0 + self.seed_bits1 + self.ef_bits + self.last_bits) / self.n as f64
    }
}

/// p = fraction of filler keys. Levels > 0 use the same scheme.
pub fn mphf(n: usize, c0: &ClassConf, c1: &ClassConf, p: f64, seed: u64, verify: bool) -> Report {
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let mut sb0 = 0.0;
    let mut sb1 = 0.0;
    let mut total_range = 0usize;
    let mut level = 0u64;
    let thr = (p * u64::MAX as f64) as u64;
    while ids.len() > 8192 {
        let k = ids.len();
        let m = k;
        let mut h0: Vec<(u64, u64)> = Vec::with_capacity(k);
        let mut h1: Vec<(u64, u64)> = Vec::with_capacity(k);
        for &id in &ids {
            let h = level_hash(id, level, seed);
            let cls = mix64(h ^ 0x5851f42d4c957f2d);
            if p > 0.0 && cls < thr {
                h1.push((h, id));
            } else {
                h0.push((h, id));
            }
        }
        h0.sort_unstable_by_key(|x| x.0);
        h1.sort_unstable_by_key(|x| x.0);
        let hs0: Vec<u64> = h0.iter().map(|x| x.0).collect();
        let hs1: Vec<u64> = h1.iter().map(|x| x.0).collect();
        let mut used = Bits::new(m + 64);
        let (g0, s0, bb0, bumped0, t0) = sweep_t(&hs0, m, c0, &mut used);
        let (g1, s1, bb1, bumped1, t1) = if hs1.is_empty() {
            (CGeom::new(1, m, c1), vec![], vec![0], 0, vec![])
        } else {
            sweep_t(&hs1, m, c1, &mut used)
        };
        if verify {
            let mut seen = Bits::new(m + 64);
            for (g, s, bb, hs, t) in [(&g0, &s0, &bb0, &hs0, &t0), (&g1, &s1, &bb1, &hs1, &t1)] {
                for b in 0..s.len() {
                    if s[b] == 0 {
                        continue;
                    }
                    for &c in &hs[bb[b]..bb[b + 1]] {
                        if fp(c) >= t[b] {
                            continue;
                        }
                        let x = g.pos(c, s[b]);
                        assert!(x < m);
                        assert!(!seen.get(x));
                        seen.set(x);
                    }
                }
            }
        }
        sb0 += s0.len() as f64 * c0.seed_bits();
        sb1 += s1.len() as f64 * c1.seed_bits();
        total_range += m;
        let mut next = Vec::with_capacity(bumped0 + bumped1);
        for b in 0..s0.len() {
            for j in bb0[b]..bb0[b + 1] {
                if s0[b] == 0 || fp(h0[j].0) >= t0[b] {
                    next.push(h0[j].1);
                }
            }
        }
        for b in 0..s1.len() {
            for j in bb1[b]..bb1[b + 1] {
                if s1[b] == 0 || fp(h1[j].0) >= t1[b] {
                    next.push(h1[j].1);
                }
            }
        }
        assert_eq!(next.len(), bumped0 + bumped1);
        levels.push((k, hs0.len(), bumped0, hs1.len(), bumped1));
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
        seed_bits0: sb0,
        seed_bits1: sb1,
        ef_bits: ef_bits(ef_entries, n),
        last_bits,
    }
}

pub fn class_conf(_c: &Conf) {}
