//! PHast+ (R patterns x D shifts) with eviction-based repair of failing
//! buckets.

use crate::plus::{Conf, Geom, level_hash};
use crate::{Cyclic, ef_bits, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const CYC_BITS: usize = 512 * 64;
const CYC_MASK: usize = CYC_BITS - 1;

#[derive(Clone, Copy, Debug)]
pub struct EvictConf {
    /// number of patterns
    pub r: u16,
    /// number of shifts per pattern
    pub d: u16,
    /// max candidate evictions tried per failing bucket
    pub max_cand: usize,
    /// max size of evicted bucket
    pub max_evict_size: usize,
    /// recursion depth for re-placing evicted buckets
    pub depth: u32,
    /// allow bumping a smaller evicted bucket
    pub sacrifice: bool,
}

#[inline(always)]
pub fn offset(c: u64, r: u16, g: &Geom) -> usize {
    if r == 0 {
        (c & g.l_mask) as usize
    } else {
        (mul_hi(c, crate::plus::PATTERN_KEYS[r as usize]) & g.l_mask) as usize
    }
}

#[inline(always)]
pub fn position(c: u64, seed: u16, g: &Geom, d: u16) -> usize {
    if d == 0 {
        // regular PHast placement
        return g.slice_begin(c)
            + (mul_hi((seed as u64).wrapping_mul(0x517cc1b727220a95), c) & g.l_mask) as usize;
    }
    let s = seed - 1;
    g.slice_begin(c) + offset(c, s / d, g) + (s % d) as usize
}

struct State<'a> {
    hashes: &'a [u64],
    bb: &'a [usize],
    g: Geom,
    ec: EvictConf,
    used: Cyclic<512>,
    owner: Vec<u32>,
    seeds: Vec<u16>,
    value_to_clear: usize,
    evictions: usize,
    repaired: usize,
    self_coll: usize,
    sacrificed: usize,
}

impl<'a> State<'a> {
    #[inline]
    fn keys(&self, b: usize) -> &'a [u64] {
        &self.hashes[self.bb[b]..self.bb[b + 1]]
    }

    fn mark(&mut self, b: usize, seed: u16) {
        let keys = self.keys(b);
        for &c in keys {
            let p = position(c, seed, &self.g, self.ec.d);
            debug_assert!(!self.used.get(p));
            self.used.set(p);
            self.owner[p & CYC_MASK] = b as u32;
        }
        self.seeds[b] = seed;
    }

    fn unmark(&mut self, b: usize) {
        let seed = self.seeds[b];
        let keys = self.keys(b);
        for &c in keys {
            let p = position(c, seed, &self.g, self.ec.d);
            self.used.clear(p);
        }
        self.seeds[b] = 0;
    }

    /// Standard best-seed search (min sum over patterns of first feasible shift).
    fn search(&self, b: usize) -> (u16, bool) {
        let keys = self.keys(b);
        if self.ec.d == 0 {
            let mut best = (usize::MAX, 0u16);
            let mut vals = [0usize; 64];
            let k = keys.len().min(64);
            'outer: for seed in 1..=255u16 {
                let mut sum = 0;
                for (i, &c) in keys.iter().take(k).enumerate() {
                    let v = position(c, seed, &self.g, 0);
                    if self.used.get(v) {
                        continue 'outer;
                    }
                    sum += v;
                    vals[i] = v;
                }
                if sum < best.0 {
                    let mut sv = vals;
                    sv[..k].sort_unstable();
                    if sv[..k].windows(2).any(|w| w[0] == w[1]) {
                        continue;
                    }
                    best = (sum, seed);
                }
            }
            return (best.1, false);
        }
        let mut best: Option<(usize, u16)> = None;
        let mut any_self = false;
        let mut bases = [0usize; 64];
        let k = keys.len().min(64);
        for r in 0..self.ec.r {
            for (i, &c) in keys.iter().take(k).enumerate() {
                bases[i] = self.g.slice_begin(c) + offset(c, r, &self.g);
            }
            let bs = &mut bases[..k];
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
                any_self = true;
                continue;
            }
            let mut shift = 0u16;
            while shift < self.ec.d {
                let mut u = 0u64;
                for &x in bs.iter() {
                    u |= self.used.get64(x + shift as usize);
                }
                if shift + 64 > self.ec.d {
                    u |= !0u64 << (self.ec.d - shift);
                }
                if u != u64::MAX {
                    let d = shift + u.trailing_ones() as u16;
                    let sum = base_sum + d as usize * k;
                    if best.map_or(true, |(s, _)| sum < s) {
                        best = Some((sum, r * self.ec.d + d + 1));
                    }
                    break;
                }
                shift += 64;
            }
        }
        (best.map_or(0, |x| x.1), any_self)
    }

    /// Lowest position of a bucket's possible placements (min base).
    fn min_base(&self, b: usize) -> usize {
        let keys = self.keys(b);
        let mut m = usize::MAX;
        if self.ec.d == 0 {
            for &c in keys {
                m = m.min(self.g.slice_begin(c));
            }
            return m;
        }
        for r in 0..self.ec.r {
            for &c in keys {
                m = m.min(self.g.slice_begin(c) + offset(c, r, &self.g));
            }
        }
        m
    }

    /// Try to place bucket b, possibly evicting. Returns true on success.
    fn place(&mut self, b: usize, depth: u32, forbidden: usize) -> bool {
        let (seed, self_coll) = self.search(b);
        if seed != 0 {
            self.mark(b, seed);
            return true;
        }
        if depth == 0 || self.ec.max_cand == 0 {
            if self_coll && depth == self.ec.depth {
                self.self_coll += 1;
            }
            return false;
        }
        // Repair: find shifts where exactly one key is blocked.
        let keys = self.keys(b);
        let k = keys.len();
        if k > 64 {
            return false;
        }
        let mut cands = 0;
        // candidate list: (sum, r, d)
        let mut list: Vec<(usize, u16, u16)> = Vec::new();
        if self.ec.d == 0 {
            'seeds: for seed in 1..=255u16 {
                let mut blocked = 0;
                let mut sum = 0;
                let mut vals = [0usize; 64];
                for (i, &c) in keys.iter().enumerate() {
                    let v = position(c, seed, &self.g, 0);
                    vals[i] = v;
                    sum += v;
                    if self.used.get(v) {
                        blocked += 1;
                        if blocked > 1 {
                            continue 'seeds;
                        }
                    }
                }
                if blocked == 1 {
                    vals[..k].sort_unstable();
                    if vals[..k].windows(2).any(|w| w[0] == w[1]) {
                        continue;
                    }
                    // encode as r = seed, d = u16::MAX marker
                    list.push((sum, seed, u16::MAX));
                }
            }
        }
        for r in 0..self.ec.r {
            let mut bases = [0usize; 64];
            for (i, &c) in keys.iter().enumerate() {
                bases[i] = self.g.slice_begin(c) + offset(c, r, &self.g);
            }
            let bs = &bases[..k];
            let mut sorted = [0usize; 64];
            sorted[..k].copy_from_slice(bs);
            sorted[..k].sort_unstable();
            if sorted[..k].windows(2).any(|w| w[0] == w[1]) {
                continue;
            }
            let base_sum: usize = bs.iter().sum();
            let mut shift = 0u16;
            while shift < self.ec.d {
                let mut ones = 0u64;
                let mut twos = 0u64;
                for &x in bs {
                    let w = self.used.get64(x + shift as usize);
                    twos |= ones & w;
                    ones |= w;
                }
                let mut exactly_one = ones & !twos;
                if shift + 64 > self.ec.d {
                    exactly_one &= !(!0u64 << (self.ec.d - shift));
                }
                while exactly_one != 0 {
                    let d = shift + exactly_one.trailing_zeros() as u16;
                    exactly_one &= exactly_one - 1;
                    list.push((base_sum + d as usize * k, r, d));
                }
                shift += 64;
            }
        }
        list.sort_unstable();
        for &(_, r, d) in &list {
            if cands >= self.ec.max_cand {
                break;
            }
            // find the blocked key and its owner
            let cand_seed = if d == u16::MAX {
                r
            } else {
                r * self.ec.d + d + 1
            };
            let mut blocker = usize::MAX;
            for &c in keys {
                let p = position(c, cand_seed, &self.g, self.ec.d);
                if self.used.get(p) {
                    blocker = self.owner[p & CYC_MASK] as usize;
                    break;
                }
            }
            if blocker == usize::MAX || blocker == forbidden {
                continue;
            }
            let ysz = self.bb[blocker + 1] - self.bb[blocker];
            if ysz > self.ec.max_evict_size {
                continue;
            }
            if self.min_base(blocker) < self.value_to_clear {
                continue;
            }
            cands += 1;
            let yseed = self.seeds[blocker];
            self.unmark(blocker);
            self.mark(b, cand_seed);
            if self.place(blocker, depth - 1, b) {
                self.evictions += 1;
                if depth == self.ec.depth {
                    self.repaired += 1;
                }
                return true;
            }
            // undo
            self.unmark(b);
            self.mark(blocker, yseed);
        }
        if !self.ec.sacrifice || depth != self.ec.depth {
            return false;
        }
        // Sacrifice: evict a smaller bucket and bump it if it cannot be re-placed.
        let mut sac: Vec<(usize, usize, u16, u16, usize)> = Vec::new();
        for &(sum, r, d) in &list {
            let mut blocker = usize::MAX;
            for &c in keys {
                let p = self.g.slice_begin(c) + offset(c, r, &self.g) + d as usize;
                if self.used.get(p) {
                    blocker = self.owner[p & CYC_MASK] as usize;
                    break;
                }
            }
            if blocker == usize::MAX {
                continue;
            }
            let ysz = self.bb[blocker + 1] - self.bb[blocker];
            if ysz >= k || self.min_base(blocker) < self.value_to_clear {
                continue;
            }
            sac.push((ysz, sum, r, d, blocker));
        }
        sac.sort_unstable();
        if let Some(&(_, _, r, d, blocker)) = sac.first() {
            self.unmark(blocker);
            self.mark(b, r * self.ec.d + d + 1);
            if !self.place(blocker, depth - 1, b) {
                self.sacrificed += 1;
            }
            return true;
        }
        false
    }
}

pub struct EvictLevel {
    pub geom: Geom,
    pub seeds: Vec<u16>,
    pub bucket_begin: Vec<usize>,
    pub bumped_keys: usize,
    pub evictions: usize,
    pub repaired: usize,
    pub self_coll: usize,
}

pub fn build_level(hashes: &[u64], m: usize, conf: &Conf, ec: &EvictConf) -> EvictLevel {
    let n = hashes.len();
    let mut conf = conf.clone();
    conf.extra = if ec.d == 0 { 0 } else { ec.d as usize - 1 };
    let g = Geom::new(n, m, &conf);
    let nb = g.buckets;
    let mut bucket_begin = vec![0usize; nb + 1];
    for &c in hashes {
        bucket_begin[g.bucket(c) + 1] += 1;
    }
    for i in 0..nb {
        bucket_begin[i + 1] += bucket_begin[i];
    }
    let mut st = State {
        hashes,
        bb: &bucket_begin,
        g,
        ec: *ec,
        used: Cyclic::default(),
        owner: vec![0; CYC_BITS],
        seeds: vec![0; nb],
        value_to_clear: 0,
        evictions: 0,
        repaired: 0,
        self_coll: 0,
        sacrificed: 0,
    };
    let size = |b: usize| bucket_begin[b + 1] - bucket_begin[b];
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
    let mut in_heap = vec![false; nb];
    let mut span_begin = 0usize;
    while span_begin < nb && size(span_begin) == 0 {
        span_begin += 1;
    }
    let mut bumped_keys = 0;
    if span_begin < nb {
        let slice_begin_of = |b: usize| g.slice_begin(hashes[bucket_begin[b]]);
        st.value_to_clear = slice_begin_of(span_begin);
        let span_end = |sb: usize| (sb + conf.window).min(nb);
        for b in span_begin..span_end(span_begin) {
            if size(b) != 0 {
                heap.push((conf.eval(b, size(b)), Reverse(b)));
                in_heap[b] = true;
            }
        }
        while let Some((_, Reverse(b))) = heap.pop() {
            in_heap[b] = false;
            if !st.place(b, ec.depth, usize::MAX) {
                bumped_keys += size(b);
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
                // Keep a margin so that evictable buckets remain tracked:
                // clear only below the min base of span_begin minus nothing.
                let end = slice_begin_of(span_begin);
                while st.value_to_clear < end {
                    st.used.clear(st.value_to_clear);
                    st.value_to_clear += 1;
                }
                for b2 in old_end..span_end(span_begin) {
                    if size(b2) != 0 {
                        heap.push((conf.eval(b2, size(b2)), Reverse(b2)));
                        in_heap[b2] = true;
                    }
                }
            }
        }
    }
    // Bumped keys may have been changed by evictions: recount.
    let bumped: usize = (0..nb).filter(|&b| st.seeds[b] == 0).map(size).sum();
    debug_assert!(bumped <= bumped_keys);
    EvictLevel {
        geom: g,
        seeds: st.seeds,
        bucket_begin: bucket_begin.clone(),
        bumped_keys: bumped,
        evictions: st.evictions,
        repaired: st.repaired,
        self_coll: st.sacrificed,
    }
}

pub struct Report {
    pub n: usize,
    /// keys, buckets, bumped, evictions, repaired, self collisions
    pub levels: Vec<(usize, usize, usize, usize, usize, usize)>,
    pub seed_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl Report {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

pub fn mphf(n: usize, conf: &Conf, ec: &EvictConf, seed: u64, verify: bool) -> Report {
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
        let hashes: Vec<u64> = h.iter().map(|x| x.0).collect();
        let k = hashes.len();
        let lv = build_level(&hashes, k, conf, ec);
        if verify {
            let mut seen = vec![false; k];
            for b in 0..lv.geom.buckets {
                let s = lv.seeds[b];
                if s == 0 {
                    continue;
                }
                for &c in &hashes[lv.bucket_begin[b]..lv.bucket_begin[b + 1]] {
                    let p = position(c, s, &lv.geom, ec.d);
                    assert!(p < k, "out of range");
                    assert!(!seen[p], "collision at {p}");
                    seen[p] = true;
                }
            }
        }
        seed_bits += (lv.geom.buckets as f64)
            * if ec.d == 0 {
                8.0
            } else {
                ((ec.r as u32 * ec.d as u32 + 1) as f64).log2().ceil()
            };
        total_range += k;
        let mut next = Vec::with_capacity(lv.bumped_keys);
        for b in 0..lv.geom.buckets {
            if lv.seeds[b] == 0 {
                for j in lv.bucket_begin[b]..lv.bucket_begin[b + 1] {
                    next.push(h[j].1);
                }
            }
        }
        levels.push((
            k,
            lv.geom.buckets,
            lv.bumped_keys,
            lv.evictions,
            lv.repaired,
            lv.self_coll,
        ));
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
