//! PHast-R placement with unbounded repair.
//!
//! Same cyclic model as [`crate::dfs`] (without chaining): buckets are
//! processed in index order or by PHast-R priority, each bucket takes its
//! minimum-sum free seed, and a bucket with no free seed takes the seed
//! minimizing the sum of the squared sizes of the buckets it collides with
//! (PtrHash-style), evicting them; evicted buckets are placed in the same way
//! (last in, first out) until no bucket is pending. Recently evicted buckets
//! are penalized to avoid cycles. An episode (the repair started by a failing
//! bucket) that exceeds the eviction cap gives up, and pending buckets are
//! bumped.

use crate::dfs::{Keys, Ring, mix64};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct WalkConf {
    pub seed_bits: u32,
    pub log2_patterns: u32,
    pub log2_slice_len: u32,
    /// Process buckets by PHast-R priority in a window of 256 buckets
    /// (otherwise, in index order).
    pub priority: bool,
    /// Maximum number of evictions per episode.
    pub cap: u64,
    /// Cost per slot of the age of an evicted bucket, that is, of the
    /// distance between its first slice beginning and that of the bucket
    /// that started the episode (in units of 1/64 of an evicted key).
    pub age_cost: u64,
    /// Cost per evicted bucket (in units of 1/64 of an evicted key).
    pub owner_cost: u64,
    /// Buckets whose first slice beginning is more than this many slots
    /// behind the frontier cannot be evicted (0: no limit).
    pub zone: usize,
    /// Evict at most one bucket per placement.
    pub single_owner: bool,
}

#[derive(Debug, Default, Clone)]
pub struct WalkStats {
    pub layers: usize,
    /// Buckets that found no free seed when first placed (episodes).
    pub episodes: u64,
    /// Total evictions.
    pub evictions: u64,
    /// Placements of evicted buckets that found a free seed.
    pub direct_replacements: u64,
    /// Maximum evictions in an episode.
    pub max_episode: u64,
    /// Histogram of the evictions per episode (bin i: [2^i . . 2^(i+1)) ).
    pub episode_hist: [u64; 32],
    /// Keys bumped by episodes that exceeded the cap.
    pub bumped_keys: usize,
    /// Empty slots at the end.
    pub holes: usize,
    pub ns_per_key: f64,
    /// Capped episodes by position of the starting bucket (tenths).
    pub capped_by_pos: [u64; 10],
}

const TABU: usize = 8;

struct State<'a> {
    keys: &'a Keys,
    c: WalkConf,
    ring: Ring,
    owner: Vec<u32>,
    seeds: Vec<u32>,
    shifts: usize,
    l_mask: u64,
    tabu: [u32; TABU],
    tabu_pos: usize,
    rng: u64,
    bases: Vec<usize>,
    counts: Vec<u8>,
    /// First slice beginning of the bucket that started the episode.
    frontier: usize,
}

const NONE: u32 = u32::MAX;

impl<'a> State<'a> {
    #[inline(always)]
    fn size(&self, j: usize) -> usize {
        self.keys.layer_begin[j + 1] - self.keys.layer_begin[j]
    }

    #[inline(always)]
    fn reduce(&self, p: usize) -> usize {
        if p >= self.keys.m { p - self.keys.m } else { p }
    }

    #[inline(always)]
    fn base(&self, x: usize, r: u32) -> usize {
        self.keys.sb[x] + ((self.keys.o[x] >> (r * self.c.log2_slice_len)) & self.l_mask) as usize
    }

    fn pos(&self, x: usize, seed: u32) -> usize {
        let r = seed / self.shifts as u32;
        let d = seed as usize % self.shifts;
        self.reduce(self.base(x, r) + d)
    }

    fn mark(&mut self, j: usize, seed: u32) {
        for x in self.keys.layer_begin[j]..self.keys.layer_begin[j + 1] {
            let p = self.pos(x, seed);
            debug_assert!(self.ring.w[p / 64] >> (p % 64) & 1 == 0);
            self.ring.set(p);
            self.owner[p] = j as u32;
        }
        self.seeds[j] = seed;
    }

    fn unmark(&mut self, j: usize) {
        let seed = self.seeds[j];
        for x in self.keys.layer_begin[j]..self.keys.layer_begin[j + 1] {
            let p = self.pos(x, seed);
            self.ring.clear(p);
        }
        self.seeds[j] = NONE;
    }

    /// Loads the bases of pattern r; returns their sum, or None if two keys
    /// collide.
    fn load(&mut self, j: usize, r: u32) -> Option<usize> {
        self.bases.clear();
        let mut sum = 0;
        for x in self.keys.layer_begin[j]..self.keys.layer_begin[j + 1] {
            let b = self.base(x, r);
            if self.bases.contains(&b) {
                return None;
            }
            sum += b;
            self.bases.push(b);
        }
        Some(sum)
    }

    /// Minimum-sum free seed.
    fn search(&mut self, j: usize) -> Option<u32> {
        let k = self.size(j);
        let mut best = (usize::MAX, NONE);
        for r in 0..(1u32 << self.c.log2_patterns) {
            let Some(sum) = self.load(j, r) else { continue };
            let mut chunk = 0;
            while chunk < self.shifts {
                if sum + chunk * k >= best.0 {
                    break;
                }
                let mut u = 0u64;
                for &b in &self.bases {
                    u |= self.ring.get64(self.reduce(b + chunk));
                }
                if self.shifts - chunk < 64 {
                    u |= !0u64 << (self.shifts - chunk);
                }
                if u != !0 {
                    let d = chunk + u.trailing_ones() as usize;
                    if sum + d * k < best.0 {
                        best = (sum + d * k, (r as usize * self.shifts + d) as u32);
                    }
                    break;
                }
                chunk += 64;
            }
        }
        (best.1 != NONE).then_some(best.1)
    }

    /// Seed minimizing the eviction cost (needs at least one collision).
    fn evict_seed(&mut self, j: usize) -> u32 {
        let k = self.size(j);
        let mut best = (u64::MAX, NONE);
        let mut owners: Vec<u32> = Vec::with_capacity(k);
        for r in 0..(1u32 << self.c.log2_patterns) {
            if self.load(j, r).is_none() {
                continue;
            }
            let mut chunk = 0;
            while chunk < self.shifts {
                let lim = (self.shifts - chunk).min(64);
                // Number of blocked keys for each shift
                self.counts.clear();
                self.counts.resize(64, 0);
                for &b in &self.bases {
                    let mut w = self.ring.get64(self.reduce(b + chunk));
                    while w != 0 {
                        self.counts[w.trailing_zeros() as usize] += 1;
                        w &= w - 1;
                    }
                }
                for d in 0..lim {
                    let blocked = self.counts[d] as u64;
                    // Each evicted key costs at least one squared unit
                    if blocked == 0 || blocked * 64 >= best.0 {
                        continue;
                    }
                    owners.clear();
                    for &b in &self.bases {
                        let p = self.reduce(b + chunk + d);
                        if self.ring.w[p / 64] >> (p % 64) & 1 != 0 {
                            let o = self.owner[p];
                            if !owners.contains(&o) {
                                owners.push(o);
                            }
                        }
                    }
                    if self.c.single_owner && owners.len() > 1 {
                        continue;
                    }
                    let mut cost = owners.len() as u64 * self.c.owner_cost;
                    let sb_j = self.frontier;
                    for &o in &owners {
                        let s = self.size(o as usize) as u64;
                        cost += s * s * 64;
                        let sb_o = self.keys.sb[self.keys.layer_begin[o as usize]];
                        // Cyclic distance, if the evicted bucket comes first
                        let age = (sb_j + self.keys.m - sb_o) % self.keys.m;
                        if age < self.keys.m / 2 {
                            cost += age as u64 * self.c.age_cost;
                            if self.c.zone != 0 && age > self.c.zone {
                                cost = u64::MAX;
                                break;
                            }
                        }
                        if self.tabu.contains(&o) {
                            cost += 1 << 20;
                        }
                    }
                    if cost == u64::MAX {
                        continue;
                    }
                    // Random tie breaking
                    self.rng = mix64(self.rng);
                    cost += self.rng & 63;
                    if cost < best.0 {
                        best = (cost, ((r as usize) * self.shifts + chunk + d) as u32);
                    }
                }
                chunk += 64;
            }
        }
        best.1
    }

    /// Places layer j; returns the number of evictions, or None if the cap
    /// was exceeded (pending buckets are bumped).
    fn episode(&mut self, j: usize, st: &mut WalkStats) -> bool {
        if let Some(seed) = self.search(j) {
            self.mark(j, seed);
            return true;
        }
        st.episodes += 1;
        self.frontier = self.keys.sb[self.keys.layer_begin[j]];
        let mut stack = vec![j as u32];
        let mut ev = 0u64;
        let mut first = true;
        let trace =
            std::env::var("WALK_TRACE").is_ok() && st.capped_by_pos.iter().sum::<u64>() == 0;
        let mut log: Vec<String> = vec![];
        while let Some(b) = stack.pop() {
            let b = b as usize;
            let rel = self.keys.sb[self.keys.layer_begin[b]] as i64 - self.frontier as i64;
            if !first {
                if let Some(seed) = self.search(b) {
                    self.mark(b, seed);
                    st.direct_replacements += 1;
                    if trace {
                        log.push(format!("{b}(k{} {rel:+}) direct", self.size(b)));
                    }
                    continue;
                }
            }
            first = false;
            if ev >= self.c.cap {
                st.bumped_keys += self.size(b);
                for &x in &stack {
                    st.bumped_keys += self.size(x as usize);
                }
                break;
            }
            let seed = self.evict_seed(b);
            if seed == NONE {
                // Self-colliding on every pattern
                st.bumped_keys += self.size(b);
                continue;
            }
            if trace {
                let mut ow = vec![];
                for x in self.keys.layer_begin[b]..self.keys.layer_begin[b + 1] {
                    let p = self.pos(x, seed);
                    if self.ring.w[p / 64] >> (p % 64) & 1 != 0 {
                        let o = self.owner[p] as usize;
                        ow.push(format!(
                            "{o}(k{} {:+})",
                            self.size(o),
                            self.keys.sb[self.keys.layer_begin[o]] as i64 - self.frontier as i64
                        ));
                    }
                }
                log.push(format!(
                    "{b}(k{} {rel:+}) evicts {} [stack {}]",
                    self.size(b),
                    ow.join(","),
                    stack.len()
                ));
            }
            for x in self.keys.layer_begin[b]..self.keys.layer_begin[b + 1] {
                let p = self.pos(x, seed);
                if self.ring.w[p / 64] >> (p % 64) & 1 != 0 {
                    let o = self.owner[p];
                    self.unmark(o as usize);
                    stack.push(o);
                    self.tabu[self.tabu_pos] = o;
                    self.tabu_pos = (self.tabu_pos + 1) % TABU;
                    ev += 1;
                }
            }
            self.mark(b, seed);
        }
        st.evictions += ev;
        st.max_episode = st.max_episode.max(ev);
        st.episode_hist[(u64::BITS - 1 - ev.max(1).leading_zeros()) as usize] += 1;
        if ev >= self.c.cap {
            if trace {
                for l in log.iter().take(40) {
                    eprintln!("{l}");
                }
                eprintln!("...");
                for l in log.iter().skip(log.len().saturating_sub(40)) {
                    eprintln!("{l}");
                }
            }
            st.capped_by_pos[j * 10 / self.keys.layers()] += 1;
        }
        ev < self.c.cap
    }
}

/// Default PHast-R weights for (S = 8, L = 512).
const WEIGHTS: [i64; 7] = [-48137, 68016, 105189, 121129, 132794, 140850, 145685];

fn priority(size: usize, b: usize) -> i64 {
    let w = if size <= 7 {
        WEIGHTS[size - 1]
    } else {
        WEIGHTS[6] + (WEIGHTS[6] - WEIGHTS[5]) * (size - 7) as i64
    };
    w - 1024 * b as i64
}

pub fn run(keys: &Keys, c: &WalkConf) -> WalkStats {
    let start = Instant::now();
    let layers = keys.layers();
    let shifts = (1usize << c.seed_bits) >> c.log2_patterns;
    let mut s = State {
        keys,
        c: *c,
        ring: Ring {
            w: vec![0; keys.m / 64],
        },
        owner: vec![NONE; keys.m],
        seeds: vec![NONE; layers],
        shifts,
        l_mask: (1u64 << c.log2_slice_len) - 1,
        tabu: [NONE; TABU],
        tabu_pos: 0,
        rng: 0x9e3779b97f4a7c15,
        bases: vec![],
        counts: vec![],
        frontier: 0,
    };
    let mut st = WalkStats {
        layers,
        ..Default::default()
    };
    if c.priority {
        // Layers are nonempty buckets, so the window is in layers
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        const WINDOW: usize = 256;
        let mut heap = BinaryHeap::new();
        let mut done = vec![false; layers];
        let mut next = 0;
        let mut low = 0;
        while low < layers {
            while next < layers && next < low + WINDOW {
                heap.push((priority(s.size(next), next), Reverse(next)));
                next += 1;
            }
            let (_, Reverse(j)) = heap.pop().unwrap();
            s.episode(j, &mut st);
            done[j] = true;
            while low < layers && done[low] {
                low += 1;
            }
        }
    } else {
        for j in 0..layers {
            s.episode(j, &mut st);
        }
    }
    st.holes = s
        .ring
        .w
        .iter()
        .map(|w| w.count_zeros() as usize)
        .sum::<usize>();
    st.ns_per_key = start.elapsed().as_nanos() as f64 / keys.sb.len() as f64;
    // Verify
    let mut seen = vec![false; keys.m];
    for j in 0..layers {
        if s.seeds[j] == NONE {
            continue;
        }
        for x in keys.layer_begin[j]..keys.layer_begin[j + 1] {
            let p = s.pos(x, s.seeds[j]);
            assert!(!seen[p], "collision");
            seen[p] = true;
        }
    }
    st
}
