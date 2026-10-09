//! CONSENSUS-style search for bump-free PHast-R placements.
//!
//! Keys are placed on a cyclic range of m = n slots: a key with hashes (h, o)
//! has slice beginning sb = ⌊hm / 2⁶⁴⌋, and the seed (r, d) places it at
//! (sb + ω_r + d) mod m, where ω_r is the r-th field of log₂ L bits of
//! mix(o ⊕ H), and H is the chained hash of its bucket (see below).
//!
//! Buckets are processed in index order by a depth-first search: each bucket
//! tries its collision-free seeds in increasing order of the sum of their
//! positions (or in seed order), and a placement is accepted only if all
//! slots that no later bucket can reach are occupied (the closure check). If
//! no seed is accepted, the search backtracks to the previous bucket.
//!
//! With chaining, H is a hash of the seeds of the previous `chain` buckets,
//! so that changing a seed rerandomizes the following buckets, as in
//! CONSENSUS (Lehmann, Sanders, Walzer, and Ziegler, 2025). Without
//! chaining, H = 0 and the placements of a bucket do not depend on the seeds
//! of the other buckets.

use std::time::Instant;

/// The high 64 bits of a 128-bit product.
#[inline(always)]
pub fn mul_hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// The SplitMix64 finalizer.
#[inline(always)]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[derive(Clone, Copy, Debug)]
pub struct DfsConf {
    pub seed_bits: u32,
    pub log2_layouts: u32,
    pub log2_slice_len: u32,
    pub lambda: f64,
    /// Number of previous seeds hashed into H (0: no chaining).
    pub chain: u32,
    /// Try seeds by increasing sum of positions (otherwise, in seed order).
    pub min_sum: bool,
    /// The search is aborted after this many placements per bucket.
    pub max_nodes_per_bucket: u64,
}

#[derive(Debug, Default, Clone)]
pub struct DfsStats {
    /// Nonempty buckets.
    pub layers: usize,
    /// Placements tried.
    pub nodes: u64,
    /// Candidate lists computed.
    pub searches: u64,
    /// Steps back to the previous bucket.
    pub backtracks: u64,
    /// Placements rejected by the closure check.
    pub closure_fails: u64,
    /// Maximum distance between the deepest bucket reached and the bucket
    /// the search backtracked to.
    pub max_retreat: usize,
    /// Histogram of retreats at each backtrack (bin i: [2^i . . 2^(i+1)) ).
    pub retreat_hist: [u64; 24],
    /// Placements tried in the final buckets, whose slices wrap around.
    pub endgame_nodes: u64,
    pub success: bool,
    pub ns_per_key: f64,
    /// Deepest layer reached.
    pub deepest: usize,
}

/// A cyclic bit vector whose length is a multiple of 64.
pub(crate) struct Ring {
    pub(crate) w: Vec<u64>,
}

impl Ring {
    #[inline(always)]
    pub(crate) fn set(&mut self, p: usize) {
        self.w[p / 64] |= 1 << (p % 64);
    }
    #[inline(always)]
    pub(crate) fn clear(&mut self, p: usize) {
        self.w[p / 64] &= !(1 << (p % 64));
    }
    /// The 64 bits starting at p (cyclically).
    #[inline(always)]
    pub(crate) fn get64(&self, p: usize) -> u64 {
        let c = p / 64;
        let b = p % 64;
        let lo = self.w[c];
        if b == 0 {
            return lo;
        }
        let hi = self.w[if c + 1 == self.w.len() { 0 } else { c + 1 }];
        (lo >> b) | (hi << (64 - b))
    }
    /// Number of zeros in [lo . . hi), with lo <= hi <= m.
    pub(crate) fn zeros(&self, lo: usize, hi: usize) -> usize {
        let mut z = 0;
        let mut p = lo;
        while p < hi {
            let take = (hi - p).min(64);
            let x = self.get64(p) | if take == 64 { 0 } else { !0u64 << take };
            z += x.count_zeros() as usize;
            p += take;
        }
        z
    }
}

/// The keys: hashes sorted by h, with precomputed slice beginnings.
pub struct Keys {
    pub sb: Vec<usize>,
    pub o: Vec<u64>,
    /// Boundaries of the nonempty buckets.
    pub layer_begin: Vec<usize>,
    pub buckets: usize,
    pub m: usize,
}

impl Keys {
    /// Generates n random keys on m = n slots (n must be a multiple of 64).
    pub fn new(n: usize, lambda: f64, seed: u64) -> Self {
        Self::with_load(n, lambda, 1.0, seed)
    }

    /// Generates n random keys on m = ⌈n / α⌉ slots, rounded up to a
    /// multiple of 64 (n must be a multiple of 64).
    pub fn with_load(n: usize, lambda: f64, alpha: f64, seed: u64) -> Self {
        assert!(n % 64 == 0);
        let m = ((n as f64 / alpha).ceil() as usize).div_ceil(64) * 64;
        let buckets = ((n as f64 / lambda).round() as usize).max(1);
        let mut h: Vec<(u64, u64)> = (0..n as u64)
            .map(|i| {
                let x = mix64(i ^ mix64(seed));
                (x, mix64(x ^ 0x5851f42d4c957f2d))
            })
            .collect();
        h.sort_unstable();
        let sb = h.iter().map(|x| mul_hi(x.0, m as u64) as usize).collect();
        let o = h.iter().map(|x| x.1).collect();
        let mut layer_begin = vec![0];
        for i in 1..n {
            if mul_hi(h[i].0, buckets as u64) != mul_hi(h[i - 1].0, buckets as u64) {
                layer_begin.push(i);
            }
        }
        layer_begin.push(n);
        Self {
            sb,
            o,
            layer_begin,
            buckets,
            m,
        }
    }

    pub fn layers(&self) -> usize {
        self.layer_begin.len() - 1
    }

    /// The range of X(t) = t − #{keys with sb < t}, t ∈ [0 . . m]. For
    /// m = n, X(b) − X(a) is the deficit of keys of the cyclic interval
    /// [a . . b), so by Hall's condition a bump-free placement with windows
    /// of W slots can exist only if the range is less than W.
    pub fn bridge_range(&self) -> usize {
        let mut x = 0i64;
        let (mut lo, mut hi) = (0i64, 0i64);
        let mut j = 0;
        for t in 0..self.m {
            x += 1;
            while j < self.sb.len() && self.sb[j] == t {
                x -= 1;
                j += 1;
            }
            lo = lo.min(x);
            hi = hi.max(x);
        }
        (hi - lo) as usize
    }
}

struct Search<'a> {
    keys: &'a Keys,
    c: DfsConf,
    ring: Ring,
    seeds: Vec<u32>,
    shifts: usize,
    l_mask: u64,
    wrap: usize,
    // scratch
    t: Vec<u64>,
    bases: Vec<usize>,
    cands: Vec<(usize, u32)>,
}

impl<'a> Search<'a> {
    fn chain_hash(&self, j: usize) -> u64 {
        if self.c.chain == 0 {
            return 0;
        }
        let mut w = 0x2545f4914f6cdd1d;
        for t in 1..=self.c.chain as usize {
            if j >= t {
                w = mix64(w ^ self.seeds[j - t] as u64 ^ ((t as u64) << 40));
            }
        }
        w
    }

    /// Loads the offset sources of layer j.
    fn load(&mut self, j: usize) {
        let h = self.chain_hash(j);
        let (a, b) = (self.keys.layer_begin[j], self.keys.layer_begin[j + 1]);
        self.t.clear();
        for x in a..b {
            let o = self.keys.o[x];
            self.t
                .push(if self.c.chain == 0 { o } else { mix64(o ^ h) });
        }
    }

    #[inline(always)]
    fn base(&self, x: usize, i: usize, r: u32) -> usize {
        self.keys.sb[x] + ((self.t[i] >> (r * self.c.log2_slice_len)) & self.l_mask) as usize
    }

    #[inline(always)]
    fn reduce(&self, p: usize) -> usize {
        if p >= self.keys.m { p - self.keys.m } else { p }
    }

    /// Computes the sorted list of collision-free seeds of layer j (after
    /// [`load`](Self::load)).
    fn candidates(&mut self, j: usize) {
        let a = self.keys.layer_begin[j];
        let k = self.t.len();
        self.cands.clear();
        for r in 0..(1u32 << self.c.log2_layouts) {
            self.bases.clear();
            let mut sum = 0;
            for i in 0..k {
                let b = self.base(a + i, i, r);
                sum += b;
                self.bases.push(b);
            }
            let bs = &self.bases;
            if (1..k).any(|i| bs[..i].contains(&bs[i])) {
                continue;
            }
            let mut chunk = 0;
            while chunk < self.shifts {
                let mut u = 0u64;
                for &b in bs.iter() {
                    u |= self.ring.get64(self.reduce(b + chunk));
                }
                if self.shifts - chunk < 64 {
                    u |= !0u64 << (self.shifts - chunk);
                }
                let mut free = !u;
                while free != 0 {
                    let d = chunk + free.trailing_zeros() as usize;
                    free &= free - 1;
                    let seed = r as usize * self.shifts + d;
                    let key = if self.c.min_sum { sum + d * k } else { seed };
                    self.cands.push((key, seed as u32));
                }
                chunk += 64;
            }
        }
        self.cands.sort_unstable();
    }

    fn place(&mut self, j: usize, seed: u32, set: bool) {
        let a = self.keys.layer_begin[j];
        let r = seed / self.shifts as u32;
        let d = seed as usize % self.shifts;
        for i in 0..self.t.len() {
            let p = self.reduce(self.base(a + i, i, r) + d);
            if set {
                self.ring.set(p);
            } else {
                self.ring.clear(p);
            }
        }
    }

    /// Checks that the slots closed by layer j are occupied.
    fn closure_ok(&self, j: usize) -> bool {
        let layers = self.keys.layers();
        if j + 1 == layers {
            return true;
        }
        let lo = self.keys.sb[self.keys.layer_begin[j]].max(self.wrap);
        let hi = self.keys.sb[self.keys.layer_begin[j + 1]];
        lo >= hi || self.ring.zeros(lo, hi) == 0
    }
}

/// Runs the search; returns statistics and, on success, the seeds of the
/// layers.
pub fn run(keys: &Keys, c: &DfsConf) -> (DfsStats, Vec<u32>) {
    let start = Instant::now();
    let layers = keys.layers();
    let shifts = (1usize << c.seed_bits) >> c.log2_layouts;
    let l = 1usize << c.log2_slice_len;
    let mut s = Search {
        keys,
        c: *c,
        ring: Ring {
            w: vec![0; keys.m / 64],
        },
        seeds: vec![0; layers],
        shifts,
        l_mask: l as u64 - 1,
        wrap: l + shifts - 1,
        t: vec![],
        bases: vec![],
        cands: vec![],
    };
    let endgame = keys
        .layer_begin
        .iter()
        .position(|&x| x < keys.sb.len() && keys.sb[x] + 2 * (l + shifts) >= keys.m)
        .unwrap_or(layers);
    let mut st = DfsStats {
        layers,
        ..Default::default()
    };
    let budget = c.max_nodes_per_bucket.saturating_mul(layers as u64);
    // Index of the next candidate to try for each layer
    let mut next = vec![0u32; layers + 1];
    let mut j = 0;
    let mut deepest = 0;
    while j < layers {
        st.deepest = st.deepest.max(j);
        if st.nodes > budget {
            st.ns_per_key = start.elapsed().as_nanos() as f64 / keys.m as f64;
            return (st, vec![]);
        }
        s.load(j);
        s.candidates(j);
        st.searches += 1;
        let mut ok = false;
        let mut idx = next[j] as usize;
        while idx < s.cands.len() {
            let seed = s.cands[idx].1;
            s.place(j, seed, true);
            st.nodes += 1;
            if j >= endgame {
                st.endgame_nodes += 1;
            }
            if s.closure_ok(j) {
                s.seeds[j] = seed;
                next[j] = idx as u32;
                ok = true;
                break;
            }
            st.closure_fails += 1;
            s.place(j, seed, false);
            idx += 1;
        }
        if ok {
            j += 1;
            next[j] = 0;
            deepest = deepest.max(j);
        } else {
            if j == 0 {
                st.ns_per_key = start.elapsed().as_nanos() as f64 / keys.m as f64;
                return (st, vec![]);
            }
            j -= 1;
            st.backtracks += 1;
            let retreat = deepest - j;
            st.max_retreat = st.max_retreat.max(retreat);
            st.retreat_hist[(usize::BITS - 1 - retreat.max(1).leading_zeros()) as usize] += 1;
            s.load(j);
            let seed = s.seeds[j];
            s.place(j, seed, false);
            next[j] += 1;
        }
    }
    st.success = true;
    st.ns_per_key = start.elapsed().as_nanos() as f64 / keys.m as f64;
    (st, s.seeds)
}

/// Verifies that the seeds define a bijection.
pub fn verify(keys: &Keys, c: &DfsConf, seeds: &[u32]) -> bool {
    let layers = keys.layers();
    let shifts = (1usize << c.seed_bits) >> c.log2_layouts;
    let l = 1usize << c.log2_slice_len;
    let mut s = Search {
        keys,
        c: *c,
        ring: Ring {
            w: vec![0; keys.m / 64],
        },
        seeds: seeds.to_vec(),
        shifts,
        l_mask: l as u64 - 1,
        wrap: 0,
        t: vec![],
        bases: vec![],
        cands: vec![],
    };
    let mut seen = vec![false; keys.m];
    for j in 0..layers {
        s.load(j);
        let a = keys.layer_begin[j];
        let r = seeds[j] / shifts as u32;
        let d = seeds[j] as usize % shifts;
        for i in 0..s.t.len() {
            let p = s.reduce(s.base(a + i, i, r) + d);
            if seen[p] {
                return false;
            }
            seen[p] = true;
        }
    }
    seen.iter().all(|&x| x)
}
