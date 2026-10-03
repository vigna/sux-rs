//! Faithful re-implementation of PHast+ (ShiftOnly, no wrapping) with
//! instrumentation.

use crate::{Cyclic, ef_bits, mix64, mul_hi};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct Conf {
    pub s: u32,
    pub lambda: f64,
    pub l: usize,
    pub window: usize,
    pub weights: [i64; 7],
    /// maximum shift added on top of the slice (usize::MAX = 2^s - 2)
    pub extra: usize,
    /// mixed bucket layout: q big buckets of mix_big keys + one of mix_small keys per group
    pub mix_q: usize,
    pub mix_big: f64,
    pub mix_small: f64,
}

/// Weights from the reference ShiftOnly::bucket_evaluator.
pub fn shift_only_weights(s: u32, l: usize) -> [i64; 7] {
    let w: [i32; 7] = if l <= 256 {
        match (s, l) {
            (..=6, ..=128) => [-98439, 68040, 81130, 86896, 91188, 93897, 296481],
            (..=6, _) => [-81980, 50520, 90817, 106897, 116472, 123937, 287280],
            (_, ..=128) => [-173163, 58917, 73926, 83423, 88222, 92168, 206758],
            (..=7, _) => [-85977, 81531, 98837, 107586, 113333, 117710, 120656],
            (_, _) => [-85787, 84108, 99553, 107291, 112859, 117377, 119965],
        }
    } else {
        match (s, l) {
            (..=7, ..=512) => [-95834, 38499, 103035, 124756, 137603, 147839, 155448],
            (_, ..=512) => [-68137, 80516, 110189, 123629, 132794, 140850, 145685],
            (..=8, ..=1024) => [-49776, 28610, 120514, 154976, 177328, 193499, 204936],
            (..=8, ..=2048) => [-14014, -11926, 63698, 144877, 194056, 353593, 360338],
            (9, ..=1024) => [-60439, 49207, 121850, 149181, 166713, 179181, 187815],
            (9, ..=2048) => [48168, 48328, 132443, 197796, 234543, 260358, 279164],
            (10, ..=1024) => [-4759, 9930, 87924, 125082, 143308, 165460, 165095],
            (10, ..=2048) => [-3419, 8042, 98860, 145429, 176433, 198538, 214441],
            (_, ..=1024) => [-1560, 25555, 96323, 156791, 189688, 201315, 198828],
            (11, ..=2048) => [-294, 2300, 161956, 227418, 278332, 344537, 342726],
            (11, ..=4096) => [-2674, 19194, 37310, 111428, 167443, 205425, 236469],
            (_, ..=2048) => [-1914, 10973, 70225, 173122, 240880, 305750, 293320],
            (_, ..=4096) => [-2651, -447, 16106, 163680, 223955, 353813, 339271],
            (_, _) => [-4309, -487, 21662, 26095, 83370, 157063, 543843],
        }
    };
    w.map(|x| x as i64)
}

pub fn shift_only_slice_len(s: u32) -> usize {
    match s {
        ..=4 => 128,
        ..=7 => 256,
        8 => 512,
        9 => 1024,
        10 => 2048,
        11 => 4096,
        _ => 8192,
    }
}

impl Conf {
    pub fn plus(s: u32, lambda: f64) -> Self {
        let l = shift_only_slice_len(s);
        Self {
            s,
            lambda,
            l,
            window: 256,
            weights: shift_only_weights(s, l),
            extra: usize::MAX,
            mix_q: 0,
            mix_big: 0.0,
            mix_small: 0.0,
        }
    }

    #[inline(always)]
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

/// Geometry of a level.
#[derive(Clone, Copy, Debug)]
pub struct Geom {
    pub m: usize,
    pub buckets: usize,
    pub num_slices: usize,
    pub l_mask: u64,
    pub extra: usize,
    /// mixed layout: (groups, q, big width, group width) in 1/256 key units; q == 0 means plain
    pub mix: (u64, u64, u64, u64),
}

impl Geom {
    pub fn new(n: usize, m: usize, conf: &Conf) -> Self {
        let extra = if conf.extra != usize::MAX {
            conf.extra
        } else {
            (1usize << conf.s) - 2
        };
        let buckets = 1.max((n as f64 / conf.lambda).round() as usize);
        let l = conf
            .l
            .min((m.saturating_sub(extra) / 2 + 1).next_power_of_two());
        let mut mix = (0, 0, 0, 0);
        let mut buckets = buckets;
        if conf.mix_q > 0 {
            let bw = (conf.mix_big * 256.0).round() as u64;
            let gw = conf.mix_q as u64 * bw + (conf.mix_small * 256.0).round() as u64;
            let groups = ((n as f64 * 256.0) / gw as f64).round().max(1.0) as u64;
            mix = (groups, conf.mix_q as u64, bw, gw);
            buckets = groups as usize * (conf.mix_q + 1);
        }
        Self {
            m,
            buckets,
            num_slices: m + 1 - l - extra,
            l_mask: l as u64 - 1,
            extra,
            mix,
        }
    }
    #[inline(always)]
    pub fn bucket(&self, c: u64) -> usize {
        if self.mix.1 == 0 {
            return mul_hi(c, self.buckets as u64) as usize;
        }
        let (groups, q, bw, gw) = self.mix;
        let y = mul_hi(c, groups * gw);
        let g = y / gw;
        let r = y % gw;
        let j = (r / bw).min(q);
        (g * (q + 1) + j) as usize
    }
    #[inline(always)]
    pub fn slice_begin(&self, c: u64) -> usize {
        mul_hi(c, self.num_slices as u64) as usize
    }
    #[inline(always)]
    pub fn base(&self, c: u64) -> usize {
        self.slice_begin(c) + (c & self.l_mask) as usize
    }
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    /// [size] -> (buckets, bumped buckets)
    pub by_size: Vec<(usize, usize)>,
    /// histogram of seeds (index = seed)
    pub seed_hist: Vec<usize>,
    /// [size][seed] histogram
    pub seed_hist_by_size: Vec<Vec<usize>>,
    pub self_collisions: usize,
}

pub struct Level {
    pub geom: Geom,
    pub seeds: Vec<u16>,
    pub bucket_begin: Vec<usize>,
    pub bumped_keys: usize,
    pub stats: Stats,
}

/// Seed chooser abstraction for experiments.
pub trait Chooser {
    /// Returns the seed (0 = bump) and marks used values.
    fn choose(
        &mut self,
        used: &mut Cyclic<512>,
        keys: &[u64],
        g: &Geom,
        s: u32,
        stats: &mut Stats,
    ) -> u16;
    fn value(&self, c: u64, seed: u16, g: &Geom) -> usize;
}

/// The PHast+ ShiftOnly chooser.
pub struct ShiftOnly;

impl Chooser for ShiftOnly {
    #[inline]
    fn choose(
        &mut self,
        used: &mut Cyclic<512>,
        keys: &[u64],
        g: &Geom,
        s: u32,
        stats: &mut Stats,
    ) -> u16 {
        let mut bases: Vec<usize> = keys.iter().map(|&c| g.base(c)).collect();
        let last_shift = (1u16 << s) - 1;
        let mut shift = 0u16;
        while shift < last_shift {
            let mut u = 0u64;
            for &b in &bases {
                u |= used.get64(b + shift as usize);
            }
            if u != u64::MAX {
                bases.sort_unstable();
                if bases.windows(2).any(|w| w[0] == w[1]) {
                    stats.self_collisions += 1;
                    return 0;
                }
                let total = shift + u.trailing_ones() as u16;
                if total >= last_shift {
                    return 0;
                }
                for &b in &bases {
                    used.set(b + total as usize);
                }
                return total + 1;
            }
            shift += 64;
        }
        0
    }
    #[inline(always)]
    fn value(&self, c: u64, seed: u16, g: &Geom) -> usize {
        g.base(c) + seed as usize - 1
    }
}

/// Builds a level over sorted hashes with output range m.
pub fn build_level<C: Chooser>(hashes: &[u64], m: usize, conf: &Conf, chooser: &mut C) -> Level {
    let n = hashes.len();
    let g = Geom::new(n, m, conf);
    let nb = g.buckets;
    let mut bucket_begin = vec![0usize; nb + 1];
    for &c in hashes {
        bucket_begin[g.bucket(c) + 1] += 1;
    }
    for i in 0..nb {
        bucket_begin[i + 1] += bucket_begin[i];
    }
    let mut seeds = vec![0u16; nb];
    let mut stats = Stats::default();
    stats.by_size = vec![(0, 0); 64];
    stats.seed_hist = vec![0; 1 << conf.s];
    stats.seed_hist_by_size = vec![vec![0; 1 << conf.s]; 16];

    let size = |b: usize| bucket_begin[b + 1] - bucket_begin[b];
    let mut used = Cyclic::<512>::default();
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::with_capacity(conf.window);
    let mut in_heap = vec![false; nb]; // simple
    let mut span_begin = 0usize;
    while span_begin < nb && size(span_begin) == 0 {
        span_begin += 1;
    }
    if span_begin == nb {
        return Level {
            geom: g,
            seeds,
            bucket_begin,
            bumped_keys: 0,
            stats,
        };
    }
    let slice_begin_of = |b: usize| g.slice_begin(hashes[bucket_begin[b]]);
    let mut value_to_clear = slice_begin_of(span_begin);
    let span_end = |sb: usize| (sb + conf.window).min(nb);
    for b in span_begin..span_end(span_begin) {
        let sz = size(b);
        if sz != 0 {
            heap.push((conf.eval(b, sz), Reverse(b)));
            in_heap[b] = true;
        }
    }
    let mut bumped_keys = 0;
    while let Some((_, Reverse(b))) = heap.pop() {
        in_heap[b] = false;
        let keys = &hashes[bucket_begin[b]..bucket_begin[b + 1]];
        let seed = chooser.choose(&mut used, keys, &g, conf.s, &mut stats);
        seeds[b] = seed;
        let sz = keys.len();
        let si = sz.min(63);
        stats.by_size[si].0 += 1;
        if seed == 0 {
            stats.by_size[si].1 += 1;
            bumped_keys += sz;
        }
        stats.seed_hist[seed as usize] += 1;
        stats.seed_hist_by_size[sz.min(15)][seed as usize] += 1;
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
            let end = slice_begin_of(span_begin);
            while value_to_clear < end {
                used.clear(value_to_clear);
                value_to_clear += 1;
            }
            for nb2 in old_end..span_end(span_begin) {
                let sz = size(nb2);
                if sz != 0 {
                    heap.push((conf.eval(nb2, sz), Reverse(nb2)));
                    in_heap[nb2] = true;
                }
            }
        }
    }
    Level {
        geom: g,
        seeds,
        bucket_begin,
        bumped_keys,
        stats,
    }
}

/// Hash for key `id` at level `level`.
#[inline(always)]
pub fn level_hash(id: u64, level: u64, seed: u64) -> u64 {
    mix64(mix64(id ^ seed) ^ level.wrapping_mul(0x9e3779b97f4a7c15))
}

pub struct MphfReport {
    pub n: usize,
    pub levels: Vec<(usize, usize, usize)>, // (keys, buckets, bumped)
    pub seed_bits: f64,
    pub ef_bits: f64,
    pub last_bits: f64,
    pub stats0: Stats,
}

impl MphfReport {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

/// Estimates the size of a full PHast+ MPHF (Function2-like) on n random keys.
pub fn mphf<C: Chooser>(n: usize, conf: &Conf, mut mk: impl FnMut() -> C, seed: u64) -> MphfReport {
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut levels = vec![];
    let mut seed_bits = 0.0;
    let mut entries = 0usize;
    let mut stats0 = None;
    let mut level = 0u64;
    while ids.len() > 8192 {
        let mut h: Vec<(u64, u64)> = ids
            .iter()
            .map(|&id| (level_hash(id, level, seed), id))
            .collect();
        h.sort_unstable_by_key(|x| x.0);
        let hashes: Vec<u64> = h.iter().map(|x| x.0).collect();
        let m = hashes.len();
        let lv = build_level(&hashes, m, conf, &mut mk());
        seed_bits += (lv.geom.buckets * conf.s as usize) as f64;
        if level > 0 {
            entries += m;
        }
        let mut next = Vec::with_capacity(lv.bumped_keys);
        for b in 0..lv.geom.buckets {
            if lv.seeds[b] == 0 {
                for k in lv.bucket_begin[b]..lv.bucket_begin[b + 1] {
                    next.push(h[k].1);
                }
            }
        }
        levels.push((m, lv.geom.buckets, lv.bumped_keys));
        if stats0.is_none() {
            stats0 = Some(lv.stats);
        }
        ids = next;
        level += 1;
    }
    // Last level: regular PHast no-bump S=8, lambda=4, range 1.2n.
    let last_n = ids.len();
    let last_range = (last_n + 10) * 120 / 100;
    let last_bits = (last_n as f64 / 4.0).ceil() * 8.0;
    entries += last_range;
    let ef = ef_bits(entries, n);
    MphfReport {
        n,
        levels,
        seed_bits,
        ef_bits: ef,
        last_bits,
        stats0: stats0.unwrap_or_default(),
    }
}

pub struct OverloadReport {
    pub n: usize,
    /// (keys, range, buckets, bumped, holes)
    pub levels: Vec<(usize, usize, usize, usize, usize)>,
    pub seed_bits: f64,
    pub ef_entries: usize,
    pub ef_bits: f64,
    pub last_bits: f64,
}

impl OverloadReport {
    pub fn bits_per_key(&self) -> f64 {
        (self.seed_bits + self.ef_bits + self.last_bits) / self.n as f64
    }
}

/// Concatenated-range layout: level i has range (1 - gamma_i) n_i, values of
/// level i are offset by the sum of previous ranges, and only values >= n are
/// remapped (EF entries = total range - n).
pub fn mphf_overload<C: Chooser>(
    n: usize,
    conf: &Conf,
    gamma: f64,
    mut mk: impl FnMut() -> C,
    seed: u64,
) -> OverloadReport {
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
        let m = ((1.0 - gamma) * k as f64).round() as usize;
        let lv = build_level(&hashes, m, conf, &mut mk());
        seed_bits += (lv.geom.buckets * conf.s as usize) as f64;
        total_range += m;
        let mut next = Vec::with_capacity(lv.bumped_keys);
        for b in 0..lv.geom.buckets {
            if lv.seeds[b] == 0 {
                for j in lv.bucket_begin[b]..lv.bucket_begin[b + 1] {
                    next.push(h[j].1);
                }
            }
        }
        let holes = m - (k - lv.bumped_keys);
        levels.push((k, m, lv.geom.buckets, lv.bumped_keys, holes));
        ids = next;
        level += 1;
    }
    let last_n = ids.len();
    let last_range = (last_n + 10) * 120 / 100;
    let last_bits = (last_n as f64 / 4.0).ceil() * 8.0;
    total_range += last_range;
    let ef_entries = total_range.saturating_sub(n);
    OverloadReport {
        n,
        levels,
        seed_bits,
        ef_entries,
        ef_bits: ef_bits(ef_entries, n),
        last_bits,
    }
}

/// R independent offset patterns x D shifts; seed-1 = r*D + d.
pub struct MultiPattern {
    pub r: u16,
    pub d: u16,
}

pub const PATTERN_KEYS: [u64; 64] = {
    let mut k = [0u64; 64];
    let mut i = 0;
    let mut x: u64 = 0x9e3779b97f4a7c15;
    while i < 64 {
        // splitmix64
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        k[i] = z | 1;
        i += 1;
    }
    k
};

impl MultiPattern {
    #[inline(always)]
    pub fn offset(c: u64, r: u16, g: &Geom) -> usize {
        if r == 0 {
            (c & g.l_mask) as usize
        } else {
            (mul_hi(c, PATTERN_KEYS[r as usize]) & g.l_mask) as usize
        }
    }
}

impl Chooser for MultiPattern {
    fn choose(
        &mut self,
        used: &mut Cyclic<512>,
        keys: &[u64],
        g: &Geom,
        _s: u32,
        stats: &mut Stats,
    ) -> u16 {
        let mut best: Option<(usize, u16, u16)> = None; // (sum, r, d)
        let mut bases: Vec<usize> = Vec::with_capacity(keys.len());
        let mut sorted: Vec<usize> = Vec::with_capacity(keys.len());
        let mut any_self = false;
        for r in 0..self.r {
            bases.clear();
            bases.extend(
                keys.iter()
                    .map(|&c| g.slice_begin(c) + Self::offset(c, r, g)),
            );
            sorted.clear();
            sorted.extend_from_slice(&bases);
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                any_self = true;
                continue;
            }
            let base_sum: usize = bases.iter().sum();
            if let Some((bs, _, _)) = best {
                // lower bound: sum at d=0
                if base_sum >= bs {
                    continue;
                }
            }
            let mut shift = 0u16;
            while shift < self.d {
                let mut u = 0u64;
                for &b in &bases {
                    u |= used.get64(b + shift as usize);
                }
                if shift + 64 > self.d {
                    u |= !0u64 << (self.d - shift);
                }
                if u != u64::MAX {
                    let d = shift + u.trailing_ones() as u16;
                    let sum = base_sum + d as usize * keys.len();
                    if best.map_or(true, |(bs, _, _)| sum < bs) {
                        best = Some((sum, r, d));
                    }
                    break;
                }
                shift += 64;
            }
        }
        match best {
            None => {
                if any_self {
                    stats.self_collisions += 1;
                }
                0
            }
            Some((_, r, d)) => {
                for &c in keys {
                    used.set(g.slice_begin(c) + Self::offset(c, r, g) + d as usize);
                }
                r * self.d + d + 1
            }
        }
    }
    fn value(&self, c: u64, seed: u16, g: &Geom) -> usize {
        let s = seed - 1;
        g.slice_begin(c) + Self::offset(c, s / self.d, g) + (s % self.d) as usize
    }
}

/// Regular PHast (SeedOnly): random placement in slice, best of all seeds.
pub struct SeedOnly;

impl SeedOnly {
    #[inline(always)]
    pub fn pos(c: u64, seed: u16, g: &Geom) -> usize {
        g.slice_begin(c)
            + (mul_hi((seed as u64).wrapping_mul(0x517cc1b727220a95), c) & g.l_mask) as usize
    }
}

impl Chooser for SeedOnly {
    fn choose(
        &mut self,
        used: &mut Cyclic<512>,
        keys: &[u64],
        g: &Geom,
        s: u32,
        stats: &mut Stats,
    ) -> u16 {
        let mut best = (usize::MAX, 0u16);
        let mut vals: Vec<usize> = Vec::with_capacity(keys.len());
        let mut any_self = false;
        'outer: for seed in 1..(1u16 << s) {
            vals.clear();
            let mut sum = 0;
            for &c in keys {
                let v = Self::pos(c, seed, g);
                if used.get(v) {
                    continue 'outer;
                }
                sum += v;
                vals.push(v);
            }
            if sum < best.0 {
                vals.sort_unstable();
                if vals.windows(2).any(|w| w[0] == w[1]) {
                    any_self = true;
                    continue;
                }
                best = (sum, seed);
            }
        }
        if best.1 == 0 {
            if any_self {
                stats.self_collisions += 1;
            }
            return 0;
        }
        for &c in keys {
            used.set(Self::pos(c, best.1, g));
        }
        best.1
    }
    fn value(&self, c: u64, seed: u16, g: &Geom) -> usize {
        Self::pos(c, seed, g)
    }
}

pub fn phast_weights_8_1024() -> [i64; 7] {
    [-50171, 59462, 109868, 141865, 163564, 181092, 192852]
}
