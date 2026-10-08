/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The sweep assigning seeds to the buckets of a level.

use super::builder::*;
use super::*;
use dsi_progress_logger::ConcurrentProgressLog;
use std::collections::BinaryHeap;

/// A source of the signatures of the keys of a level, distributed into
/// *parts* of consecutive buckets, which sweeps load in increasing order
/// (see [`Parts`]).
///
/// The signatures of the construction from a slice are kept in memory (see
/// [`Signatures`]), whereas those of the construction from a lender are
/// computed from records kept in memory or on disk (see the `stream`
/// module).
///
/// [`Signatures`]: super::sigs::Signatures
pub(super) trait PartSource: Sync {
    /// The state of a loader of parts; each sweep has its own.
    type Loader: Default;

    /// Returns the base-2 logarithm of a lower bound on the number of
    /// buckets of the parts, except possibly the last one.
    fn log2_min_buckets(&self) -> u32;

    /// Returns the part of bucket `b`.
    fn part(&self, b: usize) -> usize;

    /// Returns the first bucket of part `p`.
    fn first_bucket(&self, p: usize) -> usize;

    /// Returns the number of buckets of part `p`.
    fn buckets(&self, p: usize, g: &Geometry) -> usize;

    /// Distributes into buckets the signatures of part `p` whose bucket is
    /// in `range` (which must contain the buckets of the part only, but can
    /// be a proper subset of them), writing them to `sigs` and the position
    /// in `sigs` of the first signature of each bucket of the range,
    /// followed by the number of signatures, to `begin`; `temp` is a
    /// temporary buffer (see [`distribute`]).
    ///
    /// [`distribute`]: super::sigs::distribute
    #[allow(clippy::too_many_arguments)]
    fn load(
        &self,
        loader: &mut Self::Loader,
        p: usize,
        g: &Geometry,
        range: std::ops::Range<usize>,
        sigs: &mut Vec<u64>,
        begin: &mut Vec<usize>,
        temp: &mut Vec<u64>,
    );
}

/// Number of buckets in the window of a sweep.
pub(super) const WINDOW: usize = 256;
/// Size of the cyclic set recording the buckets in the window.
const WINDOW_BITS: usize = 1024;
/// Number of keys from which the buckets of a window are kept in a heap.
const LARGE_BUCKET: usize = 64;

/// The buckets in the window of a sweep, in order of priority.
///
/// The priority of a bucket depends on its size, and decreases with its
/// index. Since buckets enter the window in order of index, the buckets of
/// each size form a queue, and the bucket with the highest priority is the
/// first bucket of one of the queues: finding it requires just to scan the
/// nonempty queues, and adding a bucket takes constant time. Buckets with
/// at least [`LARGE_BUCKET`] keys, if any, are kept in a heap.
struct Window {
    /// The priority weights for buckets with less than [`LARGE_BUCKET`]
    /// keys.
    weights: [i64; LARGE_BUCKET],
    /// For each size below [`LARGE_BUCKET`], a cyclic buffer containing
    /// the lowest bits of the indices of the buckets of that size.
    queues: Box<[[u8; WINDOW]]>,
    /// The position in its cyclic buffer of the first bucket of each
    /// queue.
    first: [u8; LARGE_BUCKET],
    /// The length of each queue.
    len: [u16; LARGE_BUCKET],
    /// The key of the first bucket of each nonempty queue.
    keys: [i64; LARGE_BUCKET],
    /// The sizes of the nonempty queues, as a set of bits.
    nonempty: u64,
    /// The keys of the buckets with at least [`LARGE_BUCKET`] keys.
    large: BinaryHeap<i64>,
    /// The buckets in the window, as a cyclic set of bits.
    buckets: [u64; WINDOW_BITS / 64],
}

impl Window {
    fn new(weights: &[i64; 7]) -> Self {
        const { assert!(WINDOW <= 256 && WINDOW <= WINDOW_BITS) }
        Self {
            weights: std::array::from_fn(|size| Self::weight(weights, size)),
            queues: vec![[0; WINDOW]; LARGE_BUCKET].into_boxed_slice(),
            first: [0; LARGE_BUCKET],
            len: [0; LARGE_BUCKET],
            keys: [0; LARGE_BUCKET],
            nonempty: 0,
            large: BinaryHeap::new(),
            buckets: [0; WINDOW_BITS / 64],
        }
    }

    /// Returns the priority weight of the buckets of the given size:
    /// weights are given for the first seven sizes, and extrapolated
    /// linearly for the following ones.
    fn weight(weights: &[i64; 7], size: usize) -> i64 {
        if size <= 7 {
            weights[size.max(1) - 1]
        } else {
            weights[6] + (weights[6] - weights[5]) * (size - 7) as i64
        }
    }

    /// Returns the key of a bucket: its priority and, in the lowest bits,
    /// the complement of the lowest bits of its index, which give
    /// precedence to the first bucket in case of equal priorities (and
    /// identify the bucket inside the window).
    #[inline(always)]
    fn key(weight: i64, b: usize) -> i64 {
        ((weight - 1024 * b as i64) << WINDOW_BITS.ilog2())
            | (WINDOW_BITS - 1 - b % WINDOW_BITS) as i64
    }

    /// Returns whether bucket `b` is in the window.
    #[inline(always)]
    fn contains(&self, b: usize) -> bool {
        self.buckets[(b % WINDOW_BITS) / 64] >> (b % 64) & 1 != 0
    }

    /// Adds bucket `b`, which must follow all buckets added previously and
    /// have the given nonzero size.
    #[inline(always)]
    fn push(&mut self, b: usize, size: usize, weights: &[i64; 7]) {
        self.buckets[(b % WINDOW_BITS) / 64] |= 1 << (b % 64);
        if size >= LARGE_BUCKET {
            self.large.push(Self::key(Self::weight(weights, size), b));
            return;
        }
        if self.len[size] == 0 {
            self.keys[size] = Self::key(self.weights[size], b);
            self.nonempty |= 1 << size;
        }
        let last = self.first[size] as usize + self.len[size] as usize;
        self.queues[size][last % WINDOW] = b as u8;
        self.len[size] += 1;
    }

    /// Removes and returns the bucket with the highest priority, if any,
    /// given the first bucket in the window.
    #[inline(always)]
    fn pop(&mut self, span_begin: usize) -> Option<usize> {
        let (mut key, mut size) = (i64::MIN, 0);
        let mut nonempty = self.nonempty;
        while nonempty != 0 {
            let s = nonempty.trailing_zeros() as usize;
            nonempty &= nonempty - 1;
            (key, size) = if self.keys[s] > key {
                (self.keys[s], s)
            } else {
                (key, size)
            };
        }
        // Buckets are identified by the lowest bits of their index
        let bucket = |low: usize, bits: usize| span_begin + low.wrapping_sub(span_begin) % bits;
        let b = if self.large.peek().is_some_and(|&large| large > key) {
            let key = self.large.pop().unwrap();
            bucket(WINDOW_BITS - 1 - key as usize % WINDOW_BITS, WINDOW_BITS)
        } else if size == 0 {
            return None;
        } else {
            let queue = &self.queues[size];
            let b = bucket(queue[self.first[size] as usize % WINDOW] as usize, 256);
            self.first[size] = self.first[size].wrapping_add(1);
            self.len[size] -= 1;
            if self.len[size] == 0 {
                self.nonempty &= !(1 << size);
            } else {
                let next = bucket(queue[self.first[size] as usize % WINDOW] as usize, 256);
                self.keys[size] = Self::key(self.weights[size], next);
            }
            b
        };
        self.buckets[(b % WINDOW_BITS) / 64] &= !(1 << (b % 64));
        Some(b)
    }
}

/// The result of the sweep of a level.
pub(super) struct Level {
    /// The seeds of the buckets.
    pub(super) seeds: Vec<u16>,
    /// The used slots, as a set of bits.
    pub(super) occupied: Vec<u64>,
}

/// Assigns seeds to the buckets of a level that can bump keys (see
/// [`sweep_level`]), logging the progress of the sweep, in buckets, on a
/// concurrent logger obtained from `pl`; `index` is the index of the level
/// and `k` its number of keys.
pub(super) fn sweep_bumping_level<S: PartSource>(
    sigs: &S,
    g: &Geometry,
    weights: &[i64; 7],
    index: usize,
    k: usize,
    pl: &mut impl ProgressLog,
) -> Level {
    let mut cpl = pl.concurrent();
    cpl.item_name("bucket");
    cpl.expected_updates(g.buckets);
    cpl.start(format!(
        "Sweeping level {index} ({k} keys, {} buckets)...",
        g.buckets
    ));
    let out = sweep_level(sigs, g, weights, true, &mut cpl).expect("bumping sweeps cannot fail");
    cpl.done();
    out
}

/// Assigns seeds to the buckets of a level, possibly in parallel, given the
/// signatures of its keys (see [`PartSource`]).
///
/// Each sweep updates a clone of `pl` with the number of buckets of each
/// part it loads.
///
/// Returns `None` if `allow_bump` is false and some bucket could not be
/// placed.
pub(super) fn sweep_level<S: PartSource, P: ConcurrentProgressLog>(
    sigs: &S,
    g: &Geometry,
    weights: &[i64; 7],
    allow_bump: bool,
    pl: &mut P,
) -> Option<Level> {
    let nb = g.buckets;
    let l = g.l_mask as usize + 1;

    // Number of buckets between chunks whose slots cannot overlap (as in
    // PHast): the beginning of the slice grows by num_slices / nb per
    // bucket, and the slots of a key lie at most L - 1 slots after the
    // beginning of its slice, so considering the rounding of the beginning
    // of the slices it suffices that gap * num_slices / nb >= L.
    let gap = l * nb / g.num_slices as usize + 1;
    #[cfg(feature = "rayon")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    let threads = 1;
    let chunks = threads.min(nb / (64 * gap).max(4096)).max(1);

    let mut seeds = vec![0u16; nb];
    let mut occupied = vec![0u64; g.m.div_ceil(64)];
    // Each sweep returns the slots it used (and those marked as used)
    let mut merge = |swept: Swept| {
        let occupied = occupied.iter_mut().skip(swept.occupied_begin / 64);
        for (occupied, word) in occupied.zip(&swept.occupied) {
            *occupied |= word;
        }
    };

    if chunks == 1 {
        merge(Sweep::new(sigs, g, weights, 0, nb, &mut seeds).run(allow_bump, pl.clone())?);
        return Some(Level { seeds, occupied });
    }

    // Chunk boundaries; chunk i processes [bounds[i], bounds[i + 1] - gap),
    // except for the last one, which processes everything.
    let bounds: Vec<usize> = (0..=chunks).map(|i| i * nb / chunks).collect();
    let mut parts: Vec<&mut [u16]> = Vec::with_capacity(chunks);
    let mut rest = &mut seeds[..];
    for i in 0..chunks {
        let (a, r) = rest.split_at_mut(bounds[i + 1] - bounds[i]);
        parts.push(a);
        rest = r;
    }
    let pl = &*pl;
    let run_chunk = |(i, part): (usize, &mut &mut [u16])| {
        let lo = bounds[i];
        let hi = if i + 1 == chunks {
            bounds[i + 1]
        } else {
            bounds[i + 1] - gap
        };
        Sweep::new(sigs, g, weights, lo, hi, &mut part[..hi - lo]).run(allow_bump, pl.clone())
    };
    #[cfg(feature = "rayon")]
    let swept: Vec<Option<Swept>> = {
        use rayon::prelude::*;
        parts.par_iter_mut().enumerate().map(run_chunk).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let swept: Vec<Option<Swept>> = parts.iter_mut().enumerate().map(run_chunk).collect();
    drop(parts);
    for swept in swept {
        merge(swept?);
    }

    // Gaps: the seeds of the neighboring buckets are now known, and the
    // slots they use are marked.
    let run_gap = |i: usize| -> (Vec<u16>, Option<Swept>) {
        let lo = bounds[i + 1] - gap;
        let hi = bounds[i + 1];
        let mut gap_seeds = vec![0u16; hi - lo];
        let mut sw = Sweep::new(sigs, g, weights, lo, hi, &mut gap_seeds);
        let before = lo.saturating_sub(gap).max(bounds[i]);
        let after = (hi + gap).min(bounds[i + 2]);
        let first = g.first_slice(lo);
        // These buckets have been swept already, so we do not log them
        let mut parts = Parts::<S, P>::new(sigs, before..after, None);
        for b in (before..lo).chain(hi..after) {
            parts.ensure(sigs, g, b);
            for &x in parts.keys(sigs, b) {
                let s = seeds[g.bucket(x)] as usize;
                if s != 0 {
                    let p = g.pos(x, s);
                    if p >= first {
                        sw.set(p);
                    }
                }
            }
        }
        let swept = sw.run(allow_bump, pl.clone());
        (gap_seeds, swept)
    };
    #[cfg(feature = "rayon")]
    let gaps: Vec<(Vec<u16>, Option<Swept>)> = {
        use rayon::prelude::*;
        (0..chunks - 1).into_par_iter().map(run_gap).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let gaps: Vec<(Vec<u16>, Option<Swept>)> = (0..chunks - 1).map(run_gap).collect();
    for (i, (gap_seeds, swept)) in gaps.into_iter().enumerate() {
        merge(swept?);
        seeds[bounds[i + 1] - gap..bounds[i + 1]].copy_from_slice(&gap_seeds);
    }
    Some(Level { seeds, occupied })
}

/// Returns whether position `p` is set in the occupancy bitmap.
#[inline(always)]
pub(super) fn is_occupied(bits: &[u64], p: usize) -> bool {
    bits[p / 64] >> (p % 64) & 1 != 0
}

/// Returns the positions in [0 . . m) that are not set in the occupancy
/// bitmap, in increasing order.
pub(super) fn holes(bits: &[u64], m: usize) -> Vec<usize> {
    /// Number of words scanned by a parallel task.
    #[cfg(feature = "rayon")]
    const CHUNK: usize = 1 << 12;
    // Appends to out the free positions of the words starting at word first
    let scan = |first: usize, words: &[u64], out: &mut Vec<usize>| {
        for (i, &word) in words.iter().enumerate() {
            let w = first + i;
            let mut free = !word;
            let valid = m - w * 64;
            if valid < 64 {
                free &= (1u64 << valid) - 1;
            }
            while free != 0 {
                out.push(w * 64 + free.trailing_zeros() as usize);
                free &= free - 1;
            }
        }
    };
    #[cfg(feature = "rayon")]
    if parallel(m) {
        use rayon::prelude::*;
        let parts: Vec<Vec<usize>> = bits
            .par_chunks(CHUNK)
            .enumerate()
            .map(|(c, words)| {
                let mut out = vec![];
                scan(c * CHUNK, words, &mut out);
                out
            })
            .collect();
        return parts.concat();
    }
    let mut out = vec![];
    scan(0, bits, &mut out);
    out
}

/// The parameters of the rings of a level, and of the set of used slots of
/// a sweep.
///
/// The seeds of a pattern move a key through the slots of its *ring*: the
/// slots of its slice that are congruent to its first slot modulo the
/// stride (the stride is the number of patterns times the amount by which a
/// unit increase of the seed moves a key). The set of used slots is stored
/// by residue classes modulo the stride (*rows*), 64 slots of each row at a
/// time (*columns*): bit *i* of word *c* · 2^`log2_stride` + *ρ* is
/// associated with slot (64*c* + *i*) · 2^`log2_stride` + *ρ* (the number
/// of columns is a power of two, and columns are used cyclically). A ring
/// is thus a block of consecutive bits of a row, and the seed of index *j*
/// of a pattern maps a key to the bit of its ring whose index is the index
/// for the first seed of the pattern plus *j*, modulo the length of the
/// ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rings {
    /// The slice length minus one.
    l_mask: u64,
    /// The base-2 logarithm of the amount by which a unit increase of the
    /// seed moves the keys of a bucket.
    scale: u32,
    /// The number of patterns.
    patterns: usize,
    /// The distance in bits between the offsets of two consecutive
    /// patterns.
    pattern_bits: u32,
    /// The base-2 logarithm of the stride (at most that of the slice
    /// length).
    log2_stride: u32,
    /// The number of slots of a ring.
    len: usize,
    /// The number of seeds of a pattern (at least the length of a ring; if
    /// it is larger, seeds are redundant).
    pattern_seeds: usize,
    /// The number of words of the set of used slots, minus one.
    word_mask: usize,
}

impl Rings {
    /// The rings of the levels of a function with default parameters
    /// (8-bit seeds, four patterns, and slices of 1024 slots), except for
    /// the size of the set of used slots.
    const DEFAULT: Rings = Rings {
        l_mask: 1023,
        scale: DEFAULT_SCALE,
        patterns: 4,
        pattern_bits: 16,
        log2_stride: 4,
        len: 64,
        pattern_seeds: 64,
        word_mask: 0,
    };

    fn new(g: &Geometry) -> Self {
        let l = g.l_mask as usize + 1;
        // With slices shorter than the stride, a ring is a single slot
        let log2_stride = (g.log2_patterns + g.scale).min(l.ilog2());
        // The cyclic set must contain the slots reachable from the buckets
        // in the window (WINDOW buckets, whose slices begin about
        // num_slices / buckets slots apart) plus a slice; gap sweeps need
        // three slices more. Moreover, slots that are no longer reachable
        // are cleared a column at a time, so there must be room for the
        // slots of a column (in particular, there is at least a column).
        let per_bucket = (g.num_slices as usize).div_ceil(g.buckets);
        let slots =
            (4 * (l + 1) + 2 * WINDOW * per_bucket + (64 << log2_stride)).next_power_of_two();
        Self {
            l_mask: g.l_mask,
            scale: g.scale,
            patterns: 1 << g.log2_patterns,
            pattern_bits: 64 >> g.log2_patterns,
            log2_stride,
            len: l >> log2_stride,
            pattern_seeds: (1usize << g.seed_bits) >> g.log2_patterns,
            word_mask: slots / 64 - 1,
        }
    }

    /// Returns the offset in its slice of a key for the first seed of
    /// pattern `r`, given the value providing its offsets.
    #[inline(always)]
    fn offset(&self, offsets: u64, r: usize) -> usize {
        ((offsets >> (r as u32 * self.pattern_bits)).wrapping_add((r as u64) << self.scale)
            & self.l_mask) as usize
    }

    /// Returns the first slot of the ring of a key for a pattern and the
    /// index in the ring of the slot of the key for the first seed of the
    /// pattern, given the beginning of the slice of the key and its
    /// [offset] for the pattern.
    ///
    /// [offset]: Self::offset
    #[inline(always)]
    fn ring(&self, slice_begin: usize, offset: usize) -> (usize, usize) {
        (
            slice_begin + (offset & ((1 << self.log2_stride) - 1)),
            offset >> self.log2_stride,
        )
    }

    /// Returns the slot of a key for the seed of index `j` of a pattern,
    /// given the first slot of its ring and its index for the first seed
    /// of the pattern (see [`ring`]).
    ///
    /// [`ring`]: Self::ring
    #[inline(always)]
    fn slot(&self, first: usize, index: usize, j: usize) -> usize {
        first + (((index + j) & (self.len - 1)) << self.log2_stride)
    }

    /// Returns the seed of index `j` of pattern `r`.
    ///
    /// The first seed of the first pattern is zero, which marks bumped
    /// buckets; if seeds are redundant we use instead the first seed of
    /// the second turn around the ring.
    #[inline(always)]
    fn seed(&self, r: usize, j: usize) -> usize {
        if r == 0 && j == 0 {
            debug_assert!(self.len < self.pattern_seeds);
            self.len * self.patterns
        } else {
            j * self.patterns + r
        }
    }

    /// Returns the index of the word of the set of used slots containing
    /// the bit associated with slot `p` (the index of the bit in the word
    /// is `p >> log2_stride`, modulo 64).
    #[inline(always)]
    fn word(&self, p: usize) -> usize {
        let stride_mask = (1 << self.log2_stride) - 1;
        (((p >> 6) & !stride_mask) | (p & stride_mask)) & self.word_mask
    }

    /// Returns the bits of the 64 slots of the row of slot `p` starting
    /// from `p` (that is, bit *i* is associated with the slot `p` plus *i*
    /// strides).
    #[inline(always)]
    fn row64(&self, used: &[u64], p: usize) -> u64 {
        let t = self.log2_stride;
        let w = self.word(p);
        // The next word of the row is in the next column
        debug_assert_eq!(used.len(), self.word_mask + 1);
        // SAFETY: the indices are at most word_mask
        let (lo, hi) = unsafe {
            (
                *used.get_unchecked(w),
                *used.get_unchecked((w + (1 << t)) & self.word_mask),
            )
        };
        // Branchless: a 128-bit shift
        ((((hi as u128) << 64) | lo as u128) >> ((p >> t) % 64)) as u64
    }

    /// Returns the base-2 logarithm of the number of bits of the fields
    /// packing indices in a ring into words (see [`Sweep::sum`]): the sum
    /// of two indices must fit a field.
    #[inline(always)]
    fn log2_field(&self) -> u32 {
        if self.len <= 1 << 7 {
            3
        } else if self.len <= 1 << 15 {
            4
        } else {
            5
        }
    }
}

/// The maximum number of free seeds of a bucket that are examined
/// exhaustively when looking for the best seed.
const MAX_FREE: u32 = 32;

/// The state of a sweep over a range of buckets.
///
/// Buckets are processed in order of priority inside a window sliding over
/// the range, as in PHast: for each bucket we look for the seed minimizing
/// the sum of the slots of its keys. The seeds of a pattern that are
/// feasible for a bucket are obtained by rotating the ring of each key and
/// combining the results (see [`Rings`]).
struct Sweep<'a, S: PartSource> {
    /// The signatures of the keys of the level.
    sigs: &'a S,
    g: &'a Geometry,
    weights: &'a [i64; 7],
    /// First bucket of the range.
    lo: usize,
    /// End of the range.
    hi: usize,
    /// Seeds of the buckets in the range.
    seeds: &'a mut [u16],
    /// Cyclic bit set of used slots (see [`Rings`]).
    used: Box<[u64]>,
    rings: Rings,
    /// The first slot of the first column of the set of used slots; slots
    /// before this one are no longer reachable.
    first_slot: usize,
    /// The used slots that are no longer reachable, as a set of bits
    /// starting from slot `occupied_begin` (a multiple of 64).
    occupied: Vec<u64>,
    occupied_begin: usize,
    /// The state of the search for the seed of a bucket (see
    /// [`state`]).
    ///
    /// [`state`]: Self::state
    state: Vec<u64>,
    /// The nonzero words of `free`, each with its index.
    nonzero: Vec<(u64, u32)>,
    /// The indices at which some key goes back to the first slot of its
    /// ring.
    back: Vec<u32>,
}

impl<'a, S: PartSource> Sweep<'a, S> {
    fn new(
        sigs: &'a S,
        g: &'a Geometry,
        weights: &'a [i64; 7],
        lo: usize,
        hi: usize,
        seeds: &'a mut [u16],
    ) -> Self {
        debug_assert_eq!(<[u16]>::len(seeds), hi - lo);
        let rings = Rings::new(g);
        let free_words = rings.patterns * rings.len.div_ceil(64);
        // Columns contain 64 strides
        let first_slot = g.first_slice(lo) >> (rings.log2_stride + 6) << (rings.log2_stride + 6);
        Self {
            sigs,
            g,
            weights,
            lo,
            hi,
            seeds,
            used: vec![0; rings.word_mask + 1].into_boxed_slice(),
            rings,
            first_slot,
            occupied: vec![],
            occupied_begin: first_slot,
            state: Vec::with_capacity(free_words + rings.patterns * 3),
            nonzero: vec![(0, 0); free_words],
            back: Vec::with_capacity(64),
        }
    }

    /// Marks slot `p` as used.
    #[inline(always)]
    fn set(&mut self, p: usize) {
        self.used[self.rings.word(p)] |= 1 << ((p >> self.rings.log2_stride) % 64);
    }

    /// Returns the parts of the state of the search for the seed of a
    /// bucket:
    ///
    /// - for each pattern, the indices of the seeds for which no key of the
    ///   bucket is mapped to a used slot (limited to the first turn around
    ///   the ring);
    ///
    /// - for each pattern, the sum of the slots of the keys of the bucket
    ///   for the first seed of the pattern;
    ///
    /// - the indices in their rings of the slots of the keys of the bucket
    ///   for the first seed of each pattern, packed into words (see
    ///   [`sum`]): a word for each pattern contains the indices
    ///   of a group of keys.
    ///
    /// The parts are stored contiguously, so that in the innermost loops
    /// they are accessed through a single pointer.
    ///
    /// [`sum`]: Self::sum
    #[inline(always)]
    fn state<'b>(
        rings: &Rings,
        state: &'b mut [u64],
    ) -> (&'b mut [u64], &'b mut [u64], &'b mut [u64]) {
        let (free, state) = state.split_at_mut(rings.patterns * rings.len.div_ceil(64));
        let (sums, indices) = state.split_at_mut(rings.patterns);
        (free, sums, indices)
    }

    /// Removes the first column from the set of used slots, recording its
    /// slots in `occupied`: the bits of the column will be associated with
    /// the slots following those of the last column.
    #[inline(always)]
    fn retire_column(&mut self, rings: &Rings) {
        let t = rings.log2_stride;
        let w = rings.word(self.first_slot);
        let begin = (self.first_slot - self.occupied_begin) / 64;
        // A column contains a word for each row
        self.occupied.resize(begin + (1 << t), 0);
        let occupied = &mut self.occupied[begin..];
        for (row, word) in self.used[w..w + (1 << t)].iter_mut().enumerate() {
            let mut word = std::mem::take(word);
            while word != 0 {
                let p = ((word.trailing_zeros() as usize) << t) + row;
                occupied[p / 64] |= 1 << (p % 64);
                word &= word - 1;
            }
        }
        self.first_slot += 64 << t;
    }

    /// Returns the sum of the slots of the keys of a bucket of the given
    /// size for the seed of index `j` of pattern `r`, given the sum for
    /// the first seed of each pattern and the packed indices of the keys.
    ///
    /// The seed of index `j` moves each key `j` strides forward, and then
    /// a slice length backward if the key has gone past the end of its
    /// ring, that is, if its index plus `j` is at least the length of the
    /// ring: since indices are packed into fields that can contain the sum
    /// of two indices, we can count such keys a word at a time.
    #[inline(always)]
    fn sum(rings: &Rings, sums: &[u64], indices: &[u64], size: usize, r: usize, j: usize) -> usize {
        // A one in each field
        let ones = u64::MAX / (u64::MAX >> (64 - (1 << rings.log2_field())));
        let (j_in_fields, len_in_fields) = (j as u64 * ones, rings.len as u64 * ones);
        // Almost all buckets have a single group of keys
        let (first, others) = indices.split_at(rings.patterns);
        let mut back = ((first[r] + j_in_fields) & len_in_fields).count_ones() as usize;
        if !others.is_empty() {
            back += Self::back_others(others, rings.patterns, r, j_in_fields, len_in_fields);
        }
        (sums[r] as usize + ((size * j) << rings.log2_stride))
            .wrapping_sub(back * (rings.l_mask as usize + 1))
    }

    /// Returns the number of keys going past the end of their ring in the
    /// groups after the first one (see [`sum`]).
    ///
    /// [`sum`]: Self::sum
    #[cold]
    #[inline(never)]
    fn back_others(
        others: &[u64],
        patterns: usize,
        r: usize,
        j_in_fields: u64,
        len_in_fields: u64,
    ) -> usize {
        others
            .chunks_exact(patterns)
            .map(|group| ((group[r] + j_in_fields) & len_in_fields).count_ones() as usize)
            .sum()
    }

    /// Finds the seed of bucket `b` minimizing the sum of the slots of its
    /// keys, and marks the slots as used; returns zero if no seed is
    /// feasible.
    #[inline(always)]
    fn search(&mut self, rings: &Rings, keys: &[u64]) -> usize {
        let g = *self.g;
        let size = keys.len();
        let (n, patterns) = (rings.len, rings.patterns);
        // The number of words of the free seeds of a pattern
        let words = n.div_ceil(64);
        let ring_mask = if n < 64 { (1u64 << n) - 1 } else { !0 };
        // Indices are packed in groups filling a word
        let log2_field = rings.log2_field();
        let log2_group = 6 - log2_field;

        let Self {
            used,
            state,
            nonzero,
            ..
        } = self;
        state.clear();
        state.resize(patterns * (words + 1 + size.div_ceil(1 << log2_group)), 0);
        // Slices, so that pointers and lengths are loop invariants
        let (free, sums, indices) = Self::state(rings, state);
        let (used, nonzero) = (&used[..], &mut nonzero[..patterns * words]);

        // For each pattern we rotate the ring of each key by its index, so
        // that the bit of index j is associated with the slot of the key
        // for the seed of index j, and combine the results: we obtain the
        // indices of the seeds for which some key is mapped to a used slot
        for (k, &key) in keys.iter().enumerate() {
            let (slice_begin, offsets) = (g.slice_begin(key), g.offsets(key));
            let indices = &mut indices[(k >> log2_group) * patterns..][..patterns];
            let field = (k as u32 % (1 << log2_group)) << log2_field;
            for r in 0..patterns {
                let offset = rings.offset(offsets, r);
                let (first, x) = rings.ring(slice_begin, offset);
                sums[r] += (slice_begin + offset) as u64;
                indices[r] |= (x as u64) << field;
                if words == 1 {
                    // Rings of at most a word (e.g., 8-bit seeds and four
                    // patterns)
                    let ring = rings.row64(used, first) & ring_mask;
                    free[r] |= (ring >> x) | (ring << ((n - x) % 64));
                } else {
                    // Rings are sequences of whole words
                    let word = |w: usize| {
                        rings.row64(
                            used,
                            first + ((64 * (w & (words - 1))) << rings.log2_stride),
                        )
                    };
                    let (xw, xb) = (x / 64, x % 64);
                    let mut prev = word(xw);
                    for (w, blocked) in free[r * words..][..words].iter_mut().enumerate() {
                        let next = word(xw + w + 1);
                        *blocked |= if xb == 0 {
                            prev
                        } else {
                            (prev >> xb) | (next << (64 - xb))
                        };
                        prev = next;
                    }
                }
            }
        }
        // The first seed of the first pattern is zero, which marks bumped
        // buckets, unless seeds are redundant (see Rings::seed)
        if n == rings.pattern_seeds {
            free[0] |= 1;
        }

        // We complement the result, and gather the nonzero words
        let (mut count, mut total) = (0, 0);
        for (i, w) in free.iter_mut().enumerate() {
            *w = !*w & ring_mask;
            nonzero[count] = (*w, i as u32);
            count += (*w != 0) as usize;
            total += w.count_ones();
        }
        if total == 0 {
            return 0;
        }

        // The best seed is identified by the index of its bit in free
        let best = if total <= MAX_FREE {
            // Just a few free seeds (the common case): we consider all of
            // them, with a single loop that has no other branches
            let (mut best_sum, mut best) = (usize::MAX, 0);
            let mut i = 0;
            for _ in 0..total {
                let (word, index) = nonzero[i];
                let bit = index as usize * 64 + word.trailing_zeros() as usize;
                let sum = Self::sum(
                    rings,
                    sums,
                    indices,
                    size,
                    bit / (64 * words),
                    bit % (64 * words),
                );
                (best_sum, best) = if sum < best_sum {
                    (sum, bit)
                } else {
                    (best_sum, best)
                };
                // We move to the next word when no bits are left
                let word = word & (word - 1);
                nonzero[i].0 = word;
                i += (word == 0) as usize;
            }
            best
        } else {
            self.search_many(keys)
        };

        let (r, j) = (best / (64 * words), best % (64 * words));
        if self.place(rings, keys, r, j) {
            rings.seed(r, j)
        } else {
            self.search_distinct(keys)
        }
    }

    /// Returns the best free seed of the bucket being searched, as
    /// [`search`] does, when there are many free seeds.
    ///
    /// The sum of the slots increases with the index of the seed, except
    /// at the indices at which some key goes back to the first slot of its
    /// ring: between two such indices only the first free index can be the
    /// best one.
    ///
    /// [`search`]: Self::search
    #[inline(never)]
    fn search_many(&mut self, keys: &[u64]) -> usize {
        let (g, rings) = (self.g, self.rings);
        let n = rings.len;
        let words = n.div_ceil(64);
        let mut best = (usize::MAX, 0);
        let (free, sums, indices) = Self::state(&rings, &mut self.state);
        for r in 0..rings.patterns {
            let free = &free[r * words..][..words];
            self.back.clear();
            self.back.extend(
                keys.iter()
                    .map(|&key| {
                        let offset = rings.offset(g.offsets(key), r);
                        rings.ring(g.slice_begin(key), offset).1
                    })
                    .filter(|&x| x != 0)
                    .map(|x| (n - x) as u32),
            );
            self.back.sort_unstable();
            self.back.dedup();
            let mut begin = 0;
            for i in 0..=self.back.len() {
                let end = self.back.get(i).map_or(n, |&x| x as usize);
                if let Some(j) = next_set(free, begin, end) {
                    let sum = Self::sum(&rings, sums, indices, keys.len(), r, j);
                    best = best.min((sum, r * 64 * words + j));
                }
                begin = end;
            }
        }
        best.1
    }

    /// Marks as used the slots of the given keys for the seed of index `j`
    /// of pattern `r`, which must be free, unless two keys are mapped to
    /// the same slot, in which case nothing happens and the method returns
    /// `false`.
    #[inline(always)]
    fn place(&mut self, rings: &Rings, keys: &[u64], r: usize, j: usize) -> bool {
        let g = *self.g;
        let bit = |key: u64| {
            let offset = rings.offset(g.offsets(key), r);
            let (first, x) = rings.ring(g.slice_begin(key), offset);
            let p = rings.slot(first, x, j);
            (rings.word(p), 1u64 << ((p >> rings.log2_stride) % 64))
        };
        // The slots were free: a used slot has been marked by a previous
        // key of the bucket
        let used = &mut self.used[..];
        let mut collisions = 0;
        for &key in keys {
            let (w, mask) = bit(key);
            collisions |= used[w] & mask;
            used[w] |= mask;
        }
        if collisions != 0 {
            for &key in keys {
                let (w, mask) = bit(key);
                used[w] &= !mask;
            }
        }
        collisions == 0
    }

    /// Like [`search`], but considers all free seeds in
    /// order of sum of slots until one that maps the keys of the bucket to
    /// distinct slots is found (this happens rarely).
    ///
    /// [`search`]: Self::search
    #[cold]
    #[inline(never)]
    fn search_distinct(&mut self, keys: &[u64]) -> usize {
        let rings = self.rings;
        let ring_bits = 64 * rings.len.div_ceil(64);
        let mut candidates = vec![];
        let (free, sums, indices) = Self::state(&rings, &mut self.state);
        for (i, &word) in free.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let bit = i * 64 + word.trailing_zeros() as usize;
                word &= word - 1;
                let (r, j) = (bit / ring_bits, bit % ring_bits);
                let sum = Self::sum(&rings, sums, indices, keys.len(), r, j);
                candidates.push((sum, r, j));
            }
        }
        candidates.sort_unstable();
        for (_, r, j) in candidates {
            if self.place(&rings, keys, r, j) {
                return rings.seed(r, j);
            }
        }
        0
    }

    /// Processes the buckets of the range, logging them on `pl`; returns
    /// `None` if `allow_bump` is false and some bucket could not be placed
    /// (in which case the sweep stops immediately).
    fn run(self, allow_bump: bool, pl: impl ProgressLog) -> Option<Swept> {
        let rings = self.rings;
        // The parameters of the rings are used as shift amounts and masks
        // in the innermost loops: we compile a version of the sweep in
        // which those of the default configuration are constants
        let default = Rings {
            word_mask: rings.word_mask,
            ..Rings::DEFAULT
        };
        if rings == default {
            self.sweep(&default, allow_bump, pl)
        } else {
            self.sweep(&rings, allow_bump, pl)
        }
    }

    #[inline(always)]
    fn sweep(mut self, rings: &Rings, allow_bump: bool, pl: impl ProgressLog) -> Option<Swept> {
        let (lo, hi) = (self.lo, self.hi);
        let (sigs, g) = (self.sigs, self.g);
        let mut window = Window::new(self.weights);
        // The parts containing the buckets of the window
        let mut parts = Parts::new(sigs, lo..hi, Some(pl));
        let mut span_begin = lo;
        loop {
            if span_begin == hi {
                break;
            }
            if parts.size(sigs, g, span_begin) != 0 {
                break;
            }
            span_begin += 1;
        }
        let span_end = |span_begin: usize| (span_begin + WINDOW).min(hi);
        let column = 64usize << rings.log2_stride;
        if span_begin < hi {
            // The columns before the first slice in the window contain
            // slots that are no longer reachable
            while self.first_slot + column <= g.first_slice(span_begin) {
                self.retire_column(rings);
            }
            for b in span_begin..span_end(span_begin) {
                let size = parts.size(sigs, g, b);
                if size != 0 {
                    window.push(b, size, self.weights);
                }
            }
        }
        while let Some(b) = window.pop(span_begin) {
            let seed = self.search(rings, parts.keys(sigs, b));
            self.seeds[b - lo] = seed as u16;
            if seed == 0 && !allow_bump {
                return None;
            }
            if b == span_begin {
                let old_end = span_end(span_begin);
                span_begin += 1;
                while span_begin < old_end && !window.contains(span_begin) {
                    span_begin += 1;
                }
                if span_begin == old_end {
                    loop {
                        if span_begin == hi {
                            break;
                        }
                        if parts.size(sigs, g, span_begin) != 0 {
                            break;
                        }
                        span_begin += 1;
                    }
                    if span_begin == hi {
                        break;
                    }
                }
                while self.first_slot + column <= g.first_slice(span_begin) {
                    self.retire_column(rings);
                }
                for b in old_end..span_end(span_begin) {
                    let size = parts.size(sigs, g, b);
                    if size != 0 {
                        window.push(b, size, self.weights);
                    }
                }
            }
        }
        // The remaining columns
        for _ in 0..=rings.word_mask >> rings.log2_stride {
            self.retire_column(rings);
        }
        Some(Swept {
            occupied_begin: self.occupied_begin,
            occupied: self.occupied,
        })
    }
}

/// A part loaded by a sweep (see [`PartSource::load`]).
#[derive(Default)]
struct Part {
    /// The index of the part (`usize::MAX` if the buffers are unused).
    index: usize,
    /// The first loaded bucket.
    first_bucket: usize,
    /// The signatures of the loaded buckets, grouped by bucket.
    sigs: Vec<u64>,
    /// The position in `sigs` of the first signature of each loaded
    /// bucket, followed by the number of signatures.
    begin: Vec<usize>,
}

/// The parts loaded by a sweep, restricted to a range of buckets: the
/// buckets of the window are consecutive, so a few parts are enough.
struct Parts<S: PartSource, P> {
    range: std::ops::Range<usize>,
    parts: Vec<Part>,
    temp: Vec<u64>,
    loader: S::Loader,
    /// The logger updated with the number of buckets of each loaded part,
    /// if any.
    pl: Option<P>,
}

impl<S: PartSource, P: ProgressLog> Parts<S, P> {
    /// Creates a cache for the buckets of the given range, logging the
    /// buckets loaded on `pl`, if any.
    fn new(sigs: &S, range: std::ops::Range<usize>, pl: Option<P>) -> Self {
        // The window can span (WINDOW >> log2_min_buckets) + 1 parts, and we
        // keep the last one when loading the next
        let slots = (WINDOW >> sigs.log2_min_buckets()) + 2;
        Self {
            range,
            parts: (0..slots)
                .map(|_| Part {
                    index: usize::MAX,
                    ..Default::default()
                })
                .collect(),
            temp: vec![],
            loader: S::Loader::default(),
            pl,
        }
    }

    /// Loads the part of bucket `b`, if necessary, replacing the oldest
    /// loaded part.
    #[inline(always)]
    fn ensure(&mut self, sigs: &S, g: &Geometry, b: usize) {
        let p = sigs.part(b);
        if self.parts.iter().any(|part| part.index == p) {
            return;
        }
        self.load(sigs, g, p);
    }

    /// Loads part `p`, replacing the oldest loaded part.
    #[inline(never)]
    fn load(&mut self, sigs: &S, g: &Geometry, p: usize) {
        // An unused slot, or the one of the oldest part (parts are loaded
        // in increasing order)
        let slot = (0..self.parts.len())
            .find(|&i| self.parts[i].index == usize::MAX)
            .unwrap_or_else(|| {
                (0..self.parts.len())
                    .min_by_key(|&i| self.parts[i].index)
                    .unwrap()
            });
        let part = &mut self.parts[slot];
        let first = sigs.first_bucket(p).max(self.range.start);
        let end = (sigs.first_bucket(p) + sigs.buckets(p, g)).min(self.range.end);
        sigs.load(
            &mut self.loader,
            p,
            g,
            first..end,
            &mut part.sigs,
            &mut part.begin,
            &mut self.temp,
        );
        part.index = p;
        part.first_bucket = first;
        if let Some(pl) = &mut self.pl {
            pl.update_with_count(end - first);
        }
    }

    /// Returns the loaded part of bucket `b`.
    #[inline(always)]
    fn part(&self, sigs: &S, b: usize) -> &Part {
        let p = sigs.part(b);
        self.parts
            .iter()
            .find(|part| part.index == p)
            .expect("the part is loaded")
    }

    /// Returns the size of bucket `b`, loading its part if necessary.
    #[inline(always)]
    fn size(&mut self, sigs: &S, g: &Geometry, b: usize) -> usize {
        self.ensure(sigs, g, b);
        let part = self.part(sigs, b);
        let i = b - part.first_bucket;
        part.begin[i + 1] - part.begin[i]
    }

    /// Returns the signatures of the keys of bucket `b`, whose part must be
    /// loaded.
    #[inline(always)]
    fn keys(&self, sigs: &S, b: usize) -> &[u64] {
        let part = self.part(sigs, b);
        let i = b - part.first_bucket;
        &part.sigs[part.begin[i]..part.begin[i + 1]]
    }
}

/// The result of a [`Sweep`].
struct Swept {
    /// The slots used by the buckets of the range, and those marked as
    /// used beforehand, as a set of bits starting from slot
    /// `occupied_begin` (a multiple of 64).
    occupied: Vec<u64>,
    occupied_begin: usize,
}

/// Returns the index of the first bit set in `bits` in the range
/// [`begin` . . `end`), if any.
#[inline(always)]
fn next_set(bits: &[u64], begin: usize, end: usize) -> Option<usize> {
    let mut w = begin / 64;
    let mut word = bits[w] & (!0 << (begin % 64));
    loop {
        if word != 0 {
            let j = w * 64 + word.trailing_zeros() as usize;
            return (j < end).then_some(j);
        }
        w += 1;
        if w * 64 >= end {
            return None;
        }
        word = bits[w];
    }
}
