/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The builder of [`PHastR`] and the construction of the levels in memory.

use super::sigs::*;
use super::sweep::*;
use super::*;
use crate::dict::elias_fano::EliasFanoBuilder;
use anyhow::{bail, ensure};
use derive_setters::*;
use dsi_progress_logger::no_logging;
use std::time::Instant;

/// The maximum number of keys of the last level, which is built without
/// bumping.
pub(super) const LAST_LEVEL_THRESHOLD: usize = 4096;

/// A builder for [`PHastR`].
///
/// The builder is passed to the constructors [`try_new_with_builder`] and
/// [`try_par_new_with_builder`], and its fields can be set using the
/// methods of the same name.
///
/// The defaults use 8-bit seeds, four patterns, slices of length 1024, and
/// an expected bucket size of 4.5 keys: a larger size (e.g., 4.75) reduces
/// space slightly, but more keys are bumped from the first level, and
/// queries for such keys are slower.
///
/// For 10-bit seeds, good parameters are slices of length 2048 and an
/// expected bucket size of 6 keys; seeds must then be stored in a
/// [`BitFieldVec`], and the function should be converted with
/// [`TryIntoUnaligned::try_into_unaligned`] to use unaligned reads.
///
/// [`try_new_with_builder`]: PHastR::try_new_with_builder
/// [`try_par_new_with_builder`]: PHastR::try_par_new_with_builder
#[derive(Setters, Debug, Clone)]
#[setters(generate = false)]
pub struct PHastRBuilder {
    /// The number of bits of a seed.
    ///
    /// The default is 8. Seeds stored in a `Box<[u8]>` can have at most 8
    /// bits, and all other seed storages at most 16 bits (see
    /// [`SeedStoreBuild::MAX_BITS`]).
    #[setters(generate = true)]
    pub(super) seed_bits: u32,

    /// The base-2 logarithm of the number of patterns.
    ///
    /// The default is 2. The in-slice offsets of the patterns are taken from
    /// disjoint blocks of bits of a 64-bit value, so the number of patterns
    /// times the base-2 logarithm of the slice length must be at most 64.
    #[setters(generate = true)]
    pub(super) log2_patterns: u32,

    /// The base-2 logarithm of the slice length.
    ///
    /// The default is 10. This is a maximum, as small levels use shorter
    /// slices. Slices should be at least as long as the number of seeds, as
    /// otherwise different seeds of a pattern map the keys of a bucket to the
    /// same slots.
    #[setters(generate = true)]
    pub(super) log2_slice_len: u32,

    /// The expected number of keys of a bucket.
    ///
    /// The default is 4.5.
    #[setters(generate = true)]
    pub(super) bucket_size: f64,

    /// The seed used to compute the hashes of the keys.
    ///
    /// The default is 0.
    #[setters(generate = true)]
    pub(super) seed: u64,

    /// The size-dependent components of the priority of a bucket, for sizes
    /// from one to seven (larger sizes are extrapolated linearly).
    ///
    /// By default, the weights depend on the number of bits of a seed and on
    /// the slice length.
    #[setters(generate = true, strip_option)]
    pub(super) weights: Option<[i64; 7]>,

    /// Keeps the hashes of the keys in temporary files, rather than in
    /// memory.
    ///
    /// The default is false. Only the constructors reading keys from a
    /// lender honor this setting, and the constructors reading keys from a
    /// slice return an error if it is set. Files are created in the
    /// directory specified by the environment variable `TMPDIR`, if set (see
    /// [`tempfile::tempfile`]).
    #[setters(generate = true)]
    pub(super) offline: bool,
}

impl PHastRBuilder {
    /// Builds a function on the keys of a slice, computing their hashes in
    /// parallel and keeping them in memory (see [`PHastR::try_par_new`]).
    pub(super) fn try_build_from_slice<K: ?Sized + ToSig<[u64; 1]> + Sync, D: SeedStoreBuild>(
        &self,
        keys: &[impl Borrow<K> + Sync],
        pl: &mut impl ProgressLog,
    ) -> Result<PHastR<K, D>> {
        ensure!(
            !self.offline,
            "Offline mode is not supported by the constructors reading keys from a slice; use try_new_with_builder"
        );
        self.check_params::<D>()?;
        let start = Instant::now();
        self.log_params(pl);
        let (levels, remap) = self.build_levels::<K>(keys, pl)?;
        let mut levels = levels.into_iter();
        let (params0, seeds0) = levels.next().expect("there is always at least one level");
        let seeds0 = D::from_seeds(&seeds0, self.seed_bits);
        let func = self.assemble(keys.len(), params0, seeds0, levels, remap);
        log_completion(start, keys.len(), pl);
        Ok(func)
    }

    /// Checks that the parameters are compatible with the seed storage.
    pub(super) fn check_params<D: SeedStoreBuild>(&self) -> Result<()> {
        if self.seed_bits == 0 || self.seed_bits > D::MAX_BITS {
            bail!(
                "The number of seed bits must be in [1 . . {}] for this seed storage",
                D::MAX_BITS
            );
        }
        if self.log2_patterns >= self.seed_bits {
            bail!("Too many patterns for the given number of seed bits");
        }
        if self.log2_slice_len > 16 {
            bail!("The slice length must be at most 65536");
        }
        if self.log2_patterns > 6 || (self.log2_slice_len << self.log2_patterns) > 64 {
            bail!("Too many patterns for the given slice length");
        }
        Ok(())
    }

    /// Logs the parameters of the construction.
    pub(super) fn log_params(&self, pl: &mut impl ProgressLog) {
        pl.info(format_args!(
            "Seed bits: {}; patterns: {}; maximum slice length: {}; expected bucket size: {}",
            self.seed_bits,
            1 << self.log2_patterns,
            1 << self.log2_slice_len,
            self.bucket_size
        ));
    }

    /// Assembles a function with `num_keys` keys from its first level (whose
    /// seeds are already stored), the following levels, and its remapping
    /// sequence.
    pub(super) fn assemble<K: ?Sized, D: SeedStoreBuild>(
        &self,
        num_keys: usize,
        params0: LevelParams,
        seeds0: D,
        levels: impl IntoIterator<Item = (LevelParams, Vec<u16>)>,
        remap: Remap,
    ) -> PHastR<K, D> {
        let seed = self.seed;
        let mut params = vec![];
        let mut seeds = vec![];
        for (mut p, s) in levels {
            p.first_seed = seeds.len() as u64;
            seeds.extend_from_slice(&s);
            params.push(p);
        }
        PHastR {
            seed,
            num_keys,
            pattern_shift: 6 - self.log2_patterns,
            default_shifts: 6 - self.log2_patterns == DEFAULT_PATTERN_SHIFT
                && params0.scale == DEFAULT_SCALE as u64,
            fast_scale: if 6 - self.log2_patterns == DEFAULT_PATTERN_SHIFT && params0.scale <= 3 {
                params0.scale as u8 + 1
            } else {
                0
            },
            params0,
            seeds0,
            params: params.into_boxed_slice(),
            seeds: D::from_seeds(&seeds, self.seed_bits),
            remap,
            _marker: std::marker::PhantomData,
        }
    }

    /// Builds the levels on the keys of a slice, returning their parameters
    /// and seeds, and the remapping sequence.
    #[allow(clippy::type_complexity)]
    fn build_levels<K: ?Sized + ToSig<[u64; 1]> + Sync>(
        &self,
        keys: &[impl Borrow<K> + Sync],
        pl: &mut impl ProgressLog,
    ) -> Result<(Vec<(LevelParams, Vec<u16>)>, Remap)> {
        let hash = |idx: usize, seed: u64| K::to_sig(keys[idx].borrow(), seed)[0];
        let n = keys.len();
        let mut levels: Vec<(LevelParams, Vec<u16>)> = vec![];
        let mut entries: Vec<usize> = vec![];
        let weights = |g: &Geometry| self.priority_weights(g);

        if n == 0 {
            // A single empty level, so that queries need no special case
            let geom = self.geometry(0, 1, self.bucket_size);
            levels.push((geom.level(), vec![0]));
            let efb = EliasFanoBuilder::new(0, 1);
            return Ok((levels, remap(efb)));
        }

        // The first level
        pl.item_name("key");
        pl.expected_updates(n);
        pl.start(format!(
            "Computing and distributing 64-bit signatures in parallel ({} threads) in RAM using seed 0x{:016x}...",
            num_threads(),
            self.seed
        ));
        let start = Instant::now();
        let geom = self.geometry(n, n, self.bucket_size);
        let sigs = group(n, |i| hash(i, self.seed), &geom);
        pl.done_with_count(n);
        log_signatures(start, n, pl);
        let out = sweep_bumping_level(&sigs, &geom, &weights(&geom), 0, n, pl);
        // The indices of the keys of the next level
        let mut cur = bumped(&sigs, &out.seeds, &geom);
        drop(sigs);
        pl.info(format_args!(
            "Level 0: {} keys, {} bumped ({:.3}%)",
            n,
            cur.len(),
            100.0 * cur.len() as f64 / n as f64
        ));
        let holes = self::holes(&out.occupied, n);
        debug_assert_eq!(holes.len(), cur.len());
        levels.push((geom.level(), out.seeds));
        let mut hole_idx = 0;
        let mut last_hole = 0;

        // Keys with the same signature are mapped to the same slot by every
        // seed, so they are bumped, and separated by the signatures of the
        // following levels, which depend on a second hash, unless the keys
        // are equal: since equal keys are bumped from the first level, we
        // detect them in the following levels, hashing with a third seed
        // the keys of a bucket with the same signature.
        let duplicates = |sigs: &Signatures, cur: &[usize], g: &Geometry| {
            let (mut part, mut begin, mut temp) = (vec![], vec![], vec![]);
            let mut bucket = vec![];
            let mut repeated = vec![];
            for p in 0..sigs.num_parts {
                let first = sigs.first_bucket(p);
                let range = first..first + sigs.buckets(p, g);
                sigs.load(p, g, range, &mut part, &mut begin, &mut temp);
                for w in begin.windows(2) {
                    bucket.clear();
                    bucket.extend_from_slice(&part[w[0]..w[1]]);
                    bucket.sort_unstable();
                    repeated.extend(bucket.windows(2).filter(|w| w[0] == w[1]).map(|w| w[0]));
                }
            }
            if repeated.is_empty() {
                return false;
            }
            repeated.sort_unstable();
            let mut other: Vec<(u64, u64)> = vec![];
            for chunk in 0..sigs.chunks {
                sigs.walk(chunk, |i, h| {
                    if repeated.binary_search(&h).is_ok() {
                        other.push((h, hash(cur[i], self.seed ^ THIRD_HASH)));
                    }
                });
            }
            other.sort_unstable();
            other.windows(2).any(|w| w[0] == w[1])
        };

        // The following levels: signatures are computed from the two
        // hashes of each key and the salt of the level
        while !cur.is_empty() {
            let k = cur.len();
            let group = |salt: u64, geom: &Geometry| {
                group(
                    k,
                    |i| {
                        let second = hash(cur[i], self.seed ^ SECOND_HASH);
                        level_sig(hash(cur[i], self.seed), second, salt)
                    },
                    geom,
                )
            };
            let (level, seeds, occupied, next, m) = if k > LAST_LEVEL_THRESHOLD {
                let salt = level_salt(self.seed, levels.len(), 0);
                let geom = self.geometry(k, k, self.bucket_size);
                let sigs = group(salt, &geom);
                if duplicates(&sigs, &cur, &geom) {
                    bail!("Duplicate keys");
                }
                let out = sweep_bumping_level(&sigs, &geom, &weights(&geom), levels.len(), k, pl);
                let next: Vec<usize> = bumped(&sigs, &out.seeds, &geom)
                    .into_iter()
                    .map(|i| cur[i])
                    .collect();
                let mut level = geom.level();
                level.salt = salt;
                (level, out.seeds, out.occupied, next, geom.m)
            } else {
                // The last level does not bump: we enlarge the range and
                // change the salt until we succeed
                let mut attempt = 0u64;
                loop {
                    let m = k + k / 4 + 16 + (attempt as usize / 8) * (k / 8 + 8);
                    let geom = self.geometry(k, m, self.bucket_size.min(3.0));
                    let salt = level_salt(self.seed, levels.len(), attempt);
                    let sigs = group(salt, &geom);
                    if attempt == 0 && duplicates(&sigs, &cur, &geom) {
                        bail!("Duplicate keys");
                    }
                    if let Some(out) =
                        sweep_level(&sigs, &geom, &weights(&geom), false, no_logging![])
                    {
                        let mut level = geom.level();
                        level.salt = salt;
                        break (level, out.seeds, out.occupied, vec![], geom.m);
                    }
                    attempt += 1;
                    if attempt > 1000 {
                        bail!("Could not build the last level");
                    }
                }
            };

            pl.info(format_args!(
                "Level {}: {} keys, {} bumped ({:.3}%)",
                levels.len(),
                k,
                next.len(),
                100.0 * next.len() as f64 / k as f64
            ));

            let mut level = level;
            level.offset = entries.len() as u64;
            for p in 0..m {
                if is_occupied(&occupied, p) {
                    last_hole = holes[hole_idx];
                    hole_idx += 1;
                }
                entries.push(last_hole);
            }
            levels.push((level, seeds));
            cur = next;
        }
        debug_assert_eq!(hole_idx, holes.len());

        let mut efb = EliasFanoBuilder::new(entries.len(), n);
        for &e in &entries {
            efb.push(e);
        }
        Ok((levels, remap(efb)))
    }

    /// Returns the size-dependent components of the priority of a bucket for
    /// a level of the given geometry: the default ones depend on the slice
    /// length, which can be smaller for small levels.
    pub(super) fn priority_weights(&self, g: &Geometry) -> [i64; 7] {
        self.weights
            .unwrap_or_else(|| default_weights(self.seed_bits, g.l_mask as usize + 1))
    }

    /// Computes the geometry of a level with `k` keys and output range `m`.
    pub(super) fn geometry(&self, k: usize, m: usize, bucket_size: f64) -> Geometry {
        let m = m.max(1);
        // As in PHast+ with wrapping, the slices of small levels span about
        // half of the output range
        let l = (m / 2 + 1)
            .next_power_of_two()
            .min(1 << self.log2_slice_len);
        // The number of buckets must be odd, as offsets are taken from the
        // product of the signature and the number of buckets
        let buckets = (k as f64 / bucket_size).round().max(1.0) as usize | 1;
        Geometry {
            buckets,
            bucket_width: u64::MAX / buckets as u64,
            num_slices: (m + 1 - l) as u64,
            l_mask: l as u64 - 1,
            // With slices shorter than the number of seeds, seeds are
            // redundant
            scale: l.ilog2().saturating_sub(self.seed_bits),
            log2_patterns: self.log2_patterns,
            seed_bits: self.seed_bits,
            m,
        }
    }
}

impl Default for PHastRBuilder {
    fn default() -> Self {
        Self {
            seed_bits: 8,
            log2_patterns: 2,
            log2_slice_len: 10,
            bucket_size: 4.5,
            seed: 0,
            weights: None,
            offline: false,
        }
    }
}

/// The geometry of a level during construction.
#[derive(Debug, Clone, Copy)]
pub(super) struct Geometry {
    /// The number of buckets (always odd).
    pub(super) buckets: usize,
    /// A lower bound for the number of signatures of a bucket.
    pub(super) bucket_width: u64,
    /// The number of slices.
    pub(super) num_slices: u64,
    /// The slice length minus one.
    pub(super) l_mask: u64,
    /// The base-2 logarithm of the amount by which a unit increase of the
    /// seed moves the keys of a bucket.
    pub(super) scale: u32,
    /// The base-2 logarithm of the number of patterns.
    pub(super) log2_patterns: u32,
    /// The number of bits of a seed.
    pub(super) seed_bits: u32,
    /// The output range.
    pub(super) m: usize,
}

impl Geometry {
    /// Returns the bucket of a key with the given signature.
    #[inline(always)]
    pub(super) fn bucket(&self, h: u64) -> usize {
        mul_hi(h, self.buckets as u64) as usize
    }

    /// Returns the beginning of the slice of a key with the given
    /// signature.
    #[inline(always)]
    pub(super) fn slice_begin(&self, h: u64) -> usize {
        mul_hi(h, self.num_slices) as usize
    }

    /// Returns a lower bound for the beginning of the slices of the keys
    /// of bucket `b` and of the following buckets.
    #[inline(always)]
    pub(super) fn first_slice(&self, b: usize) -> usize {
        // The signatures of bucket b are at least b · 2⁶⁴ / buckets
        mul_hi(b as u64 * self.bucket_width, self.num_slices) as usize
    }

    /// Returns the value providing the in-slice offsets of the patterns:
    /// the lower half of the product of the signature and the number of
    /// buckets, whose upper half is the bucket (it is thus uniform among
    /// the keys of a bucket).
    #[inline(always)]
    pub(super) fn offsets(&self, h: u64) -> u64 {
        h.wrapping_mul(self.buckets as u64)
    }

    /// Returns the slot of a key with the given signature when its bucket
    /// has the given seed (see [`PHastR::pos`]).
    #[inline(always)]
    pub(super) fn pos(&self, h: u64, seed: usize) -> usize {
        let r = seed & ((1 << self.log2_patterns) - 1);
        let offset = (self.offsets(h) >> (r as u32 * (64 >> self.log2_patterns)))
            .wrapping_add((seed as u64) << self.scale);
        self.slice_begin(h) + (offset & self.l_mask) as usize
    }

    /// Returns the parameters of a level with this geometry; the offset in
    /// the remapping sequence, the salt, and the index of the first seed are
    /// zero.
    pub(super) fn level(&self) -> LevelParams {
        LevelParams {
            buckets: self.buckets as u64,
            num_slices: self.num_slices,
            l_mask: self.l_mask,
            scale: self.scale as u64,
            offset: 0,
            salt: 0,
            first_seed: 0,
        }
    }
}

/// Default size-dependent priority weights: those of PHast+ with wrapping
/// and multiplier 3 in the implementation by Piotr Beling.
#[rustfmt::skip]
fn default_weights(seed_bits: u32, slice_len: usize) -> [i64; 7] {
    let w: [i32; 7] = match (seed_bits, slice_len) {
        (_, ..=64) => [-81342, 97738, 103193, 106305, 108524, 109876, 112382],
        (_, ..=128) => [-82883, 89250, 99246, 105030, 108983, 111224, 117058],
        (..=6, ..=256) => [-143420, 70364, 89794, 100431, 107778, 113842, 253543],
        (..=6, ..=512) => [-118906, 41451, 83177, 104570, 119520, 131788, 197543],
        (_, ..=256) => [-82828, 77192, 94710, 105243, 112716, 118768, 136225],
        (7, ..=512) => [-11540, 68580, 98218, 115370, 128607, 139118, 145832],
        (_, ..=512) => [25100, 89361, 117113, 134755, 147369, 154606, 172378],
        (8, ..=1024) => [-50649, 63792, 110014, 139267, 161285, 176594, 188305],
        (..=8, ..=2048) => [-3427, 10388, 90470, 141895, 179413, 208576, 232553],
        (9, ..=1024) => [-41757, 60279, 113069, 143467, 162892, 179091, 188139],
        (..=9, ..=2048) => [-3753, 11840, 77702, 132696, 169641, 200687, 218764],
        (10, ..=1024) => [-2394, 29640, 81921, 108732, 126229, 141102, 150457],
        (..=10, ..=2048) => [-3417, 13564, 81208, 133035, 168506, 198114, 214382],
        (11, ..=1024) => [-1555, 25982, 126717, 155711, 174202, 191358, 198247],
        (11, ..=2048) => [-2229, 21208, 88554, 137643, 169905, 200075, 213746],
        (11, _) => [-3267, 25041, 24325, 40786, 100528, 155125, 182822],
        (_, ..=1024) => [-2206, 33628, 110901, 143147, 161228, 177559, 183794],
        (_, ..=2048) => [-2665, 16252, 98048, 149519, 183487, 214959, 227347],
        (_, _) => [-3356, 26074, 26278, 44692, 94747, 143426, 168599],
    };
    w.map(|x| x as i64)
}

/// Returns the salt of a level after the first one (see [`level_sig`]): a
/// value depending on the seed of the function, on the level, and, for the
/// last level, on the attempt.
#[inline(always)]
pub(super) const fn level_salt(seed: u64, level: usize, attempt: u64) -> u64 {
    mix(
        seed.wrapping_add(level as u64).wrapping_add(attempt << 32) | 1,
        0x9E37_79B9_7F4A_7C15,
    )
}

/// Logs the completion of the computation of the signatures of `n` keys,
/// started at `start`.
pub(super) fn log_signatures(start: Instant, n: usize, pl: &mut impl ProgressLog) {
    pl.info(format_args!(
        "Computation of signatures from inputs completed in {:.3} seconds ({} keys, {:.3} ns/key)",
        start.elapsed().as_secs_f64(),
        n,
        start.elapsed().as_nanos() as f64 / n.max(1) as f64
    ));
}

/// Logs the completion of the construction of a function on `n` keys,
/// started at `start`.
pub(super) fn log_completion(start: Instant, n: usize, pl: &mut impl ProgressLog) {
    pl.info(format_args!(
        "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
        start.elapsed().as_secs_f64(),
        n,
        start.elapsed().as_nanos() as f64 / n.max(1) as f64
    ));
}

/// Returns the number of threads of the current rayon pool (one without
/// the `rayon` feature).
pub(super) fn num_threads() -> usize {
    #[cfg(feature = "rayon")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "rayon"))]
    {
        1
    }
}

/// Returns whether to process `len` elements using rayon, that is, whether
/// `len` is large enough and the current pool has more than one thread
/// (with a single thread, dispatching work to the pool only adds overhead,
/// and if the process is restricted to a single CPU the thread waiting for
/// the pool competes with the worker).
#[cfg(feature = "rayon")]
#[inline]
pub(super) fn parallel(len: usize) -> bool {
    len >= PAR_MIN_LEN && rayon::current_num_threads() > 1
}

/// Minimum number of elements for which a pass is run in parallel: below
/// this size the overhead of parallelism exceeds its benefits.
#[cfg(feature = "rayon")]
const PAR_MIN_LEN: usize = 1 << 17;

/// Builds a [`Remap`].
pub(super) fn remap(efb: EliasFanoBuilder<usize>) -> Remap {
    // SAFETY: the selection structure is built on the high bits
    unsafe { efb.build().map_high_bits(SelectAdaptConst::new) }
}
