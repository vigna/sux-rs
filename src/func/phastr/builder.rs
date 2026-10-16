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
use std::path::PathBuf;
use std::time::Instant;

/// The maximum number of keys of the last level, which is built without
/// bumping.
pub(super) const LAST_LEVEL_THRESHOLD: usize = 4096;

/// The maximum number of attempts at building the last level.
pub(super) const LAST_LEVEL_ATTEMPTS: u64 = 1000;

/// The maximum number of keys: on 32-bit platforms, the indices of the
/// slots used during a sweep can overflow on almost 2³² keys.
pub(super) const MAX_KEYS: usize = usize::MAX / 2;

/// Returns an error if a level with `k` keys bumped all of them, as then the
/// following levels would do the same.
pub(super) fn check_progress(
    level: usize,
    k: usize,
    bumped: usize,
    bucket_size: f64,
) -> Result<()> {
    ensure!(
        bumped < k,
        "No bucket of level {level} ({k} keys) could be placed: the expected bucket size ({bucket_size}) is too large for the slice length"
    );
    Ok(())
}

/// Logs the number of keys of the level of index `index` and the number of
/// keys bumped from it.
pub(super) fn log_level(index: usize, k: usize, bumped: usize, pl: &mut impl ProgressLog) {
    pl.info(format_args!(
        "Level {index}: {k} keys, {bumped} bumped ({:.3}%)",
        100.0 * bumped as f64 / k as f64
    ));
}

/// A builder for [`PHastR`].
///
/// The builder is passed to the constructors [`try_new_with_builder`] and
/// [`try_par_new_with_builder`] of [`PHastR`], and to the constructors with
/// the same names of a [`SignedFunc`] wrapping a [`PHastR`] (see its
/// [supported combinations]); its fields can be set using the methods of the
/// same name.
///
/// The defaults use 8-bit seeds, four layouts, slices of length 1024, and
/// an expected bucket size of 4.25 keys; see the documentation of [`PHastR`]
/// for [when to change them].
///
/// [`try_new_with_builder`]: PHastR::try_new_with_builder
/// [`try_par_new_with_builder`]: PHastR::try_par_new_with_builder
/// [`SignedFunc`]: crate::func::SignedFunc
/// [supported combinations]: crate::func::SignedFunc#supported-combinations
/// [when to change them]: PHastR#changing-the-defaults
#[derive(Setters, Debug, Clone)]
#[setters(generate = false)]
pub struct PHastRBuilder {
    /// The number of bits of a bucket seed.
    ///
    /// The default is 8. Seeds stored in a `Box<[u8]>` can have at most 8
    /// bits, and seeds stored in a [`BitFieldVec`] at most 16 bits (see
    /// [`SeedStoreBuild::MAX_BITS`]). The number of bits of a seed must be at
    /// least two, and larger than the [base-2 logarithm of the number of
    /// layouts]; with fewer than five bits, construction might fail,
    /// in particular with a single layout.
    ///
    /// [base-2 logarithm of the number of layouts]: Self::log2_layouts
    #[setters(generate = true)]
    pub(super) seed_bits: u32,

    /// The base-2 logarithm of the number of layouts.
    ///
    /// The default is 2, and the maximum is 6. The in-slice offsets of the
    /// layouts use disjoint bits of a 64-bit value, so the number of layouts
    /// times the base-2 logarithm of the slice length must be at most 64.
    #[setters(generate = true)]
    pub(super) log2_layouts: u32,

    /// The base-2 logarithm of the slice length.
    ///
    /// The default is 10, the minimum is 3, and the maximum is 16. This is a
    /// maximum, as small levels use shorter slices. Slices should have at
    /// least as many slots as there are seeds, as otherwise different
    /// rotations of a layout map the keys of a bucket to the same slots;
    /// with slices shorter than 64, construction might fail, in particular
    /// with a single layout.
    #[setters(generate = true)]
    pub(super) log2_slice_len: u32,

    /// The expected number of keys of a bucket.
    ///
    /// The default is 4.25. The size must be in [1 . . 64], and at most
    /// one eighth of the slice length; construction fails if a level cannot
    /// place any bucket.
    #[setters(generate = true)]
    pub(super) bucket_size: f64,

    /// The hash seed, used to compute the signatures of the keys (see
    /// [`ToSig`]).
    ///
    /// The default is 0.
    #[setters(generate = true)]
    pub(super) seed: u64,

    /// The weights of the priority of a bucket, which depend on its size,
    /// for sizes from one to seven (larger sizes are extrapolated linearly).
    ///
    /// By default, the weights depend only on the number of bits of a seed:
    /// they were tuned for 8-bit seeds and for 10-bit seeds. Weights must
    /// be in [−2³² . . 2³²].
    #[setters(generate = true, strip_option)]
    pub(super) weights: Option<[i64; 7]>,

    /// Keeps the signatures of the keys in temporary files, rather than in
    /// memory.
    ///
    /// The default is false. Only the constructors reading keys from a
    /// lender honor this setting, and the constructors reading keys from a
    /// slice return an error if it is set. Files are created in the
    /// [temporary directory], which must have about sixteen bytes of free
    /// space per key.
    ///
    /// Offline construction uses 256 MiB of buffers up to about 8.6 · 10⁹
    /// keys, and then about a sixteenth of a byte per key (at most 16 GiB),
    /// so it is useful only on large key sets (see [`PHastR`] for the memory
    /// used by the construction).
    ///
    /// [temporary directory]: Self::temp_dir
    #[setters(generate = true)]
    pub(super) offline: bool,

    /// The directory of the temporary files of [offline mode].
    ///
    /// By default, the default temporary directory (see
    /// [`tempfile::env::temp_dir`]; on Unix, the directory specified by the
    /// environment variable `TMPDIR`, if set); note that on some systems the
    /// default temporary directory is in memory (e.g., a `tmpfs`). Libraries
    /// should use this setting to let their users choose where temporary
    /// files are written, as the default temporary directory can be changed
    /// only process-wide.
    ///
    /// [offline mode]: Self::offline
    #[setters(generate = true, strip_option, into)]
    pub(super) temp_dir: Option<PathBuf>,

    /// The maximum number of chunks in which a level is swept in parallel.
    ///
    /// By default, the number of threads of the current [rayon] pool (one
    /// without the `rayon` feature). Since the function built depends on
    /// the number of chunks, by default it depends on the size of the pool:
    /// setting this value makes it independent of the pool (the chunks are
    /// still processed by its threads). Levels too small to be split into
    /// this number of chunks are swept in fewer chunks, depending only on
    /// their size.
    ///
    /// With fewer chunks than threads, sweeps use fewer threads, and if the
    /// number of chunks is not a multiple of the number of threads, some
    /// threads are idle at the end of a sweep. Each additional chunk
    /// requires the sweep of a short gap, which makes construction slower
    /// and the function slightly larger: for example, on 10⁸ keys with one
    /// thread, construction in 16 chunks is about 3% slower than in one
    /// chunk, and in 64 chunks about 10% slower, using 0.003 more bits per
    /// key. To build the same function with any number of threads, a good
    /// choice is thus the largest number of threads that will be used.
    ///
    /// [rayon]: https://docs.rs/rayon
    #[setters(generate = true, strip_option)]
    pub(super) sweep_chunks: Option<usize>,

    /// The expected number of keys, used only to log the progress of the
    /// constructors reading keys from a lender.
    ///
    /// The constructors reading keys from a slice know the number of keys,
    /// and ignore this setting. The function built does not depend on it.
    #[setters(generate = true, strip_option)]
    pub(super) expected_num_keys: Option<usize>,
}

impl PHastRBuilder {
    /// Builds a function on a slice of keys, computing their signatures in
    /// parallel and keeping them in memory (see [`PHastR::try_par_new`]).
    pub(super) fn try_build_from_slice<
        K: ?Sized + ToSig<[u64; 1]> + Sync,
        D: SeedStoreBuild + MemSize + FlatType,
    >(
        &self,
        keys: &[impl Borrow<K> + Sync],
        pl: &mut impl ProgressLog,
    ) -> Result<PHastR<K, D>> {
        ensure!(
            !self.offline,
            "Offline mode is not supported by the constructors reading keys from a slice; use try_new_with_builder"
        );
        self.check_params::<D>()?;
        ensure!(
            keys.len() <= MAX_KEYS,
            "Too many keys for this platform ({})",
            keys.len()
        );
        let start = Instant::now();
        self.log_params(pl);
        let (levels, remap) = self.build_levels::<K>(keys, pl)?;
        let mut levels = levels.into_iter();
        let (params0, seeds0) = levels.next().expect("there is always at least one level");
        let seeds0 = D::from_seeds(&seeds0, self.seed_bits);
        let func = self.assemble(keys.len(), params0, seeds0, levels, remap);
        log_completion("Construction", start, keys.len(), pl);
        log_bits_per_key(&func, pl);
        Ok(func)
    }

    /// Checks that the parameters are valid and compatible with the seed
    /// storage.
    pub(super) fn check_params<D: SeedStoreBuild>(&self) -> Result<()> {
        // Seeds are kept in 16-bit integers during construction
        let max_bits = D::MAX_BITS.min(16);
        if self.seed_bits < 2 || self.seed_bits > max_bits {
            bail!(
                "The number of seed bits must be in [2..{max_bits}] for this seed storage (got {})",
                self.seed_bits
            );
        }
        if self.log2_layouts >= self.seed_bits {
            bail!(
                "Too many layouts ({}) for {}-bit seeds",
                1u64 << self.log2_layouts.min(63),
                self.seed_bits
            );
        }
        if !(3..=16).contains(&self.log2_slice_len) {
            bail!(
                "The base-2 logarithm of the slice length must be in [3..16] (got {})",
                self.log2_slice_len
            );
        }
        if self.log2_layouts > 6 || (self.log2_slice_len << self.log2_layouts) > 64 {
            bail!(
                "Too many layouts ({}) for slices of length {}",
                1u64 << self.log2_layouts.min(63),
                1 << self.log2_slice_len
            );
        }
        // Buckets must be much smaller than slices, as otherwise levels place
        // few buckets, or none
        let max_bucket_size = ((1u64 << self.log2_slice_len) / 8).min(64) as f64;
        if !(1.0..=max_bucket_size).contains(&self.bucket_size) {
            bail!(
                "The expected bucket size must be in [1..{max_bucket_size}] for slices of length {} (got {})",
                1 << self.log2_slice_len,
                self.bucket_size
            );
        }
        if let Some(weights) = self.weights {
            if weights.iter().any(|w| w.unsigned_abs() > 1 << 32) {
                bail!("The weights must be in [-2^32..2^32] (got {weights:?})");
            }
        }
        if self.sweep_chunks == Some(0) {
            bail!("The number of sweep chunks must be positive");
        }
        Ok(())
    }

    /// Returns the directory of the temporary files, if in offline mode, or
    /// `None`.
    pub(super) fn offline_dir(&self) -> Option<PathBuf> {
        self.offline.then(|| {
            self.temp_dir
                .clone()
                .unwrap_or_else(tempfile::env::temp_dir)
        })
    }

    /// Returns the maximum number of chunks of a parallel sweep (see
    /// [`sweep_chunks`](Self::sweep_chunks)).
    fn max_sweep_chunks(&self) -> usize {
        self.sweep_chunks.unwrap_or_else(num_threads)
    }

    /// Logs the parameters of the construction.
    pub(super) fn log_params(&self, pl: &mut impl ProgressLog) {
        pl.info(format_args!(
            "Seed bits: {}; layouts: {}; maximum slice length: {}; expected bucket size: {}; maximum sweep chunks: {}",
            self.seed_bits,
            1 << self.log2_layouts,
            1 << self.log2_slice_len,
            self.bucket_size,
            self.max_sweep_chunks()
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
        let layout_shift = 6 - self.log2_layouts;
        let default_layouts = layout_shift == DEFAULT_LAYOUT_SHIFT;
        PHastR {
            seed,
            num_keys,
            layout_shift,
            default_shifts: default_layouts && params0.scale == DEFAULT_SCALE as u64,
            fast_scale: if default_layouts && params0.scale <= 3 {
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
        // The sets of used slots and the output ranges of the levels after
        // the first one
        let mut used: Vec<(UsedSlots, usize)> = vec![];
        let weights = self.priority_weights();

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
        // As in group
        #[cfg(feature = "rayon")]
        let par = parallel(n);
        #[cfg(not(feature = "rayon"))]
        let par = false;
        pl.start(format!(
            "Computing and distributing 64-bit signatures {} in RAM using seed 0x{:016x}...",
            if par {
                format!("in parallel ({} threads)", num_threads())
            } else {
                "sequentially".to_owned()
            },
            self.seed
        ));
        let start = Instant::now();
        let geom = self.geometry(n, n, self.bucket_size);
        let sigs = group(n, |i| hash(i, self.seed), &geom);
        pl.done_with_count(n);
        log_completion("Computation of signatures from inputs", start, n, pl);
        let out = sweep_bumping_level(&sigs, &geom, &weights, 0, n, pl);
        // The indices of the keys of the next level
        let mut cur = bumped(&sigs, &out.seeds, &geom);
        drop(sigs);
        log_level(0, n, cur.len(), pl);
        let holes = self::holes(&out.occupied, n);
        debug_assert_eq!(holes.len(), cur.len());
        levels.push((geom.level(), out.seeds));

        // Keys with the same signature are mapped to the same slot by every
        // seed, so they are bumped, and separated by the signatures of the
        // following levels, which depend on a second hash, unless the keys
        // are equal: since equal keys are bumped from the first level, we
        // detect them in the second level, hashing with a third seed the
        // keys of a bucket with the same signature.
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
            for batch in 0..sigs.batches {
                sigs.walk(batch, |i, h| {
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
        let mut offset = 0;
        while !cur.is_empty() {
            let k = cur.len();
            let index = levels.len();
            let sig = |i: usize, salt: u64| {
                let second = hash(cur[i], self.seed ^ SECOND_HASH);
                level_sig(hash(cur[i], self.seed), second, salt)
            };
            let (mut level, seeds, occupied, next, m) = if k > LAST_LEVEL_THRESHOLD {
                let salt = level_salt(self.seed, index, 0);
                let geom = self.geometry(k, k, self.bucket_size);
                let sigs = group(k, |i| sig(i, salt), &geom);
                if index == 1 && duplicates(&sigs, &cur, &geom) {
                    bail!("Duplicate keys");
                }
                let out = sweep_bumping_level(&sigs, &geom, &weights, index, k, pl);
                let next: Vec<usize> = bumped(&sigs, &out.seeds, &geom)
                    .into_iter()
                    .map(|i| cur[i])
                    .collect();
                check_progress(index, k, next.len(), self.bucket_size)?;
                let mut level = geom.level();
                level.salt = salt;
                (level, out.seeds, out.occupied, next, geom.m)
            } else {
                let (level, out, m) = self.last_level(index, k, &weights, sig, |sigs, geom| {
                    duplicates(sigs, &cur, geom)
                })?;
                (level, out.seeds, out.occupied, vec![], m)
            };
            log_level(index, k, next.len(), pl);
            level.offset = offset as u64;
            offset += m;
            levels.push((level, seeds));
            used.push((occupied, m));
            cur = next;
        }

        Ok((levels, build_remap(holes, &used, n)))
    }

    /// Builds the last level, of index `index` and with `k` keys, which does
    /// not bump keys: we change the salt until we succeed, enlarging the
    /// output range every eight attempts. The signature of the key of index
    /// *i* with a given salt is `sig(i, salt)`.
    ///
    /// If the level is the second one, `duplicates` is called at the first
    /// attempt on the signatures and on the geometry of the level, and an
    /// error is returned if it returns true (see [`build_levels`]).
    ///
    /// Returns the parameters of the level, its seeds and its used slots, and
    /// its output range.
    ///
    /// [`build_levels`]: Self::build_levels
    pub(super) fn last_level(
        &self,
        index: usize,
        k: usize,
        weights: &[i64; 7],
        sig: impl Fn(usize, u64) -> u64 + Sync,
        duplicates: impl Fn(&Signatures, &Geometry) -> bool,
    ) -> Result<(LevelParams, Level, usize)> {
        for attempt in 0..LAST_LEVEL_ATTEMPTS {
            let m = k + k / 4 + 16 + (attempt as usize / 8) * (k / 8 + 8);
            let geom = self.geometry(k, m, self.bucket_size.min(3.0));
            let salt = level_salt(self.seed, index, attempt);
            let sigs = group(k, |i| sig(i, salt), &geom);
            if index == 1 && attempt == 0 && duplicates(&sigs, &geom) {
                bail!("Duplicate keys");
            }
            if let Some(out) = sweep_level(&sigs, &geom, weights, false, no_logging![]) {
                let mut level = geom.level();
                level.salt = salt;
                return Ok((level, out, geom.m));
            }
        }
        bail!(
            "Could not build the last level ({k} keys) in {LAST_LEVEL_ATTEMPTS} attempts: the slice length or the number of bits of a seed is probably too small"
        )
    }

    /// Returns the weights of the priority of a bucket, which depend on its
    /// size: the default ones depend only on the number of bits of a seed.
    pub(super) fn priority_weights(&self) -> [i64; 7] {
        self.weights
            .unwrap_or_else(|| default_weights(self.seed_bits))
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
        let num_slices = (m + 1 - l) as u64;
        Geometry {
            buckets,
            slices_per_bucket: ((num_slices as u128) << 64) / buckets as u128,
            num_slices,
            l_mask: l as u64 - 1,
            // With slices having fewer slots than there are seeds, the seed
            // is not shifted, and rotations are redundant
            scale: l.ilog2().saturating_sub(self.seed_bits),
            log2_layouts: self.log2_layouts,
            seed_bits: self.seed_bits,
            m,
            max_chunks: self.max_sweep_chunks(),
        }
    }
}

impl Default for PHastRBuilder {
    fn default() -> Self {
        Self {
            seed_bits: 8,
            log2_layouts: 2,
            log2_slice_len: 10,
            bucket_size: 4.25,
            seed: 0,
            weights: None,
            offline: false,
            temp_dir: None,
            sweep_chunks: None,
            expected_num_keys: None,
        }
    }
}

/// The geometry of a level during construction.
#[derive(Debug, Clone, Copy)]
pub(super) struct Geometry {
    /// The number of buckets (always odd).
    pub(super) buckets: usize,
    /// The number of possible slice starts divided by the number of buckets,
    /// in 64.64 fixed point (rounded down).
    pub(super) slices_per_bucket: u128,
    /// The number of possible slice starts (the output range minus the slice
    /// length, plus one).
    pub(super) num_slices: u64,
    /// The slice length minus one.
    pub(super) l_mask: u64,
    /// The base-2 logarithm of the amount *u* by which a unit increase of
    /// the seed moves the keys of a bucket.
    pub(super) scale: u32,
    /// The base-2 logarithm of the number of layouts.
    pub(super) log2_layouts: u32,
    /// The number of bits of a seed.
    pub(super) seed_bits: u32,
    /// The output range.
    pub(super) m: usize,
    /// The maximum number of chunks of a parallel sweep (see
    /// [`sweep_chunks`]).
    ///
    /// [`sweep_chunks`]: Self::sweep_chunks
    pub(super) max_chunks: usize,
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
    /// of bucket `b` and of the following buckets; the slices of the keys
    /// of bucket `b` begin at most [`max_slice_distance`] slots after it.
    ///
    /// [`max_slice_distance`]: Self::max_slice_distance
    #[inline(always)]
    pub(super) fn first_slice(&self, b: usize) -> usize {
        // We return ⌊b · slices_per_bucket / 2⁶⁴⌋. The signatures of bucket b
        // are at least b · 2⁶⁴ / buckets, so their slices begin at least at
        // ⌊b · num_slices / buckets⌋, which is at least our value, and at
        // most one more, as the error of slices_per_bucket is less than one
        let (hi, lo) = (
            (self.slices_per_bucket >> 64) as u64,
            self.slices_per_bucket as u64,
        );
        (b as u64 * hi + mul_hi(b as u64, lo)) as usize
    }

    /// Returns the maximum distance between the [first slice] of bucket `b`
    /// and the beginning of the slice of a key of bucket `b`.
    ///
    /// The signatures of bucket `b` are less than (`b` + 1) · 2⁶⁴ /
    /// `buckets`, so their slices begin at most at ⌊`b` · `num_slices` /
    /// `buckets`⌋ + ⌈`num_slices` / `buckets`⌉, and the first slice of
    /// bucket `b` is at least ⌊`b` · `num_slices` / `buckets`⌋ − 1.
    ///
    /// [first slice]: Self::first_slice
    #[inline(always)]
    pub(super) fn max_slice_distance(&self) -> usize {
        (self.num_slices as usize).div_ceil(self.buckets) + 1
    }

    /// Returns the number of buckets between chunks of a parallel sweep
    /// whose slots cannot overlap (as in PHast): the beginning of the slice
    /// grows by `num_slices` / `buckets` per bucket, and the slots of a key
    /// lie at most *L* − 1 slots after the beginning of its slice, so
    /// considering the rounding of the beginning of the slices it suffices
    /// that the gap times `num_slices` / `buckets` is at least *L*.
    #[inline(always)]
    pub(super) fn chunk_gap(&self) -> u64 {
        // In 128 bits, as the product might overflow (in particular, on
        // 32-bit platforms, it does in 32 bits)
        ((self.l_mask as u128 + 1) * self.buckets as u128 / self.num_slices as u128) as u64 + 1
    }

    /// Returns the number of chunks of a parallel sweep: at most
    /// [`max_chunks`], and such that each chunk has at least 4096 buckets
    /// and 64 times the [gap between chunks]; it thus depends only on the
    /// geometry.
    ///
    /// [`max_chunks`]: Self::max_chunks
    /// [gap between chunks]: Self::chunk_gap
    pub(super) fn sweep_chunks(&self) -> usize {
        // In 64 bits, as the gap of small levels can be very large
        let max_by_size = self.buckets as u64 / (64 * self.chunk_gap()).max(4096);
        (self.max_chunks as u64).min(max_by_size).max(1) as usize
    }

    /// Returns the value providing the in-slice offsets of the layouts:
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
        let r = seed & ((1 << self.log2_layouts) - 1);
        let offset = (self.offsets(h) >> (r as u32 * (64 >> self.log2_layouts)))
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

/// Default size-dependent priority weights, used for all slice lengths.
///
/// The weights were tuned for PHastR with four layouts by coordinate
/// descent on the space of single-threaded constructions on 10⁷ keys: the
/// weights for seeds of at most eight bits with 8-bit seeds, slices of length
/// 1024 and an expected bucket size of 4.25, and the weights for longer seeds
/// with 10-bit seeds, slices of length 2048 and an expected bucket size of
/// 5.75.
const fn default_weights(seed_bits: u32) -> [i64; 7] {
    if seed_bits <= 8 {
        [-52000, 53315, 105861, 138629, 156944, 175176, 186591]
    } else {
        [46000, 55041, 137806, 184081, 217321, 241846, 256427]
    }
}

/// Returns the salt of a level after the first one (see [`level_sig`]): a
/// value depending on the seed of the function, on the level, and, for the
/// last level, on the attempt.
#[inline(always)]
pub(super) const fn level_salt(seed: u64, level: usize, attempt: u64) -> u64 {
    folded_multiply(
        seed.wrapping_add(level as u64).wrapping_add(attempt << 32),
        0x9E37_79B9_7F4A_7C15,
    )
}

/// Logs the completion of a phase (e.g., the computation of the signatures,
/// or the whole construction) on `n` keys, started at `start`.
pub(super) fn log_completion(phase: &str, start: Instant, n: usize, pl: &mut impl ProgressLog) {
    pl.info(format_args!(
        "{phase} completed in {:.3} seconds ({} keys, {:.3} ns/key)",
        start.elapsed().as_secs_f64(),
        n,
        start.elapsed().as_nanos() as f64 / n.max(1) as f64
    ));
}

/// Logs the space used by a function, in bits per key.
pub(super) fn log_bits_per_key<K: ?Sized, D: MemSize + FlatType>(
    func: &PHastR<K, D>,
    pl: &mut impl ProgressLog,
) {
    pl.info(format_args!(
        "Bits/key: {:.3}",
        func.mem_size(SizeFlags::default()) as f64 * 8.0 / func.len().max(1) as f64
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

/// Builds the remapping sequence of a function with `n` keys, given the free
/// slots of its first level, in increasing order, and the used slots and the
/// output range of each following level: the used slots of the following
/// levels are mapped, in order, to the free slots of the first level, and
/// their free slots to the last free slot assigned.
pub(super) fn build_remap(
    holes: impl IntoIterator<Item = usize>,
    used: &[(UsedSlots, usize)],
    n: usize,
) -> Remap {
    let mut efb = EliasFanoBuilder::new(used.iter().map(|&(_, m)| m).sum(), n);
    let mut holes = holes.into_iter();
    let mut last_hole = 0;
    for (occupied, m) in used {
        for p in 0..*m {
            if is_occupied(occupied, p) {
                last_hole = holes
                    .next()
                    .expect("there is a free slot for each bumped key");
            }
            efb.push(last_hole);
        }
    }
    debug_assert!(holes.next().is_none());
    remap(efb)
}

/// Builds a [`Remap`], whose low bits are padded with a zero word, so that
/// conversion to [unaligned reads] needs no reallocation (as for the seeds).
///
/// [unaligned reads]: TryIntoUnaligned
pub(super) fn remap(efb: EliasFanoBuilder<usize>) -> Remap {
    // SAFETY: the selection structure is built on the high bits, and the
    // padded low bits contain the same values
    unsafe {
        efb.build()
            .map_high_bits(SelectAdaptConst::new)
            .map_low_bits(|low| {
                let (bits, bit_width, len) = low.into_raw_parts();
                let mut bits = bits.into_vec();
                bits.reserve_exact(1);
                bits.push(0);
                crate::bits::BitFieldVec::from_raw_parts(bits.into_boxed_slice(), bit_width, len)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The geometries of levels with `n` keys, for the values of `n` that
    /// fit a `usize`, and some expected bucket sizes and slice lengths.
    fn geometries(sizes: &[u64]) -> impl Iterator<Item = Geometry> + '_ {
        sizes
            .iter()
            .filter_map(|&n| usize::try_from(n).ok())
            .flat_map(|n| {
                [(4.25, 10), (5.75, 11), (1.0, 3), (64.0, 16)]
                    .into_iter()
                    .map(move |(bucket_size, log2_slice_len)| {
                        PHastRBuilder::default()
                            .log2_slice_len(log2_slice_len)
                            .geometry(n, n, bucket_size)
                    })
            })
    }

    const SIZES: [u64; 9] = [
        1,
        10,
        4097,
        20_000_000,
        1 << 32,
        20_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
        1 << 50,
    ];

    #[test]
    fn test_first_slice() {
        // The first slice of a bucket is at most the beginning of the
        // slice of its smallest signature, and at most max_slice_distance
        // slots before that of its largest signature (sweeps size their
        // tables accordingly), even on huge levels
        for g in geometries(&SIZES) {
            let nb = g.buckets as u128;
            let mut buckets = vec![
                0,
                1,
                2,
                nb / 2,
                nb.saturating_sub(3),
                nb.saturating_sub(2),
                nb - 1,
            ];
            buckets.retain(|&b| b < nb);
            for b in buckets {
                let first = ((b << 64).div_ceil(nb)) as u64;
                let last = if b == nb - 1 {
                    u64::MAX
                } else {
                    (((b + 1) << 64).div_ceil(nb) - 1) as u64
                };
                let b = b as usize;
                assert_eq!(g.bucket(first), b);
                assert_eq!(g.bucket(last), b);
                assert!(g.first_slice(b) <= g.slice_begin(first));
                assert!(
                    g.slice_begin(last) - g.first_slice(b) <= g.max_slice_distance(),
                    "bucket {b} of {nb}: {} - {} > {}",
                    g.slice_begin(last),
                    g.first_slice(b),
                    g.max_slice_distance()
                );
            }
        }
    }

    #[test]
    fn test_chunk_gap() {
        // The gap is computed without overflow on 32-bit platforms
        for g in geometries(&SIZES) {
            let expected = (g.l_mask as u128 + 1) * g.buckets as u128 / g.num_slices as u128 + 1;
            assert_eq!(g.chunk_gap() as u128, expected);
        }
    }

    #[test]
    fn test_level_salt() {
        // Different levels and attempts have different salts
        for seed in [0, 1, u64::MAX] {
            let mut salts: Vec<u64> = (1..1000)
                .map(|level| level_salt(seed, level, 0))
                .chain((1..1000).map(|attempt| level_salt(seed, 5, attempt)))
                .collect();
            let len = salts.len();
            salts.sort_unstable();
            salts.dedup();
            assert_eq!(salts.len(), len);
        }
    }
}
