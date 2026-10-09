/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! PHast-R: bucket-placement minimal perfect hashing with rings of patterns.
//!
//! PHast-R is a minimal perfect hash function based on *bucket placement*:
//! keys are hashed to buckets, and each bucket stores a fixed-width *seed*
//! that places the keys of the bucket in the output range so that no two keys
//! collide. For almost all keys, a query needs just a hash and the access to
//! a seed.
//!
//! The bucket of a key is a linear function of its 64-bit hash, and the seed
//! of a bucket places each of its keys inside the *slice* of the key, a small
//! range of consecutive slots whose start grows linearly with the hash. A seed
//! is *feasible* for a bucket if it maps its keys to distinct free slots.
//! Buckets are processed by a sweep that favors large buckets, and each
//! bucket gets the feasible seed minimizing the sum of the slots of its keys.
//! Buckets for which no seed is feasible are *bumped* to the next level, and
//! an [Elias–Fano] sequence maps the outputs of the following levels to the
//! free slots of the first one. Bumped keys cost space and make queries
//! slower, so the quality of a placement is measured by how few keys it
//! bumps.
//!
//! PHast-R uses a placement that bumps few keys and makes it possible to find
//! feasible seeds with bit-parallel operations. Each key has *R* independent
//! in-slice offsets, one for each of *R* *patterns*. The lowest bits of a
//! seed choose a pattern, which gives each key of the bucket a *base
//! position*, and the remaining bits choose a *rotation*, which moves all the
//! keys of the bucket by the same multiple of a *stride* *T*, wrapping around
//! the end of their slices. As the rotation varies, a key goes exactly once
//! through the slots of its slice whose distance from its base position is a
//! multiple of *T*: we call these slots its *ring*. During construction the
//! set of used slots is stored by residue classes modulo the stride, so the
//! bits of the ring of a key are a block of consecutive bits, and the
//! feasible rotations of a pattern are found by rotating and combining by a
//! bitwise OR one such block for each key.
//!
//! Two keys of a bucket with the same base position move together, and thus
//! collide for (almost) all rotations, but they will not, in general, collide
//! in the other patterns: such *self-collisions* thus disappear, and the
//! patterns provide (almost) independent trials.
//!
//! This placement improves on those of PHast and PHast+ (see the
//! [references]). PHast uses a pseudorandom placement, which bumps few keys,
//! but finding a feasible seed requires trying all seeds of the bucket, so
//! construction is more than ten times slower. PHast+ moves the keys of a
//! bucket along a single rigid pattern, so feasible seeds can be found with
//! bit-parallel operations, but self-collisions cannot be avoided: two keys
//! with the same base position collide for (almost) all seeds, and their
//! bucket must be bumped. With 8-bit seeds, self-collisions cause almost 40%
//! of the bumping of PHast+ with wrapping.
//!
//! The in-slice offsets of the patterns are consecutive blocks of bits of the
//! lower half of the product of the hash and the number of buckets, whose
//! upper half is the bucket: such a value is uniform among the keys of a
//! bucket, and it is computed anyway. Queries thus need a hash, a seed access,
//! two multiplications, and a handful of shifts, additions, and masks; a
//! small fraction of the keys accesses further levels and the Elias–Fano
//! sequence.
//!
//! The signatures of the levels after the first one depend also on a second
//! hash of the key, so 64-bit signatures suffice for any number of keys: keys
//! with the same signature are bumped from the first level and separated at
//! the following ones. Duplicate keys are detected and reported as errors.
//!
//! With the default parameters (8-bit seeds, four patterns, slices of length
//! 1024, and an expected bucket size of 4.25 keys) space is about 1.96 bits
//! per key, essentially the same as PHast+ with wrapping, which however
//! takes about twice as long to build and has much slower queries; PHast
//! uses 1.92 bits per key, but it takes more than ten times as long to build
//! and has slower queries. With 10-bit seeds stored in a [`BitFieldVec`]
//! (see [`PHastRBuilder::seed_bits`]) space is about 1.85 bits per key; in this
//! case, queries are faster after converting the function with
//! [`TryIntoUnaligned::try_into_unaligned`], so that seeds are accessed with
//! [unaligned reads].
//!
//! Functions are built by [`PHastR::try_new`], which reads the keys once
//! from a lender, and by [`PHastR::try_par_new`], which computes the hashes of
//! the keys of a slice in parallel; their variants with a builder make it
//! possible to configure the construction using a [`PHastRBuilder`]. In
//! particular, in [offline mode] the hashes of the keys are kept on disk, and
//! construction needs in memory less than one byte per key. The function
//! built does not depend on the constructor, provided that the current
//! [rayon] pool has the same number of threads.
//!
//! # References
//!
//! Piotr Beling and Peter Sanders. [PHast — Perfect Hashing made fast].
//! CoRR, abs/2504.17918, 2025.
//!
//! [references]: #references
//! [PHast — Perfect Hashing made fast]: https://arxiv.org/abs/2504.17918
//! [Elias–Fano]: crate::dict::elias_fano
//! [unaligned reads]: BitFieldVec::get_unaligned
//! [offline mode]: PHastRBuilder::offline
//! [rayon]: https://docs.rs/rayon

use core::error::Error;
use std::borrow::Borrow;

use anyhow::Result;
use dsi_progress_logger::ProgressLog;
use lender::{FallibleLender, FallibleLending};
use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::bits::{BitFieldVec, BitVec};
use crate::dict::elias_fano::EliasFano;
use crate::rank_sel::SelectAdaptConst;
use crate::traits::{TryIntoUnaligned, Unaligned, UnalignedConversionError};
use crate::utils::ToSig;

mod builder;
pub use builder::PHastRBuilder;

mod seeds;
pub use seeds::*;

mod sigs;
mod stream;
mod sweep;

/// Returns the most significant 64 bits of the 128-bit product of `a` and
/// `b`.
#[inline(always)]
const fn mul_hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// The base-2 logarithm of the distance in bits between the offsets of two
/// consecutive patterns with the default number of patterns (four).
const DEFAULT_PATTERN_SHIFT: u32 = 4;

/// The scale of large levels with the default parameters (8-bit seeds and
/// slices of length 1024).
const DEFAULT_SCALE: u32 = 2;

/// The value combined with the seed of the function to obtain the seed of
/// the second hash of a key (see [`level_sig`]).
const SECOND_HASH: u64 = 0x5851_F42D_4C95_7F2D;

/// The value combined with the seed of the function to obtain the seed of
/// the hash used to tell apart keys with the same signature when detecting
/// duplicates.
const THIRD_HASH: u64 = 0xD6E8_FEB8_6659_FD93;

/// Returns the upper half of the 128-bit product of `a` and `b` xored with
/// the lower half.
#[inline(always)]
const fn mix(a: u64, b: u64) -> u64 {
    let p = a as u128 * b as u128;
    (p >> 64) as u64 ^ p as u64
}

/// Returns the signature of a key for a level after the first one, given
/// its signature for the first level, a second hash of the key, and the salt
/// of the level.
///
/// Keys with the same signature for the first level have different
/// signatures for the following ones, as the second hash is independent
/// from the first one; on the other hand, the key itself is not necessary
/// after computing the two hashes, and this makes it possible to handle the
/// keys bumped from the first level out of line without keeping the key
/// around.
#[inline(always)]
const fn level_sig(h: u64, second: u64, salt: u64) -> u64 {
    mix(h ^ salt, second ^ 0xD6E8_FEB8_6659_FD93)
}

/// The default remapping sequence of a [`PHastR`]: an Elias–Fano sequence
/// whose selection inventory is sparser than that of [`EfSeq`] (one entry
/// every 4096 ones instead of 2048), as it is accessed only by keys bumped
/// from the first level.
///
/// [`EfSeq`]: crate::dict::elias_fano::EfSeq
pub type Remap = EliasFano<usize, SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>>;

/// The parameters of a level.
#[derive(Debug, Clone, Copy, MemSize, MemDbg)]
#[mem_size(flat)]
#[repr(C)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(zero_copy))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LevelParams {
    /// The number of buckets (always odd).
    buckets: u64,
    /// The number of slices (the output range minus the slice length, plus
    /// one).
    num_slices: u64,
    /// The slice length minus one.
    l_mask: u64,
    /// The base-2 logarithm of the amount by which a unit increase of the
    /// seed moves the keys of a bucket.
    scale: u64,
    /// The offset of the outputs of this level in the remapping sequence.
    offset: u64,
    /// The salt used to compute the signatures of the keys of this level
    /// from their hashes (unused for the first level).
    salt: u64,
    /// The index of the first seed of this level in the seed storage.
    first_seed: u64,
}

/// A bucket-placement minimal perfect hash function with rings of patterns.
///
/// A *minimal perfect hash function* maps bijectively a set of *n* keys to
/// the integers in [0 . . *n*): querying a key outside of the original set
/// will lead to an arbitrary result. Values are retrieved using the [`get`]
/// method. See the [module documentation] for a description of the
/// algorithm.
///
/// Instances of this structure are immutable; they are built using
/// [`try_new`], [`try_par_new`], or one of their variants, and can be
/// serialized using [ε-serde] or [`serde`].
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Generics
///
/// * `K` - the type of the keys, which must be hashable to 64-bit signatures
///   (see [`ToSig`]).
///
/// * `D` - the storage of the seeds (see [`SeedStore`]). The default is
///   `Box<[u8]>`, which supports seeds of at most eight bits, and provides
///   the fastest queries.
///
/// * `P` - the parameters of the levels after the first one. The default is
///   `Box<[LevelParams]>`.
///
/// * `R` - the sequence remapping the outputs of the levels after the first
///   one to the free slots of the first one. The default is [`Remap`].
///
/// The last three parameters make it possible to deserialize with ε-serde
/// without copying: for example, [`deserialize_eps`] on a `PHastR<K>`
/// returns a structure whose seeds are a `&[u8]`, whose parameters are a
/// `&[LevelParams]`, and whose remapping sequence is an Elias–Fano sequence
/// over slices.
///
/// # Examples
///
/// See [`try_new`] and [`try_par_new`].
///
/// [`get`]: PHastR::get
/// [module documentation]: self
/// [`try_new`]: PHastR::try_new
/// [`try_par_new`]: PHastR::try_par_new
/// [ε-serde]: https://crates.io/crates/epserde
/// [`serde`]: https://crates.io/crates/serde
/// [`deserialize_eps`]: https://docs.rs/epserde/latest/epserde/deser/trait.Deserialize.html#tymethod.deserialize_eps
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(phantom(K)))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PHastR<K: ?Sized, D = Box<[u8]>, P = Box<[LevelParams]>, R = Remap> {
    /// The seed used to compute signatures.
    seed: u64,
    /// The number of keys.
    num_keys: usize,
    /// The base-2 logarithm of the distance in bits between the offsets of
    /// two consecutive patterns (i.e., six minus the base-2 logarithm of
    /// the number of patterns).
    pattern_shift: u32,
    /// Whether the shifts of the first level are those of the default
    /// parameters (see [`DEFAULT_PATTERN_SHIFT`] and [`DEFAULT_SCALE`]), in
    /// which case queries use constants.
    default_shifts: bool,
    /// The scale of the first level plus one, if there are four patterns
    /// and the scale is at most three, in which case queries use constants
    /// even if the shifts are not the default ones; zero otherwise.
    fast_scale: u8,
    /// The parameters of the first level.
    params0: LevelParams,
    /// The seeds of the first level.
    seeds0: D,
    /// The parameters of the following levels.
    params: P,
    /// The seeds of the following levels, concatenated.
    seeds: D,
    /// Maps outputs of levels after the first one to the free slots of the
    /// first one.
    remap: R,
    _marker: std::marker::PhantomData<*const K>,
}

// SAFETY: K occurs only inside _marker, so no value of this type is ever
// stored: the auto traits depend just on the remaining fields. The *const K
// in the marker is a Sized placeholder standing in for a possibly unsized K,
// not a pointer we own; without these impls its !Send/!Sync-ness would
// propagate to PHastR and to everything containing it.
unsafe impl<K: ?Sized, D: Send, P: Send, R: Send> Send for PHastR<K, D, P, R> {}
unsafe impl<K: ?Sized, D: Sync, P: Sync, R: Sync> Sync for PHastR<K, D, P, R> {}

impl<K: ?Sized, D, P, R> PHastR<K, D, P, R> {
    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.num_keys
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.num_keys == 0
    }
}

impl<K: ?Sized, D, P: AsRef<[LevelParams]>, R> PHastR<K, D, P, R> {
    /// Returns the number of levels.
    pub fn num_levels(&self) -> usize {
        self.params.as_ref().len() + 1
    }
}

impl<
    K: ?Sized + ToSig<[u64; 1]>,
    D: SeedStore,
    P: AsRef<[LevelParams]>,
    R: SliceByValue<Value = usize>,
> PHastR<K, D, P, R>
{
    /// Returns the output of a key in the given level, given its hash *h* for
    /// the level, the lower half `lo` of the product of *h* and the number of
    /// buckets (whose upper half is the bucket of the key), the seed of the
    /// bucket, the base-2 logarithm of the distance in bits between the
    /// offsets of two consecutive patterns, and the scale of the level.
    ///
    /// The pattern is given by the lowest bits of the seed, and its in-slice
    /// offset starts at bit 64*r*/*R* of `lo`, where *r* is the pattern and
    /// *R* the number of patterns: since shifts are taken modulo 64, shifting
    /// the seed left by log₂(64/*R*) yields the first bit of the offset
    /// without extracting the pattern.
    #[inline(always)]
    fn pos(
        lv: &LevelParams,
        h: u64,
        lo: u64,
        seed: usize,
        pattern_shift: u32,
        scale: u32,
    ) -> usize {
        let s = seed as u64;
        let offset = lo
            .wrapping_shr((s << pattern_shift) as u32)
            .wrapping_add(s << scale);
        mul_hi(h, lv.num_slices) as usize + (offset & lv.l_mask) as usize
    }

    /// Returns the output of a key in the first level (see [`pos`]).
    ///
    /// With the default shifts we use constants, so that the compiler can
    /// fuse the scaling of the seed with the addition (the test is on a
    /// dedicated field because a test on the shifts themselves would be
    /// optimized away; queries in a loop perform it just once). We use
    /// constants also for the other scales that can be fused when there
    /// are four patterns (e.g., with 10-bit seeds and slices of length
    /// 2048), albeit in this case the test is not moved out of loops.
    ///
    /// [`pos`]: Self::pos
    #[inline(always)]
    fn pos0(&self, h: u64, lo: u64, seed: usize) -> usize {
        let lv = &self.params0;
        if self.default_shifts {
            return Self::pos(lv, h, lo, seed, DEFAULT_PATTERN_SHIFT, DEFAULT_SCALE);
        }
        match self.fast_scale {
            1 => Self::pos(lv, h, lo, seed, DEFAULT_PATTERN_SHIFT, 0),
            2 => Self::pos(lv, h, lo, seed, DEFAULT_PATTERN_SHIFT, 1),
            4 => Self::pos(lv, h, lo, seed, DEFAULT_PATTERN_SHIFT, 3),
            _ => Self::pos(lv, h, lo, seed, self.pattern_shift, lv.scale as u32),
        }
    }

    /// Returns the value associated with the given key.
    ///
    /// The returned value is in the range [0 . . *n*), where *n* is the
    /// number of keys, and different keys of the original set are mapped to
    /// different values. If the key was not in the original set, the result
    /// is arbitrary.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> usize {
        let key = key.borrow();
        let h = K::to_sig(key, self.seed)[0];
        let lv = &self.params0;
        let p = h as u128 * lv.buckets as u128;
        // SAFETY: the upper half of the product is smaller than the number
        // of buckets, which is the number of seeds
        let s = unsafe { self.seeds0.get_seed((p >> 64) as usize) };
        if s != 0 {
            return self.pos0(h, p as u64, s);
        }
        self.get_slow(h, K::to_sig(key, self.seed ^ SECOND_HASH)[0])
    }

    /// Returns the values associated with a batch of keys, prefetching the
    /// seeds of the first level of all keys before computing the values
    /// (experimental).
    #[doc(hidden)]
    #[inline(always)]
    pub fn get_batch<const B: usize>(&self, keys: [&K; B]) -> [usize; B] {
        let lv = &self.params0;
        let hs: [u64; B] = std::array::from_fn(|i| {
            let h = K::to_sig(keys[i], self.seed)[0];
            self.seeds0.prefetch_seed(mul_hi(h, lv.buckets) as usize);
            h
        });
        std::array::from_fn(|i| {
            let h = hs[i];
            let p = h as u128 * lv.buckets as u128;
            // SAFETY: the upper half of the product is smaller than the
            // number of buckets, which is the number of seeds
            let s = unsafe { self.seeds0.get_seed((p >> 64) as usize) };
            if s != 0 {
                return self.pos0(h, p as u64, s);
            }
            self.get_slow(h, K::to_sig(keys[i], self.seed ^ SECOND_HASH)[0])
        })
    }

    /// Returns whether the given key is bumped from the first level (for
    /// benchmarking).
    #[doc(hidden)]
    pub fn is_bumped(&self, key: impl Borrow<K>) -> bool {
        let h = K::to_sig(key.borrow(), self.seed)[0];
        // SAFETY: mul_hi(h, buckets) < buckets, which is the number of seeds
        unsafe {
            self.seeds0
                .get_seed(mul_hi(h, self.params0.buckets) as usize)
                == 0
        }
    }

    /// Handles keys bumped from the first level, given the signature for
    /// the first level and the second hash of the key (see [`level_sig`]).
    #[cold]
    #[inline(never)]
    fn get_slow(&self, h: u64, second: u64) -> usize {
        for lv in self.params.as_ref() {
            let h = level_sig(h, second, lv.salt);
            let p = h as u128 * lv.buckets as u128;
            // SAFETY: the seeds of the level are stored consecutively
            let s = unsafe {
                self.seeds
                    .get_seed(lv.first_seed as usize + (p >> 64) as usize)
            };
            if s != 0 {
                // SAFETY: by construction, the remapping sequence contains
                // one entry for each output of each level after the first
                let pos = Self::pos(lv, h, p as u64, s, self.pattern_shift, lv.scale as u32);
                return unsafe { self.remap.get_value_unchecked(lv.offset as usize + pos) };
            }
        }
        // Only keys outside the original set can reach this point
        0
    }
}

// ── Constructors ────────────────────────────────────────────────────

impl<K: ?Sized + ToSig<[u64; 1]>, D: SeedStoreBuild> PHastR<K, D> {
    /// Builds a [`PHastR`] from keys using default [`PHastRBuilder`]
    /// settings.
    ///
    /// This is a convenience wrapper around [`try_new_with_builder`] with
    /// `PHastRBuilder::default()`.
    ///
    /// If keys are available as a slice, [`try_par_new`] parallelizes the
    /// computation of their hashes for faster construction.
    ///
    /// * `keys` - a [`FallibleLender`] returning the keys, which is read just
    ///   once. The [`lenders`] module provides easy ways to build such
    ///   lenders.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::PHastR;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let phf = <PHastR<usize>>::try_new(
    ///     FromCloneableIntoIterator::new(0..100_000),
    ///     no_logging![],
    /// )?;
    ///
    /// let mut seen = vec![false; phf.len()];
    /// for key in 0..100_000_usize {
    ///     let v = phf.get(key);
    ///     assert!(!seen[v]);
    ///     seen[v] = true;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`try_new_with_builder`]: Self::try_new_with_builder
    /// [`try_par_new`]: Self::try_par_new
    /// [`lenders`]: crate::utils::lenders
    pub fn try_new<B: ?Sized + Borrow<K>>(
        keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        pl: &mut impl ProgressLog,
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, PHastRBuilder::default(), pl)
    }

    /// Builds a [`PHastR`] from keys using the given [`PHastRBuilder`]
    /// configuration.
    ///
    /// The builder controls construction parameters such as the [number of
    /// bits of a seed], the [slice length], the [expected bucket size], and
    /// [offline mode].
    ///
    /// Hashes are kept in memory, or on disk in offline mode, using sixteen
    /// bytes per key; in offline mode, construction needs in memory less than
    /// one byte per key. The function is identical to that built by
    /// [`try_par_new_with_builder`] with the same builder, provided that the
    /// current [rayon] pool has the same number of threads.
    ///
    /// * `keys` - a [`FallibleLender`] returning the keys, which is read just
    ///   once. The [`lenders`] module provides easy ways to build such
    ///   lenders: for example, a [`LineLender`] reads the keys from a file.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{PHastR, PHastRBuilder};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let phf = <PHastR<usize>>::try_new_with_builder(
    ///     FromCloneableIntoIterator::new(0..100_000),
    ///     PHastRBuilder::default().offline(true),
    ///     no_logging![],
    /// )?;
    ///
    /// let mut seen = vec![false; phf.len()];
    /// for key in 0..100_000_usize {
    ///     let v = phf.get(key);
    ///     assert!(!seen[v]);
    ///     seen[v] = true;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [number of bits of a seed]: PHastRBuilder::seed_bits
    /// [slice length]: PHastRBuilder::log2_slice_len
    /// [expected bucket size]: PHastRBuilder::bucket_size
    /// [offline mode]: PHastRBuilder::offline
    /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
    /// [rayon]: https://docs.rs/rayon
    /// [`lenders`]: crate::utils::lenders
    /// [`LineLender`]: crate::utils::LineLender
    pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
        keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        builder: PHastRBuilder,
        pl: &mut impl ProgressLog,
    ) -> Result<Self> {
        builder.try_build_from_lender(keys, pl)
    }

    /// Builds a [`PHastR`] from the keys of a slice, computing their hashes
    /// in parallel, using default [`PHastRBuilder`] settings.
    ///
    /// This is a convenience wrapper around [`try_par_new_with_builder`]
    /// with `PHastRBuilder::default()`.
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new`] instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::PHastR;
    /// # use dsi_progress_logger::no_logging;
    /// let keys: Vec<u64> = (0..100_000).collect();
    /// let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
    ///
    /// let mut seen = vec![false; keys.len()];
    /// for key in &keys {
    ///     let v = phf.get(key);
    ///     assert!(!seen[v]);
    ///     seen[v] = true;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
    /// [`try_new`]: Self::try_new
    pub fn try_par_new(keys: &[impl Borrow<K> + Sync], pl: &mut impl ProgressLog) -> Result<Self>
    where
        K: Sync,
    {
        Self::try_par_new_with_builder(keys, PHastRBuilder::default(), pl)
    }

    /// Builds a [`PHastR`] from the keys of a slice, computing their hashes
    /// in parallel, using the given [`PHastRBuilder`] configuration.
    ///
    /// The builder controls construction parameters such as the [number of
    /// bits of a seed], the [slice length], and the [expected bucket size];
    /// [offline mode] is not supported, and an error is returned if it is
    /// set.
    ///
    /// The hashes of the keys are kept in memory, using about nine bytes per
    /// key, and the keys bumped from the first level are hashed again.
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new_with_builder`] instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{PHastR, PHastRBuilder};
    /// # use sux::bits::BitFieldVec;
    /// # use dsi_progress_logger::no_logging;
    /// let keys: Vec<u64> = (0..100_000).collect();
    /// // 10-bit seeds
    /// let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
    ///     &keys,
    ///     PHastRBuilder::default()
    ///         .seed_bits(10)
    ///         .log2_slice_len(11)
    ///         .bucket_size(6.0),
    ///     no_logging![],
    /// )?;
    ///
    /// let mut seen = vec![false; keys.len()];
    /// for key in &keys {
    ///     let v = phf.get(key);
    ///     assert!(!seen[v]);
    ///     seen[v] = true;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [number of bits of a seed]: PHastRBuilder::seed_bits
    /// [slice length]: PHastRBuilder::log2_slice_len
    /// [expected bucket size]: PHastRBuilder::bucket_size
    /// [offline mode]: PHastRBuilder::offline
    /// [`try_new_with_builder`]: Self::try_new_with_builder
    pub fn try_par_new_with_builder(
        keys: &[impl Borrow<K> + Sync],
        builder: PHastRBuilder,
        pl: &mut impl ProgressLog,
    ) -> Result<Self>
    where
        K: Sync,
    {
        builder.try_build_from_slice(keys, pl)
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

/// Converts the seed storage and the remapping sequence to [unaligned
/// reads]; this is useful only for seeds stored in a [`BitFieldVec`], as
/// slices of bytes or of 16-bit values are left unchanged.
///
/// [unaligned reads]: BitFieldVec::get_unaligned
impl<K: ?Sized, D: TryIntoUnaligned, P, R: TryIntoUnaligned> TryIntoUnaligned
    for PHastR<K, D, P, R>
{
    type Unaligned = PHastR<K, Unaligned<D>, P, Unaligned<R>>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        Ok(PHastR {
            seed: self.seed,
            num_keys: self.num_keys,
            pattern_shift: self.pattern_shift,
            default_shifts: self.default_shifts,
            fast_scale: self.fast_scale,
            params0: self.params0,
            seeds0: self.seeds0.try_into_unaligned()?,
            params: self.params,
            seeds: self.seeds.try_into_unaligned()?,
            remap: self.remap.try_into_unaligned()?,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<K: ?Sized, P> From<Unaligned<PHastR<K, BitFieldVec<Box<[usize]>>, P, Remap>>>
    for PHastR<K, BitFieldVec<Box<[usize]>>, P, Remap>
{
    fn from(f: Unaligned<PHastR<K, BitFieldVec<Box<[usize]>>, P, Remap>>) -> Self {
        PHastR {
            seed: f.seed,
            num_keys: f.num_keys,
            pattern_shift: f.pattern_shift,
            default_shifts: f.default_shifts,
            fast_scale: f.fast_scale,
            params0: f.params0,
            seeds0: f.seeds0.into(),
            params: f.params,
            seeds: f.seeds.into(),
            remap: f.remap.into(),
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::FromSlice;
    use dsi_progress_logger::no_logging;

    /// Builds a function on the first `n` integers from a slice, and checks
    /// that it is a minimal perfect hash function.
    fn check<D: SeedStoreBuild>(n: usize, builder: PHastRBuilder) -> Result<()> {
        let keys: Vec<u64> = (0..n as u64).collect();
        let phf = <PHastR<u64, D>>::try_par_new_with_builder(&keys, builder, no_logging![])?;
        assert_eq!(phf.len(), n);
        let mut seen = vec![false; n];
        for key in &keys {
            let v = phf.get(key);
            assert!(v < n, "{v} >= {n}");
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
        Ok(())
    }

    #[test]
    fn test_small() -> Result<()> {
        for n in [0, 1, 2, 3, 10, 100, 1000, 5000, 10000] {
            check::<Box<[u8]>>(n, PHastRBuilder::default())?;
            check::<BitFieldVec<Box<[usize]>>>(n, PHastRBuilder::default().seed_bits(10))?;
            for log2_patterns in 0..=3 {
                check::<Box<[u8]>>(
                    n,
                    PHastRBuilder::default()
                        .log2_patterns(log2_patterns)
                        .log2_slice_len(8),
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn test_medium() -> Result<()> {
        check::<Box<[u8]>>(1_000_000, PHastRBuilder::default())?;
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().bucket_size(5.0))?;
        // Rings of four, two and one words
        check::<BitFieldVec<Box<[usize]>>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        )?;
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_patterns(1))?;
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_patterns(0))?;
        check::<Box<[u16]>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(11)
                .log2_slice_len(12)
                .bucket_size(6.75),
        )?;
        // Longer slices, and thus a larger stride
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(12))?;
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(16))?;
        // Rings shorter than a word: a pattern has four rotations
        check::<Box<[u8]>>(
            100_000,
            PHastRBuilder::default()
                .seed_bits(4)
                .log2_slice_len(8)
                .bucket_size(2.0),
        )?;
        // Slices with fewer slots than there are seeds: rotations are
        // redundant
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(6))
    }

    #[test]
    fn test_large_buckets() -> Result<()> {
        // Large buckets (kept in a heap, and with several groups of keys)
        // and very long rings
        for log2_patterns in [0, 2] {
            check::<Box<[u16]>>(
                20_000,
                PHastRBuilder::default()
                    .seed_bits(16)
                    .log2_patterns(log2_patterns)
                    .log2_slice_len(16)
                    .bucket_size(70.0),
            )?;
        }
        check::<Box<[u8]>>(100_000, PHastRBuilder::default().bucket_size(70.0))
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_many_chunks() -> Result<()> {
        // Many threads, and hence many chunks and gaps, with long slices
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build()?;
        pool.install(|| {
            check::<BitFieldVec<Box<[usize]>>>(
                2_000_000,
                PHastRBuilder::default()
                    .seed_bits(10)
                    .log2_slice_len(11)
                    .bucket_size(6.0),
            )?;
            check::<Box<[u8]>>(2_000_000, PHastRBuilder::default())
        })
    }

    #[test]
    fn test_parameter_checks() {
        let keys: Vec<u64> = (0..1000).collect();
        let build = |builder: PHastRBuilder| {
            <PHastR<u64>>::try_par_new_with_builder(&keys, builder, no_logging![])
        };
        // Seeds too large for the storage
        assert!(build(PHastRBuilder::default().seed_bits(10)).is_err());
        // Too many patterns for the slice length
        assert!(build(PHastRBuilder::default().log2_patterns(3)).is_err());
        // Too many patterns for the number of seed bits
        assert!(build(PHastRBuilder::default().seed_bits(2)).is_err());
        // Offline mode needs a lender
        assert!(build(PHastRBuilder::default().offline(true)).is_err());
    }

    #[test]
    fn test_lender() -> Result<()> {
        // The constructors reading keys from a lender, in memory and offline
        let keys: Vec<u64> = (0..100_000).collect();
        for builder in [
            PHastRBuilder::default(),
            PHastRBuilder::default().offline(true),
        ] {
            let phf =
                <PHastR<u64>>::try_new_with_builder(FromSlice::new(&keys), builder, no_logging![])?;
            assert_eq!(phf.len(), keys.len());
            let mut seen = vec![false; keys.len()];
            for key in &keys {
                let v = phf.get(key);
                assert!(!seen[v], "duplicate output {v}");
                seen[v] = true;
            }
        }
        let phf = <PHastR<u64>>::try_new(FromSlice::new(&keys), no_logging![])?;
        assert_eq!(phf.len(), keys.len());
        Ok(())
    }

    #[test]
    fn test_unaligned() -> Result<()> {
        for (bits, log2_slice_len, n) in [
            (10, 11, 300_000),
            (12, 12, 300_000),
            (16, 12, 10_000),
            (10, 11, 0),
        ] {
            let keys: Vec<u64> = (0..n as u64).collect();
            let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
                &keys,
                PHastRBuilder::default()
                    .seed_bits(bits)
                    .log2_slice_len(log2_slice_len)
                    .bucket_size(6.0),
                no_logging![],
            )?;
            let expected: Vec<usize> = keys.iter().map(|k| phf.get(k)).collect();
            let phf = phf.try_into_unaligned()?;
            for (k, &e) in keys.iter().zip(&expected) {
                assert_eq!(phf.get(k), e);
            }
            let phf: PHastR<u64, BitFieldVec<Box<[usize]>>> = phf.into();
            for (k, &e) in keys.iter().zip(&expected) {
                assert_eq!(phf.get(k), e);
            }
        }
        Ok(())
    }

    #[test]
    fn test_batch() -> Result<()> {
        let keys: Vec<u64> = (0..100_000).collect();
        let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
        for chunk in keys.chunks_exact(8) {
            let batch: [&u64; 8] = std::array::from_fn(|i| &chunk[i]);
            let values = phf.get_batch(batch);
            for (key, value) in chunk.iter().zip(values) {
                assert_eq!(value, phf.get(key));
            }
        }
        Ok(())
    }

    #[test]
    fn test_duplicates() {
        let keys: Vec<u64> = vec![1, 2, 3, 2];
        assert!(<PHastR<u64>>::try_par_new(&keys, no_logging![]).is_err());
    }

    /// A key whose 64-bit signature with seed 0 collides with that of another
    /// key for the first 2 · `COLLIDING` keys (pairs 2*i*, 2*i* + 1); other
    /// seeds give independent signatures.
    #[derive(Clone, Copy)]
    pub(super) struct CollidingKey(pub(super) u64);
    const COLLIDING: u64 = 2000;

    impl ToSig<[u64; 1]> for CollidingKey {
        fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
            let k = key.borrow().0;
            if seed == 0 && k < 2 * COLLIDING {
                <u64 as ToSig<[u64; 1]>>::to_sig(k / 2, 0)
            } else {
                <u64 as ToSig<[u64; 1]>>::to_sig(k, seed)
            }
        }
    }

    #[test]
    fn test_signature_collisions() -> Result<()> {
        // Colliding signatures are bumped and separated by the signatures of
        // the following levels, with all configurations of the following
        // levels (in particular, with only the last level)
        for n in [5_000u64, 300_000] {
            let keys: Vec<CollidingKey> = (0..n).map(CollidingKey).collect();
            let phf = <PHastR<CollidingKey>>::try_par_new(&keys, no_logging![])?;
            let mut seen = vec![false; keys.len()];
            for &key in &keys {
                if key.0 < 2 * COLLIDING {
                    assert!(phf.is_bumped(key));
                }
                let v = phf.get(key);
                assert!(!seen[v], "duplicate output {v}");
                seen[v] = true;
            }
        }
        // Equal keys are still detected
        let mut keys: Vec<CollidingKey> = (0..100_000).map(CollidingKey).collect();
        keys.push(CollidingKey(50_000));
        assert!(<PHastR<CollidingKey>>::try_par_new(&keys, no_logging![]).is_err());
        Ok(())
    }

    /// Serializes, deserializes zero-copy (checking at compile time that
    /// seeds, level parameters, and the remapping sequence are borrowed),
    /// and compares outputs.
    #[cfg(feature = "epserde")]
    macro_rules! check_epserde {
        ($d:ty, $deser_d:ty, $builder:expr) => {{
            use epserde::prelude::*;
            use epserde::utils::AlignedCursor;
            let keys: Vec<u64> = (0..200_000).collect();
            let phf = <PHastR<u64, $d>>::try_par_new_with_builder(&keys, $builder, no_logging![])?;
            let mut cursor = <AlignedCursor<Aligned64>>::new();
            // SAFETY: phf was built by its constructor
            unsafe { phf.serialize(&mut cursor) }?;
            let len = cursor.len();
            cursor.set_position(0);
            // SAFETY: we just serialized a valid structure into this buffer
            let case = unsafe { <PHastR<u64, $d>>::read_mem(&mut cursor, len) }?;
            let des = case.uncase();
            let _: &$deser_d = &des.seeds0;
            let _: &$deser_d = &des.seeds;
            let _: &&[LevelParams] = &des.params;
            let _: &crate::dict::EliasFano<
                usize,
                crate::rank_sel::SelectAdaptConst<crate::bits::BitVec<&[usize]>, &[usize], 12, 3>,
                BitFieldVec<&[usize]>,
            > = &des.remap;
            assert_eq!(des.len(), keys.len());
            for key in &keys {
                assert_eq!(des.get(key), phf.get(key));
            }
        }};
    }

    /// A memory-mapped function must take little memory beyond the mapping:
    /// with the default flags mem_dbg does not follow references, so it
    /// counts only the fields of the structure, whereas following references
    /// it must count the same data as the original.
    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_mem_size() -> Result<()> {
        use epserde::prelude::*;
        let keys: Vec<u64> = (0..1_000_000).collect();
        let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
        let file = tempfile::NamedTempFile::new()?;
        // SAFETY: phf was built by its constructor
        unsafe { phf.store(file.path()) }?;
        // SAFETY: we just stored a valid structure into the file
        let case = unsafe { <PHastR<u64>>::mmap(file.path(), Flags::empty()) }?;
        let mapped = case.uncase();
        let owned_size = phf.mem_size(SizeFlags::default());
        let mapped_size = mapped.mem_size(SizeFlags::default());
        let mapped_followed = mapped.mem_size(SizeFlags::FOLLOW_REFS);
        assert!(
            mapped_size < 1024,
            "mapped structure takes {mapped_size} bytes"
        );
        assert!(
            mapped_followed >= owned_size - 1024 && mapped_followed <= owned_size + 1024,
            "owned {owned_size} bytes, mapped following references {mapped_followed} bytes"
        );
        for key in &keys {
            assert_eq!(mapped.get(key), phf.get(key));
        }
        Ok(())
    }

    #[cfg(feature = "epserde")]
    #[test]
    fn test_epserde() -> Result<()> {
        check_epserde!(Box<[u8]>, &[u8], PHastRBuilder::default());
        check_epserde!(
            Box<[u16]>,
            &[u16],
            PHastRBuilder::default().seed_bits(11).log2_slice_len(12)
        );
        check_epserde!(
            BitFieldVec<Box<[usize]>>,
            BitFieldVec<&[usize]>,
            PHastRBuilder::default().seed_bits(10).log2_slice_len(11)
        );
        // Unaligned reads
        use crate::bits::BitFieldVecU;
        use epserde::prelude::*;
        use epserde::utils::AlignedCursor;
        type U = Unaligned<PHastR<u64, BitFieldVec<Box<[usize]>>>>;
        let keys: Vec<u64> = (0..200_000).collect();
        let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
            &keys,
            PHastRBuilder::default().seed_bits(10).log2_slice_len(11),
            no_logging![],
        )?;
        let phf: U = phf.try_into_unaligned()?;
        let mut cursor = <AlignedCursor<Aligned64>>::new();
        // SAFETY: phf was built by its constructor
        unsafe { phf.serialize(&mut cursor) }?;
        let len = cursor.len();
        cursor.set_position(0);
        // SAFETY: we just serialized a valid structure into this buffer
        let case = unsafe { <U>::read_mem(&mut cursor, len) }?;
        let des = case.uncase();
        let _: &BitFieldVecU<&[usize]> = &des.seeds0;
        for key in &keys {
            assert_eq!(des.get(key), phf.get(key));
        }
        Ok(())
    }

    #[test]
    fn test_strings() -> Result<()> {
        let keys: Vec<String> = (0..200_000).map(|i| format!("key{i}")).collect();
        let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let phf = <PHastR<str>>::try_par_new(&refs, no_logging![])?;
        let mut seen = vec![false; keys.len()];
        for key in &keys {
            let v = phf.get(key.as_str());
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
        Ok(())
    }
}
