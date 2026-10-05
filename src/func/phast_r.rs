/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! PHast-R: minimal perfect hashing with additive placement over rings of
//! patterns.
//!
//! This structure is a variant of PHast+ ([Beling and Sanders, *PHast —
//! Perfect Hashing made fast*]). As in PHast+, keys are hashed to buckets
//! using a linear function of their hash, each bucket stores a fixed-width
//! seed, and the keys of a bucket are mapped inside a small *slice* of the
//! output range to an in-slice offset that depends on the key, moved by an
//! amount that depends on the seed only, so that feasible seeds can be found
//! with bit-parallel operations. Buckets for which no seed is found are
//! *bumped* to the next level, and an [Elias–Fano] sequence maps the outputs
//! of the following levels to the free slots of the first one.
//!
//! In PHast+ all seeds move the keys of a bucket by the same amount (modulo
//! the slice length, in the variant with wrapping): thus, two keys of a
//! bucket that are mapped to the same slot by a seed are mapped to the same
//! slot by all seeds, and their bucket must be bumped. With the parameters
//! suggested for 8-bit seeds this *self-collision* happens to buckets
//! containing 1.7% of the keys, that is, to almost 40% of the bumped keys.
//!
//! In PHast-R the lowest bits of a seed *s* select one of *R* *patterns*,
//! that is, one of *R* independent in-slice offsets for each key, and the
//! offset is moved by *s* · *L* / 2<sup>*S*</sup> modulo *L*, where *L* is
//! the slice length and *S* the number of bits of a seed. Since the seeds of
//! a pattern differ by multiples of *R*, they move a key by multiples of the
//! *stride* *T* = *RL* / 2<sup>*S*</sup>, and as they vary the key goes
//! exactly once through the slots of the slice that are congruent to its
//! first slot modulo *T*: we call such slots a *ring*. Two keys of a bucket
//! colliding in a pattern will not, in general, collide in the others, so
//! self-collisions disappear, and the patterns provide (almost) independent
//! trials.
//!
//! The in-slice offsets of the patterns are consecutive blocks of bits of the
//! lower half of the product of the hash and the number of buckets, whose
//! upper half is the bucket: such a value is uniform among the keys of a
//! bucket, and it is computed anyway. Queries thus need a hash, a seed access,
//! two multiplications, and a handful of shifts, additions, and masks—just
//! two shifts more than PHast+ with wrapping; a small fraction of the keys
//! accesses further levels and the Elias–Fano sequence.
//!
//! During construction the set of used slots is stored by residue classes
//! modulo the stride, so the ring of a key is a block of consecutive bits:
//! the seeds of a pattern that are feasible for a bucket are obtained by
//! rotating and combining one such block for each key.
//!
//! The signatures of the levels after the first one depend also on a second
//! hash of the key, so, as in PHast+, 64-bit signatures suffice for any
//! number of keys: keys with the same signature are bumped from the first
//! level and separated at the following ones. Duplicate keys are detected
//! and reported as errors.
//!
//! With the default parameters (8-bit seeds, four patterns, slices of length
//! 1024) space is about 1.93 bits per key, against the 1.97 bits per key of
//! PHast+ with wrapping, construction is about twice as fast, and queries
//! are slightly faster. With 10-bit seeds stored in a [`BitFieldVec`] (see
//! [`PHastRBuilder::seed_bits`]) space is about 1.86 bits per key; in this
//! case, queries are faster after converting the function with
//! [`TryIntoUnaligned::try_into_unaligned`], so that seeds are accessed with
//! [unaligned reads].
//!
//! [Beling and Sanders, *PHast — Perfect Hashing made fast*]: https://arxiv.org/abs/2504.17918
//! [Elias–Fano]: crate::dict::elias_fano
//! [unaligned reads]: BitFieldVec::get_unaligned

use std::borrow::Borrow;
use std::collections::BinaryHeap;
use std::mem::MaybeUninit;

use anyhow::{Result, bail};
use dsi_progress_logger::ProgressLog;
use mem_dbg::*;
use value_traits::slices::{SliceByValue, SliceByValueMut};

use crate::bits::{BitFieldVec, BitFieldVecU, BitVec};
use crate::dict::elias_fano::{EliasFano, EliasFanoBuilder};
use crate::rank_sel::SelectAdaptConst;
use crate::traits::{TryIntoUnaligned, Unaligned, UnalignedConversionError};
use crate::utils::ToSig;

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

/// Returns the salt of a level after the first one (see [`level_sig`]): a
/// value depending on the seed of the function, on the level, and, for the
/// last level, on the attempt.
#[inline(always)]
const fn level_salt(seed: u64, level: usize, attempt: u64) -> u64 {
    mix(
        seed.wrapping_add(level as u64).wrapping_add(attempt << 32) | 1,
        0x9E37_79B9_7F4A_7C15,
    )
}

/// The value combined with the seed of the function to obtain the seed of
/// the second hash of a key (see [`level_sig`]).
const SECOND_HASH: u64 = 0x5851_F42D_4C95_7F2D;

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

/// Returns whether to process `len` elements using rayon, that is, whether
/// `len` is large enough and the current pool has more than one thread (with a single thread, dispatching
/// work to the pool only adds overhead, and if the process is restricted to
/// a single CPU the thread waiting for the pool competes with the worker).
#[cfg(feature = "rayon")]
#[inline]
fn parallel(len: usize) -> bool {
    len >= PAR_MIN_LEN && rayon::current_num_threads() > 1
}

/// Minimum number of elements for which a pass is run in parallel: below
/// this size the overhead of parallelism exceeds its benefits.
#[cfg(feature = "rayon")]
const PAR_MIN_LEN: usize = 1 << 17;

/// Storage for the seeds of a level (query side).
///
/// Implementations are provided for `Box<[u8]>` (at most 8 bits per seed,
/// the fastest option), `Box<[u16]>`, [`BitFieldVec`] (any width up to 16
/// bits), [`BitFieldVecU`] (the same, with unaligned reads, obtained by
/// [`TryIntoUnaligned::try_into_unaligned`]), and for the corresponding
/// borrowed types obtained by ε-serde deserialization.
pub trait SeedStore {
    /// Returns the seed of index `i`.
    ///
    /// # Safety
    ///
    /// `i` must be smaller than the number of seeds.
    unsafe fn get_seed(&self, i: usize) -> usize;

    /// Prefetches the seed of index `i` (by default, does nothing).
    #[inline(always)]
    fn prefetch_seed(&self, _i: usize) {}
}

/// Storage for the seeds of a level (construction side).
pub trait SeedStoreBuild: SeedStore + Sized {
    /// The maximum number of bits per seed supported.
    const MAX_BITS: u32;

    /// Builds the storage from the given seeds, which use at most `bits` bits.
    fn from_seeds(seeds: &[u16], bits: u32) -> Self;
}

macro_rules! impl_seed_store_slice {
    ($ty:ty, $bits:expr) => {
        impl SeedStore for Box<[$ty]> {
            #[inline(always)]
            unsafe fn get_seed(&self, i: usize) -> usize {
                // SAFETY: by the contract of this method
                unsafe { *<[$ty]>::get_unchecked(self, i) as usize }
            }

            #[inline(always)]
            fn prefetch_seed(&self, i: usize) {
                crate::utils::prefetch_index(self, i);
            }
        }

        impl SeedStore for &[$ty] {
            #[inline(always)]
            unsafe fn get_seed(&self, i: usize) -> usize {
                // SAFETY: by the contract of this method
                unsafe { *<[$ty]>::get_unchecked(self, i) as usize }
            }
        }

        impl SeedStoreBuild for Box<[$ty]> {
            const MAX_BITS: u32 = $bits;

            fn from_seeds(seeds: &[u16], _bits: u32) -> Self {
                seeds.iter().map(|&s| s as $ty).collect()
            }
        }
    };
}

impl_seed_store_slice!(u8, 8);
impl_seed_store_slice!(u16, 16);

impl<B: crate::traits::Backend<Word = usize> + AsRef<[usize]>> SeedStore for BitFieldVec<B> {
    #[inline(always)]
    unsafe fn get_seed(&self, i: usize) -> usize {
        // SAFETY: by the contract of this method
        unsafe { self.get_value_unchecked(i) }
    }

    #[inline(always)]
    fn prefetch_seed(&self, i: usize) {
        let bw = crate::traits::BitWidth::bit_width(self);
        crate::utils::prefetch_index(self.as_slice(), i * bw / usize::BITS as usize);
    }
}

impl<B: crate::traits::Backend<Word = usize> + AsRef<[usize]>> SeedStore for BitFieldVecU<B> {
    #[inline(always)]
    unsafe fn get_seed(&self, i: usize) -> usize {
        // SAFETY: by the contract of this method (reads are unaligned)
        unsafe { self.get_value_unchecked(i) }
    }

    #[inline(always)]
    fn prefetch_seed(&self, i: usize) {
        let bw = crate::traits::BitWidth::bit_width(self);
        crate::utils::prefetch_index(self.as_ref(), i * bw / usize::BITS as usize);
    }
}

impl SeedStoreBuild for BitFieldVec<Box<[usize]>> {
    const MAX_BITS: u32 = 16;

    fn from_seeds(seeds: &[u16], bits: u32) -> Self {
        // Padded, so that conversion to unaligned reads needs no reallocation
        let mut bfv = BitFieldVec::<Box<[usize]>>::new_padded(bits as usize, seeds.len());
        for (i, &s) in seeds.iter().enumerate() {
            bfv.set_value(i, s as usize);
        }
        bfv
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
            n: self.n,
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
            n: f.n,
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

/// The default remapping sequence of a [`PHastR`]: an Elias–Fano sequence
/// whose selection inventory is sparser than that of
/// [`EfSeq`](crate::dict::elias_fano::EfSeq) (one entry every 4096 ones
/// instead of 2048), as it is accessed only by keys bumped from the first
/// level.
pub type Remap = EliasFano<usize, SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>>;

/// Builds a [`Remap`].
fn remap(efb: EliasFanoBuilder<usize>) -> Remap {
    // SAFETY: the selection structure is built on the high bits
    unsafe { efb.build().map_high_bits(SelectAdaptConst::new) }
}

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

/// A minimal perfect hash function based on PHast+ with rings of patterns.
///
/// See the [module documentation](self) for a description of the algorithm.
/// Instances are built using [`PHastRBuilder`].
///
/// # Type Parameters
///
/// - `K`: the type of the keys, which must be hashable to 64-bit signatures
///   (see [`ToSig`]).
/// - `D`: the storage of the seeds (see [`SeedStore`]); default `Box<[u8]>`.
/// - `P`: the parameters of the levels after the first one; default
///   `Box<[LevelParams]>`.
/// - `R`: the sequence remapping the outputs of the levels after the first
///   one to the free slots of the first one; default [`Remap`].
///
/// The last three parameters make it possible to deserialize with ε-serde
/// without copying: for example, [`deserialize_eps`] on a `PHastR<K>`
/// returns a structure whose seeds are a `&[u8]`, whose parameters are a
/// `&[LevelParams]`, and whose remapping sequence is an Elias–Fano sequence
/// over slices.
///
/// [`deserialize_eps`]: https://docs.rs/epserde/latest/epserde/deser/trait.Deserialize.html#tymethod.deserialize_eps
///
/// # Examples
///
/// ```rust
/// # fn main() -> anyhow::Result<()> {
/// # use sux::func::phast_r::*;
/// # use dsi_progress_logger::no_logging;
/// let keys: Vec<u64> = (0..100_000).collect();
/// let phf = <PHastR<u64>>::try_new(&keys, no_logging![])?;
/// let mut seen = vec![false; keys.len()];
/// for key in &keys {
///     let v = phf.get(key);
///     assert!(!seen[v]);
///     seen[v] = true;
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(phantom(K)))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PHastR<K: ?Sized, D = Box<[u8]>, P = Box<[LevelParams]>, R = Remap> {
    /// The seed used to compute signatures.
    seed: u64,
    /// The number of keys.
    n: usize,
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

// SAFETY: K occurs only inside _marker (see VFunc).
unsafe impl<K: ?Sized, D: Send, P: Send, R: Send> Send for PHastR<K, D, P, R> {}
unsafe impl<K: ?Sized, D: Sync, P: Sync, R: Sync> Sync for PHastR<K, D, P, R> {}

impl<K: ?Sized, D, P: AsRef<[LevelParams]>, R> PHastR<K, D, P, R> {
    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

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

    /// Returns the output of a key in the first level (see
    /// [`pos`](Self::pos)).
    ///
    /// With the default shifts we use constants, so that the compiler can
    /// fuse the scaling of the seed with the addition (the test is on a
    /// dedicated field because a test on the shifts themselves would be
    /// optimized away; queries in a loop perform it just once). We use
    /// constants also for the other scales that can be fused when there
    /// are four patterns (e.g., with 10-bit seeds and slices of length
    /// 2048), albeit in this case the test is not moved out of loops.
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

impl<K: ?Sized + ToSig<[u64; 1]>, D: SeedStore> PHastR<K, D> {
    /// Builds a function using default parameters.
    pub fn try_new<B: Borrow<K> + Sync>(keys: &[B], pl: &mut impl ProgressLog) -> Result<Self>
    where
        K: Sync,
        D: SeedStoreBuild,
    {
        PHastRBuilder::default().try_build(keys, pl)
    }
}

/// Builder for [`PHastR`].
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
#[derive(Debug, Clone)]
pub struct PHastRBuilder {
    seed_bits: u32,
    log2_patterns: u32,
    log2_slice_len: u32,
    bucket_size: f64,
    seed: u64,
    weights: Option<[i64; 7]>,
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
        }
    }
}

impl PHastRBuilder {
    /// Sets the number of bits per seed (default: 8; at most 8 for byte
    /// seeds, at most 16 otherwise).
    pub fn seed_bits(mut self, seed_bits: u32) -> Self {
        self.seed_bits = seed_bits;
        self
    }

    /// Sets the base-2 logarithm of the number of patterns (default: 2).
    ///
    /// The in-slice offsets of the patterns are taken from disjoint blocks
    /// of bits of a 64-bit value, so the number of patterns times the base-2
    /// logarithm of the slice length must be at most 64.
    pub fn log2_patterns(mut self, log2_patterns: u32) -> Self {
        self.log2_patterns = log2_patterns;
        self
    }

    /// Sets the base-2 logarithm of the slice length (default: 10).
    ///
    /// This is a maximum: small levels use shorter slices. Slices should be
    /// at least as long as the number of seeds, as otherwise different
    /// seeds of a pattern map the keys of a bucket to the same slots.
    pub fn log2_slice_len(mut self, log2_slice_len: u32) -> Self {
        self.log2_slice_len = log2_slice_len;
        self
    }

    /// Sets the expected number of keys per bucket (default: 4.5).
    pub fn bucket_size(mut self, bucket_size: f64) -> Self {
        self.bucket_size = bucket_size;
        self
    }

    /// Sets the seed used to compute signatures (default: 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the size-dependent components of the bucket priority (sizes 1 to
    /// 7; larger sizes are extrapolated linearly).
    pub fn weights(mut self, weights: [i64; 7]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Builds a function on the given keys.
    pub fn try_build<K: ?Sized + ToSig<[u64; 1]> + Sync, D: SeedStoreBuild, B: Borrow<K> + Sync>(
        &self,
        keys: &[B],
        pl: &mut impl ProgressLog,
    ) -> Result<PHastR<K, D>> {
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

        let seed = self.seed;
        let (levels, remap) = self.build_levels::<K, B>(keys, pl)?;
        let mut levels = levels.into_iter();
        let (params0, seeds0) = levels.next().expect("there is always at least one level");
        let mut params = vec![];
        let mut seeds = vec![];
        for (mut p, s) in levels {
            p.first_seed = seeds.len() as u64;
            seeds.extend_from_slice(&s);
            params.push(p);
        }
        Ok(PHastR {
            seed,
            n: keys.len(),
            pattern_shift: 6 - self.log2_patterns,
            default_shifts: 6 - self.log2_patterns == DEFAULT_PATTERN_SHIFT
                && params0.scale == DEFAULT_SCALE as u64,
            fast_scale: if 6 - self.log2_patterns == DEFAULT_PATTERN_SHIFT && params0.scale <= 3 {
                params0.scale as u8 + 1
            } else {
                0
            },
            params0,
            seeds0: D::from_seeds(&seeds0, self.seed_bits),
            params: params.into_boxed_slice(),
            seeds: D::from_seeds(&seeds, self.seed_bits),
            remap,
            _marker: std::marker::PhantomData,
        })
    }

    /// Builds the levels.
    ///
    /// Signatures are stored with the index of their key, so that the
    /// signatures of the keys bumped from a level can be computed for the
    /// next one.
    #[allow(clippy::type_complexity)]
    fn build_levels<K: ?Sized + ToSig<[u64; 1]> + Sync, B: Borrow<K> + Sync>(
        &self,
        keys: &[B],
        pl: &mut impl ProgressLog,
    ) -> Result<(Vec<(LevelParams, Vec<u16>)>, Remap)> {
        let hash = |idx: u64, seed: u64| K::to_sig(keys[idx as usize].borrow(), seed)[0];
        let n = keys.len();
        let mut levels: Vec<(LevelParams, Vec<u16>)> = vec![];
        let mut entries: Vec<usize> = vec![];
        // Priority weights depend on the slice length of each level
        let weights = |g: &Geometry| {
            self.weights
                .unwrap_or_else(|| default_weights(self.seed_bits, g.l_mask as usize + 1))
        };

        if n == 0 {
            // A single empty level, so that queries need no special case
            let geom = self.geometry(0, 1, self.bucket_size);
            levels.push((geom.level(), vec![0]));
            let efb = EliasFanoBuilder::new(0, 1);
            return Ok((levels, remap(efb)));
        }

        // The first level
        pl.info(format_args!("Computing signatures..."));
        let geom = self.geometry(n, n, self.bucket_size);
        let (sigs, bucket_begin) = group(
            n,
            |i| Sig {
                h: hash(i as u64, self.seed),
                idx: i as u64,
            },
            &geom,
        );
        let out = sweep_level(&sigs, &bucket_begin, &geom, &weights(&geom), true)
            .expect("bumping sweeps cannot fail");
        drop((sigs, bucket_begin));
        pl.info(format_args!(
            "Level 0: {} keys, {} bumped ({:.3}%)",
            n,
            out.bumped.len(),
            100.0 * out.bumped.len() as f64 / n as f64
        ));
        let holes = self::holes(&out.occupied, n);
        debug_assert_eq!(holes.len(), out.bumped.len());
        levels.push((geom.level(), out.seeds));
        let mut hole_idx = 0;
        let mut last_hole = 0;

        // Keys with the same signature are mapped to the same slot by every
        // seed, so they are bumped, and separated by the signatures of the
        // following levels, which depend on a second hash, unless the keys
        // are equal: since equal keys are bumped from the first level, we
        // detect them in the following levels, hashing keys with the same
        // signature with a third seed.
        let duplicates = |sigs: &[Sig], bucket_begin: &[usize]| {
            let mut bucket = vec![];
            bucket_begin.windows(2).any(|w| {
                bucket.clear();
                bucket.extend_from_slice(&sigs[w[0]..w[1]]);
                bucket.sort_unstable_by_key(|x| x.h);
                bucket.chunk_by(|x, y| x.h == y.h).any(|equal| {
                    let mut other: Vec<u64> = equal
                        .iter()
                        .map(|x| hash(x.idx, self.seed ^ 0xD6E8_FEB8_6659_FD93))
                        .collect();
                    other.sort_unstable();
                    other.windows(2).any(|w| w[0] == w[1])
                })
            })
        };

        // The following levels: signatures are computed from the two
        // hashes of each key and the salt of the level
        let mut cur = out.bumped;
        while !cur.is_empty() {
            let k = cur.len();
            let group = |salt: u64, geom: &Geometry| {
                group(
                    k,
                    |i| {
                        let idx = cur[i].idx;
                        let second = hash(idx, self.seed ^ SECOND_HASH);
                        Sig {
                            h: level_sig(hash(idx, self.seed), second, salt),
                            idx,
                        }
                    },
                    geom,
                )
            };
            let (level, seeds, occupied, bumped, m) = if k > LAST_LEVEL_THRESHOLD {
                let salt = level_salt(self.seed, levels.len(), 0);
                let geom = self.geometry(k, k, self.bucket_size);
                let (sigs, bucket_begin) = group(salt, &geom);
                if duplicates(&sigs, &bucket_begin) {
                    bail!("Duplicate keys");
                }
                let out = sweep_level(&sigs, &bucket_begin, &geom, &weights(&geom), true)
                    .expect("bumping sweeps cannot fail");
                let mut level = geom.level();
                level.salt = salt;
                (level, out.seeds, out.occupied, out.bumped, geom.m)
            } else {
                // The last level does not bump: we enlarge the range and
                // change the salt until we succeed
                let mut attempt = 0u64;
                loop {
                    let m = k + k / 4 + 16 + (attempt as usize / 8) * (k / 8 + 8);
                    let geom = self.geometry(k, m, self.bucket_size.min(3.0));
                    let salt = level_salt(self.seed, levels.len(), attempt);
                    let (sigs, bucket_begin) = group(salt, &geom);
                    if attempt == 0 && duplicates(&sigs, &bucket_begin) {
                        bail!("Duplicate keys");
                    }
                    if let Some(out) =
                        sweep_level(&sigs, &bucket_begin, &geom, &weights(&geom), false)
                    {
                        let mut level = geom.level();
                        level.salt = salt;
                        break (level, out.seeds, out.occupied, out.bumped, geom.m);
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
                bumped.len(),
                100.0 * bumped.len() as f64 / k as f64
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
            cur = bumped;
        }
        debug_assert_eq!(hole_idx, holes.len());

        let mut efb = EliasFanoBuilder::new(entries.len(), n);
        for &e in &entries {
            efb.push(e);
        }
        Ok((levels, remap(efb)))
    }

    /// Computes the geometry of a level with `k` keys and output range `m`.
    fn geometry(&self, k: usize, m: usize, bucket_size: f64) -> Geometry {
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

/// Number of keys below which a level is built without bumping.
const LAST_LEVEL_THRESHOLD: usize = 4096;

/// A key during construction: its signature for the current level and its
/// index (used to compute its signatures for the following levels).
#[derive(Debug, Clone, Copy)]
struct Sig {
    h: u64,
    idx: u64,
}

/// The base-2 logarithm of the number of keys of the parts that [`group`]
/// distributes into buckets: such parts should fit the second-level cache.
const LOG2_PART_KEYS: u32 = 14;
/// The base-2 logarithm of the maximum number of parts into which
/// [`group`] distributes signatures in a single pass.
const MAX_LOG2_PARTS: u32 = 11;

/// Computes the signatures of the keys of a level and groups them by
/// bucket.
///
/// Returns the signatures, with those of each bucket in consecutive
/// positions and buckets in order, and the position of the first signature
/// of each bucket, followed by the number of keys. The signature of the
/// key of index *i* in the level is `sig(i)`.
///
/// Signatures need not be sorted, so we just distribute them: first into
/// parts made of a power of two of consecutive buckets, and then each part
/// into its buckets, counting in both cases the number of signatures with
/// each destination beforehand. The first phase processes a chunk of the
/// keys for each thread, and writes sequentially into each part; the second
/// phase processes parts in parallel, gathering the signatures of a part
/// from all chunks, and writes into a region that fits the cache. With very
/// large key sets (more than 2<sup>[`LOG2_PART_KEYS`] +
/// [`MAX_LOG2_PARTS`]</sup> keys) that would require too many parts: in
/// that case parts are larger, and the second phase distributes each part
/// into smaller parts before distributing the latter into buckets.
fn group(n: usize, sig: impl Fn(usize) -> Sig + Sync, g: &Geometry) -> (Vec<Sig>, Vec<usize>) {
    group_with(n, sig, g, LOG2_PART_KEYS, MAX_LOG2_PARTS)
}

/// Implements [`group`] with the given parameters in place of
/// [`LOG2_PART_KEYS`] and [`MAX_LOG2_PARTS`].
fn group_with(
    n: usize,
    sig: impl Fn(usize) -> Sig + Sync,
    g: &Geometry,
    log2_part_keys: u32,
    max_log2_parts: u32,
) -> (Vec<Sig>, Vec<usize>) {
    let nb = g.buckets;
    // The number of parts that fit the cache: if they are too many for a
    // single pass we use two passes with about the same number of parts
    let log2_small_parts = (n >> log2_part_keys).next_power_of_two().ilog2();
    let log2_parts = if log2_small_parts <= max_log2_parts {
        log2_small_parts
    } else {
        log2_small_parts.div_ceil(2)
    };
    // The part of a bucket is given by its index shifted right by this
    // amount
    let shift = (usize::BITS - (nb - 1).leading_zeros()).saturating_sub(log2_parts);
    let parts = ((nb - 1) >> shift) + 1;
    #[cfg(feature = "rayon")]
    let par = parallel(n);
    #[cfg(feature = "rayon")]
    let chunk_len = if par {
        n.div_ceil(rayon::current_num_threads())
    } else {
        n
    };
    #[cfg(not(feature = "rayon"))]
    let chunk_len = n;
    let chunk_len = chunk_len.max(1);

    // The signatures are written here by the first phase in the order of
    // the keys, and then by the second phase grouped by bucket
    let mut sigs: Vec<Sig> = Vec::with_capacity(n);

    // First phase: each chunk of the keys is hashed, and its signatures are
    // distributed into parts; we return the signatures of the chunk and
    // the position of the first signature of each part
    type Chunk<'a> = (usize, &'a mut [MaybeUninit<Sig>]);
    let distribute = |(chunk, sigs): Chunk| -> (Vec<Sig>, Vec<usize>) {
        let mut part_begin = vec![0usize; parts + 1];
        for (i, x) in sigs.iter_mut().enumerate() {
            let x = x.write(sig(chunk * chunk_len + i));
            part_begin[(g.bucket(x.h) >> shift) + 1] += 1;
        }
        for p in 0..parts {
            part_begin[p + 1] += part_begin[p];
        }
        let mut next = part_begin.clone();
        let mut distributed = Vec::with_capacity(sigs.len());
        let spare = distributed.spare_capacity_mut();
        for x in sigs.iter() {
            // SAFETY: we have just written this element
            let x = unsafe { x.assume_init() };
            let next = &mut next[g.bucket(x.h) >> shift];
            spare[*next].write(x);
            *next += 1;
        }
        // SAFETY: we counted the signatures of each part, so each of the
        // first sigs.len() positions has been written exactly once
        unsafe { distributed.set_len(sigs.len()) };
        (distributed, part_begin)
    };
    let spare = &mut sigs.spare_capacity_mut()[..n];
    #[cfg(feature = "rayon")]
    let chunks: Vec<(Vec<Sig>, Vec<usize>)> = if par {
        use rayon::prelude::*;
        spare
            .par_chunks_mut(chunk_len)
            .enumerate()
            .map(distribute)
            .collect()
    } else {
        spare
            .chunks_mut(chunk_len)
            .enumerate()
            .map(distribute)
            .collect()
    };
    #[cfg(not(feature = "rayon"))]
    let chunks: Vec<(Vec<Sig>, Vec<usize>)> = spare
        .chunks_mut(chunk_len)
        .enumerate()
        .map(distribute)
        .collect();

    // Second phase: the signatures of each part are distributed into
    // buckets
    let mut part_begin = vec![0usize; parts + 1];
    for (_, chunk_part_begin) in &chunks {
        for p in 0..parts {
            part_begin[p + 1] += chunk_part_begin[p + 1] - chunk_part_begin[p];
        }
    }
    for p in 0..parts {
        part_begin[p + 1] += part_begin[p];
    }
    let mut bucket_begin = vec![0usize; nb + 1];
    let mut part_sigs = Vec::with_capacity(parts);
    let mut spare = &mut sigs.spare_capacity_mut()[..n];
    for p in 0..parts {
        let (part, rest) = spare.split_at_mut(part_begin[p + 1] - part_begin[p]);
        part_sigs.push(part);
        spare = rest;
    }
    type Part<'a, 'b> = (usize, (&'a mut &'b mut [MaybeUninit<Sig>], &'a mut [usize]));
    let distribute = |(p, (sigs, begin)): Part| {
        let first_bucket = p << shift;
        let part = || {
            chunks
                .iter()
                .map(move |(sigs, part_begin)| &sigs[part_begin[p]..part_begin[p + 1]])
        };
        let log2_subparts = (sigs.len() >> log2_part_keys)
            .next_power_of_two()
            .ilog2()
            .min(shift);
        if log2_subparts <= 1 {
            // The part fits the cache (or almost)
            into_buckets(part(), sigs, begin, first_bucket, part_begin[p], g);
            return;
        }
        // We distribute the signatures of the part into smaller parts made
        // of a power of two of consecutive buckets, and then each smaller
        // part into its buckets
        let sub_shift = shift - log2_subparts;
        let subparts = ((begin.len() - 1) >> sub_shift) + 1;
        let mut subpart_begin = vec![0usize; subparts + 1];
        for run in part() {
            for x in run {
                subpart_begin[((g.bucket(x.h) - first_bucket) >> sub_shift) + 1] += 1;
            }
        }
        for s in 0..subparts {
            subpart_begin[s + 1] += subpart_begin[s];
        }
        let mut next = subpart_begin.clone();
        for run in part() {
            for &x in run {
                let next = &mut next[(g.bucket(x.h) - first_bucket) >> sub_shift];
                sigs[*next].write(x);
                *next += 1;
            }
        }
        let mut subpart: Vec<Sig> = vec![];
        for (s, begin) in begin.chunks_mut(1 << sub_shift).enumerate() {
            let sigs = &mut sigs[subpart_begin[s]..subpart_begin[s + 1]];
            subpart.clear();
            // SAFETY: we have just written these elements
            subpart.extend(sigs.iter().map(|x| unsafe { x.assume_init() }));
            into_buckets(
                std::iter::once(&subpart[..]),
                sigs,
                begin,
                first_bucket + (s << sub_shift),
                part_begin[p] + subpart_begin[s],
                g,
            );
        }
    };
    #[cfg(feature = "rayon")]
    if par {
        use rayon::prelude::*;
        part_sigs
            .par_iter_mut()
            .zip(bucket_begin[..nb].par_chunks_mut(1 << shift))
            .enumerate()
            .for_each(distribute);
    } else {
        part_sigs
            .iter_mut()
            .zip(bucket_begin[..nb].chunks_mut(1 << shift))
            .enumerate()
            .for_each(distribute);
    }
    #[cfg(not(feature = "rayon"))]
    part_sigs
        .iter_mut()
        .zip(bucket_begin[..nb].chunks_mut(1 << shift))
        .enumerate()
        .for_each(distribute);
    bucket_begin[nb] = n;
    // SAFETY: we counted the signatures of each part and of each bucket,
    // so each of the first n positions has been written exactly once
    unsafe { sigs.set_len(n) };
    (sigs, bucket_begin)
}

/// Distributes into buckets the signatures of a sequence of runs whose
/// buckets are consecutive and start from `first_bucket`.
///
/// The signatures are written to `sigs`, and the position of the first
/// signature of each bucket to `begin`, assuming that `sigs` starts at
/// position `offset`.
#[inline(always)]
fn into_buckets<'a>(
    runs: impl Iterator<Item = &'a [Sig]> + Clone,
    sigs: &mut [MaybeUninit<Sig>],
    begin: &mut [usize],
    first_bucket: usize,
    offset: usize,
    g: &Geometry,
) {
    for run in runs.clone() {
        for x in run {
            begin[g.bucket(x.h) - first_bucket] += 1;
        }
    }
    let mut sum = 0;
    for begin in begin.iter_mut() {
        let size = *begin;
        *begin = sum;
        sum += size;
    }
    for run in runs {
        for &x in run {
            let next = &mut begin[g.bucket(x.h) - first_bucket];
            sigs[*next].write(x);
            *next += 1;
        }
    }
    // Each element is now the end of its bucket, that is, the beginning of
    // the following one
    let mut prev = 0;
    for begin in begin.iter_mut() {
        let end = *begin;
        *begin = offset + prev;
        prev = end;
    }
}

/// The geometry of a level during construction.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// The number of buckets (always odd).
    buckets: usize,
    /// A lower bound for the number of signatures of a bucket.
    bucket_width: u64,
    /// The number of slices.
    num_slices: u64,
    /// The slice length minus one.
    l_mask: u64,
    /// The base-2 logarithm of the amount by which a unit increase of the
    /// seed moves the keys of a bucket.
    scale: u32,
    /// The base-2 logarithm of the number of patterns.
    log2_patterns: u32,
    /// The number of bits of a seed.
    seed_bits: u32,
    /// The output range.
    m: usize,
}

impl Geometry {
    #[inline(always)]
    fn bucket(&self, h: u64) -> usize {
        mul_hi(h, self.buckets as u64) as usize
    }

    #[inline(always)]
    fn slice_begin(&self, h: u64) -> usize {
        mul_hi(h, self.num_slices) as usize
    }

    /// Returns a lower bound for the beginning of the slices of the keys
    /// of bucket `b` and of the following buckets.
    #[inline(always)]
    fn first_slice(&self, b: usize) -> usize {
        // The signatures of bucket b are at least b · 2⁶⁴ / buckets
        mul_hi(b as u64 * self.bucket_width, self.num_slices) as usize
    }

    /// Returns the value providing the in-slice offsets of the patterns:
    /// the lower half of the product of the signature and the number of
    /// buckets, whose upper half is the bucket (it is thus uniform among
    /// the keys of a bucket).
    #[inline(always)]
    fn offsets(&self, h: u64) -> u64 {
        h.wrapping_mul(self.buckets as u64)
    }

    /// Returns the slot of a key with the given signature when its bucket
    /// has the given seed (see [`PHastR::pos`]).
    #[inline(always)]
    fn pos(&self, h: u64, seed: usize) -> usize {
        let r = seed & ((1 << self.log2_patterns) - 1);
        let offset = (self.offsets(h) >> (r as u32 * (64 >> self.log2_patterns)))
            .wrapping_add((seed as u64) << self.scale);
        self.slice_begin(h) + (offset & self.l_mask) as usize
    }

    fn level(&self) -> LevelParams {
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

/// Number of buckets in the window of a sweep.
const WINDOW: usize = 256;
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
struct Level {
    /// The seeds of the buckets.
    seeds: Vec<u16>,
    /// The used slots, as a set of bits.
    occupied: Vec<u64>,
    /// The keys of the buckets that could not be placed.
    bumped: Vec<Sig>,
}

/// Assigns seeds to the buckets of a level, possibly in parallel.
///
/// Returns `None` if `allow_bump` is false and some bucket could not be
/// placed.
fn sweep_level(
    keys: &[Sig],
    bucket_begin: &[usize],
    g: &Geometry,
    weights: &[i64; 7],
    allow_bump: bool,
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
    let mut bumped = vec![];
    // Each sweep returns the slots it used (and those marked as used) and
    // the keys it bumped
    let mut merge = |swept: Swept| {
        let occupied = occupied.iter_mut().skip(swept.occupied_begin / 64);
        for (occupied, word) in occupied.zip(&swept.occupied) {
            *occupied |= word;
        }
        bumped.extend(swept.bumped);
    };

    if chunks == 1 {
        merge(Sweep::new(keys, bucket_begin, g, weights, 0, nb, &mut seeds).run(allow_bump)?);
        return Some(Level {
            seeds,
            occupied,
            bumped,
        });
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
    let run_chunk = |(i, part): (usize, &mut &mut [u16])| {
        let lo = bounds[i];
        let hi = if i + 1 == chunks {
            bounds[i + 1]
        } else {
            bounds[i + 1] - gap
        };
        Sweep::new(keys, bucket_begin, g, weights, lo, hi, &mut part[..hi - lo]).run(allow_bump)
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
        let mut sw = Sweep::new(keys, bucket_begin, g, weights, lo, hi, &mut gap_seeds);
        let before = lo.saturating_sub(gap).max(bounds[i]);
        let after = (hi + gap).min(bounds[i + 2]);
        let first = g.first_slice(lo);
        for (from, to) in [(before, lo), (hi, after)] {
            for x in &keys[bucket_begin[from]..bucket_begin[to]] {
                let s = seeds[g.bucket(x.h)] as usize;
                if s != 0 {
                    let p = g.pos(x.h, s);
                    if p >= first {
                        sw.set(p);
                    }
                }
            }
        }
        let swept = sw.run(allow_bump);
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
    Some(Level {
        seeds,
        occupied,
        bumped,
    })
}

/// Returns whether position `p` is set in the occupancy bitmap.
#[inline(always)]
fn is_occupied(bits: &[u64], p: usize) -> bool {
    bits[p / 64] >> (p % 64) & 1 != 0
}

/// Returns the positions in [0 . . m) that are not set in the occupancy
/// bitmap, in increasing order.
fn holes(bits: &[u64], m: usize) -> Vec<usize> {
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
    /// [offset](Self::offset) for the pattern.
    #[inline(always)]
    fn ring(&self, slice_begin: usize, offset: usize) -> (usize, usize) {
        (
            slice_begin + (offset & ((1 << self.log2_stride) - 1)),
            offset >> self.log2_stride,
        )
    }

    /// Returns the slot of a key for the seed of index `j` of a pattern,
    /// given the first slot of its ring and its index for the first seed
    /// of the pattern (see [`ring`](Self::ring)).
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
struct Sweep<'a> {
    /// The keys of the level, grouped by bucket.
    keys: &'a [Sig],
    /// The position in `keys` of the first key of each bucket.
    bucket_begin: &'a [usize],
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
    /// The keys of the buckets that could not be placed.
    bumped: Vec<Sig>,
    /// The state of the search for the seed of a bucket (see
    /// [`state`](Self::state)).
    state: Vec<u64>,
    /// The nonzero words of `free`, each with its index.
    nonzero: Vec<(u64, u32)>,
    /// The indices at which some key goes back to the first slot of its
    /// ring.
    back: Vec<u32>,
}

impl<'a> Sweep<'a> {
    fn new(
        keys: &'a [Sig],
        bucket_begin: &'a [usize],
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
            keys,
            bucket_begin,
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
            bumped: vec![],
            state: Vec::with_capacity(free_words + rings.patterns * 3),
            nonzero: vec![(0, 0); free_words],
            back: Vec::with_capacity(64),
        }
    }

    #[inline(always)]
    fn size(&self, b: usize) -> usize {
        self.bucket_begin[b + 1] - self.bucket_begin[b]
    }

    #[inline(always)]
    fn bucket_keys(&self, b: usize) -> &'a [Sig] {
        &self.keys[self.bucket_begin[b]..self.bucket_begin[b + 1]]
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
    ///   [`sum`](Self::sum)): a word for each pattern contains the indices
    ///   of a group of keys.
    ///
    /// The parts are stored contiguously, so that in the innermost loops
    /// they are accessed through a single pointer.
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
    /// groups after the first one (see [`sum`](Self::sum)).
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
    fn search(&mut self, rings: &Rings, b: usize) -> usize {
        let g = *self.g;
        let keys = self.bucket_keys(b);
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
        for (k, key) in keys.iter().enumerate() {
            let (slice_begin, offsets) = (g.slice_begin(key.h), g.offsets(key.h));
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

        // The best seed is identified by the index of its bit in `free`
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
    /// [`search`](Self::search) does, when there are many free seeds.
    ///
    /// The sum of the slots increases with the index of the seed, except
    /// at the indices at which some key goes back to the first slot of its
    /// ring: between two such indices only the first free index can be the
    /// best one.
    #[inline(never)]
    fn search_many(&mut self, keys: &[Sig]) -> usize {
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
                    .map(|key| {
                        let offset = rings.offset(g.offsets(key.h), r);
                        rings.ring(g.slice_begin(key.h), offset).1
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
    fn place(&mut self, rings: &Rings, keys: &[Sig], r: usize, j: usize) -> bool {
        let g = *self.g;
        let bit = |key: &Sig| {
            let offset = rings.offset(g.offsets(key.h), r);
            let (first, x) = rings.ring(g.slice_begin(key.h), offset);
            let p = rings.slot(first, x, j);
            (rings.word(p), 1u64 << ((p >> rings.log2_stride) % 64))
        };
        // The slots were free: a used slot has been marked by a previous
        // key of the bucket
        let used = &mut self.used[..];
        let mut collisions = 0;
        for key in keys {
            let (w, mask) = bit(key);
            collisions |= used[w] & mask;
            used[w] |= mask;
        }
        if collisions != 0 {
            for key in keys {
                let (w, mask) = bit(key);
                used[w] &= !mask;
            }
        }
        collisions == 0
    }

    /// Like [`search`](Self::search), but considers all free seeds in
    /// order of sum of slots until one that maps the keys of the bucket to
    /// distinct slots is found (this happens rarely).
    #[cold]
    #[inline(never)]
    fn search_distinct(&mut self, keys: &[Sig]) -> usize {
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

    /// Processes the buckets of the range; returns `None` if `allow_bump`
    /// is false and some bucket could not be placed (in which case the
    /// sweep stops immediately).
    fn run(self, allow_bump: bool) -> Option<Swept> {
        let rings = self.rings;
        // The parameters of the rings are used as shift amounts and masks
        // in the innermost loops: we compile a version of the sweep in
        // which those of the default configuration are constants
        let default = Rings {
            word_mask: rings.word_mask,
            ..Rings::DEFAULT
        };
        if rings == default {
            self.sweep(&default, allow_bump)
        } else {
            self.sweep(&rings, allow_bump)
        }
    }

    #[inline(always)]
    fn sweep(mut self, rings: &Rings, allow_bump: bool) -> Option<Swept> {
        let (lo, hi) = (self.lo, self.hi);
        let mut window = Window::new(self.weights);
        let mut span_begin = lo;
        while span_begin < hi && self.size(span_begin) == 0 {
            span_begin += 1;
        }
        let span_end = |span_begin: usize| (span_begin + WINDOW).min(hi);
        let column = 64usize << rings.log2_stride;
        if span_begin < hi {
            // The columns before the first slice in the window contain
            // slots that are no longer reachable
            while self.first_slot + column <= self.g.first_slice(span_begin) {
                self.retire_column(rings);
            }
            for b in span_begin..span_end(span_begin) {
                let size = self.size(b);
                if size != 0 {
                    window.push(b, size, self.weights);
                }
            }
        }
        while let Some(b) = window.pop(span_begin) {
            let seed = self.search(rings, b);
            self.seeds[b - lo] = seed as u16;
            if seed == 0 {
                if !allow_bump {
                    return None;
                }
                self.bumped.extend_from_slice(self.bucket_keys(b));
            }
            if b == span_begin {
                let old_end = span_end(span_begin);
                span_begin += 1;
                while span_begin < old_end && !window.contains(span_begin) {
                    span_begin += 1;
                }
                if span_begin == old_end {
                    while span_begin < hi && self.size(span_begin) == 0 {
                        span_begin += 1;
                    }
                    if span_begin == hi {
                        break;
                    }
                }
                while self.first_slot + column <= self.g.first_slice(span_begin) {
                    self.retire_column(rings);
                }
                for b in old_end..span_end(span_begin) {
                    let size = self.size(b);
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
            bumped: self.bumped,
        })
    }
}

/// The result of a [`Sweep`].
struct Swept {
    /// The slots used by the buckets of the range, and those marked as
    /// used beforehand, as a set of bits starting from slot
    /// `occupied_begin` (a multiple of 64).
    occupied: Vec<u64>,
    occupied_begin: usize,
    /// The keys of the buckets that could not be placed.
    bumped: Vec<Sig>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use dsi_progress_logger::no_logging;

    fn check<D: SeedStoreBuild>(n: usize, builder: PHastRBuilder) {
        let keys: Vec<u64> = (0..n as u64).collect();
        let phf: PHastR<u64, D> = builder.try_build(&keys, no_logging![]).unwrap();
        let mut seen = vec![false; n];
        for key in &keys {
            let v = phf.get(key);
            assert!(v < n, "{v} >= {n}");
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
    }

    #[test]
    fn test_small() {
        for n in [0, 1, 2, 3, 10, 100, 1000, 5000, 10000] {
            check::<Box<[u8]>>(n, PHastRBuilder::default());
            check::<BitFieldVec<Box<[usize]>>>(n, PHastRBuilder::default().seed_bits(10));
            for log2_patterns in 0..=3 {
                check::<Box<[u8]>>(
                    n,
                    PHastRBuilder::default()
                        .log2_patterns(log2_patterns)
                        .log2_slice_len(8),
                );
            }
        }
    }

    #[test]
    fn test_medium() {
        check::<Box<[u8]>>(1_000_000, PHastRBuilder::default());
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().bucket_size(5.0));
        // Rings of four, two and one words
        check::<BitFieldVec<Box<[usize]>>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        );
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_patterns(1));
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_patterns(0));
        check::<Box<[u16]>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(11)
                .log2_slice_len(12)
                .bucket_size(6.75),
        );
        // Longer slices, and thus a larger stride
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(12));
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(16));
        // Rings shorter than a word: a pattern has four seeds
        check::<Box<[u8]>>(
            100_000,
            PHastRBuilder::default()
                .seed_bits(4)
                .log2_slice_len(8)
                .bucket_size(2.0),
        );
        // Slices shorter than the number of seeds: seeds are redundant
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(6));
    }

    #[test]
    fn test_group() {
        // One, two and three passes, with skewed signatures, too
        for (n, log2_part_keys, max_log2_parts) in [
            (1, 14, 11),
            (1000, 14, 11),
            (100_000, 6, 11),
            (100_000, 4, 5),
            (100_000, 2, 3),
            (300_000, 5, 4),
        ] {
            for skew in [false, true] {
                let g = PHastRBuilder::default().geometry(n, n, 4.5);
                let sig = |i: usize| {
                    let h = mix(i as u64 + 1, 0x9E37_79B9_7F4A_7C15);
                    Sig {
                        h: if skew { h >> (i % 8) } else { h },
                        idx: i as u64,
                    }
                };
                let (sigs, bucket_begin) = group_with(n, sig, &g, log2_part_keys, max_log2_parts);
                assert_eq!(sigs.len(), n);
                assert_eq!(bucket_begin.len(), g.buckets + 1);
                assert_eq!(bucket_begin[g.buckets], n);
                let mut seen = vec![false; n];
                for b in 0..g.buckets {
                    for x in &sigs[bucket_begin[b]..bucket_begin[b + 1]] {
                        assert_eq!(g.bucket(x.h), b);
                        assert_eq!(x.h, sig(x.idx as usize).h);
                        assert!(!std::mem::replace(&mut seen[x.idx as usize], true));
                    }
                }
                assert!(seen.iter().all(|&x| x));
            }
        }
    }

    #[test]
    fn test_large_buckets() {
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
            );
        }
        check::<Box<[u8]>>(100_000, PHastRBuilder::default().bucket_size(70.0));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_many_chunks() {
        // Many threads, and hence many chunks and gaps, with long slices
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            check::<BitFieldVec<Box<[usize]>>>(
                2_000_000,
                PHastRBuilder::default()
                    .seed_bits(10)
                    .log2_slice_len(11)
                    .bucket_size(6.0),
            );
            check::<Box<[u8]>>(2_000_000, PHastRBuilder::default());
        });
    }

    #[test]
    fn test_parameter_checks() {
        let keys: Vec<u64> = (0..1000).collect();
        // Seeds too large for the storage
        let r: Result<PHastR<u64>> = PHastRBuilder::default()
            .seed_bits(10)
            .try_build(&keys, no_logging![]);
        assert!(r.is_err());
        // Too many patterns for the slice length
        let r: Result<PHastR<u64>> = PHastRBuilder::default()
            .log2_patterns(3)
            .try_build(&keys, no_logging![]);
        assert!(r.is_err());
        // Too many patterns for the number of seed bits
        let r: Result<PHastR<u64>> = PHastRBuilder::default()
            .seed_bits(2)
            .try_build(&keys, no_logging![]);
        assert!(r.is_err());
    }

    #[test]
    fn test_unaligned() -> Result<()> {
        for (bits, ll, n) in [
            (10, 11, 300_000),
            (12, 12, 300_000),
            (16, 12, 10_000),
            (10, 11, 0),
        ] {
            let keys: Vec<u64> = (0..n as u64).collect();
            let phf: PHastR<u64, BitFieldVec<Box<[usize]>>> = PHastRBuilder::default()
                .seed_bits(bits)
                .log2_slice_len(ll)
                .bucket_size(6.0)
                .try_build(&keys, no_logging![])?;
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
    fn test_batch() {
        let keys: Vec<u64> = (0..100_000).collect();
        let phf: PHastR<u64> = PHastRBuilder::default()
            .try_build(&keys, no_logging![])
            .unwrap();
        for chunk in keys.chunks_exact(8) {
            let batch: [&u64; 8] = std::array::from_fn(|i| &chunk[i]);
            let values = phf.get_batch(batch);
            for (key, value) in chunk.iter().zip(values) {
                assert_eq!(phf.get(key), value);
            }
        }
    }

    #[test]
    fn test_duplicates() {
        let keys: Vec<u64> = vec![1, 2, 3, 2];
        let r: Result<PHastR<u64>> = PHastRBuilder::default().try_build(&keys, no_logging![]);
        assert!(r.is_err());
    }

    /// A key whose 64-bit signature with seed 0 collides with that of another
    /// key for the first 2 · `COLLIDING` keys (pairs 2*i*, 2*i* + 1); other
    /// seeds give independent signatures.
    #[derive(Clone, Copy)]
    struct CollidingKey(u64);
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
    fn test_signature_collisions() {
        // Colliding signatures are bumped and separated by the signatures of
        // the following levels, with all configurations of the following
        // levels (in particular, with only the last level)
        for n in [5_000u64, 300_000] {
            let keys: Vec<CollidingKey> = (0..n).map(CollidingKey).collect();
            let phf: PHastR<CollidingKey> = PHastRBuilder::default()
                .try_build(&keys, no_logging![])
                .unwrap();
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
        let r: Result<PHastR<CollidingKey>> =
            PHastRBuilder::default().try_build(&keys, no_logging![]);
        assert!(r.is_err());
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
            let phf: PHastR<u64, $d> = $builder.try_build(&keys, no_logging![]).unwrap();
            let mut cursor = <AlignedCursor<Aligned64>>::new();
            // SAFETY: phf was built by its constructor
            unsafe { phf.serialize(&mut cursor) }.unwrap();
            let len = cursor.len();
            cursor.set_position(0);
            // SAFETY: we just serialized a valid structure into this buffer
            let case = unsafe { <PHastR<u64, $d>>::read_mem(&mut cursor, len) }.unwrap();
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
                assert_eq!(phf.get(key), des.get(key));
            }
        }};
    }

    /// A memory-mapped function must take little memory beyond the mapping:
    /// with the default flags mem_dbg does not follow references, so it
    /// counts only the fields of the structure, whereas following references
    /// it must count the same data as the original.
    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_mem_size() {
        use epserde::prelude::*;
        let keys: Vec<u64> = (0..1_000_000).collect();
        let phf: PHastR<u64> = PHastRBuilder::default()
            .try_build(&keys, no_logging![])
            .unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        // SAFETY: phf was built by its constructor
        unsafe { phf.store(file.path()) }.unwrap();
        // SAFETY: we just stored a valid structure into the file
        let case = unsafe { <PHastR<u64>>::mmap(file.path(), Flags::empty()) }.unwrap();
        let mapped = case.uncase();
        let owned_size = phf.mem_size(SizeFlags::default());
        let mapped_size = mapped.mem_size(SizeFlags::default());
        let mapped_followed = mapped.mem_size(SizeFlags::FOLLOW_REFS);
        eprintln!(
            "owned {owned_size} bytes, mapped {mapped_size} bytes, mapped following references {mapped_followed} bytes"
        );
        assert!(
            mapped_size < 1024,
            "mapped structure takes {mapped_size} bytes"
        );
        assert!(mapped_followed >= owned_size - 1024 && mapped_followed <= owned_size + 1024);
        for key in &keys {
            assert_eq!(phf.get(key), mapped.get(key));
        }
    }

    #[cfg(feature = "epserde")]
    #[test]
    fn test_epserde() {
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
        {
            use epserde::prelude::*;
            use epserde::utils::AlignedCursor;
            type U = Unaligned<PHastR<u64, BitFieldVec<Box<[usize]>>>>;
            let keys: Vec<u64> = (0..200_000).collect();
            let phf: PHastR<u64, BitFieldVec<Box<[usize]>>> = PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .try_build(&keys, no_logging![])
                .unwrap();
            let phf: U = phf.try_into_unaligned().unwrap();
            let mut cursor = <AlignedCursor<Aligned64>>::new();
            // SAFETY: phf was built by its constructor
            unsafe { phf.serialize(&mut cursor) }.unwrap();
            let len = cursor.len();
            cursor.set_position(0);
            // SAFETY: we just serialized a valid structure into this buffer
            let case = unsafe { <U>::read_mem(&mut cursor, len) }.unwrap();
            let des = case.uncase();
            let _: &BitFieldVecU<&[usize]> = &des.seeds0;
            for key in &keys {
                assert_eq!(phf.get(key), des.get(key));
            }
        }
    }

    #[test]
    fn test_strings() {
        let keys: Vec<String> = (0..200_000).map(|i| format!("key{i}")).collect();
        let phf: PHastR<str> = PHastRBuilder::default()
            .try_build(
                &keys.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                no_logging![],
            )
            .unwrap();
        let mut seen = vec![false; keys.len()];
        for key in &keys {
            let v = phf.get(key.as_str());
            assert!(!seen[v]);
            seen[v] = true;
        }
    }
}
