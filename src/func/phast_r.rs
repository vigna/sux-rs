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
//! one operation more than PHast+ with wrapping; a small fraction of the keys
//! accesses further levels and the Elias–Fano sequence.
//!
//! During construction the set of used slots is stored by residue classes
//! modulo the stride, so the ring of a key is a block of consecutive bits:
//! the seeds of a pattern that are feasible for a bucket are obtained by
//! rotating and combining one such block for each key.
//!
//! As in PHast+, keys are hashed again with a different seed at each level
//! after the first, so 64-bit signatures suffice for any number of keys: keys
//! with the same signature are bumped from the first level and separated at
//! the following ones. Duplicate keys are detected and reported as errors.
//!
//! With the default parameters (8-bit seeds, four patterns, slices of length
//! 1024) space is about 1.92 bits per key, against the 1.97 bits per key of
//! PHast+ with wrapping, construction is faster, and queries take about the
//! same time. With 10-bit seeds stored in a [`BitFieldVec`] (see
//! [`PHastRBuilder::seed_bits`]) space is about 1.86 bits per key; in this
//! case, queries are faster after converting the function with
//! [`TryIntoUnaligned::try_into_unaligned`], so that seeds are accessed with
//! [unaligned reads].
//!
//! [Beling and Sanders, *PHast — Perfect Hashing made fast*]: https://arxiv.org/abs/2504.17918
//! [Elias–Fano]: crate::dict::elias_fano
//! [unaligned reads]: BitFieldVec::get_unaligned

use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

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

/// Returns the seed used to hash keys for a level: the first level uses the
/// seed of the function; level *ℓ* > 0 uses the seed plus *ℓ*, and attempt *a*
/// of the last level adds *a* · 2³² to it.
#[inline(always)]
const fn level_seed(seed: u64, level: usize, attempt: u64) -> u64 {
    seed.wrapping_add(level as u64).wrapping_add(attempt << 32)
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
    /// The seed used to hash keys for this level (unused for the first
    /// level, which uses the seed of the function).
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
    /// optimized away; queries in a loop perform it just once).
    #[inline(always)]
    fn pos0(&self, h: u64, lo: u64, seed: usize) -> usize {
        let lv = &self.params0;
        if self.default_shifts {
            Self::pos(lv, h, lo, seed, DEFAULT_PATTERN_SHIFT, DEFAULT_SCALE)
        } else {
            Self::pos(lv, h, lo, seed, self.pattern_shift, lv.scale as u32)
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
        self.get_slow(key)
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
            self.get_slow(keys[i])
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

    /// Handles keys bumped from the first level: at each level, the key is
    /// hashed again with the seed of the level.
    #[cold]
    #[inline(never)]
    fn get_slow(&self, key: &K) -> usize {
        for lv in self.params.as_ref() {
            let h = K::to_sig(key, lv.salt)[0];
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
/// an expected bucket size of 4.75 keys.
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
            bucket_size: 4.75,
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

    /// Sets the expected number of keys per bucket (default: 4.75).
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

        pl.info(format_args!("Computing signatures..."));
        let seed = self.seed;
        let hash = |(i, k): (usize, &B)| Sig {
            h: K::to_sig(k.borrow(), seed)[0],
            idx: i as u64,
        };
        #[cfg(feature = "rayon")]
        let sigs: Vec<Sig> = if parallel(keys.len()) {
            use rayon::prelude::*;
            keys.par_iter()
                .enumerate()
                .with_min_len(crate::RAYON_MIN_LEN)
                .map(hash)
                .collect()
        } else {
            keys.iter().enumerate().map(hash).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let sigs: Vec<Sig> = keys.iter().enumerate().map(hash).collect();

        let (levels, remap) = self.build_levels::<K, B>(keys, sigs, pl)?;
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
            params0,
            seeds0: D::from_seeds(&seeds0, self.seed_bits),
            params: params.into_boxed_slice(),
            seeds: D::from_seeds(&seeds, self.seed_bits),
            remap,
            _marker: std::marker::PhantomData,
        })
    }

    /// Builds the levels from the signatures of the first level (each with
    /// the index of its key, so that keys bumped from a level can be hashed
    /// again for the next one).
    #[allow(clippy::type_complexity)]
    fn build_levels<K: ?Sized + ToSig<[u64; 1]> + Sync, B: Borrow<K> + Sync>(
        &self,
        keys: &[B],
        mut cur: Vec<Sig>,
        pl: &mut impl ProgressLog,
    ) -> Result<(Vec<(LevelParams, Vec<u16>)>, Remap)> {
        // Hashes again the keys of the given signatures with the given seed
        let hash = |idx: u64, seed: u64| K::to_sig(keys[idx as usize].borrow(), seed)[0];
        let rehash = |v: &mut [Sig], seed: u64| {
            let f = |x: &mut Sig| x.h = hash(x.idx, seed);
            #[cfg(feature = "rayon")]
            if parallel(v.len()) {
                use rayon::prelude::*;
                v.par_iter_mut().with_min_len(1 << 16).for_each(f);
                return;
            }
            v.iter_mut().for_each(f);
        };
        let n = cur.len();
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

        // Keys with the same signature are adjacent after sorting. They are
        // bumped (they are mapped to the same slot by every seed), and
        // separated by the signatures of the following levels, unless they
        // are equal: we detect duplicate keys hashing again keys with the
        // same signature with a different seed.
        sort_sigs(&mut cur);
        let equal = |w: &[Sig]| (w[0].h == w[1].h).then_some(w[0].h);
        #[cfg(feature = "rayon")]
        let mut equal_h: Vec<u64> = if parallel(cur.len()) {
            use rayon::prelude::*;
            cur.par_windows(2).filter_map(equal).collect()
        } else {
            cur.windows(2).filter_map(equal).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let mut equal_h: Vec<u64> = cur.windows(2).filter_map(equal).collect();
        equal_h.dedup();
        for h in equal_h {
            let start = cur.partition_point(|x| x.h < h);
            let end = cur.partition_point(|x| x.h <= h);
            let mut other: Vec<u64> = cur[start..end]
                .iter()
                .map(|x| hash(x.idx, self.seed ^ 0xD6E8_FEB8_6659_FD93))
                .collect();
            other.sort_unstable();
            if other.windows(2).any(|w| w[0] == w[1]) {
                bail!("Duplicate keys");
            }
        }

        // The first level
        let geom = self.geometry(n, n, self.bucket_size);
        let out =
            sweep_level(&cur, &geom, &weights(&geom), true).expect("bumping sweeps cannot fail");
        drop(cur);
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

        // The following levels: keys are hashed again with the seed of the
        // level
        let mut cur = out.bumped;
        while !cur.is_empty() {
            let k = cur.len();
            let (level, seeds, occupied, bumped, m) = if k > LAST_LEVEL_THRESHOLD {
                let seed = level_seed(self.seed, levels.len(), 0);
                rehash(&mut cur, seed);
                sort_sigs(&mut cur);
                let geom = self.geometry(k, k, self.bucket_size);
                let out = sweep_level(&cur, &geom, &weights(&geom), true)
                    .expect("bumping sweeps cannot fail");
                let mut level = geom.level();
                level.salt = seed;
                (level, out.seeds, out.occupied, out.bumped, geom.m)
            } else {
                // The last level does not bump: we enlarge the range and
                // hash the keys with a different seed until we succeed
                let mut attempt = 0u64;
                loop {
                    let m = k + k / 4 + 16 + (attempt as usize / 8) * (k / 8 + 8);
                    let geom = self.geometry(k, m, self.bucket_size.min(3.0));
                    let seed = level_seed(self.seed, levels.len(), attempt);
                    let mut sigs = cur.clone();
                    rehash(&mut sigs, seed);
                    sort_sigs(&mut sigs);
                    if let Some(out) = sweep_level(&sigs, &geom, &weights(&geom), false) {
                        let mut level = geom.level();
                        level.salt = seed;
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
        Geometry {
            // The number of buckets must be odd, as offsets are taken from
            // the product of the signature and the number of buckets
            buckets: (k as f64 / bucket_size).round().max(1.0) as usize | 1,
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
/// index (used to hash it again for the following levels).
#[derive(Debug, Clone, Copy, Default)]
struct Sig {
    h: u64,
    idx: u64,
}

/// Ordering by signature (and index, to break ties), needed by
/// [`voracious_radix_sort`].
impl PartialOrd for Sig {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((self.h, self.idx).cmp(&(other.h, other.idx)))
    }
}

impl PartialEq for Sig {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.h == other.h && self.idx == other.idx
    }
}

impl voracious_radix_sort::Radixable<u64> for Sig {
    type Key = u64;
    #[inline(always)]
    fn key(&self) -> u64 {
        self.h
    }
}

/// Sorts by signature using [`voracious_radix_sort`] (multithreaded with the
/// `rayon` feature).
fn sort_sigs(v: &mut [Sig]) {
    use voracious_radix_sort::RadixSort;
    #[cfg(feature = "rayon")]
    if parallel(v.len()) {
        v.voracious_mt_sort(rayon::current_num_threads());
        return;
    }
    v.voracious_sort();
}

/// The geometry of a level during construction.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// The number of buckets (always odd).
    buckets: usize,
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

/// Size of the cyclic set recording the buckets in the priority queue.
const HEAP_BITS: usize = 1024;
/// Number of buckets in the window of a sweep.
const WINDOW: usize = 256;

struct SweepOut {
    seeds: Vec<u16>,
    /// Occupancy bitmap of the output range.
    occupied: Vec<u64>,
    bumped: Vec<Sig>,
}

/// Assigns seeds to the buckets of a level, possibly in parallel.
///
/// Returns `None` if `allow_bump` is false and some bucket could not be
/// placed.
fn sweep_level(
    keys: &[Sig],
    g: &Geometry,
    weights: &[i64; 7],
    allow_bump: bool,
) -> Option<SweepOut> {
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
    let ok = if chunks == 1 {
        Sweep::new(keys, g, weights, 0, nb, &mut seeds).run(allow_bump)
    } else {
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
            Sweep::new(keys, g, weights, lo, hi, &mut part[..hi - lo]).run(allow_bump)
        };
        #[cfg(feature = "rayon")]
        let ok1 = {
            use rayon::prelude::*;
            parts
                .par_iter_mut()
                .enumerate()
                .map(run_chunk)
                .reduce(|| true, |a, b| a && b)
        };
        #[cfg(not(feature = "rayon"))]
        let ok1 = parts
            .iter_mut()
            .enumerate()
            .map(run_chunk)
            .fold(true, |a, b| a && b);
        drop(parts);
        if !ok1 && !allow_bump {
            return None;
        }

        // Gaps: the seeds of the neighboring buckets are now known, and the
        // slots they use are marked.
        let seeds_ro: &[u16] = &seeds;
        let run_gap = |i: usize| -> (usize, Vec<u16>, bool) {
            let lo = bounds[i + 1] - gap;
            let hi = bounds[i + 1];
            let mut gap_seeds = vec![0u16; hi - lo];
            let mut sw = Sweep::new(keys, g, weights, lo, hi, &mut gap_seeds);
            let before = lo.saturating_sub(gap).max(bounds[i]);
            let after = (hi + gap).min(bounds[i + 2]);
            let first = sw.first_slice();
            for (from, to) in [(before, lo), (hi, after)] {
                let k_lo = keys.partition_point(|x| g.bucket(x.h) < from);
                let k_hi = keys.partition_point(|x| g.bucket(x.h) < to);
                for x in &keys[k_lo..k_hi] {
                    let s = seeds_ro[g.bucket(x.h)] as usize;
                    if s != 0 {
                        let p = g.pos(x.h, s);
                        if p >= first {
                            sw.set(p);
                        }
                    }
                }
            }
            let ok = sw.run(allow_bump);
            (lo, gap_seeds, ok)
        };
        #[cfg(feature = "rayon")]
        let gaps: Vec<(usize, Vec<u16>, bool)> = {
            use rayon::prelude::*;
            (0..chunks - 1).into_par_iter().map(run_gap).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let gaps: Vec<(usize, Vec<u16>, bool)> = (0..chunks - 1).map(run_gap).collect();
        let mut ok = ok1;
        for (lo, gs, gok) in gaps {
            seeds[lo..lo + gs.len()].copy_from_slice(&gs);
            ok &= gok;
        }
        ok
    };
    if !ok && !allow_bump {
        return None;
    }

    // Occupancy and bumped keys. Keys are sorted by bucket, so the slots of
    // a part of the keys lie in a range that overlaps those of the other
    // parts only at its ends: each part fills a private segment of the
    // occupancy bitmap, and segments are then merged (no atomic operations).
    let words = g.m.div_ceil(64);
    let seeds_ro: &[u16] = &seeds;
    let process = |part: &[Sig]| -> (usize, Vec<u64>, Vec<Sig>) {
        let mut bumped = vec![];
        let (Some(first), Some(last)) = (part.first(), part.last()) else {
            return (0, vec![], bumped);
        };
        let lo = g.slice_begin(first.h) / 64;
        let hi = (g.slice_begin(last.h) + l).div_ceil(64).min(words);
        let mut seg = vec![0u64; hi - lo];
        for &x in part {
            let s = seeds_ro[g.bucket(x.h)] as usize;
            if s == 0 {
                bumped.push(x);
            } else {
                let p = g.pos(x.h, s);
                debug_assert!(p < g.m);
                debug_assert!(seg[p / 64 - lo] & (1 << (p % 64)) == 0, "collision at {p}");
                seg[p / 64 - lo] |= 1 << (p % 64);
            }
        }
        (lo, seg, bumped)
    };
    let part_len = keys.len().div_ceil(threads.max(1)).max(1 << 16);
    #[cfg(feature = "rayon")]
    let parts: Vec<(usize, Vec<u64>, Vec<Sig>)> = if parallel(keys.len()) {
        use rayon::prelude::*;
        keys.par_chunks(part_len).map(process).collect()
    } else {
        keys.chunks(part_len).map(process).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let parts: Vec<(usize, Vec<u64>, Vec<Sig>)> = keys.chunks(part_len).map(process).collect();
    let mut occupied = vec![0u64; words];
    let mut bumped = Vec::with_capacity(parts.iter().map(|p| p.2.len()).sum());
    for (lo, seg, b) in parts {
        for (o, w) in occupied[lo..lo + seg.len()].iter_mut().zip(seg) {
            debug_assert!(*o & w == 0, "collision between parts");
            *o |= w;
        }
        bumped.extend(b);
    }
    if !allow_bump && !bumped.is_empty() {
        return None;
    }
    Some(SweepOut {
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

/// The state of a sweep over a range of buckets.
///
/// Buckets are processed in order of priority inside a window sliding over
/// the range, as in PHast: for each bucket we look for the seed minimizing
/// the sum of the slots of its keys.
///
/// The seeds of a pattern move a key through the slots of its *ring*: the
/// slots of its slice that are congruent to its first slot modulo the
/// stride (the stride is the number of patterns times the amount by which a
/// unit increase of the seed moves a key). The set of used slots is stored
/// by residue classes modulo the stride (*rows*), so a ring is a block of
/// consecutive bits of a row, and the seed of index *j* of a pattern maps a
/// key to the bit of its ring whose index is the index for the first seed of
/// the pattern plus *j*, modulo the length of the ring: the seeds that are
/// feasible for a bucket are thus obtained by rotating the ring of each key
/// and combining the results.
struct Sweep<'a> {
    /// The keys of the buckets in the range.
    keys: &'a [Sig],
    /// The beginning of each bucket of the range in `keys`.
    bucket_begin: Vec<usize>,
    g: &'a Geometry,
    weights: &'a [i64; 7],
    /// First bucket of the range.
    lo: usize,
    /// End of the range.
    hi: usize,
    /// Seeds of the buckets in the range.
    seeds: &'a mut [u16],
    /// Cyclic bit set of used slots, stored by rows: bit *q* of row *ρ* is
    /// associated with slot *q* · 2^`log2_stride` + *ρ* (modulo the size of
    /// the set).
    used: Box<[u64]>,
    /// The base-2 logarithm of the stride (at most that of the slice
    /// length).
    log2_stride: u32,
    /// The base-2 logarithm of the number of bits of a row.
    log2_row: u32,
    /// The number of slots of a ring.
    ring_len: usize,
    /// The number of seeds of a pattern (at least the length of a ring; if
    /// it is larger, seeds are redundant).
    pattern_seeds: usize,
    /// Slots before this value are no longer reachable.
    value_to_clear: usize,
    /// The beginning of the slice and the value providing the offsets of
    /// each key of the bucket being searched.
    bucket: Vec<(usize, u64)>,
    /// For each key of the bucket and the pattern being searched, the first
    /// slot of its ring.
    first: Vec<usize>,
    /// For each key of the bucket and the pattern being searched, the index
    /// in its ring of its slot for the first seed of the pattern.
    index: Vec<u32>,
    /// The indices of the seeds of the pattern being searched for which no
    /// key of the bucket is mapped to a used slot.
    free: Vec<u64>,
    /// The indices at which some key goes back to the first slot of its
    /// ring.
    back: Vec<u32>,
    /// Candidate indices of seeds for the pattern being searched, with the
    /// sum of the slots.
    candidates: Vec<(usize, usize)>,
    /// The slots of the keys of the bucket for a candidate.
    slots: Vec<usize>,
    /// The slots of the keys of the bucket for the seed returned by the
    /// last search.
    best: Vec<usize>,
}

impl<'a> Sweep<'a> {
    fn new(
        keys: &'a [Sig],
        g: &'a Geometry,
        weights: &'a [i64; 7],
        lo: usize,
        hi: usize,
        seeds: &'a mut [u16],
    ) -> Self {
        debug_assert_eq!(<[u16]>::len(seeds), hi - lo);
        let l = g.l_mask as usize + 1;
        // With slices shorter than the stride, a ring is a single slot
        let log2_stride = (g.log2_patterns + g.scale).min(l.ilog2());
        // The cyclic set must contain the slots reachable from the buckets
        // in the priority queue (WINDOW buckets, whose slices begin about
        // num_slices / buckets slots apart) plus a slice; gap sweeps need
        // three slices more. Moreover, rows must contain at least a word.
        let per_bucket = (g.num_slices as usize).div_ceil(g.buckets);
        let cyc_bits = (4 * (l + 1) + 2 * WINDOW * per_bucket)
            .next_power_of_two()
            .max(64 << log2_stride);
        // Keys are sorted by bucket
        let k_lo = keys.partition_point(|x| g.bucket(x.h) < lo);
        let k_hi = keys.partition_point(|x| g.bucket(x.h) < hi);
        let keys = &keys[k_lo..k_hi];
        let mut bucket_begin = vec![0usize; hi - lo + 1];
        for x in keys {
            bucket_begin[g.bucket(x.h) - lo + 1] += 1;
        }
        for i in 0..hi - lo {
            bucket_begin[i + 1] += bucket_begin[i];
        }
        Self {
            keys,
            bucket_begin,
            g,
            weights,
            lo,
            hi,
            seeds,
            used: vec![0; cyc_bits / 64].into_boxed_slice(),
            log2_stride,
            log2_row: cyc_bits.ilog2() - log2_stride,
            ring_len: l >> log2_stride,
            pattern_seeds: (1usize << g.seed_bits) >> g.log2_patterns,
            value_to_clear: 0,
            bucket: Vec::with_capacity(64),
            first: Vec::with_capacity(64),
            index: Vec::with_capacity(64),
            free: Vec::with_capacity(4),
            back: Vec::with_capacity(64),
            candidates: Vec::with_capacity(64),
            slots: Vec::with_capacity(64),
            best: Vec::with_capacity(64),
        }
    }

    #[inline(always)]
    fn size(&self, b: usize) -> usize {
        self.bucket_begin[b - self.lo + 1] - self.bucket_begin[b - self.lo]
    }

    #[inline(always)]
    fn bucket_keys(&self, b: usize) -> &'a [Sig] {
        &self.keys[self.bucket_begin[b - self.lo]..self.bucket_begin[b - self.lo + 1]]
    }

    /// Returns the beginning of the slice of the first key of the first
    /// nonempty bucket of the range.
    fn first_slice(&self) -> usize {
        let mut b = self.lo;
        while b < self.hi && self.size(b) == 0 {
            b += 1;
        }
        if b == self.hi {
            usize::MAX
        } else {
            self.g
                .slice_begin(self.keys[self.bucket_begin[b - self.lo]].h)
        }
    }

    #[inline(always)]
    fn priority(&self, b: usize, size: usize) -> i64 {
        let w = if size <= 7 {
            self.weights[size - 1]
        } else {
            let l = self.weights[6];
            let p = self.weights[5];
            l + (l - p) * (size - 7) as i64
        };
        w - 1024 * b as i64
    }

    /// Returns the index of the bit associated with slot `p` in the set of
    /// used slots.
    #[inline(always)]
    fn bit(&self, p: usize) -> usize {
        let t = self.log2_stride;
        ((p & ((1 << t) - 1)) << self.log2_row) | ((p >> t) & ((1 << self.log2_row) - 1))
    }

    /// Marks slot `p` as used.
    #[inline(always)]
    fn set(&mut self, p: usize) {
        let q = self.bit(p);
        self.used[q / 64] |= 1 << (q % 64);
    }

    /// Marks slot `p` as free.
    #[inline(always)]
    fn clear(&mut self, p: usize) {
        let q = self.bit(p);
        self.used[q / 64] &= !(1 << (q % 64));
    }

    /// Returns the bits of the 64 slots of the row of slot `p` starting
    /// from `p` (that is, bit *i* is associated with the slot `p` plus *i*
    /// strides).
    #[inline(always)]
    fn row64(&self, p: usize) -> u64 {
        let q = self.bit(p);
        // Rows are sequences of whole words
        let row_mask = (1usize << (self.log2_row - 6)) - 1;
        let (row, w) = ((q / 64) & !row_mask, (q / 64) & row_mask);
        // Branchless: a 128-bit shift
        let lo = self.used[row | w] as u128;
        let hi = self.used[row | ((w + 1) & row_mask)] as u128;
        ((hi << 64 | lo) >> (q % 64)) as u64
    }

    /// Returns 64 bits of the ring whose first slot is `first` starting
    /// from index `x` (cyclically); if the ring is shorter than a word,
    /// only the lowest bits are valid.
    #[inline(always)]
    fn ring64(&self, first: usize, x: usize) -> u64 {
        let n = self.ring_len;
        if n >= 64 {
            let w = self.row64(first + (x << self.log2_stride));
            let avail = n - x;
            if avail >= 64 {
                w
            } else {
                (w & ((1 << avail) - 1)) | (self.row64(first) << avail)
            }
        } else {
            let ring = self.row64(first) & ((1 << n) - 1);
            (ring >> x) | (ring << (n - x))
        }
    }

    /// Finds the seed of bucket `b` minimizing the sum of the slots of its
    /// keys, and leaves the slots in `self.best`; returns zero if no seed
    /// is feasible.
    fn search(&mut self, b: usize) -> usize {
        let keys = self.bucket_keys(b);
        let k = keys.len();
        let g = self.g;
        let t = self.log2_stride;
        let stride_mask = (1usize << t) - 1;
        let n = self.ring_len;
        let words = n.div_ceil(64);
        let pattern_bits = 64 >> g.log2_patterns;
        // The indices of the seeds of a pattern we consider are those of the
        // first turn around the ring
        let valid = if n < 64 { (1u64 << n) - 1 } else { !0 };

        self.bucket.clear();
        self.bucket
            .extend(keys.iter().map(|x| (g.slice_begin(x.h), g.offsets(x.h))));
        self.first.clear();
        self.first.resize(k, 0);
        self.index.clear();
        self.index.resize(k, 0);
        self.free.clear();
        self.free.resize(words, 0);

        let mut best_sum = usize::MAX;
        let mut best_seed = 0;
        for r in 0..1usize << g.log2_patterns {
            let shift = r as u32 * pattern_bits;
            let first_seed = (r as u64) << g.scale;
            // The sum of the slots for the first seed of the pattern
            let mut sum = 0;
            self.free.fill(0);
            for i in 0..k {
                let (slice_begin, offsets) = self.bucket[i];
                let offset = ((offsets >> shift).wrapping_add(first_seed) & g.l_mask) as usize;
                let first = slice_begin + (offset & stride_mask);
                let index = offset >> t;
                self.first[i] = first;
                self.index[i] = index as u32;
                sum += first + (index << t);
                if n == 64 {
                    // The seeds of the pattern are the rotations of a word
                    let used = self.row64(first).rotate_right(index as u32);
                    self.free[0] |= used;
                } else {
                    for w in 0..words {
                        let used = self.ring64(first, (index + 64 * w) & (n - 1));
                        self.free[w] |= used;
                    }
                }
            }
            let mut count = 0;
            for w in &mut self.free {
                *w = !*w & valid;
                count += w.count_ones();
            }
            // The first seed of the first pattern is zero, which marks
            // bumped buckets; if seeds are redundant we can use instead the
            // first seed of the second turn around the ring
            if r == 0 && n == self.pattern_seeds && self.free[0] & 1 != 0 {
                self.free[0] &= !1;
                count -= 1;
            }
            if count == 0 {
                continue;
            }

            // The sum of the slots for the seed of index j is that for the
            // first seed plus kj strides, minus the length of a ring in
            // strides for each key that went back to the first slot of its
            // ring. Thus, between two indices at which some key goes back
            // only the first free index can be the best one.
            self.candidates.clear();
            if count <= 4 {
                // Just a few free indices (the common case): we consider
                // all of them
                for w in 0..words {
                    let mut free = self.free[w];
                    while free != 0 {
                        let j = w * 64 + free.trailing_zeros() as usize;
                        free &= free - 1;
                        let back = self.index.iter().filter(|&&x| x as usize + j >= n).count();
                        self.candidates
                            .push((sum + ((k * j) << t) - ((n * back) << t), j));
                    }
                }
            } else {
                self.back.clear();
                self.back.extend(
                    self.index
                        .iter()
                        .filter(|&&x| x != 0)
                        .map(|&x| n as u32 - x),
                );
                self.back.sort_unstable();
                let (mut begin, mut back, mut i) = (0, 0, 0);
                loop {
                    while i < k && self.back.get(i) == Some(&(begin as u32)) {
                        back += 1;
                        i += 1;
                    }
                    let end = self.back.get(i).map_or(n, |&x| x as usize);
                    if let Some(j) = next_set(&self.free, begin, end) {
                        self.candidates
                            .push((sum + ((k * j) << t) - ((n * back) << t), j));
                    }
                    if end == n {
                        break;
                    }
                    begin = end;
                }
            }

            for c in 0..self.candidates.len() {
                let (sum, j) = self.candidates[c];
                if sum >= best_sum {
                    continue;
                }
                self.slots.clear();
                self.slots.extend(
                    self.first
                        .iter()
                        .zip(&self.index)
                        .map(|(&first, &index)| first + (((index as usize + j) & (n - 1)) << t)),
                );
                // Two keys of the bucket might be mapped to the same slot
                if distinct(&mut self.slots) {
                    best_sum = sum;
                    best_seed = if r == 0 && j == 0 {
                        n << g.log2_patterns
                    } else {
                        (j << g.log2_patterns) | r
                    };
                    std::mem::swap(&mut self.best, &mut self.slots);
                }
            }
        }
        best_seed
    }

    /// Places bucket `b`. Returns `true` on success.
    fn place(&mut self, b: usize) -> bool {
        let seed = self.search(b);
        if seed == 0 {
            return false;
        }
        for i in 0..self.best.len() {
            self.set(self.best[i]);
        }
        self.seeds[b - self.lo] = seed as u16;
        true
    }

    /// Processes the buckets of the range; returns `false` if some bucket
    /// could not be placed (in which case, if `allow_bump` is false, the
    /// sweep stops immediately).
    fn run(mut self, allow_bump: bool) -> bool {
        let (lo, hi) = (self.lo, self.hi);
        let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::with_capacity(WINDOW);
        let mut in_heap = [0u64; HEAP_BITS / 64];
        let in_heap_get = |s: &[u64], b: usize| s[(b % HEAP_BITS) / 64] >> (b % 64) & 1 != 0;
        let mut span_begin = lo;
        while span_begin < hi && self.size(span_begin) == 0 {
            span_begin += 1;
        }
        if span_begin == hi {
            return true;
        }
        let slice_begin_of =
            |s: &Self, b: usize| s.g.slice_begin(s.keys[s.bucket_begin[b - s.lo]].h);
        let span_end = |sb: usize| (sb + WINDOW).min(hi);
        let mut ok = true;
        self.value_to_clear = slice_begin_of(&self, span_begin);
        for b in span_begin..span_end(span_begin) {
            let sz = self.size(b);
            if sz != 0 {
                heap.push((self.priority(b, sz), Reverse(b)));
                in_heap[(b % HEAP_BITS) / 64] |= 1 << (b % 64);
            }
        }
        while let Some((_, Reverse(b))) = heap.pop() {
            in_heap[(b % HEAP_BITS) / 64] &= !(1 << (b % 64));
            if !self.place(b) {
                ok = false;
                if !allow_bump {
                    return false;
                }
            }
            if b == span_begin {
                let old_end = span_end(span_begin);
                span_begin += 1;
                while span_begin < old_end && !in_heap_get(&in_heap, span_begin) {
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
                // Slots before the first slice in the window are no longer
                // reachable, and their bits will be used for other slots
                let end = slice_begin_of(&self, span_begin);
                for p in self.value_to_clear..end {
                    self.clear(p);
                }
                self.value_to_clear = self.value_to_clear.max(end);
                for b2 in old_end..span_end(span_begin) {
                    let sz = self.size(b2);
                    if sz != 0 {
                        heap.push((self.priority(b2, sz), Reverse(b2)));
                        in_heap[(b2 % HEAP_BITS) / 64] |= 1 << (b2 % 64);
                    }
                }
            }
        }
        ok
    }
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

/// Returns whether the given values are distinct (possibly permuting them).
#[inline(always)]
fn distinct(v: &mut [usize]) -> bool {
    if v.len() <= 12 {
        for i in 1..v.len() {
            let x = v[i];
            for &y in &v[..i] {
                if x == y {
                    return false;
                }
            }
        }
        true
    } else {
        v.sort_unstable();
        v.windows(2).all(|w| w[0] != w[1])
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
