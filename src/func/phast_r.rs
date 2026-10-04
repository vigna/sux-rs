/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! PHast-R: minimal perfect hashing with additive placement over multiple
//! patterns and cuckoo repair.
//!
//! This structure is a variant of PHast+ ([Beling and Sanders, *PHast —
//! Perfect Hashing made fast*]). As in PHast+, keys are hashed to buckets
//! using a linear function of their hash, each bucket stores a fixed-width
//! seed, and the seed selects an additive shift for all keys of the bucket
//! inside small overlapping slices of the output range, so that feasible seeds
//! can be found with bit-parallel operations. Buckets for which no seed is
//! found are *bumped* to the next level, and an [Elias–Fano] sequence maps the
//! outputs of the following levels to the free slots of the first one.
//!
//! PHast-R differs from PHast+ in two respects:
//!
//! - the seed selects not only a shift, but also one of *R* independent
//!   offset *patterns*, i.e., one of *R* independent sets of in-slice
//!   positions for the keys of a bucket; this decorrelates the trials and
//!   eliminates the *self-collisions* (two keys of the same bucket with the
//!   same in-slice position) that additive placement can never resolve;
//!
//! - when no seed is feasible, construction tries to *repair* the situation:
//!   using a bit-parallel half adder it finds the shifts at which exactly one
//!   key of the bucket is blocked, evicts the bucket owning the blocking slot,
//!   places the current bucket, and tries to place the evicted bucket again
//!   (possibly recursively), as in cuckoo hashing.
//!
//! Queries perform the same operations as in PHast+ (a hash, a seed access,
//! and a couple of multiplications and shifts), plus a shift and a mask; a
//! small fraction of the keys accesses further levels and the Elias–Fano
//! sequence. With byte seeds (the default) queries are as fast as in PHast+.
//!
//! As in PHast+, keys are hashed again with a different seed at each level
//! after the first, so 64-bit signatures suffice for any number of keys: keys
//! with the same signature are bumped from the first level and separated at
//! the following ones. Duplicate keys are detected and reported as errors.
//!
//! With the default parameters (8-bit seeds, depth-1 repair) space is about
//! 1.93 bits per key; depth-2 repair (see [`PHastRBuilder::repair_depth`])
//! reaches the space of PHast (about 1.92 bits per key) at about 1.6 times
//! the construction time, and 10-bit seeds stored in a [`BitFieldVec`] (see
//! [`PHastRBuilder::seed_bits`]) reach about 1.86 bits per key; in this
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

use crate::bits::BitVec;
use crate::bits::{BitFieldVec, BitFieldVecU};
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

/// Derives from the hash *h* of a key for the first level the value *o*
/// providing the in-slice offsets of the patterns (at the following levels,
/// the offsets are provided by *o* ⊕ *h*, where *h* is the hash of the level).
///
/// Offsets are derived with a single multiplication by an odd constant: bit
/// *j* of the product depends on the bits of the hash up to *j*, so the
/// offsets of all patterns depend on the lower bits of the hash, which vary
/// independently among the keys of a bucket (whose upper bits coincide).
#[inline(always)]
const fn offsets(h: u64) -> u64 {
    h.wrapping_mul(0x5BD1_E995)
}

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
            log2_patterns: self.log2_patterns,
            log2_slice_len: self.log2_slice_len,
            wrap: self.wrap,
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
            log2_patterns: f.log2_patterns,
            log2_slice_len: f.log2_slice_len,
            wrap: f.wrap,
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
    /// The number of buckets.
    buckets: u64,
    /// The number of slices (the output range minus the slice length and the
    /// maximum shift).
    num_slices: u64,
    /// The slice length minus one.
    l_mask: u64,
    /// The offset of the outputs of this level in the remapping sequence.
    offset: u64,
    /// The seed used to hash keys for this level (unused for the first
    /// level, which uses the seed of the function).
    salt: u64,
    /// The index of the first seed of this level in the seed storage.
    first_seed: u64,
}

/// A minimal perfect hash function based on PHast+ with multiple patterns and
/// cuckoo repair.
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
    /// The base-2 logarithm of the number of patterns.
    log2_patterns: u32,
    /// The base-2 logarithm of the (maximum) slice length.
    log2_slice_len: u32,
    /// The multiplier of shifts with wrapping (0 if patterns are used).
    wrap: u32,
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
    /// the level, the value *o* providing its in-slice offsets, and the seed of
    /// its bucket.
    #[inline(always)]
    fn pos(&self, lv: &LevelParams, h: u64, o: u64, seed: usize) -> usize {
        // The pattern is the lowest log2_patterns bits of the seed, and its
        // offset starts at bit pattern * 64 / R of o: since shifts are
        // taken modulo 64, shifting the whole seed left by log2(64 / R)
        // yields the shift count without extracting the pattern
        let count = (seed as u32) << (6 - self.log2_patterns);
        mul_hi(h, lv.num_slices) as usize
            + (o.wrapping_shr(count) & lv.l_mask) as usize
            + (seed >> self.log2_patterns)
    }

    /// Returns the output of a key in the given level with wrapping, given
    /// its hash *h* for the level and the seed of its bucket.
    #[inline(always)]
    fn pos_wrap(&self, lv: &LevelParams, h: u64, seed: usize) -> usize {
        mul_hi(h, lv.num_slices) as usize
            + (h.wrapping_add((seed * self.wrap as usize) as u64) & lv.l_mask) as usize
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
        if self.wrap != 0 {
            // SAFETY: mul_hi(h, buckets) < buckets, which is the number of seeds
            let s = unsafe { self.seeds0.get_seed(mul_hi(h, lv.buckets) as usize) };
            if s != 0 {
                return self.pos_wrap(lv, h, s);
            }
            return self.get_slow(key, 0);
        }
        let o = offsets(h);
        // SAFETY: mul_hi(h, buckets) < buckets, which is the number of seeds
        let s = unsafe { self.seeds0.get_seed(mul_hi(h, lv.buckets) as usize) };
        if s != 0 {
            return self.pos(lv, h, o, s);
        }
        self.get_slow(key, o)
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
            // SAFETY: mul_hi(h, buckets) < buckets, which is the number of seeds
            let s = unsafe { self.seeds0.get_seed(mul_hi(h, lv.buckets) as usize) };
            if self.wrap != 0 {
                if s != 0 {
                    return self.pos_wrap(lv, h, s);
                }
                return self.get_slow(keys[i], 0);
            }
            let o = offsets(h);
            if s != 0 {
                return self.pos(lv, h, o, s);
            }
            self.get_slow(keys[i], o)
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
    /// hashed again with the seed of the level to find its bucket and its
    /// slice, and the in-slice offsets are provided by `o ^ h`, where `o` is
    /// the value of the first level: the offsets of a key thus change at each
    /// level, at the cost of an exclusive or.
    #[cold]
    #[inline(never)]
    fn get_slow(&self, key: &K, o: u64) -> usize {
        for lv in self.params.as_ref() {
            let h = K::to_sig(key, lv.salt)[0];
            // SAFETY: the seeds of the level are stored consecutively
            let s = unsafe {
                self.seeds
                    .get_seed((lv.first_seed + mul_hi(h, lv.buckets)) as usize)
            };
            if s != 0 {
                // SAFETY: by construction, the remapping sequence contains
                // one entry for each output of each level after the first
                let p = if self.wrap != 0 {
                    self.pos_wrap(lv, h, s)
                } else {
                    self.pos(lv, h, o ^ h, s)
                };
                return unsafe { self.remap.get_value_unchecked(lv.offset as usize + p) };
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
/// The defaults use 8-bit seeds, four patterns of 64 shifts, slices of length
/// 1024, an expected bucket size of 4.75 keys, and repairs of depth one trying at
/// most 16 candidates.
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
    repair_candidates: usize,
    repair_depth: u32,
    repair_by_size: bool,
    seed: u64,
    weights: Option<[i64; 7]>,
    wrap: u32,
}

impl Default for PHastRBuilder {
    fn default() -> Self {
        Self {
            seed_bits: 8,
            log2_patterns: 2,
            log2_slice_len: 10,
            bucket_size: 4.75,
            repair_candidates: 16,
            repair_depth: 1,
            repair_by_size: false,
            seed: 0,
            weights: None,
            wrap: 0,
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
    pub fn log2_patterns(mut self, log2_patterns: u32) -> Self {
        self.log2_patterns = log2_patterns;
        self
    }

    /// Sets the base-2 logarithm of the slice length (default: 10).
    pub fn log2_slice_len(mut self, log2_slice_len: u32) -> Self {
        self.log2_slice_len = log2_slice_len;
        self
    }

    /// Sets the expected number of keys per bucket (default: 4.75).
    pub fn bucket_size(mut self, bucket_size: f64) -> Self {
        self.bucket_size = bucket_size;
        self
    }

    /// Sets the maximum number of eviction candidates tried for a failing
    /// bucket (default: 16; 0 disables repair).
    pub fn repair_candidates(mut self, repair_candidates: usize) -> Self {
        self.repair_candidates = repair_candidates;
        self
    }

    /// Sets the maximum depth of eviction chains (default: 1; 0 disables
    /// repair).
    pub fn repair_depth(mut self, repair_depth: u32) -> Self {
        self.repair_depth = repair_depth;
        self
    }

    /// Sets whether eviction candidates are ordered by the size of the
    /// evicted bucket (smaller first) rather than by the sum of the positions
    /// of the current bucket.
    pub fn repair_by_size(mut self, repair_by_size: bool) -> Self {
        self.repair_by_size = repair_by_size;
        self
    }

    /// Uses shifts with wrapping inside the slice, as in PHast+ with
    /// wrapping, instead of patterns: seed *s* maps a key with hash *h* to
    /// its slice plus (*h* + *s* · `multiplier`) mod *L* (default: 0, that
    /// is, patterns are used; when wrapping is used, the number of patterns
    /// is ignored).
    pub fn wrap(mut self, multiplier: u32) -> Self {
        self.wrap = multiplier;
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

    /// Returns the number of shifts per pattern.
    fn shifts(&self) -> usize {
        (1usize << self.seed_bits) >> self.log2_patterns
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
        if self.wrap == 0 {
            if self.log2_patterns > 6 || (1u32 << self.log2_patterns) * self.log2_slice_len > 64 {
                bail!("Too many patterns for the given slice length");
            }
            if self.shifts() == 0 {
                bail!("Too many patterns for the given number of seed bits");
            }
        } else if self.wrap > 64 {
            bail!("The multiplier of shifts with wrapping must be at most 64");
        }
        if self.log2_slice_len > 13 {
            bail!("The slice length must be at most 8192");
        }

        pl.info(format_args!("Computing signatures..."));
        let seed = self.seed;
        let hash = |(i, k): (usize, &B)| Ho {
            h: K::to_sig(k.borrow(), seed)[0],
            idx: i as u64,
        };
        #[cfg(feature = "rayon")]
        let hos: Vec<Ho> = if parallel(keys.len()) {
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
        let hos: Vec<Ho> = keys.iter().enumerate().map(hash).collect();

        let (levels, remap) = self.build_levels::<K, B>(keys, hos, pl)?;
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
            log2_patterns: if self.wrap != 0 {
                0
            } else {
                self.log2_patterns
            },
            log2_slice_len: self.log2_slice_len,
            wrap: self.wrap,
            params0,
            seeds0: D::from_seeds(&seeds0, self.seed_bits),
            params: params.into_boxed_slice(),
            seeds: D::from_seeds(&seeds, self.seed_bits),
            remap,
            _marker: std::marker::PhantomData,
        })
    }

    /// Builds the levels from the hashes of the first level (each with the
    /// index of its key, so that keys bumped from a level can be hashed again
    /// for the next one).
    #[allow(clippy::type_complexity)]
    fn build_levels<K: ?Sized + ToSig<[u64; 1]> + Sync, B: Borrow<K> + Sync>(
        &self,
        keys: &[B],
        mut cur: Vec<Ho>,
        pl: &mut impl ProgressLog,
    ) -> Result<(Vec<(LevelParams, Vec<u16>)>, Remap)> {
        // Hashes again the keys of the given records with the given seed
        let hash = |idx: u64, seed: u64| K::to_sig(keys[idx as usize].borrow(), seed)[0];
        let rehash = |v: &mut [Hoi], seed: u64| {
            let f = |x: &mut Hoi| x.h = hash(x.idx, seed);
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
            self.weights.unwrap_or_else(|| {
                if self.wrap != 0 {
                    wrap_weights(self.seed_bits, self.wrap, g.l_mask as usize + 1)
                } else {
                    default_weights(self.seed_bits, g.l_mask as usize + 1)
                }
            })
        };

        if n == 0 {
            // A single empty level, so that queries need no special case
            let geom = self.geometry(0, 1, self.bucket_size);
            levels.push((geom.level(), vec![0]));
            let efb = EliasFanoBuilder::new(0, 1);
            return Ok((levels, remap(efb)));
        }

        // Keys with the same hash are adjacent after sorting. They are bumped
        // (their bucket self-collides for every seed), and separated by the
        // hashes of the following levels, unless they are equal: we detect
        // duplicate keys hashing again keys with the same hash with a
        // different seed.
        sort_ho(&mut cur);
        let equal = |w: &[Ho]| (w[0].h == w[1].h).then_some(w[0].h);
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
        let out = sweep_level(&cur, &geom, self, &weights(&geom), true)
            .expect("bumping sweeps cannot fail");
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
        // level to find their bucket and slice; their offsets are given by
        // the value of the first level xored with the hash of the level
        let mut cur: Vec<Hoi> = out
            .bumped
            .iter()
            .map(|x| Hoi {
                h: 0,
                o: x.o(),
                idx: x.idx,
            })
            .collect();
        while !cur.is_empty() {
            let k = cur.len();
            let (level, seeds, occupied, bumped, m) = if k > LAST_LEVEL_THRESHOLD {
                let seed = level_seed(self.seed, levels.len(), 0);
                rehash(&mut cur, seed);
                sort_ho(&mut cur);
                let geom = self.geometry(k, k, self.bucket_size);
                let out = sweep_level(&cur, &geom, self, &weights(&geom), true)
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
                    let mut hos = cur.clone();
                    rehash(&mut hos, seed);
                    sort_ho(&mut hos);
                    if let Some(out) = sweep_level(&hos, &geom, self, &weights(&geom), false) {
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
        // Small levels need shorter slices, as the first and last L + D
        // positions are reached by fewer slices (thresholds on the output
        // range tuned for 8-bit seeds; the slice length set by the user is a
        // maximum)
        let max_l = 1usize << self.log2_slice_len;
        if self.wrap != 0 {
            // As in PHast+ with wrapping: shifts wrap inside the slice, so
            // there is no extra range for shifts
            let l = if m < 4096 {
                (m / 2 + 1).next_power_of_two().min(max_l)
            } else {
                max_l
            };
            let l = l.min(m.max(1));
            return Geometry {
                buckets: (k as f64 / bucket_size).round().max(1.0) as usize,
                num_slices: (m.max(1) + 1 - l) as u64,
                l_mask: l as u64 - 1,
                shifts: 1,
                patterns: 1,
                log2_patterns: 0,
                wrap: self.wrap as usize,
                max_seed: (1 << self.seed_bits) - 1,
                m,
            };
        }
        let target = match m {
            0..2500 => 64,
            2500..4000 => 128,
            4000..70_000 => 256,
            70_000..600_000 => 512,
            _ => usize::MAX,
        };
        let mut l = max_l.min(target);
        while l > 1 && l > m / 2 {
            l /= 2;
        }
        let mut d = self.shifts();
        if l + d - 1 > m {
            d = (m + 1 - l).max(1);
        }
        Geometry {
            buckets: (k as f64 / bucket_size).round().max(1.0) as usize,
            num_slices: (m + 1 - l - (d - 1)) as u64,
            l_mask: l as u64 - 1,
            shifts: d,
            patterns: 1 << self.log2_patterns,
            log2_patterns: self.log2_patterns,
            wrap: 0,
            max_seed: (1 << self.seed_bits) - 1,
            m,
        }
    }
}

/// Number of keys below which a level is built without bumping.
const LAST_LEVEL_THRESHOLD: usize = 4096;

/// A key during construction: the hash *h* determining its bucket and its
/// slice, and the value *o* providing its in-slice offsets.
trait Rec:
    Copy + Default + Send + Sync + PartialOrd + voracious_radix_sort::Radixable<u64, Key = u64>
{
    fn h(&self) -> u64;
    fn o(&self) -> u64;
}

/// A key of the first level: its hash *h* and the index of the key (used to
/// hash it again for the following levels); *o* is derived from *h*.
#[derive(Debug, Clone, Copy, Default)]
struct Ho {
    h: u64,
    idx: u64,
}

impl Rec for Ho {
    #[inline(always)]
    fn h(&self) -> u64 {
        self.h
    }
    #[inline(always)]
    fn o(&self) -> u64 {
        offsets(self.h)
    }
}

/// Ordering by *h* (and index, to break ties), needed by
/// [`voracious_radix_sort`].
impl PartialOrd for Ho {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((self.h, self.idx).cmp(&(other.h, other.idx)))
    }
}

impl PartialEq for Ho {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.h == other.h && self.idx == other.idx
    }
}

impl voracious_radix_sort::Radixable<u64> for Ho {
    type Key = u64;
    #[inline(always)]
    fn key(&self) -> u64 {
        self.h
    }
}

/// A key of a level after the first: its hash *h* for the level, the value
/// *o* of the first level, and the index of the key (offsets are provided by
/// *o* ⊕ *h*).
#[derive(Debug, Clone, Copy, Default)]
struct Hoi {
    h: u64,
    o: u64,
    idx: u64,
}

impl Rec for Hoi {
    #[inline(always)]
    fn h(&self) -> u64 {
        self.h
    }
    #[inline(always)]
    fn o(&self) -> u64 {
        self.o ^ self.h
    }
}

impl PartialOrd for Hoi {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((self.h, self.idx).cmp(&(other.h, other.idx)))
    }
}

impl PartialEq for Hoi {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.h == other.h && self.idx == other.idx
    }
}

impl voracious_radix_sort::Radixable<u64> for Hoi {
    type Key = u64;
    #[inline(always)]
    fn key(&self) -> u64 {
        self.h
    }
}

/// Sorts by *h* using [`voracious_radix_sort`] (multithreaded with the
/// `rayon` feature).
fn sort_ho<T: Rec>(v: &mut [T]) {
    use voracious_radix_sort::RadixSort;
    #[cfg(feature = "rayon")]
    if parallel(v.len()) {
        v.voracious_mt_sort(rayon::current_num_threads());
        return;
    }
    v.voracious_sort();
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    buckets: usize,
    num_slices: u64,
    l_mask: u64,
    shifts: usize,
    patterns: usize,
    log2_patterns: u32,
    /// The multiplier of shifts with wrapping (0 if patterns are used).
    wrap: usize,
    /// The maximum seed.
    max_seed: usize,
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

    #[inline(always)]
    fn base<T: Rec>(&self, x: T, r: usize) -> usize {
        self.slice_begin(x.h()) + ((x.o() >> (r as u32 * self.stride())) & self.l_mask) as usize
    }

    /// The distance in bits between the offsets of consecutive patterns.
    #[inline(always)]
    fn stride(&self) -> u32 {
        64 >> self.log2_patterns
    }

    #[inline(always)]
    fn pos<T: Rec>(&self, x: T, seed: usize) -> usize {
        if self.wrap != 0 {
            return self.slice_begin(x.h())
                + (x.h().wrapping_add((seed * self.wrap) as u64) & self.l_mask) as usize;
        }
        let r = seed & (self.patterns - 1);
        self.base(x, r) + (seed >> self.log2_patterns)
    }

    /// Returns the seed of pattern `r` and shift `d`; the pair (0, 0) is
    /// not valid, as seed 0 marks bumped buckets.
    #[inline(always)]
    fn seed_of(&self, r: usize, d: usize) -> usize {
        debug_assert!(r != 0 || d != 0);
        (d << self.log2_patterns) | r
    }

    fn level(&self) -> LevelParams {
        LevelParams {
            buckets: self.buckets as u64,
            num_slices: self.num_slices,
            l_mask: self.l_mask,
            offset: 0,
            salt: 0,
            first_seed: 0,
        }
    }
}

/// Size of the cyclic set recording buckets in the priority queue.
const HEAP_BITS: usize = 1024;
/// Number of buckets in the window.
const WINDOW: usize = 256;

struct SweepOut<T> {
    seeds: Vec<u16>,
    /// Occupancy bitmap of the output range.
    occupied: Vec<u64>,
    bumped: Vec<T>,
}

/// Assigns seeds to the buckets of a level, possibly in parallel.
///
/// Returns `None` if `allow_bump` is false and some bucket could not be
/// placed.
fn sweep_level<T: Rec>(
    keys: &[T],
    g: &Geometry,
    b: &PHastRBuilder,
    weights: &[i64; 7],
    allow_bump: bool,
) -> Option<SweepOut<T>> {
    let nb = g.buckets;

    // Number of buckets between chunks whose positions cannot overlap (the
    // same formula as in PHast): the slice start grows by num_slices / nb per
    // bucket, and the positions of a key lie at most L + D - 2 slots after
    // its slice start (D is the number of shifts), so considering the
    // rounding of slice starts it suffices that gap * num_slices / nb >= L +
    // D - 1.
    let span = g.l_mask as usize + 1 + g.shifts;
    let gap = (span - 1) * nb / g.num_slices.max(1) as usize + 1;
    #[cfg(feature = "rayon")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    let threads = 1;
    let chunks = threads.min(nb / (64 * gap).max(4096)).max(1);

    let mut seeds = vec![0u16; nb];
    let ok = if chunks == 1 {
        Sweep::new(keys, g, b, weights, 0, nb, &mut seeds).run(allow_bump)
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
            Sweep::new(keys, g, b, weights, lo, hi, &mut part[..hi - lo]).run(allow_bump)
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
        // positions they occupy are marked as non-evictable.
        let seeds_ro: &[u16] = &seeds;
        let run_gap = |i: usize| -> (usize, Vec<u16>, bool) {
            let lo = bounds[i + 1] - gap;
            let hi = bounds[i + 1];
            let mut gap_seeds = vec![0u16; hi - lo];
            let mut sw = Sweep::new(keys, g, b, weights, lo, hi, &mut gap_seeds);
            let before = lo.saturating_sub(gap).max(bounds[i]);
            let after = (hi + gap).min(bounds[i + 2]);
            let first = sw.first_slice();
            for (from, to) in [(before, lo), (hi, after)] {
                let k_lo = keys.partition_point(|x| g.bucket(x.h()) < from);
                let k_hi = keys.partition_point(|x| g.bucket(x.h()) < to);
                for &x in &keys[k_lo..k_hi] {
                    let s = seeds_ro[g.bucket(x.h())] as usize;
                    if s != 0 {
                        let p = g.pos(x, s);
                        if p >= first {
                            sw.premark(p);
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

    // Occupancy and bumped keys. Keys are sorted by bucket, so the positions
    // of a part of the keys lie in a range that overlaps those of the other
    // parts only at its ends: each part fills a private segment of the
    // occupancy bitmap, and segments are then merged (no atomic operations).
    let words = g.m.div_ceil(64);
    let span = g.l_mask as usize + 1 + g.shifts;
    let seeds_ro: &[u16] = &seeds;
    let process = |part: &[T]| -> (usize, Vec<u64>, Vec<T>) {
        let mut bumped = vec![];
        let (Some(first), Some(last)) = (part.first(), part.last()) else {
            return (0, vec![], bumped);
        };
        let lo = g.slice_begin(first.h()) / 64;
        let hi = (g.slice_begin(last.h()) + span).div_ceil(64).min(words);
        let mut seg = vec![0u64; hi - lo];
        for &x in part {
            let s = seeds_ro[g.bucket(x.h())] as usize;
            if s == 0 {
                bumped.push(x);
            } else {
                let p = g.pos(x, s);
                debug_assert!(p < g.m);
                debug_assert!(seg[p / 64 - lo] & (1 << (p % 64)) == 0, "collision at {p}");
                seg[p / 64 - lo] |= 1 << (p % 64);
            }
        }
        (lo, seg, bumped)
    };
    let part_len = keys.len().div_ceil(threads.max(1)).max(1 << 16);
    #[cfg(feature = "rayon")]
    let parts: Vec<(usize, Vec<u64>, Vec<T>)> = if parallel(keys.len()) {
        use rayon::prelude::*;
        keys.par_chunks(part_len).map(process).collect()
    } else {
        keys.chunks(part_len).map(process).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let parts: Vec<(usize, Vec<u64>, Vec<T>)> = keys.chunks(part_len).map(process).collect();
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
struct Sweep<'a, T: Rec> {
    /// The keys of the buckets in the range.
    keys: &'a [T],
    /// The beginning of each bucket of the range in `keys`.
    bucket_begin: Vec<usize>,
    g: &'a Geometry,
    weights: &'a [i64; 7],
    repair_candidates: usize,
    repair_depth: u32,
    repair_by_size: bool,
    /// First bucket of the range.
    lo: usize,
    /// End of the range.
    hi: usize,
    /// Seeds of the buckets in the range.
    seeds: &'a mut [u16],
    /// Cyclic bit set of used slots.
    used: Box<[u64]>,
    /// Size of the cyclic window of slots minus one (the size is a power of
    /// two).
    cyc_mask: usize,
    /// Number of words of the cyclic window minus one.
    cyc_wmask: usize,
    /// Owner (low 32 bits of the bucket index) of each used slot.
    owner: Box<[u32]>,
    /// Slots before this value are no longer reachable.
    value_to_clear: usize,
    bases: Vec<usize>,
    sb: Vec<usize>,
    oo: Vec<u64>,
    /// Repair candidates: sum of positions, pattern, and shift, packed so
    /// that their natural order is lexicographic.
    cands: Vec<u128>,
    evict: Vec<(usize, usize, usize, usize)>,
    /// Bases of all patterns of a bucket being repaired.
    all_bases: Vec<usize>,
    /// Positions of an evicted bucket.
    ypos: Vec<usize>,
    /// Bases of the best pattern found by the last search.
    best_bases: Vec<usize>,
    /// Shift of the best pattern found by the last search.
    best_d: usize,
    /// With wrapping, the total shift at the start of each segment of the
    /// last candidate collection (segments play the role of patterns).
    seg_t0: Vec<usize>,
}

impl<'a, T: Rec> Sweep<'a, T> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        keys: &'a [T],
        g: &'a Geometry,
        b: &PHastRBuilder,
        weights: &'a [i64; 7],
        lo: usize,
        hi: usize,
        seeds: &'a mut [u16],
    ) -> Self {
        debug_assert_eq!(<[u16]>::len(seeds), hi - lo);
        // Keys are sorted by bucket
        // The cyclic window must contain the positions reachable from the
        // buckets in the priority queue (WINDOW buckets, whose slices start
        // about num_slices / buckets slots apart) plus a slice and the shifts;
        // gap sweeps need (L + D) times three more. A small window keeps the
        // owner array in the L2 cache, which matters for parallel sweeps.
        let per_bucket = (g.num_slices as usize).div_ceil(g.buckets.max(1));
        let span = g.l_mask as usize + 1 + g.shifts;
        let cyc_bits = (4 * span + 2 * WINDOW * per_bucket)
            .next_power_of_two()
            .max(128);
        let k_lo = keys.partition_point(|x| g.bucket(x.h()) < lo);
        let k_hi = keys.partition_point(|x| g.bucket(x.h()) < hi);
        let keys = &keys[k_lo..k_hi];
        let mut bucket_begin = vec![0usize; hi - lo + 1];
        for x in keys {
            bucket_begin[g.bucket(x.h()) - lo + 1] += 1;
        }
        for i in 0..hi - lo {
            bucket_begin[i + 1] += bucket_begin[i];
        }
        Self {
            keys,
            bucket_begin,
            g,
            weights,
            repair_candidates: b.repair_candidates,
            repair_depth: b.repair_depth,
            repair_by_size: b.repair_by_size,
            lo,
            hi,
            seeds,
            used: vec![0; cyc_bits / 64].into_boxed_slice(),
            owner: vec![0; cyc_bits].into_boxed_slice(),
            cyc_mask: cyc_bits - 1,
            cyc_wmask: cyc_bits / 64 - 1,
            value_to_clear: 0,
            bases: Vec::with_capacity(64),
            sb: Vec::with_capacity(64),
            oo: Vec::with_capacity(64),
            cands: Vec::with_capacity(256),
            ypos: Vec::with_capacity(64),
            best_bases: Vec::with_capacity(64),
            best_d: 0,
            seg_t0: Vec::with_capacity(64),
            evict: Vec::with_capacity(64),
            all_bases: Vec::with_capacity(256),
        }
    }

    #[inline(always)]
    fn size(&self, b: usize) -> usize {
        self.bucket_begin[b - self.lo + 1] - self.bucket_begin[b - self.lo]
    }

    #[inline(always)]
    fn bucket_keys(&self, b: usize) -> &'a [T] {
        &self.keys[self.bucket_begin[b - self.lo]..self.bucket_begin[b - self.lo + 1]]
    }

    /// Returns the first position of the first nonempty bucket of the range.
    fn first_slice(&self) -> usize {
        let mut b = self.lo;
        while b < self.hi && self.size(b) == 0 {
            b += 1;
        }
        if b == self.hi {
            usize::MAX
        } else {
            self.g
                .slice_begin(self.keys[self.bucket_begin[b - self.lo]].h())
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

    #[inline(always)]
    fn get(&self, p: usize) -> bool {
        let p = p & self.cyc_mask;
        self.used[p / 64] >> (p % 64) & 1 != 0
    }

    #[inline(always)]
    fn get64(&self, p: usize) -> u64 {
        let w = (p / 64) & self.cyc_wmask;
        // Branchless: a 128-bit shift (shrd on x86)
        let lo = self.used[w] as u128;
        let hi = self.used[(w + 1) & self.cyc_wmask] as u128;
        ((hi << 64 | lo) >> (p % 64)) as u64
    }

    #[inline(always)]
    fn set(&mut self, p: usize, b: usize) {
        let q = p & self.cyc_mask;
        self.used[q / 64] |= 1 << (q % 64);
        self.owner[q] = b as u32;
    }

    /// Sets a slot as used without recording its owner.
    #[inline(always)]
    fn set_bit(&mut self, p: usize) {
        let q = p & self.cyc_mask;
        self.used[q / 64] |= 1 << (q % 64);
    }

    #[inline(always)]
    fn clear(&mut self, p: usize) {
        let q = p & self.cyc_mask;
        self.used[q / 64] &= !(1 << (q % 64));
    }

    /// Marks a slot used by a bucket outside of the range (which therefore
    /// cannot be evicted).
    fn premark(&mut self, p: usize) {
        self.set(p, self.lo.wrapping_sub(1));
    }

    /// Recovers a full bucket index from the low 32 bits stored in the owner
    /// array, using a nearby bucket as reference.
    #[inline(always)]
    fn owner_of(&self, p: usize, near: usize) -> usize {
        let low = self.owner[p & self.cyc_mask] as usize;
        #[cfg(target_pointer_width = "64")]
        {
            let mut b = (near & !0xFFFF_FFFF) | low;
            if b > near.wrapping_add(1 << 31) {
                b = b.wrapping_sub(1 << 32);
            } else if b.wrapping_add(1 << 31) < near {
                b = b.wrapping_add(1 << 32);
            }
            b
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            // Bucket indices fit in 32 bits
            let _ = near;
            low
        }
    }

    fn mark(&mut self, b: usize, seed: usize) {
        for &x in self.bucket_keys(b) {
            let p = self.g.pos(x, seed);
            self.set(p, b);
        }
        self.seeds[b - self.lo] = seed as u16;
    }

    /// Marks bucket `b` with the seed just returned by
    /// [`search`](Self::search), using the bases it computed.
    fn mark_best(&mut self, b: usize, seed: usize) {
        let d = self.best_d;
        let best_bases = std::mem::take(&mut self.best_bases);
        for &x in &best_bases {
            self.set(x + d, b);
        }
        self.best_bases = best_bases;
        self.seeds[b - self.lo] = seed as u16;
    }

    fn unmark(&mut self, b: usize) {
        let seed = self.seeds[b - self.lo] as usize;
        for &x in self.bucket_keys(b) {
            let p = self.g.pos(x, seed);
            self.clear(p);
        }
        self.seeds[b - self.lo] = 0;
    }

    /// Loads the slice beginnings and offset sources of the keys of a bucket.
    #[inline]
    fn load_bucket(&mut self, keys: &[T]) {
        let g = self.g;
        self.sb.clear();
        self.sb.extend(keys.iter().map(|x| g.slice_begin(x.h())));
        self.oo.clear();
        self.oo.extend(keys.iter().map(|x| x.o()));
    }

    /// Fills `self.bases` with the bases for pattern `r` of the keys of the
    /// bucket last passed to [`load_bucket`](Self::load_bucket), and returns
    /// their sum.
    #[inline]
    fn fill_bases(&mut self, r: usize) -> usize {
        let shift = r as u32 * self.g.stride();
        let mask = self.g.l_mask;
        self.bases.clear();
        self.bases.extend(
            self.sb
                .iter()
                .zip(self.oo.iter())
                .map(|(&sb, &o)| sb + ((o >> shift) & mask) as usize),
        );
        self.bases.iter().sum()
    }

    fn self_collides(&self) -> bool {
        let k = self.bases.len();
        if k <= 1 {
            return false;
        }
        if k <= 12 {
            let b = &self.bases;
            for i in 1..k {
                let x = b[i];
                for &y in &b[..i] {
                    if x == y {
                        return true;
                    }
                }
            }
            false
        } else {
            let mut s = self.bases.clone();
            s.sort_unstable();
            s.windows(2).any(|w| w[0] == w[1])
        }
    }

    /// Finds the best seed (minimum sum of positions) for bucket `b`, or 0.
    fn search(&mut self, b: usize) -> usize {
        if self.g.wrap != 0 {
            return self.search_wrap(b);
        }
        let keys = self.bucket_keys(b);
        let k = keys.len();
        self.load_bucket(keys);
        let d_max = self.g.shifts;
        let mut best_sum = usize::MAX;
        let mut best_seed = 0;
        for r in 0..self.g.patterns {
            let base_sum = self.fill_bases(r);
            if base_sum >= best_sum {
                continue;
            }
            let mut shift = 0;
            'scan: while shift < d_max {
                let mut u = 0u64;
                for &x in &self.bases {
                    u |= self.get64(x + shift);
                }
                if shift + 64 > d_max {
                    u |= !0u64 << (d_max - shift);
                }
                if r == 0 && shift == 0 {
                    // Seed 0 marks bumped buckets
                    u |= 1;
                }
                if u != u64::MAX {
                    let d = shift + u.trailing_ones() as usize;
                    let sum = base_sum + d * k;
                    if sum < best_sum && !self.self_collides() {
                        best_sum = sum;
                        best_seed = self.g.seed_of(r, d);
                        self.best_d = d;
                        self.best_bases.clear();
                        self.best_bases.extend_from_slice(&self.bases);
                    }
                    break 'scan;
                }
                shift += 64;
                if base_sum + shift * k >= best_sum {
                    break;
                }
            }
        }
        best_seed
    }

    /// Returns the seed of a configuration found by
    /// [`place`](Self::place): a pattern, or a segment with wrapping.
    #[inline(always)]
    fn conf_seed(&self, seg_t0: &[usize], r: usize, d: usize) -> usize {
        match self.g.wrap {
            0 => self.g.seed_of(r, d),
            m => (seg_t0[r] + d) / m + 1,
        }
    }

    /// With wrapping, loads in `self.sb` the slice beginnings and in
    /// `self.oo` the in-slice offsets for seed 1 of the keys of bucket `b`.
    fn load_bucket_wrap(&mut self, b: usize) {
        let keys = self.bucket_keys(b);
        let g = self.g;
        let m = g.wrap as u64;
        self.sb.clear();
        self.sb.extend(keys.iter().map(|x| g.slice_begin(x.h())));
        self.oo.clear();
        self.oo
            .extend(keys.iter().map(|x| x.h().wrapping_add(m) & g.l_mask));
    }

    /// With wrapping, iterates over the segments of shifts of the bucket
    /// last loaded by [`load_bucket_wrap`](Self::load_bucket_wrap), as in
    /// PHast+ with wrapping: in each segment no key wraps, so positions are
    /// the bases of the segment plus a shift. For each segment, fills
    /// `self.bases` and calls `f` with the total shift at the start of the
    /// segment, its length, and the sum of the bases; `f` returns `false` to
    /// stop.
    #[inline(always)]
    fn for_each_segment(&mut self, mut f: impl FnMut(&mut Self, usize, usize, usize) -> bool) {
        let l = self.g.l_mask as usize + 1;
        let m = self.g.wrap;
        // Seeds 1 . . max_seed are the total shifts 0, m, . . .
        let total_end = m * self.g.max_seed;
        // Rounds up to a multiple of m without divisions: for x < 2^16 and
        // m <= 64, ⌊x · ⌈2^32 / m⌉ / 2^32⌋ = ⌊x / m⌋
        let inv = (1u64 << 32).div_ceil(m as u64);
        let round_up = |x: usize| (((x + m - 1) as u64 * inv) >> 32) as usize * m;
        let mut t0 = 0;
        loop {
            let max_off = self.oo.iter().copied().max().unwrap_or(0) as usize;
            let mut len = round_up(l - max_off);
            let last = t0 + len >= total_end;
            if last {
                len = total_end - t0;
            }
            self.bases.clear();
            let sum = {
                let (sb, oo, bases) = (&self.sb, &self.oo, &mut self.bases);
                bases.extend(sb.iter().zip(oo).map(|(&s, &o)| s + o as usize));
                bases.iter().sum()
            };
            if !f(self, t0, len, sum) || last {
                return;
            }
            t0 += len;
            // When the slice is shorter than the multiplier, a key can wrap
            // more than once
            for o in &mut self.oo {
                *o = (*o + len as u64) & (l as u64 - 1);
            }
        }
    }

    /// Like [`search`](Self::search), with wrapping.
    fn search_wrap(&mut self, b: usize) -> usize {
        let k = self.size(b);
        self.load_bucket_wrap(b);
        let m = self.g.wrap;
        let (step, disallowed) = wrap_step_mask(m);
        let mut best_sum = usize::MAX;
        let mut best_seed = 0;
        self.for_each_segment(|s, t0, len, base_sum| {
            if base_sum >= best_sum {
                return true;
            }
            let mut shift = 0;
            while shift < len {
                let mut u = disallowed;
                for &x in &s.bases {
                    u |= s.get64(x + shift);
                }
                if shift + 64 > len {
                    u |= !0u64 << (len - shift);
                }
                if u != u64::MAX {
                    let d = shift + u.trailing_ones() as usize;
                    let sum = base_sum + d * k;
                    if sum < best_sum && !s.self_collides() {
                        best_sum = sum;
                        best_seed = (t0 + d) / m + 1;
                        s.best_d = d;
                        s.best_bases.clear();
                        s.best_bases.extend_from_slice(&s.bases);
                    }
                    break;
                }
                shift += step;
                if base_sum + shift * k >= best_sum {
                    break;
                }
            }
            true
        });
        best_seed
    }

    /// With wrapping, collects the repair candidates of bucket `b` (see
    /// [`place`](Self::place)), using segments as patterns.
    fn wrap_candidates(&mut self, b: usize, cands: &mut Vec<u128>, all_bases: &mut Vec<usize>) {
        let k = self.size(b);
        self.load_bucket_wrap(b);
        let (step, disallowed) = wrap_step_mask(self.g.wrap);
        self.seg_t0.clear();
        self.for_each_segment(|s, t0, len, base_sum| {
            let r = s.seg_t0.len();
            s.seg_t0.push(t0);
            all_bases.extend_from_slice(&s.bases);
            if s.self_collides() {
                return true;
            }
            let mut shift = 0;
            while shift < len {
                let mut ones = 0u64;
                let mut twos = 0u64;
                for &x in &s.bases {
                    let w = s.get64(x + shift);
                    twos |= ones & w;
                    ones |= w;
                }
                let mut one = ones & !twos & !disallowed;
                if shift + 64 > len {
                    one &= !(!0u64 << (len - shift));
                }
                while one != 0 {
                    let d = shift + one.trailing_zeros() as usize;
                    one &= one - 1;
                    cands.push(((base_sum + d * k) as u128) << 32 | (r as u128) << 16 | d as u128);
                }
                shift += step;
            }
            true
        });
    }

    /// Places bucket `b`, possibly evicting other buckets. Returns `true` on
    /// success.
    fn place(&mut self, b: usize, depth: u32, forbidden: usize) -> bool {
        let seed = self.search(b);
        if seed != 0 {
            self.mark_best(b, seed);
            return true;
        }
        if depth == 0 || self.repair_candidates == 0 {
            return false;
        }
        let k = self.size(b);
        let d_max = self.g.shifts;
        // Collect the shifts at which exactly one key is blocked, keeping the
        // bases of all patterns (pattern r at all_bases[r * k..][..k]).
        let mut cands = std::mem::take(&mut self.cands);
        let mut all_bases = std::mem::take(&mut self.all_bases);
        cands.clear();
        all_bases.clear();
        if self.g.wrap != 0 {
            self.wrap_candidates(b, &mut cands, &mut all_bases);
        }
        // Nested repairs overwrite the segments
        let seg_t0 = std::mem::take(&mut self.seg_t0);
        for r in 0..if self.g.wrap != 0 { 0 } else { self.g.patterns } {
            let base_sum = self.fill_bases(r);
            all_bases.extend_from_slice(&self.bases);
            if self.self_collides() {
                continue;
            }
            let mut shift = 0;
            while shift < d_max {
                let mut ones = 0u64;
                let mut twos = 0u64;
                for &x in &self.bases {
                    let w = self.get64(x + shift);
                    twos |= ones & w;
                    ones |= w;
                }
                let mut one = ones & !twos;
                if shift + 64 > d_max {
                    one &= !(!0u64 << (d_max - shift));
                }
                if r == 0 && shift == 0 {
                    one &= !1;
                }
                while one != 0 {
                    let d = shift + one.trailing_zeros() as usize;
                    one &= one - 1;
                    cands.push(((base_sum + d * k) as u128) << 32 | (r as u128) << 16 | d as u128);
                }
                shift += 64;
            }
        }
        let num_cands = cands.len().min(self.repair_candidates * 4);
        if cands.len() > num_cands {
            cands.select_nth_unstable(num_cands - 1);
            cands.truncate(num_cands);
        }
        cands.sort_unstable();
        // Nested repairs (for evicted buckets) try fewer candidates
        let max_tries = if depth == self.repair_depth {
            self.repair_candidates
        } else {
            (self.repair_candidates / 4).max(1)
        };
        // Resolve blockers in order of candidate, keeping for each blocker
        // only its first (best) candidate, and discarding blockers that
        // cannot be evicted. Unless candidates are ordered by size, evictions
        // are tried as soon as their blocker is resolved: since a failed
        // eviction restores the state, the result is the same as resolving
        // all blockers first.
        let mut evict = std::mem::take(&mut self.evict);
        evict.clear();
        let mut tried = 0;
        let mut ok = false;
        for &c in &cands {
            if tried == max_tries && !self.repair_by_size {
                break;
            }
            let (sum, r, d) = (
                (c >> 32) as usize,
                (c >> 16) as usize & 0xFFFF,
                c as usize & 0xFFFF,
            );
            // Find the blocked key and the owner of its slot
            let mut blocker = usize::MAX;
            for &x in &all_bases[r * k..][..k] {
                let p = x + d;
                if self.get(p) {
                    blocker = self.owner_of(p, b);
                    break;
                }
            }
            if blocker == usize::MAX
                || blocker == forbidden
                || blocker < self.lo
                || blocker >= self.hi
                || evict
                    .iter()
                    .any(|e: &(usize, usize, usize, usize)| e.3 == blocker)
            {
                continue;
            }
            // The evicted bucket must have all its potential positions in
            // the active part of the cyclic window.
            let first = self.keys[self.bucket_begin[blocker - self.lo]];
            if self.g.slice_begin(first.h()) < self.value_to_clear {
                continue;
            }
            let key = if self.repair_by_size {
                (self.size(blocker) << 40) | sum.min((1 << 40) - 1)
            } else {
                sum
            };
            evict.push((key, r, d, blocker));
            if !self.repair_by_size {
                tried += 1;
                let seed = self.conf_seed(&seg_t0, r, d);
                if self.try_evict(b, depth, seed, d, blocker, &all_bases[r * k..][..k]) {
                    ok = true;
                    break;
                }
            }
        }
        if self.repair_by_size {
            evict.sort_unstable();
            for &(_, r, d, blocker) in evict.iter().take(max_tries) {
                let seed = self.conf_seed(&seg_t0, r, d);
                if self.try_evict(b, depth, seed, d, blocker, &all_bases[r * k..][..k]) {
                    ok = true;
                    break;
                }
            }
        }
        self.all_bases = all_bases;
        self.seg_t0 = seg_t0;
        self.cands = cands;
        self.evict = evict;
        ok
    }

    /// Evicts `blocker`, places `b` with seed `b_seed`, that is, with shift
    /// `d` from the bases `b_bases`, and tries to place `blocker` again; on
    /// failure, restores the previous state.
    fn try_evict(
        &mut self,
        b: usize,
        depth: u32,
        b_seed: usize,
        d: usize,
        blocker: usize,
        b_bases: &[usize],
    ) -> bool {
        let old_seed = self.seeds[blocker - self.lo] as usize;
        debug_assert!(old_seed != 0);
        if depth > 1 {
            // Nested repairs need the owners of all slots
            self.unmark(blocker);
            self.mark(b, b_seed);
            if self.place(blocker, depth - 1, b) {
                return true;
            }
            self.unmark(b);
            self.mark(blocker, old_seed);
            return false;
        }
        // The evicted bucket cannot repair, so owners are not needed during
        // the trial: we just flip occupancy bits, and write the owners of the
        // slots of b only on success (the owners of the slots of the evicted
        // bucket are never overwritten, so a failure needs no restore).
        let mut ypos = std::mem::take(&mut self.ypos);
        ypos.clear();
        let g = self.g;
        ypos.extend(
            self.bucket_keys(blocker)
                .iter()
                .map(|&x| g.pos(x, old_seed)),
        );
        for &p in &ypos {
            self.clear(p);
        }
        for &x in b_bases {
            self.set_bit(x + d);
        }
        let seed = self.search(blocker);
        let ok = seed != 0;
        if ok {
            for &x in b_bases {
                self.owner[(x + d) & self.cyc_mask] = b as u32;
            }
            self.seeds[b - self.lo] = b_seed as u16;
            self.mark_best(blocker, seed);
        } else {
            for &x in b_bases {
                self.clear(x + d);
            }
            for &p in &ypos {
                self.set_bit(p);
            }
        }
        self.ypos = ypos;
        ok
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
            |s: &Self, b: usize| s.g.slice_begin(s.keys[s.bucket_begin[b - s.lo]].h());
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
        let depth = self.repair_depth;
        while let Some((_, Reverse(b))) = heap.pop() {
            in_heap[(b % HEAP_BITS) / 64] &= !(1 << (b % 64));
            if !self.place(b, depth, usize::MAX) {
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
                let end = slice_begin_of(&self, span_begin);
                while self.value_to_clear < end {
                    if self.value_to_clear % 64 == 0 && self.value_to_clear + 64 <= end {
                        self.used[(self.value_to_clear / 64) & self.cyc_wmask] = 0;
                        self.value_to_clear += 64;
                    } else {
                        let v = self.value_to_clear;
                        self.clear(v);
                        self.value_to_clear += 1;
                    }
                }
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

/// Default size-dependent priority weights. They are the weights of the
/// PHast+ implementation by Piotr Beling, except for 8-bit seeds with slices
/// of length 512 or 1024 and 10-bit seeds with slices of length 2048, which
/// have been retuned for PHast-R by coordinate descent.
fn default_weights(seed_bits: u32, slice_len: usize) -> [i64; 7] {
    let w: [i32; 7] = if slice_len <= 256 {
        match (seed_bits, slice_len) {
            (..=6, ..=128) => [-98439, 68040, 81130, 86896, 91188, 93897, 296481],
            (..=6, _) => [-81980, 50520, 90817, 106897, 116472, 123937, 287280],
            (_, ..=128) => [-173163, 58917, 73926, 83423, 88222, 92168, 206758],
            (..=7, _) => [-85977, 81531, 98837, 107586, 113333, 117710, 120656],
            (_, _) => [-85787, 84108, 99553, 107291, 112859, 117377, 119965],
        }
    } else {
        match (seed_bits, slice_len) {
            (..=7, ..=512) => [-95834, 38499, 103035, 124756, 137603, 147839, 155448],
            (_, ..=512) => [31863, 68016, 105189, 121129, 132794, 140850, 145685],
            (8, ..=1024) => [-50000, 20000, 90000, 130000, 150000, 165000, 175000],
            (..=8, ..=1024) => [-49776, 28610, 120514, 154976, 177328, 193499, 204936],
            (..=8, ..=2048) => [-14014, -11926, 63698, 144877, 194056, 353593, 360338],
            (9, ..=1024) => [-60439, 49207, 121850, 149181, 166713, 179181, 187815],
            (9, ..=2048) => [48168, 48328, 132443, 197796, 234543, 260358, 279164],
            (10, ..=1024) => [-4759, 9930, 87924, 125082, 143308, 165460, 165095],
            (10, ..=2048) => [-3419, 3042, 88860, 135429, 176433, 198538, 214441],
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

/// With wrapping and multiplier `m`, returns the step between the words of
/// a scan of shifts and the mask of the shifts in a word that are not
/// multiples of `m` (as in PHast+ with wrapping, the step is a multiple of
/// `m`, so the mask is the same for all words).
#[inline(always)]
fn wrap_step_mask(m: usize) -> (usize, u64) {
    let mut allowed = 0u64;
    let mut i = 0;
    while i < 64 {
        allowed |= 1 << i;
        i += m;
    }
    (64 - 64 % m, !allowed)
}

/// Returns the default priority weights with wrapping (those of PHast+ with
/// wrapping, for multipliers 1, 2, and at least 3).
fn wrap_weights(seed_bits: u32, multiplier: u32, slice_len: usize) -> [i64; 7] {
    match multiplier {
        1 => wrap_weights_m1(seed_bits, slice_len),
        2 => wrap_weights_m2(seed_bits, slice_len),
        _ => wrap_weights_m3(seed_bits, slice_len),
    }
    .map(|x| x as i64)
}

/// Weights of PHast+ with wrapping and multiplier 1 (from `ph`).
#[rustfmt::skip]
fn wrap_weights_m1(bits_per_seed: u32, slice_len: usize) -> [i32; 7] {
    match (bits_per_seed, slice_len) {
        (_, ..=64) => [-76520, 97960, 103626, 106759, 109053, 110149, 112662],   // 8, 4.1, 64
        (_, ..=128) => [-80872, 90492, 100641, 105939, 109960, 112290, 118119], // 8, 4.1, 128
        (..=6, ..=256) => [-76632, 59701, 89939, 103115, 111040, 117906, 283652], // 6, 3.0, slice=256
        (..=6, ..=512) => [-102171, 30195, 95877, 122987, 138980, 152173, 206055],  // 6, 3.0, slice=512
        (_, ..=256) => [-84425, 81165, 96951, 106065, 112137, 117421, 122309],  // 7, 3.5, slice=256
        (7, ..=512) => [-69271, 61152, 101770, 119869, 132454, 141236, 148273],  // 7, 3.5, slice=512
        (8, ..=512) => [-66903, 81776, 107154, 122354, 132033, 140641, 146584], // 4.1, slice=512
        (8, ..=1024) => [-50666, 55977, 116129, 145446, 164172, 180129, 192120],  // 4.1, slice=1024
        (_, ..=512) => [-45845, 91690, 122225, 138169, 149160, 155706, 164757],  // 5.1, slice=512
        (9, ..=1024) => [-51695, 68190, 121468, 146481, 164082, 178054, 186488],  // 5.1, slice=1024
        (..=9, ..=2048) => [-3365, 12300, 85113, 138418, 170087, 197668, 215654],  // 9, 5.1, slice=2048
        (10, ..=1024) => [-4011, 15045, 106558, 112844, 133305, 145623, 154991], // 5.7, slice=1024
        (..=10, ..=2048) => [-3301, 12449, 83323, 139924, 169323, 198105, 212187],   // 10, 5.7, slice=2048
        (11, ..=1024) => [-1524, 23928, 115028, 153353, 187370, 191075, 197861],    // 6.3, slice=1024 USELESS
        (11, ..=2048) => [-1777, 22788, 106158, 139632, 174143, 200775, 214797],  // 6.3, slice=2048
        (11, _) => [-4924, 19116, 22394, 59714, 110668, 154482, 181404],
        (_, ..=1024) => [-2190, 30393, 114587, 141471, 162103, 177602, 183787], // 12, 6.8, slice=1024 USELESS
        (_, ..=2048) => [-2355, 16099, 113987, 153868, 183912, 213486, 226897],   // 12, 6.8, slice=2048 USELESS
        (_, _) => [-3938, 38130, 29589, 52311, 96328, 147014, 172193], // 12, 6.8, 4096
    }
}

/// Weights of PHast+ with wrapping and multiplier 2 (from `ph`).
#[rustfmt::skip]
fn wrap_weights_m2(bits_per_seed: u32, slice_len: usize) -> [i32; 7] {
    match (bits_per_seed, slice_len) {
        (_, ..=64) => [-78586, 98418, 103824, 106532, 108539, 109981, 111063],  // 8, 4.1, 64
        (_, ..=128) => [-85534, 90593, 100261, 105362, 108728, 111329, 113115], // 8, 4.1, 128
        (..=6, ..=256) => [-113309, 70659, 92719, 103205, 111784, 117218, 121395], // 6, 3.0, slice=256
        (..=6, ..=512) => [-113437, 36479, 87740, 109716, 124793, 137012, 209528], // 6, 3.0, slice=512
        (_, ..=256) => [-83108, 76805, 93889, 104574, 111919, 117701, 137200],  // 7, 3.5, slice=256
        (7, ..=512) => [-11364, 71851, 100238, 116988, 128732, 138656, 145275],  // 7, 3.5, slice=512
        (8, ..=512) => [-67763, 78133, 104489, 121464, 133392, 140946, 155107], // 4.1, slice=512
        (8, ..=1024) => [-50137, 65904, 111782, 139890, 159029, 175922, 186995],  // 4.1, slice=1024
        (..=8, ..=2048) => [-3445, 11224, 85129, 138005, 176794, 209479, 234058], // 8, 4.1, slice=2048
        (_, ..=512) => [-45845, 91690, 122225, 138169, 149160, 155706, 164757],  // 5.1, slice=512
        (9, ..=1024) => [-49692, 70537, 115707, 143201, 163216, 178981, 188448],  // 5.1, slice=1024
        (..=9, ..=2048) => [-3514, 12002, 85880, 136849, 171127, 197699, 215787],  // 9, 5.1, slice=2048
        (10, ..=1024) => [-4383, 16783, 82867, 112554, 131931, 148281, 156013], // 5.7, slice=1024 USELESS
        (..=10, ..=2048) => [-3386, 13051, 82525, 133516, 169004, 198445, 214518],   // 10, 5.7, slice=2048
        (11, ..=1024) => [-1562, 24796, 122828, 155722, 174139, 191420, 198085],    // 6.3, slice=1024 USELESS
        (11, ..=2048) => [-2243, 21043, 83146, 136599, 172186, 200713, 215298],  // 6.3, slice=2048 USELESS
        (11, _) => [-4045, 8964, 9362, 21128, 86855, 136683, 166640],   // 11, 6.3, slice=4096
        (_, ..=1024) => [-2244, 32777, 107196, 142051, 161424, 177763, 183475], // 12, 6.8, slice=1024 USELESS
        (_, ..=2048) => [-2808, 16026, 90346, 150206, 185508, 214963, 227887],   // 12, 6.8, slice=2048 USELESS
        (_, _ /*..=4096*/) => [-4044, 7164, 10158, 22096, 92563, 142914, 171123],    // 12, 6.8, slice=4096  USELESS?? pure performance
        //(_, _) => [-4849, 12371, 19420, 27337, 28560, 51301, 103428]    // 12, 6.8, slice=8192, TODO optimize
    }
}

/// Weights of PHast+ with wrapping and multiplier 3 (from `ph`).
#[rustfmt::skip]
fn wrap_weights_m3(bits_per_seed: u32, slice_len: usize) -> [i32; 7] {
    match (bits_per_seed, slice_len) { // multiplier=3, almost the same result as for multiplier=2 weights
        (_, ..=64) => [-81342, 97738, 103193, 106305, 108524, 109876, 112382],  // 8, 4.1, 64
        (_, ..=128) => [-82883, 89250, 99246, 105030, 108983, 111224, 117058], // 8, 4.1, 128
        (..=6, ..=256) => [-143420, 70364, 89794, 100431, 107778, 113842, 253543], // 6, 3.0, slice=256
        (..=6, ..=512) => [-118906, 41451, 83177, 104570, 119520, 131788, 197543], // 6, 3.0, slice=512
        (_, ..=256) => [-82828, 77192, 94710, 105243, 112716, 118768, 136225],  // 7, 3.5, slice=256
        (7, ..=512) => [-11540, 68580, 98218, 115370, 128607, 139118, 145832],  // 7, 3.5, slice=512
        (_, ..=512) => [25100, 89361, 117113, 134755, 147369, 154606, 172378], // 4.1, slice=512
        (8, ..=1024) => [-50649, 63792, 110014, 139267, 161285, 176594, 188305],  // 4.1, slice=1024
        (..=8, ..=2048) => [-3427, 10388, 90470, 141895, 179413, 208576, 232553], // 8, 4.1, slice=2048
        (9, ..=1024) => [-41757, 60279, 113069, 143467, 162892, 179091, 188139],  // 5.1, slice=1024
        (..=9, ..=2048) => [-3753, 11840, 77702, 132696, 169641, 200687, 218764],  // 9, 5.1, slice=2048
        (10, ..=1024) => [-2394, 29640, 81921, 108732, 126229, 141102, 150457], // 5.7, slice=1024 USELESS
        (..=10, ..=2048) => [-3417, 13564, 81208, 133035, 168506, 198114, 214382],   // 10, 5.7, slice=2048
        (11, ..=1024) => [-1555, 25982, 126717, 155711, 174202, 191358, 198247],    // 6.3, slice=1024 USELESS
        (11, ..=2048) => [-2229, 21208, 88554, 137643, 169905, 200075, 213746],  // 6.3, slice=2048 USELESS
        (11, _) => [-3267, 25041, 24325, 40786, 100528, 155125, 182822],  // 11, 6.3, slice=4096
        (_, ..=1024) => [-2206, 33628, 110901, 143147, 161228, 177559, 183794], // 12, 6.8, slice=1024 USELESS
        (_, ..=2048) => [-2665, 16252, 98048, 149519, 183487, 214959, 227347],   // 12, 6.8, slice=2048 USELESS
        (_, _) => [-3356, 26074, 26278, 44692, 94747, 143426, 168599],  // 12, 6.8, slice=4096  USELESS
    }
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
        }
    }

    #[test]
    fn test_wrap() {
        for n in [0, 1, 2, 3, 10, 100, 1000, 5000, 10000] {
            check::<Box<[u8]>>(n, PHastRBuilder::default().wrap(3).bucket_size(5.0));
        }
        for m in 1..=3 {
            for depth in 0..=2 {
                check::<Box<[u8]>>(
                    300_000,
                    PHastRBuilder::default()
                        .wrap(m)
                        .bucket_size(5.0)
                        .repair_depth(depth)
                        .repair_candidates(if depth == 0 { 0 } else { 16 }),
                );
            }
        }
        check::<BitFieldVec<Box<[usize]>>>(
            300_000,
            PHastRBuilder::default()
                .wrap(1)
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        );
        check::<Box<[u8]>>(1_000_000, PHastRBuilder::default().wrap(3).bucket_size(5.0));
    }

    #[test]
    fn test_medium() {
        check::<Box<[u8]>>(1_000_000, PHastRBuilder::default());
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().repair_candidates(0));
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().repair_depth(1));
        check::<BitFieldVec<Box<[usize]>>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        );
        check::<Box<[u16]>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(11)
                .log2_slice_len(12)
                .bucket_size(6.75),
        );
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_patterns(0));
        check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_patterns(1));
        // Few shifts per pattern (pattern 0 has one shift less than the
        // others, as seed 0 marks bumped buckets)
        check::<Box<[u8]>>(
            100_000,
            PHastRBuilder::default()
                .seed_bits(4)
                .log2_patterns(2)
                .log2_slice_len(8)
                .bucket_size(2.0),
        );
        // Eight patterns, whose offsets are eight bits apart
        check::<Box<[u8]>>(
            300_000,
            PHastRBuilder::default().log2_patterns(3).log2_slice_len(8),
        );
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
            check::<Box<[u8]>>(2_000_000, PHastRBuilder::default().repair_depth(1));
        });
    }

    #[test]
    fn test_seed_storage_check() {
        let keys: Vec<u64> = (0..1000).collect();
        let r: Result<PHastR<u64>> = PHastRBuilder::default()
            .seed_bits(10)
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
        // Colliding signatures are bumped and separated by the hashes of the
        // following levels, with all configurations of the following levels
        // (in particular, with only the last level)
        for n in [5_000u64, 300_000] {
            let keys: Vec<CollidingKey> = (0..n).map(CollidingKey).collect();
            for builder in [
                PHastRBuilder::default(),
                PHastRBuilder::default()
                    .repair_depth(0)
                    .repair_candidates(0),
                PHastRBuilder::default().wrap(3).bucket_size(5.0),
            ] {
                let phf: PHastR<CollidingKey> = builder.try_build(&keys, no_logging![]).unwrap();
                let mut seen = vec![false; keys.len()];
                for &key in &keys {
                    let v = phf.get(key);
                    assert!(!seen[v], "duplicate output {v}");
                    seen[v] = true;
                }
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
