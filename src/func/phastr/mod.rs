/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! PHastR: minimal perfect hashing with rotated layouts.
//!
//! See [`PHastR`] for a description of the structure, of its construction,
//! and of when to change the default parameters.

use core::error::Error;
use std::borrow::Borrow;

use anyhow::Result;
use dsi_progress_logger::ProgressLog;
use lender::{FallibleLender, FallibleLending};
use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::bits::BitVec;
// For documentation and tests
#[cfg(any(doc, test))]
use crate::bits::BitFieldVec;
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
/// consecutive layouts with the default number of layouts (four).
const DEFAULT_LAYOUT_SHIFT: u32 = 4;

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

/// The value combined with the second hash of a key to obtain the multiplier
/// of [`level_sig`].
const LEVEL_SIG: u64 = 0xE703_7ED1_A0B4_28DB;

/// Returns the upper half of the 128-bit product of `a` and `b` xored with
/// the lower half.
#[inline(always)]
const fn folded_multiply(a: u64, b: u64) -> u64 {
    let p = a as u128 * b as u128;
    (p >> 64) as u64 ^ p as u64
}

/// Returns the signature of a key for a level after the first one, given
/// its signature for the first level, a second hash of the key, and the salt
/// of the level.
///
/// Keys with the same signature for the first level have, with high
/// probability, different signatures for the following ones, as the second
/// hash is independent from the first one; on the other hand, the key
/// itself is not necessary
/// after computing the two hashes, and this makes it possible to handle the
/// keys bumped from the first level out of line without keeping the key
/// around.
#[inline(always)]
const fn level_sig(h: u64, second: u64, salt: u64) -> u64 {
    folded_multiply(h ^ salt, second ^ LEVEL_SIG)
}

/// The default remapping sequence of a [`PHastR`]: an Elias–Fano whose
/// parameters are fixed, so the serialization format does not depend on the
/// defaults of [`EfSeq`].
///
/// [`EfSeq`]: crate::dict::elias_fano::EfSeq
pub type Remap = EliasFano<usize, SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>>;

/// The parameters of a level.
///
/// This type is opaque: it is public only because it appears in the type
/// parameter `P` of [`PHastR`] (e.g., `&[LevelParams]` after ε-serde
/// deserialization).
#[derive(Debug, Clone, Copy, MemSize, MemDbg)]
#[mem_size(flat)]
#[repr(C)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(zero_copy))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LevelParams {
    /// The number of buckets (always odd).
    buckets: u64,
    /// The number of possible slice starts (the output range minus the slice
    /// length, plus one).
    num_slices: u64,
    /// The slice length minus one.
    l_mask: u64,
    /// The base-2 logarithm of the amount *u* by which a unit increase of
    /// the seed moves the keys of a bucket.
    scale: u64,
    /// The offset of the outputs of this level in the remapping sequence.
    offset: u64,
    /// The salt used to compute the signatures of the keys of this level
    /// from their hashes (unused for the first level).
    salt: u64,
    /// The index of the first seed of this level in the seed storage.
    first_seed: u64,
}

/// A bucket-placement minimal perfect hash function with rotated layouts.
///
/// [`PHastR`] is a *minimal perfect hash function*, which maps bijectively a
/// set of *n* keys to the integers in [0 . . *n*): querying a key outside of
/// the original set will lead to an arbitrary result (wrap the function in
/// a [`SignedFunc`] to detect such keys with high probability). Like
/// [PHast], it is based on *bucket placement*: keys are hashed to buckets,
/// and each bucket stores a fixed-width *seed* that maps each key of the
/// bucket to a slot of the output range so that no two keys collide. For
/// almost all keys, a query needs just the computation of a signature and
/// the access to a seed.
///
/// With the default parameters (8-bit seeds, four layouts, slices of length
/// 1024, and an expected bucket size of 4.25 keys) space is about 1.94 bits per
/// key, and PHastR is one of the fastest minimal perfect hash functions
/// available, both to build and to query.
///
/// Seeds are stored by default in a `Box<[u8]>`, which provides the fastest
/// queries; seeds of more than eight bits must be stored in a
/// [`BitFieldVec`] (see [`SeedStore`]).
///
/// [`PHastR`] implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access: this is
/// useful for seeds stored in a [`BitFieldVec`], which are then accessed with
/// [unaligned reads].
///
/// Functions can be serialized with [ε-serde], which makes it possible to map
/// them in memory, or with [`serde`].
///
/// # Structure
///
/// Each key is hashed, using the *hash seed*, to a 64-bit *signature*, which
/// determines its *bucket* and its *slice*, a window of consecutive slots of
/// the output range (by default, of 1024 slots). The *seed* of a bucket
/// selects one of the *layouts* (by default, four), which give each key an
/// offset in its slice, and a rotation of the offsets; seeds are chosen so
/// that no two keys have the same slot. Seed zero marks the buckets that
/// could not be placed, whose keys are *bumped* to the following *level*, a
/// smaller function whose slots are remapped to the free slots of the first
/// level.
///
/// # Construction
///
/// To build a function from a slice of keys, use [`PHastR::try_par_new`],
/// which computes their signatures in parallel; to read the keys from a
/// [lender] (for example, from a file), use [`PHastR::try_new`], which reads
/// them just once. Their variants [`PHastR::try_par_new_with_builder`] and
/// [`PHastR::try_new_with_builder`] make it possible to configure the
/// construction using a [`PHastRBuilder`]. Construction uses the threads of
/// the current [rayon] pool, and the function built does not depend on the
/// constructor, provided that the pool has the same number of threads.
///
/// Construction from a slice of keys needs about 10 bytes per key besides
/// the keys with one thread, and about 12 with eight threads. Construction
/// from a lender keeps instead a *record* of sixteen bytes for each key,
/// containing two 64-bit signatures of the key (computed with different hash
/// seeds) from which its signatures for all levels are computed: in memory,
/// construction needs between 16 and 32 bytes per key, depending on the
/// number of keys, as records are collected in vectors whose capacity
/// doubles (plus the overhead of the allocator); in [offline mode], records
/// are kept in temporary files, which take about sixteen bytes per key of
/// disk space, and on large key sets construction needs in memory less than
/// half a byte per key with a few threads, besides buffers that take
/// 256 MiB up to about 8.6 · 10⁹ keys, and then about a sixteenth of a byte
/// per key (at most 16 GiB). With one thread offline construction is only
/// slightly slower than construction in memory, but it does not scale as
/// well with the number of threads.
///
/// # Changing the defaults
///
/// The defaults are an excellent choice for basically all uses, but you
/// might want to change:
///
/// - the [number of bits of a seed]: for example, with 10-bit seeds stored in a
///   [`BitFieldVec`], slices of length 2048, and an expected bucket size of
///   5.75, space is about 1.82 bits per key, but construction takes about 60%
///   more time, and queries are about a quarter slower, even after converting
///   the function with [`TryIntoUnaligned::try_into_unaligned`] (see
///   [`PHastR::try_par_new_with_builder`] for an example);
///
/// - the [expected bucket size]: a larger size reduces space and makes
///   construction faster, but more keys are bumped from the first level to
///   the following ones, and queries for such keys are slower (with 8-bit
///   seeds and a size of 4.5, space decreases by about 0.04 bits per key, but
///   queries on large key sets are about 8% slower); a smaller size makes
///   queries slightly faster, but it increases space significantly;
///
/// - the [hash seed], if you need different functions on the same keys.
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
/// * `P` - the parameters of the following levels. The default is
///   `Box<[LevelParams]>`.
///
/// * `R` - the remapping, which maps the slots of the following levels to
///   the holes of the first one, that is, to the slots left free by its
///   bumped keys. The default is [`Remap`].
///
/// The last three parameters make it possible to deserialize with ε-serde
/// without copying: for example, [`deserialize_eps`] on a `PHastR<K>`
/// returns a structure whose seeds are a `&[u8]`, whose parameters are a
/// `&[LevelParams]`, and whose remapping sequence is an Elias–Fano sequence
/// whose components are `&[usize]`.
///
/// # Command-line utility
///
/// The `phastr` binary (feature `cli`) builds a function on the lines of a
/// file (option `-f`), or on the integers [0 . . *n*) (option `-n` alone),
/// and stores it with ε-serde. The type to load is `PHastR<str>` in the
/// first case, and `PHastR<usize>` in the second case; with `--seed-bits`
/// larger than eight seeds are stored in a bit-field vector (e.g.,
/// `PHastR<str, BitFieldVec<Box<[usize]>>>`); with `--hash-type`, the
/// function is signed (e.g., `SignedFunc<PHastR<str>, Box<[u16]>>` with
/// `--hash-type u16`); and with `--unaligned` the function is stored in
/// unaligned form (e.g., `Unaligned<PHastR<str>>`; see [`Unaligned`]).
///
/// # Examples
///
/// We build a function on a slice of strings, we store it, and we map it in
/// memory using ε-serde (see also [`try_new`] for reading the keys from a
/// lender):
///
/// ```rust
/// # #[cfg(feature = "mmap")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// use epserde::prelude::*;
/// use sux::func::PHastR;
/// use sux::traits::{TryIntoUnaligned, Unaligned};
///
/// let keys: Vec<String> = (0..100_000).map(|i| format!("key{i}")).collect();
/// let phf = <PHastR<str>>::try_par_new(&keys, no_logging![])?;
///
/// let mut seen = vec![false; keys.len()];
/// for key in &keys {
///     let v = phf.get(key.as_str());
///     assert!(!seen[v]);
///     seen[v] = true;
/// }
///
/// let file = tempfile::NamedTempFile::new()?;
/// // SAFETY: the function was built by its constructor
/// unsafe { phf.store(file.path()) }?;
/// // SAFETY: the file contains a serialized PHastR<str>
/// let mapped = unsafe { <PHastR<str>>::mmap(file.path(), Flags::empty()) }?;
/// let mapped = mapped.uncase();
/// for key in &keys {
///     assert_eq!(mapped.get(key.as_str()), phf.get(key.as_str()));
/// }
///
/// // After conversion to unaligned reads, the type changes
/// let phf = phf.try_into_unaligned()?;
/// // SAFETY: the function was built by its constructor
/// unsafe { phf.store(file.path()) }?;
/// // SAFETY: the file contains a serialized Unaligned<PHastR<str>>
/// let loaded = unsafe { <Unaligned<PHastR<str>>>::load_full(file.path()) }?;
/// for key in &keys {
///     assert_eq!(loaded.get(key.as_str()), phf.get(key.as_str()));
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "mmap"))]
/// # fn main() {}
/// ```
///
/// To reject keys outside the original set with high probability, wrap the
/// function in a [`SignedFunc`] using one of the [constructors for
/// `PHastR`]:
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// use sux::func::{PHastR, SignedFunc};
///
/// let keys: Vec<String> = (0..100_000).map(|i| format!("key{i}")).collect();
/// // 16-bit verification hashes: false positives happen with probability 2⁻¹⁶
/// let func = <SignedFunc<PHastR<str>, Box<[u16]>>>::try_par_new(&keys, no_logging![])?;
///
/// for key in &keys {
///     assert!(func.get(key.as_str()).is_some());
/// }
/// assert_eq!(func.get("foo"), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// [PHast]: https://doi.org/10.1137/1.9781611978957.1
/// [`SignedFunc`]: crate::func::SignedFunc
/// [constructors for `PHastR`]: crate::func::SignedFunc#supported-combinations
/// [lender]: crate::utils::lenders
/// [rayon]: https://docs.rs/rayon
/// [offline mode]: PHastRBuilder::offline
/// [ε-serde]: https://crates.io/crates/epserde
/// [`serde`]: https://crates.io/crates/serde
/// [number of bits of a seed]: PHastRBuilder::seed_bits
/// [expected bucket size]: PHastRBuilder::bucket_size
/// [hash seed]: PHastRBuilder::seed
/// [unaligned reads]: BitFieldVec::get_unaligned
/// [`try_new`]: PHastR::try_new
/// [`deserialize_eps`]: https://docs.rs/epserde/latest/epserde/deser/trait.Deserialize.html#tymethod.deserialize_eps
#[doc(alias = "mphf")]
#[doc(alias = "minimal")]
#[doc(alias = "perfect")]
#[derive(MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(phantom(K)))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PHastR<K: ?Sized, D = Box<[u8]>, P = Box<[LevelParams]>, R = Remap> {
    /// The seed used to compute signatures.
    pub(crate) seed: u64,
    /// The number of keys.
    num_keys: usize,
    /// The base-2 logarithm of the distance in bits between the offsets of
    /// two consecutive layouts (i.e., six minus the base-2 logarithm of
    /// the number of layouts).
    layout_shift: u32,
    /// Whether the shifts of the first level are those of the default
    /// parameters (see [`DEFAULT_LAYOUT_SHIFT`] and [`DEFAULT_SCALE`]), in
    /// which case queries use constants.
    default_shifts: bool,
    /// The scale of the first level plus one, if there are four layouts
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
    /// The remapping, which maps the slots of the following levels to the
    /// holes of the first one.
    remap: R,
    _marker: std::marker::PhantomData<*const K>,
}

// SAFETY: K occurs only inside _marker, so no value of this type is ever
// stored: the auto traits depend just on the remaining fields. The *const K
// in the marker is a Sized placeholder standing in for a possibly unsized K,
// not a pointer we own; without these impls its !Send/!Sync-ness would
// propagate to PHastR and to everything containing it.
unsafe impl<K: ?Sized, D: Send, P: Send, R: Send> Send for PHastR<K, D, P, R> {}
// SAFETY: as for Send
unsafe impl<K: ?Sized, D: Sync, P: Sync, R: Sync> Sync for PHastR<K, D, P, R> {}

// Clone and Debug are implemented by hand, as derived implementations would
// require K to implement them, too (e.g., str is not Clone)

impl<K: ?Sized, D: Clone, P: Clone, R: Clone> Clone for PHastR<K, D, P, R> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed,
            num_keys: self.num_keys,
            layout_shift: self.layout_shift,
            default_shifts: self.default_shifts,
            fast_scale: self.fast_scale,
            params0: self.params0,
            seeds0: self.seeds0.clone(),
            params: self.params.clone(),
            seeds: self.seeds.clone(),
            remap: self.remap.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K: ?Sized, D: core::fmt::Debug, P: core::fmt::Debug, R: core::fmt::Debug> core::fmt::Debug
    for PHastR<K, D, P, R>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PHastR")
            .field("seed", &self.seed)
            .field("num_keys", &self.num_keys)
            .field("layout_shift", &self.layout_shift)
            .field("default_shifts", &self.default_shifts)
            .field("fast_scale", &self.fast_scale)
            .field("params0", &self.params0)
            .field("seeds0", &self.seeds0)
            .field("params", &self.params)
            .field("seeds", &self.seeds)
            .field("remap", &self.remap)
            .finish()
    }
}

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

    /// Makes queries use the generic code, rather than the code with
    /// constant shifts used with four layouts (for benchmarking, so that
    /// structures with different numbers of layouts are queried by the
    /// same code).
    #[doc(hidden)]
    pub fn generic_queries(mut self) -> Self {
        self.default_shifts = false;
        self.fast_scale = 0;
        self
    }
}

impl<
    K: ?Sized + ToSig<[u64; 1]>,
    D: SeedStore,
    P: AsRef<[LevelParams]>,
    R: SliceByValue<Value = usize>,
> PHastR<K, D, P, R>
{
    /// Returns the output of a key in the given level, given its signature
    /// *h* for the level, the lower half `lo` of the product of *h* and the
    /// number of buckets (whose upper half is the bucket of the key), the
    /// seed of the bucket, the base-2 logarithm of the width *f* of the field
    /// of each layout in `lo`, and the base-2 logarithm of the amount *u* by
    /// which a unit increase of the seed moves a key (the scale of the
    /// level).
    ///
    /// The layout *r* is given by the lowest bits of the seed *s*, and its
    /// in-slice offset is in field *r* of `lo`, which starts at bit *rf*:
    /// since *sf* mod 64 = *rf* and shifts are taken modulo 64, shifting `lo`
    /// right by *sf* yields the offset without extracting the layout.
    #[inline(always)]
    fn pos(lv: &LevelParams, h: u64, lo: u64, seed: usize, layout_shift: u32, scale: u32) -> usize {
        let s = seed as u64;
        let offset = lo
            .wrapping_shr((s << layout_shift) as u32)
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
    /// are four layouts (e.g., with 10-bit seeds and slices of length
    /// 2048), albeit in this case the test is not moved out of loops.
    ///
    /// [`pos`]: Self::pos
    #[inline(always)]
    fn pos0(&self, h: u64, lo: u64, seed: usize) -> usize {
        let lv = &self.params0;
        if self.default_shifts {
            return Self::pos(lv, h, lo, seed, DEFAULT_LAYOUT_SHIFT, DEFAULT_SCALE);
        }
        match self.fast_scale {
            1 => Self::pos(lv, h, lo, seed, DEFAULT_LAYOUT_SHIFT, 0),
            2 => Self::pos(lv, h, lo, seed, DEFAULT_LAYOUT_SHIFT, 1),
            4 => Self::pos(lv, h, lo, seed, DEFAULT_LAYOUT_SHIFT, 3),
            _ => Self::pos(lv, h, lo, seed, self.layout_shift, lv.scale as u32),
        }
    }

    /// Returns the value associated with the given key.
    ///
    /// The returned value is in the range [0 . . *n*), where *n* is the
    /// number of keys, and different keys of the original set are mapped to
    /// different values. If the key was not in the original set, the result
    /// is arbitrary (but still in [0 . . *n*), unless the function is
    /// empty, in which case it is zero).
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

    /// Returns the value associated with the given key, given its signature
    /// (which must be `K::to_sig(key, self.seed)`), as [`get`] does.
    ///
    /// [`SignedFunc`] uses the signature also to compute the verification
    /// hash of the key.
    ///
    /// [`get`]: Self::get
    /// [`SignedFunc`]: crate::func::SignedFunc
    #[inline(always)]
    pub(crate) fn get_by_sig(&self, key: &K, sig: [u64; 1]) -> usize {
        let h = sig[0];
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
                let pos = Self::pos(lv, h, p as u64, s, self.layout_shift, lv.scale as u32);
                // SAFETY: by construction, the remapping sequence contains
                // one entry for each output of each level after the first
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
    /// If you have a slice of keys, [`try_par_new`] computes their
    /// signatures in parallel for faster construction.
    ///
    /// * `keys` - a [`FallibleLender`] returning the keys, which is read just
    ///   once. The [`lenders`] module provides easy ways to build such
    ///   lenders.
    ///
    /// * `pl` - a progress logger, which logs the parameters and the phases
    ///   of the construction.
    ///
    /// # Errors
    ///
    /// Returns an error if there are duplicate keys, if the lender returns
    /// an error, or if there are too many keys for the platform.
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
    /// The two signatures of each key are kept in memory, or on disk in
    /// offline mode, as a record of sixteen bytes (see [`PHastR`] for the
    /// memory used). The function is identical to that built by
    /// [`try_par_new_with_builder`] with the same builder (apart from offline
    /// mode, which that constructor does not support), provided that the
    /// current [rayon] pool has the same number of threads.
    ///
    /// * `keys` - a [`FallibleLender`] returning the keys, which is read just
    ///   once. The [`lenders`] module provides easy ways to build such
    ///   lenders: for example, a [`LineLender`] reads the keys from a file.
    ///
    /// * `builder` - the configuration of the construction.
    ///
    /// * `pl` - a progress logger, which logs the parameters and the phases
    ///   of the construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameters of the builder are not valid, if
    /// there are duplicate keys, if the lender returns an error, if an I/O
    /// error occurs in offline mode, if there are too many keys for the
    /// platform, or if construction fails with the given parameters (which
    /// can happen only with very few bits per seed, very short slices, or
    /// very large buckets).
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

    /// Builds a [`PHastR`] from a slice of keys, computing their signatures
    /// in parallel, using default [`PHastRBuilder`] settings.
    ///
    /// This is a convenience wrapper around [`try_par_new_with_builder`]
    /// with `PHastRBuilder::default()`.
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new`] instead.
    ///
    /// * `keys` - the keys.
    ///
    /// * `pl` - a progress logger, which logs the parameters and the phases
    ///   of the construction.
    ///
    /// # Errors
    ///
    /// Returns an error if there are duplicate keys, or if there are too many
    /// keys for the platform.
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

    /// Builds a [`PHastR`] from a slice of keys, computing their signatures
    /// in parallel, using the given [`PHastRBuilder`] configuration.
    ///
    /// The builder controls construction parameters such as the [number of
    /// bits of a seed], the [slice length], and the [expected bucket size];
    /// [offline mode] is not supported, and an error is returned if it is
    /// set.
    ///
    /// The signatures of the keys are kept in memory, together with a byte
    /// per key recording their part, and the keys bumped from the first level
    /// are hashed again: construction needs about 10 bytes per key besides
    /// the keys with one thread, and about 12 with eight threads.
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new_with_builder`] instead.
    ///
    /// * `keys` - the keys.
    ///
    /// * `builder` - the configuration of the construction.
    ///
    /// * `pl` - a progress logger, which logs the parameters and the phases
    ///   of the construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameters of the builder are not valid (in
    /// particular, if offline mode is set), if there are duplicate keys, if
    /// there are too many keys for the platform, or if construction fails
    /// with the given parameters (which can happen only with very few bits
    /// per seed, very short slices, or very large buckets).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{PHastR, PHastRBuilder};
    /// # use sux::bits::BitFieldVec;
    /// # use sux::traits::TryIntoUnaligned;
    /// # use dsi_progress_logger::no_logging;
    /// let keys: Vec<u64> = (0..100_000).collect();
    /// // 10-bit seeds (see "Changing the defaults" in the documentation of
    /// // PHastR)
    /// let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
    ///     &keys,
    ///     PHastRBuilder::default()
    ///         .seed_bits(10)
    ///         .log2_slice_len(11)
    ///         .bucket_size(5.75),
    ///     no_logging![],
    /// )?;
    /// // Faster queries using unaligned reads
    /// let phf = phf.try_into_unaligned()?;
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
/// reads]; this is useful mainly for seeds stored in a [`BitFieldVec`], as
/// slices of bytes are left unchanged, and the remapping sequence is
/// accessed only for keys bumped from the first level.
///
/// The seed storage must implement [`SeedStoreBuild`], whose safety
/// contract guarantees that the converted storage contains the same seeds.
///
/// [unaligned reads]: BitFieldVec::get_unaligned
impl<K: ?Sized, D: SeedStoreBuild + TryIntoUnaligned, P, R: TryIntoUnaligned> TryIntoUnaligned
    for PHastR<K, D, P, R>
{
    type Unaligned = PHastR<K, Unaligned<D>, P, Unaligned<R>>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        Ok(PHastR {
            seed: self.seed,
            num_keys: self.num_keys,
            layout_shift: self.layout_shift,
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

/// Converts back a function using [unaligned reads].
///
/// The seed storage must implement [`SeedStoreBuild`], whose safety
/// contract guarantees that the storage obtained by conversion contains the
/// same seeds.
///
/// [unaligned reads]: BitFieldVec::get_unaligned
impl<K: ?Sized, D: SeedStoreBuild + TryIntoUnaligned + From<Unaligned<D>>, P>
    From<Unaligned<PHastR<K, D, P, Remap>>> for PHastR<K, D, P, Remap>
{
    fn from(f: Unaligned<PHastR<K, D, P, Remap>>) -> Self {
        PHastR {
            seed: f.seed,
            num_keys: f.num_keys,
            layout_shift: f.layout_shift,
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

// The other tests that use only the public interface are in
// tests/test_phastr.rs; this one needs the private fields
#[cfg(all(test, feature = "epserde"))]
mod tests {
    use super::*;
    use dsi_progress_logger::no_logging;

    /// Serializes, deserializes zero-copy (checking at compile time that
    /// seeds, level parameters, and the remapping sequence are borrowed),
    /// and compares outputs.
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

    /// Functions with seeds stored in bytes, in bit-field vectors, and in
    /// bit-field vectors with unaligned reads are deserialized zero-copy,
    /// and give the same outputs.
    #[test]
    fn test_epserde() -> Result<()> {
        check_epserde!(Box<[u8]>, &[u8], PHastRBuilder::default());
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
}
