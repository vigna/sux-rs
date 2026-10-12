/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::type_complexity, private_bounds)]

//! Signed static functions.
//!
//! A signed function stores for each key a hash. When querying a key, the
//! function first computes the hash of the key and compares it with the stored
//! hash for the returned index (for this to make sense, the function must be a
//! minimal perfect hash). If the hashes match, the index is returned;
//! otherwise, [`None`] is returned. This allows the function to reject queries
//! for keys outside the original set, with a false-positive probability
//! depending on the size of the stored hashes.
//!
//! [`SignedFunc`] is generic over the inner function `F` (which must implement
//! [`SignableFunc`]) and the hash storage `H` (which must implement
//! [`TruncateHash`]). The hash storage can be a full-width boxed slice (e.g.,
//! `Box<[u64]>`, giving false-positive probability 2⁻⁶⁴) or a
//! [`BitFieldVec`] with a caller-chosen bit width (e.g., 8 bits for ≈0.4%
//! false positives). Per-inner-type `get` methods are provided via
//! monomorphized `impl` blocks.
//!
//! Use concrete types directly, like `SignedFunc<LcpMmphfStr, Box<[u64]>>`,
//! `SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>`, or
//! `SignedFunc<PHastR<str>, Box<[u16]>>`. For the combinations of inner
//! function and hash storage listed in the documentation of [`SignedFunc`],
//! this module provides `try_new`/`try_par_new` constructors (and their
//! `_with_builder` variants) and a `get` method; signed [`VFunc`]s also
//! provide a `get_by_sig` method. [`SignedFunc::from_parts`] assembles an
//! instance from an inner function and a hash store. All `try_*`
//! constructors require the `rayon` feature.

use std::borrow::Borrow;

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::PHastR;
use crate::func::VFunc;
use crate::func::lcp_mmphf::{BitPrefix, IntBitPrefix, LcpMmphf, LcpMmphfInt};
use crate::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use crate::func::phast_r::{LevelParams, SeedStore};
use crate::func::shard_edge::ShardEdge;
use crate::traits::Backend;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::{PrimitiveInteger, PrimitiveNumber, PrimitiveNumberAs};
use value_traits::slices::SliceByValue;

#[cfg(feature = "rayon")]
fn validate_hash_width<H: Word>(hash_width: usize) -> anyhow::Result<()> {
    let max_width = 64.min(H::BITS as usize);
    anyhow::ensure!(
        (1..=max_width).contains(&hash_width),
        "hash_width ({hash_width}) must be in [1..={max_width}]"
    );
    Ok(())
}

/// Truncates a hash to a given width.
///
/// For full-word collections (e.g., `Box<[u16]>`), this is a type
/// conversion; for sub-word collections ([`BitFieldVec`]), this uses
/// the stored mask.
pub trait TruncateHash<W> {
    /// Returns the given hash truncated to the bit width of the stored
    /// values.
    fn truncate_hash(&self, hash: u64) -> W;
}

// The bound is Word, not PrimitiveNumber: an as-style conversion to a
// float or signed word type would silently collapse distinct hashes,
// degrading the documented false-positive rate.
impl<W: Word> TruncateHash<W> for Box<[W]>
where
    u64: PrimitiveNumberAs<W>,
{
    #[inline(always)]
    fn truncate_hash(&self, hash: u64) -> W {
        hash.as_to::<W>()
    }
}

// epserde's zero-copy deserialization turns a Box<[W]> hash store into a
// borrowed &[W]; provide the same full-width truncation so mapped signed
// functions remain queryable.
impl<W: Word> TruncateHash<W> for &[W]
where
    u64: PrimitiveNumberAs<W>,
{
    #[inline(always)]
    fn truncate_hash(&self, hash: u64) -> W {
        hash.as_to::<W>()
    }
}

impl<B: Backend<Word: Word>> TruncateHash<B::Word> for BitFieldVec<B>
where
    u64: PrimitiveNumberAs<B::Word>,
{
    #[inline(always)]
    fn truncate_hash(&self, hash: u64) -> B::Word {
        hash.as_to::<B::Word>() & self.mask()
    }
}

impl<B: Backend<Word: Word>> TruncateHash<B::Word> for BitFieldVecU<B>
where
    u64: PrimitiveNumberAs<B::Word>,
{
    #[inline(always)]
    fn truncate_hash(&self, hash: u64) -> B::Word {
        hash.as_to::<B::Word>() & self.mask()
    }
}

/// Common interface for inner functions used by signed wrappers.
///
/// This trait is not intended to be implemented by users; it is an internal
/// abstraction to allow the signed wrappers to work with different static
/// functions. It provides access to the seed, [signature type], verification
/// hash, and key count, so that [`SignedFunc`] can verify hashes without
/// knowing which specific type of function it wraps.
///
/// [signature type]: Sig
pub trait SignableFunc {
    /// The signature type used by the inner function (e.g., `[u64; 2]`).
    type Sig: Sig;

    /// Returns the seed used to hash keys into signatures.
    fn seed(&self) -> u64;
    /// Returns the verification hash of a signature.
    ///
    /// Functions using a [`ShardEdge`] return its
    /// [remixed hash](ShardEdge::remixed_hash).
    fn remixed_hash(&self, sig: Self::Sig) -> u64;
    /// Returns the number of keys stored in the function.
    fn len(&self) -> usize;
    /// Returns whether the function contains no keys.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<
    K: PrimitiveInteger,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignableFunc for LcpMmphfInt<K, D, S0, E0, S1, E1>
{
    type Sig = S0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.lcp_len_offset.seed
    }
    #[inline(always)]
    fn remixed_hash(&self, sig: S0) -> u64 {
        self.lcp_len_offset.shard_edge.remixed_hash(sig)
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<
    K: ?Sized,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignableFunc for LcpMmphf<K, D, S0, E0, S1, E1>
{
    type Sig = S0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.lcp_len_offset.seed
    }
    #[inline(always)]
    fn remixed_hash(&self, sig: S0) -> u64 {
        self.lcp_len_offset.shard_edge.remixed_hash(sig)
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<
    K: PrimitiveInteger,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignableFunc for Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1>
{
    type Sig = S0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.lcp_freq_len_offset.seed
    }
    #[inline(always)]
    fn remixed_hash(&self, sig: S0) -> u64 {
        self.lcp_freq_len_offset.shard_edge.remixed_hash(sig)
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<
    K: ?Sized,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignableFunc for Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>
{
    type Sig = S0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.lcp_freq_len_offset.seed
    }
    #[inline(always)]
    fn remixed_hash(&self, sig: S0) -> u64 {
        self.lcp_freq_len_offset.shard_edge.remixed_hash(sig)
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<K: ?Sized, D: SliceByValue, S: Sig, E: ShardEdge<S, 3>> SignableFunc for VFunc<K, D, S, E> {
    type Sig = S;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.seed
    }
    #[inline(always)]
    fn remixed_hash(&self, sig: S) -> u64 {
        self.shard_edge.remixed_hash(sig)
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.num_keys
    }
}

impl<K: ?Sized, D, P, R> SignableFunc for PHastR<K, D, P, R> {
    type Sig = [u64; 1];

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns a remix of the signature.
    ///
    /// The signature determines the output of a key of the first level (its
    /// upper bits select the bucket and the slice, its lower bits the offset
    /// in the slice), so a key outside the original set that is mapped to
    /// the output of a key tends to agree with its signature on those bits:
    /// storing the signature itself would increase the false-positive rate.
    #[inline(always)]
    fn remixed_hash(&self, sig: [u64; 1]) -> u64 {
        crate::utils::mix64(sig[0])
    }
    #[inline(always)]
    fn len(&self) -> usize {
        PHastR::len(self)
    }
}

/// A signed function using a [`TruncateHash`] to store verification hashes.
///
/// Wraps an inner function `F` (any type implementing [`SignableFunc`]) and adds
/// per-key verification hashes so that queries for keys outside the original
/// set return `None`. The false-positive probability depends on the hash
/// storage: for `Box<[W]>` it is 2<sup>−min(`W::BITS`, 64)</sup>; for
/// [`BitFieldVec`] it is 2<sup>−`hash_width`</sup>.
///
/// The verification hashes are `mix64` of a single 64-bit signature word (the
/// verification hash is 64-bit in every edge variant). With the default sharded
/// fuse edges (`FuseLge3Shards`) that same word also determines the local edge,
/// so two distinct keys that share it land in the same slots with the same
/// expected hash and are indistinguishable, an irreducible false positive that
/// no larger hash width can remove. `FuseLge3FullSigs` computes the edge from
/// the other signature word as well, so keys sharing the hash word usually land
/// in different slots, avoiding that collision (the verification hash itself
/// stays 64-bit either way). For a [`PHastR`], the verification hash is a
/// remix (`mix64`) of the 64-bit signature of the key, which determines its
/// output if the key is not bumped from the first level, so the same
/// irreducible false positive happens for keys sharing the signature.
///
/// Instances of this structure are immutable; they are built using the
/// constructors listed below, or assembled using [`from_parts`], and can be
/// serialized using [ε-serde] or [`serde`].
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Supported combinations
///
/// Constructors and query methods are available for the inner functions in
/// the table below. Each inner function can be combined with hashes stored
/// in a `Box<[W]>` (full-width hashes) or in a `BitFieldVec<Box<[W]>>`, in
/// which case the constructors take an additional argument `hash_width`, the
/// number of bits of a hash (at most `min(W::BITS, 64)`); `W` is an unsigned
/// integer type, which must not be `u128` for a [`BitFieldVec`], as hashes
/// are written in parallel using atomic operations.
///
/// Each combination provides the constructors `try_new` and
/// `try_new_with_builder`, which read the keys from a
/// [`FallibleRewindableLender`], and `try_par_new` and
/// `try_par_new_with_builder`, which read the keys from a slice and
/// parallelize construction and signing. The `_with_builder` variants take
/// a builder of the type listed below, and the constructors of the LCP-based
/// functions reading keys from a lender take the number of keys `n`.
///
/// | Inner function | Builder | Keys | Queries |
/// |----------------|---------|------|---------|
/// | [`VFunc`]`<K, BitFieldVec<Box<[usize]>>, S, E>` | [`VBuilder`] | any key; the value of a key is its position | `get(impl Borrow<K>)`, `get_by_sig(S)` |
/// | [`LcpMmphfInt`]`<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>` | [`VBuilder`] | integers in strictly increasing order; the value of a key is its rank | `get(K)` |
/// | [`LcpMmphf`]`<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>` (e.g., [`LcpMmphfStr`]) | [`VBuilder`] | byte sequences in strictly increasing lexicographic order, either without zero bytes or prefix-free; the value of a key is its rank | `get(&K)` |
/// | [`Lcp2MmphfInt`]`<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>` | [`VBuilder`] | as for [`LcpMmphfInt`] | `get(K)` |
/// | [`Lcp2Mmphf`]`<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>` (e.g., [`Lcp2MmphfStr`]) | [`VBuilder`] | as for [`LcpMmphf`] | `get(&K)` |
/// | [`PHastR`]`<K, D>` | [`PHastRBuilder`] | any key; values are a bijection with [0 . . *n*) | `get(impl Borrow<K>)` |
///
/// All queries return `Option<usize>`.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// use sux::func::{PHastR, SignedFunc};
/// let keys: Vec<String> = (0..1000).map(|i| format!("key{i}")).collect();
/// let func = <SignedFunc<PHastR<str>, Box<[u16]>>>::try_par_new(&keys, no_logging![])?;
///
/// let mut seen = vec![false; keys.len()];
/// for key in &keys {
///     let v = func.get(key.as_str()).unwrap();
///     assert!(!seen[v]);
///     seen[v] = true;
/// }
/// // With 16-bit hashes, false positives happen with probability 2⁻¹⁶
/// assert_eq!(func.get("missing"), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// [`from_parts`]: SignedFunc::from_parts
/// [`FallibleRewindableLender`]: crate::utils::FallibleRewindableLender
/// [`VBuilder`]: crate::func::VBuilder
/// [`PHastRBuilder`]: crate::func::PHastRBuilder
/// [`LcpMmphfStr`]: crate::func::LcpMmphfStr
/// [`Lcp2MmphfStr`]: crate::func::Lcp2MmphfStr
/// [ε-serde]: https://crates.io/crates/epserde
/// [`serde`]: https://crates.io/crates/serde
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SignedFunc<F, H> {
    pub(crate) func: F,
    pub(crate) hashes: H,
}
impl<F: SignableFunc, H: SliceByValue> SignedFunc<F, H> {
    /// Creates a new `SignedFunc` from a function and its hash storage.
    ///
    /// # Panics
    ///
    /// Panics if the hash storage length differs from the function length.
    pub fn from_parts(func: F, hashes: H) -> Self {
        assert_eq!(
            func.len(),
            hashes.len(),
            "hash storage length must match the function length"
        );
        Self { func, hashes }
    }
}

// ── Unified query helpers ────────────────────────────────────────

impl<F: SignableFunc, H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>>
    SignedFunc<F, H>
{
    /// Verifies that the stored hash matches the remixed hash for the
    /// given index and signature.
    #[inline(always)]
    fn verify<V: PrimitiveNumber>(&self, index: V, sig: F::Sig) -> Option<V> {
        let expected = self.hashes.truncate_hash(self.func.remixed_hash(sig));
        let stored = self.hashes.get_value(index.as_to::<usize>())?;
        if stored == expected {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the number of keys in the function.
    pub fn len(&self) -> usize {
        self.func.len()
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty()
    }
}

// ── VFunc get ──────────────────────────────────────────────────

impl<
    K: ?Sized + ToSig<S>,
    D: SliceByValue<Value: Word + BinSafe>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
> SignedFunc<VFunc<K, D, S, E>, H>
{
    /// Returns the index of a key associated with the given signature, if there
    /// was such a key in the list provided at construction time; otherwise,
    /// returns `None`.
    ///
    /// False positives happen with probability defined at construction time.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> Option<D::Value> {
        self.verify(self.func.get_by_sig(sig), sig)
    }

    /// Returns the index of the given key, if the key was in the list provided at
    /// construction time; otherwise, returns `None`.
    ///
    /// False positives happen with probability defined at construction time.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> Option<D::Value> {
        self.get_by_sig(K::to_sig(key.borrow(), self.func.seed()))
    }
}

// ── LcpMmphfInt get ────────────────────────────────────────────

impl<
    K: PrimitiveInteger + ToSig<S0> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignedFunc<LcpMmphfInt<K, D, S0, E0, S1, E1>, H>
where
    IntBitPrefix<K>: ToSig<S1>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match. As for all signed
    /// functions, a key outside the original set returns `Some` (a false
    /// positive) with the probability defined at construction time.
    #[inline]
    pub fn get(&self, key: K) -> Option<usize> {
        let sig = K::to_sig(key, self.func.seed());
        self.verify(self.func.get_by_sig(key, sig), sig)
    }
}

// ── LcpMmphf get ──────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignedFunc<LcpMmphf<K, D, S0, E0, S1, E1>, H>
where
    BitPrefix: ToSig<S1>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match. As for all signed
    /// functions, a key outside the original set returns `Some` (a false
    /// positive) with the probability defined at construction time.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let sig = K::to_sig(key, self.func.seed());
        self.verify(self.func.get_by_sig(key, sig), sig)
    }
}

// ── Lcp2MmphfInt get ──────────────────────────────────────────

impl<
    K: PrimitiveInteger + ToSig<S0> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignedFunc<Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1>, H>
where
    IntBitPrefix<K>: ToSig<S1>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match. As for all signed
    /// functions, a key outside the original set returns `Some` (a false
    /// positive) with the probability defined at construction time.
    #[inline]
    pub fn get(&self, key: K) -> Option<usize> {
        let sig = K::to_sig(key, self.func.seed());
        self.verify(self.func.get_by_sig(key, sig), sig)
    }
}

// ── Lcp2Mmphf get ─────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> SignedFunc<Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>, H>
where
    BitPrefix: ToSig<S1>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match. As for all signed
    /// functions, a key outside the original set returns `Some` (a false
    /// positive) with the probability defined at construction time.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let sig = K::to_sig(key, self.func.seed());
        self.verify(self.func.get_by_sig(key, sig), sig)
    }
}

// ── PHastR get ────────────────────────────────────────────────

impl<
    K: ?Sized + ToSig<[u64; 1]>,
    D: SeedStore,
    P: AsRef<[LevelParams]>,
    R: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
> SignedFunc<PHastR<K, D, P, R>, H>
{
    /// Returns the value associated with the given key, if the key was in
    /// the list provided at construction time; otherwise, returns `None`.
    ///
    /// False positives happen with probability defined at construction time.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> Option<usize> {
        let key = key.borrow();
        let sig = K::to_sig(key, self.func.seed);
        self.verify(self.func.get_by_sig(key, sig), sig)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors: helper functions
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::bits::AtomicBitFieldVec;
    use crate::func::VBuilder;
    use crate::func::phast_r::{PHastRBuilder, SeedStoreBuild};
    use crate::traits::AtomicBitFieldSlice;
    use crate::utils::FallibleRewindableLender;
    use anyhow::Result;
    use atomic_primitive::{AtomicPrimitive, PrimitiveAtomicUnsigned};
    use core::error::Error;
    use dsi_progress_logger::ProgressLog;
    use lender::*;
    use rayon::prelude::*;
    use rdst::RadixKey;
    use std::ops::{BitXor, BitXorAssign};
    use std::sync::atomic::Ordering;
    use sync_cell_slice::SyncSlice;
    use value_traits::slices::SliceByValueMut;

    /// Signs the keys returned by a lender, passing to `set` the index of
    /// each key and the verification hash of its signature.
    ///
    /// The closure `index_sig` returns the index and the signature of a key,
    /// given its position in the lender.
    ///
    /// # Errors
    ///
    /// Returns an error if the lender returns an error, or if it returns
    /// fewer keys than the function contains.
    fn sign<F: SignableFunc, K: ?Sized, B: ?Sized + Borrow<K>>(
        func: &F,
        mut keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        index_sig: impl Fn(usize, &K) -> (usize, F::Sig),
        mut set: impl FnMut(usize, u64),
        pl: &mut impl ProgressLog,
    ) -> Result<()> {
        let n = func.len();
        pl.push_log_target(" ▸ hashes");
        let mut run = || -> Result<()> {
            pl.item_name("key");
            pl.expected_updates(Some(n));
            pl.log_level(log::Level::Info);
            pl.start("Signing keys...");
            for i in 0..n {
                let key = keys.next()?.ok_or_else(|| {
                    anyhow::anyhow!("expected {n} keys while signing, but got {i}")
                })?;
                let (index, sig) = index_sig(i, key.borrow());
                set(index, func.remixed_hash(sig));
                pl.light_update();
            }
            pl.done();
            Ok(())
        };
        let result = run();
        pl.pop_log_target();
        result
    }

    /// Signs the keys of a slice in parallel, passing to `set` the index of
    /// each key and the verification hash of its signature.
    ///
    /// The closure `index_sig` returns the index and the signature of a key,
    /// given its position in the slice. The indices of the keys are
    /// distinct, so `set` can write them concurrently.
    fn par_sign<F: SignableFunc + Sync, K: ?Sized, B: Borrow<K> + Sync>(
        func: &F,
        keys: &[B],
        index_sig: impl Fn(usize, &K) -> (usize, F::Sig) + Sync,
        set: impl Fn(usize, u64) + Sync,
        pl: &mut impl ProgressLog,
    ) {
        debug_assert_eq!(keys.len(), func.len());
        pl.push_log_target(" ▸ hashes");
        pl.item_name("key");
        pl.expected_updates(Some(keys.len()));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        keys.par_iter().enumerate().for_each(|(i, key)| {
            let (index, sig) = index_sig(i, key.borrow());
            set(index, func.remixed_hash(sig));
        });
        pl.update_with_count(keys.len());
        pl.done();
        pl.pop_log_target();
    }

    /// Signs in parallel the keys of a [`VFunc`] mapping each key to its
    /// index, given the store of their signatures and indices used to build
    /// it, passing to `set` the index of each key and the verification hash
    /// of its signature.
    ///
    /// The store is drained, so shards are read (in offline mode, from disk)
    /// as threads need them. The indices of the keys are distinct, so `set`
    /// can write them concurrently.
    fn par_sign_shards<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
        shard_edge: &E,
        store: &mut impl ShardStore<S, usize>,
        set: impl Fn(usize, u64) + Sync,
        pl: &mut impl ProgressLog,
    ) {
        pl.push_log_target(" ▸ hashes");
        let mut cpl = pl.concurrent();
        cpl.item_name("shard");
        cpl.expected_updates(Some(1 << shard_edge.shard_high_bits()));
        cpl.log_level(log::Level::Info);
        cpl.start("Signing shards...");
        store
            .drain()
            .par_bridge()
            .for_each_with(cpl.clone(), |cpl, shard| {
                for sig_val in shard.iter() {
                    set(sig_val.val, shard_edge.remixed_hash(sig_val.sig));
                }
                cpl.update();
            });
        cpl.done();
        pl.pop_log_target();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<VFunc<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3> + MemSize + FlatType,
        H: Word,
    > SignedFunc<VFunc<K, BitFieldVec<Box<[usize]>>, S, E>, Box<[H]>>
    where
        SigVal<S, usize>: RadixKey,
        SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using
        /// default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores full-word hashes for verification, giving a false-positive
        /// rate of 2<sup>-min(`H::BITS`, 64)</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// * `keys` - a [`FallibleRewindableLender`].
        ///   The [`lenders`] module provides easy ways to build
        ///   such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// use sux::func::{SignedFunc, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type SFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>>;
        /// let func: SFunc = SFunc::try_new(
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using the
        /// given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores full-word hashes for verification, giving a false-positive
        /// rate of 2<sup>-min(`H::BITS`, 64)</sup>.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed]. The
        /// hashes are computed in parallel from the signatures used to build
        /// the function, reading them one shard at a time.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// * `keys` - a [`FallibleRewindableLender`].
        ///   The [`lenders`] module provides easy ways to build
        ///   such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// use sux::func::{SignedFunc, VBuilder, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type SFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>>;
        /// let func: SFunc = SFunc::try_new_with_builder(
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [offline mode]: VBuilder::offline
        /// [thread count]: VBuilder::max_num_threads
        /// [sharding overhead]: VBuilder::eps
        /// [PRNG seed]: VBuilder::seed
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let (func, mut store, _) = builder.try_build_func_and_store(
                keys,
                FromCloneableIntoIterator::from(0..),
                BitFieldVec::new_padded,
                pl,
            )?;
            let mut hashes = vec![H::ZERO; func.len()];
            let sync_hashes = hashes.as_sync_slice();
            // SAFETY: the indices of the keys are distinct, so each cell is
            // written by a single thread
            par_sign_shards(
                &func.shard_edge,
                &mut store,
                |i, h| unsafe { sync_hashes[i].set(H::as_from(h)) },
                pl,
            );
            Ok(SignedFunc {
                func,
                hashes: hashes.into_boxed_slice(),
            })
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing hash computation and store population with
        /// rayon, using default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores full-word hashes for verification, giving a false-positive
        /// rate of 2<sup>-min(`H::BITS`, 64)</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// use sux::func::{SignedFunc, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type SFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>>;
        /// let keys: Vec<usize> = (0..100).collect();
        /// let func: SFunc = SFunc::try_par_new(&keys, no_logging![])?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            u64: PrimitiveNumberAs<H>,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing hash computation and store population with
        /// rayon, using the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores full-word hashes for verification, giving a false-positive
        /// rate of 2<sup>-min(`H::BITS`, 64)</sup>.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// use sux::func::{SignedFunc, VBuilder, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type SFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>>;
        /// let keys: Vec<usize> = (0..100).collect();
        /// let func: SFunc = SFunc::try_par_new_with_builder(
        ///     &keys,
        ///     VBuilder::default(),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [offline mode]: VBuilder::offline
        /// [thread count]: VBuilder::max_num_threads
        /// [sharding overhead]: VBuilder::eps
        /// [PRNG seed]: VBuilder::seed
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            u64: PrimitiveNumberAs<H>,
        {
            let values: Vec<usize> = (0..keys.len()).collect();
            let func = <VFunc<K, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                keys, &values, builder, pl,
            )?;
            let seed = func.seed();
            let mut hashes = vec![H::ZERO; func.len()];
            let sync_hashes = hashes.as_sync_slice();
            // SAFETY: the indices of the keys are distinct, so each cell is
            // written by a single thread
            par_sign(
                &func,
                keys,
                |i, key: &K| (i, K::to_sig(key, seed)),
                |i, h| unsafe { sync_hashes[i].set(h.as_to::<H>()) },
                pl,
            );
            Ok(SignedFunc {
                func,
                hashes: hashes.into_boxed_slice(),
            })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<VFunc<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3> + MemSize + FlatType,
        H: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned<Value = H>>,
    > SignedFunc<VFunc<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
    where
        SigVal<S, usize>: RadixKey,
        SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using
        /// default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around `try_new_with_builder` (in
        /// this impl block) with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, `try_par_new` (in this impl
        /// block) parallelizes the hash computation for faster construction.
        ///
        /// * `keys` - a [`FallibleRewindableLender`].
        ///   The [`lenders`] module provides easy ways to build
        ///   such lenders.
        /// * `hash_width` - the number of hash bits per key
        ///   (`1..=min(H::BITS, 64)`).
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// use sux::func::{SignedFunc, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type BSFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;
        /// let func: BSFunc = BSFunc::try_new(
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     8,
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            u64: PrimitiveNumberAs<H>,
        {
            Self::try_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using
        /// the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed]. The
        /// hashes are computed in parallel from the signatures used to build
        /// the function, reading them one shard at a time.
        ///
        /// See also `try_par_new_with_builder` (in this impl block) for
        /// parallel hash computation from slices.
        ///
        /// * `keys` - a [`FallibleRewindableLender`].
        ///   The [`lenders`] module provides easy ways to build
        ///   such lenders.
        /// * `hash_width` - the number of hash bits per key
        ///   (`1..=min(H::BITS, 64)`).
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// use sux::func::{SignedFunc, VBuilder, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type BSFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;
        /// let func: BSFunc = BSFunc::try_new_with_builder(
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     8,
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [offline mode]: VBuilder::offline
        /// [thread count]: VBuilder::max_num_threads
        /// [sharding overhead]: VBuilder::eps
        /// [PRNG seed]: VBuilder::seed
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            u64: PrimitiveNumberAs<H>,
        {
            validate_hash_width::<H>(hash_width)?;
            let (func, mut store, _) = builder.try_build_func_and_store(
                keys,
                FromCloneableIntoIterator::from(0..),
                BitFieldVec::<Box<[usize]>>::new_padded,
                pl,
            )?;
            let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
            let mask = hashes.mask();
            let atomic_hashes =
                AtomicBitFieldVec::<&mut [H::Atomic]>::try_from(BitFieldVec::<&mut [H]>::from(
                    &mut hashes,
                ))?;
            // The indices of the keys are distinct, but the fields of
            // different indices may share a word: we write atomically
            par_sign_shards(
                &func.shard_edge,
                &mut store,
                |i, h| atomic_hashes.set_atomic(i, h.as_to::<H>() & mask, Ordering::Relaxed),
                pl,
            );
            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around `try_par_new_with_builder`
        /// (in this impl block) with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// `try_new` (in this impl block) instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// use sux::func::{SignedFunc, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type BSFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;
        /// let keys: Vec<usize> = (0..100).collect();
        /// let func: BSFunc = BSFunc::try_par_new(&keys, 8, no_logging![])?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            u64: PrimitiveNumberAs<H>,
        {
            Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// `try_new_with_builder` (in this impl block) instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// use sux::func::{SignedFunc, VBuilder, VFunc};
        /// use sux::bits::BitFieldVec;
        /// type BSFunc = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;
        /// let keys: Vec<usize> = (0..100).collect();
        /// let func: BSFunc = BSFunc::try_par_new_with_builder(
        ///     &keys,
        ///     8,
        ///     VBuilder::default(),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(i), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [offline mode]: VBuilder::offline
        /// [thread count]: VBuilder::max_num_threads
        /// [sharding overhead]: VBuilder::eps
        /// [PRNG seed]: VBuilder::seed
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            u64: PrimitiveNumberAs<H>,
        {
            validate_hash_width::<H>(hash_width)?;
            let values: Vec<usize> = (0..keys.len()).collect();
            let func = <VFunc<K, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                keys, &values, builder, pl,
            )?;
            let seed = func.seed();
            let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
            let mask = hashes.mask();
            let atomic_hashes =
                AtomicBitFieldVec::<&mut [H::Atomic]>::try_from(BitFieldVec::<&mut [H]>::from(
                    &mut hashes,
                ))?;
            // The indices of the keys are distinct, but the fields of
            // different indices may share a word: we write atomically
            par_sign(
                &func,
                keys,
                |i, key: &K| (i, K::to_sig(key, seed)),
                |i, h| atomic_hashes.set_atomic(i, h.as_to::<H>() & mask, Ordering::Relaxed),
                pl,
            );
            Ok(SignedFunc { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<LcpMmphfInt<...>>, SignedFunc<LcpMmphf<...>>,
    // SignedFunc<Lcp2MmphfInt<...>>, and SignedFunc<Lcp2Mmphf<...>>
    // ═══════════════════════════════════════════════════════════════════

    /// Implements the constructors of a [`SignedFunc`] wrapping an LCP-based
    /// MMPHF on integers (`$int`) and on byte sequences (`$bytes`), with
    /// hashes stored in a boxed slice and in a [`BitFieldVec`].
    ///
    /// `$params`, `$args`, and `$bounds` are the generic parameters following
    /// the type of the keys, the corresponding type arguments, and the bounds
    /// of the where clause common to all constructors; `$str` (the type
    /// alias for string keys), `$name`, and `$inner` are used in the
    /// documentation.
    macro_rules! impl_signed_lcp {
        (
            $int:ident, $bytes:ident, $str:ident,
            name: $name:literal,
            inner: $inner:literal,
            params: [$($params:tt)*],
            args: [$($args:ident),*],
            bounds: [$($bounds:tt)*]
        ) => {
            impl<
                K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
                W: Word,
                $($params)*
            > SignedFunc<$int<K, BitFieldVec<Box<[usize]>>, $($args),*>, Box<[W]>>
            where
                IntBitPrefix<K>: ToSig<S1>,
                $($bounds)*
                u64: PrimitiveNumberAs<W>,
            {
                #[doc = concat!("Creates a new signed ", $name, " for integers.")]
                ///
                /// Full-word hashes are stored for verification, giving a
                /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
                ///
                /// The keys must be in strictly increasing order.
                ///
                /// This is a convenience wrapper around `try_new_with_builder`
                /// (in this impl block) with `VBuilder::default()`.
                ///
                /// If keys are available as a slice, `try_par_new` (in this
                /// impl block) parallelizes construction and signing.
                ///
                /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
                ///   module provides easy ways to build such lenders.
                /// * `n` - the number of keys.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($int), "};")]
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// # use sux::utils::FromSlice;
                /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
                #[doc = concat!("type SFunc = SignedFunc<", stringify!($int), "<u64>, Box<[u16]>>;")]
                /// let func = SFunc::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_new(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
                    n: usize,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
                }

                /// Like `try_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// After building the function, the lender is rewound, and
                /// the keys are read again to compute their hashes.
                ///
                /// See also `try_par_new_with_builder` (in this impl block)
                /// for parallel construction and signing from slices.
                pub fn try_new_with_builder(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
                    n: usize,
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    let (func, keys) = $int::try_new_inner(keys, n, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = vec![W::ZERO; func.len()].into_boxed_slice();
                    sign(
                        &func,
                        keys.rewind()?,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| hashes[i] = h.as_to::<W>(),
                        pl,
                    )?;
                    Ok(Self { func, hashes })
                }

                #[doc = concat!("Creates a new signed ", $name, " for integers from a slice,")]
                /// parallelizing construction and signing, using default
                /// [`VBuilder`] settings.
                ///
                /// Full-word hashes are stored for verification, giving a
                /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
                ///
                /// The keys must be in strictly increasing order.
                ///
                /// This is a convenience wrapper around
                /// `try_par_new_with_builder` (in this impl block) with
                /// `VBuilder::default()`.
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($int), "};")]
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
                #[doc = concat!("type SFunc = SignedFunc<", stringify!($int), "<u64>, Box<[u16]>>;")]
                /// let func = SFunc::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new(
                    keys: &[K],
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
                }

                /// Like `try_par_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new_with_builder` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($int), ", VBuilder};")]
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
                #[doc = concat!("type SFunc = SignedFunc<", stringify!($int), "<u64>, Box<[u16]>>;")]
                /// let func = SFunc::try_par_new_with_builder(&keys, VBuilder::default(), no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new_with_builder(
                    keys: &[K],
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    let func = $int::try_par_new_with_builder(keys, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = vec![W::ZERO; func.len()];
                    let sync_hashes = hashes.as_sync_slice();
                    // SAFETY: the indices of the keys are distinct, so each
                    // cell is written by a single thread
                    par_sign(
                        &func,
                        keys,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| unsafe { sync_hashes[i].set(h.as_to::<W>()) },
                        pl,
                    );
                    Ok(Self {
                        func,
                        hashes: hashes.into_boxed_slice(),
                    })
                }
            }

            impl<
                K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
                W: Word,
                $($params)*
            > SignedFunc<$bytes<K, BitFieldVec<Box<[usize]>>, $($args),*>, Box<[W]>>
            where
                BitPrefix: ToSig<S1>,
                $($bounds)*
                u64: PrimitiveNumberAs<W>,
            {
                #[doc = concat!("Creates a new signed ", $name, " for byte-sequence keys.")]
                ///
                /// Full-word hashes are stored for verification, giving a
                /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
                ///
                /// The keys must be in strictly increasing lexicographic
                /// order (byte-level comparison), and must either contain no
                /// zero bytes or be prefix-free (a virtual zero byte is
                /// appended internally).
                ///
                /// This is a convenience wrapper around `try_new_with_builder`
                /// (in this impl block) with `VBuilder::default()`.
                ///
                /// If keys are available as a slice, `try_par_new` (in this
                /// impl block) parallelizes construction and signing.
                ///
                /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
                ///   module provides easy ways to build such lenders.
                /// * `n` - the number of keys.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($str), "};")]
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// # use sux::utils::FromSlice;
                /// let keys = vec!["alpha", "beta", "delta", "gamma"];
                #[doc = concat!("type SFunc = SignedFunc<", stringify!($str), ", Box<[u64]>>;")]
                /// let func = SFunc::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// assert_eq!(func.get("missing"), None);
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
                    n: usize,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
                }

                /// Like `try_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// After building the function, the lender is rewound, and
                /// the keys are read again to compute their hashes.
                ///
                /// See also `try_par_new_with_builder` (in this impl block)
                /// for parallel construction and signing from slices.
                pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
                    n: usize,
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    let (func, keys) = $bytes::try_new_inner(keys, n, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = vec![W::ZERO; func.len()].into_boxed_slice();
                    sign(
                        &func,
                        keys.rewind()?,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| hashes[i] = h.as_to::<W>(),
                        pl,
                    )?;
                    Ok(Self { func, hashes })
                }

                #[doc = concat!("Creates a new signed ", $name, " for byte-sequence keys from a")]
                /// slice, parallelizing construction and signing, using
                /// default [`VBuilder`] settings.
                ///
                /// Full-word hashes are stored for verification, giving a
                /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
                ///
                /// The keys must be in strictly increasing lexicographic
                /// order (byte-level comparison), and must either contain no
                /// zero bytes or be prefix-free (a virtual zero byte is
                /// appended internally).
                ///
                /// This is a convenience wrapper around
                /// `try_par_new_with_builder` (in this impl block) with
                /// `VBuilder::default()`.
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($str), "};")]
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys = vec!["alpha", "beta", "delta", "gamma"];
                #[doc = concat!("type SFunc = SignedFunc<", stringify!($str), ", Box<[u64]>>;")]
                /// let func = SFunc::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// assert_eq!(func.get("missing"), None);
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
                    keys: &[B],
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self>
                where
                    K: Sync,
                {
                    Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
                }

                /// Like `try_par_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new_with_builder` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($str), ", VBuilder};")]
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys = vec!["alpha", "beta", "delta", "gamma"];
                #[doc = concat!("type SFunc = SignedFunc<", stringify!($str), ", Box<[u64]>>;")]
                /// let func = SFunc::try_par_new_with_builder(&keys, VBuilder::default(), no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// assert_eq!(func.get("missing"), None);
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
                    keys: &[B],
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self>
                where
                    K: Sync,
                {
                    let func = $bytes::try_par_new_with_builder(keys, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = vec![W::ZERO; func.len()];
                    let sync_hashes = hashes.as_sync_slice();
                    // SAFETY: the indices of the keys are distinct, so each
                    // cell is written by a single thread
                    par_sign(
                        &func,
                        keys,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| unsafe { sync_hashes[i].set(h.as_to::<W>()) },
                        pl,
                    );
                    Ok(Self {
                        func,
                        hashes: hashes.into_boxed_slice(),
                    })
                }
            }

            impl<
                K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
                H: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned<Value = H>>,
                $($params)*
            > SignedFunc<$int<K, BitFieldVec<Box<[usize]>>, $($args),*>, BitFieldVec<Box<[H]>>>
            where
                IntBitPrefix<K>: ToSig<S1>,
                $($bounds)*
                u64: PrimitiveNumberAs<H>,
            {
                #[doc = concat!("Creates a new signed ", $name, " for integers with")]
                /// sub-word-width hashes.
                ///
                /// `hash_width`-bit hashes are stored for verification,
                /// giving a false-positive rate of 2<sup>−`hash_width`</sup>.
                ///
                /// The keys must be in strictly increasing order.
                ///
                /// This is a convenience wrapper around `try_new_with_builder`
                /// (in this impl block) with `VBuilder::default()`.
                ///
                /// If keys are available as a slice, `try_par_new` (in this
                /// impl block) parallelizes construction and signing.
                ///
                /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
                ///   module provides easy ways to build such lenders.
                /// * `n` - the number of keys.
                /// * `hash_width` - the number of hash bits per key
                ///   (`1..=min(H::BITS, 64)`).
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($int), "};")]
                /// # use sux::bits::BitFieldVec;
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// # use sux::utils::FromSlice;
                /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
                #[doc = concat!("type BSFunc = SignedFunc<", stringify!($int), "<u64>, BitFieldVec<Box<[usize]>>>;")]
                /// let func = BSFunc::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_new(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
                    n: usize,
                    hash_width: usize,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
                }

                /// Like `try_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// After building the function, the lender is rewound, and
                /// the keys are read again to compute their hashes.
                ///
                /// See also `try_par_new_with_builder` (in this impl block)
                /// for parallel construction and signing from slices.
                pub fn try_new_with_builder(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
                    n: usize,
                    hash_width: usize,
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    validate_hash_width::<H>(hash_width)?;
                    let (func, keys) = $int::try_new_inner(keys, n, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
                    sign(
                        &func,
                        keys.rewind()?,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| {
                            let h = hashes.truncate_hash(h);
                            hashes.set_value(i, h)
                        },
                        pl,
                    )?;
                    Ok(Self { func, hashes })
                }

                #[doc = concat!("Creates a new signed ", $name, " for integers with")]
                /// sub-word-width hashes from a slice, parallelizing
                /// construction and signing, using default [`VBuilder`]
                /// settings.
                ///
                /// `hash_width`-bit hashes are stored for verification,
                /// giving a false-positive rate of 2<sup>−`hash_width`</sup>.
                ///
                /// The keys must be in strictly increasing order.
                ///
                /// This is a convenience wrapper around
                /// `try_par_new_with_builder` (in this impl block) with
                /// `VBuilder::default()`.
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($int), "};")]
                /// # use sux::bits::BitFieldVec;
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
                #[doc = concat!("type BSFunc = SignedFunc<", stringify!($int), "<u64>, BitFieldVec<Box<[usize]>>>;")]
                /// let func = BSFunc::try_par_new(&keys, 8, no_logging![])?.try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new(
                    keys: &[K],
                    hash_width: usize,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
                }

                /// Like `try_par_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new_with_builder` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($int), ", VBuilder};")]
                /// # use sux::bits::BitFieldVec;
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
                #[doc = concat!("type BSFunc = SignedFunc<", stringify!($int), "<u64>, BitFieldVec<Box<[usize]>>>;")]
                /// let func = BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default(), no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new_with_builder(
                    keys: &[K],
                    hash_width: usize,
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    validate_hash_width::<H>(hash_width)?;
                    let func = $int::try_par_new_with_builder(keys, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
                    let mask = hashes.mask();
                    let atomic_hashes = AtomicBitFieldVec::<&mut [H::Atomic]>::try_from(
                        BitFieldVec::<&mut [H]>::from(&mut hashes),
                    )?;
                    // The indices of the keys are distinct, but the fields of
                    // different indices may share a word: we write atomically
                    par_sign(
                        &func,
                        keys,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| {
                            atomic_hashes.set_atomic(i, h.as_to::<H>() & mask, Ordering::Relaxed)
                        },
                        pl,
                    );
                    Ok(Self { func, hashes })
                }
            }

            impl<
                K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
                H: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned<Value = H>>,
                $($params)*
            > SignedFunc<$bytes<K, BitFieldVec<Box<[usize]>>, $($args),*>, BitFieldVec<Box<[H]>>>
            where
                BitPrefix: ToSig<S1>,
                $($bounds)*
                u64: PrimitiveNumberAs<H>,
            {
                #[doc = concat!("Creates a new signed ", $name, " for byte-sequence keys with")]
                /// sub-word-width hashes.
                ///
                /// `hash_width`-bit hashes are stored for verification,
                /// giving a false-positive rate of 2<sup>−`hash_width`</sup>.
                ///
                /// The keys must be in strictly increasing lexicographic
                /// order (byte-level comparison), and must either contain no
                /// zero bytes or be prefix-free (a virtual zero byte is
                /// appended internally).
                ///
                /// This is a convenience wrapper around `try_new_with_builder`
                /// (in this impl block) with `VBuilder::default()`.
                ///
                /// If keys are available as a slice, `try_par_new` (in this
                /// impl block) parallelizes construction and signing.
                ///
                /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
                ///   module provides easy ways to build such lenders.
                /// * `n` - the number of keys.
                /// * `hash_width` - the number of hash bits per key
                ///   (`1..=min(H::BITS, 64)`).
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($str), "};")]
                /// # use sux::bits::BitFieldVec;
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// # use sux::utils::FromSlice;
                /// let keys = vec!["alpha", "beta", "delta", "gamma"];
                #[doc = concat!("type BSFunc = SignedFunc<", stringify!($str), ", BitFieldVec<Box<[usize]>>>;")]
                /// let func = BSFunc::try_new(FromSlice::new(&keys), keys.len(), 12, no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
                    n: usize,
                    hash_width: usize,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
                }

                /// Like `try_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// After building the function, the lender is rewound, and
                /// the keys are read again to compute their hashes.
                ///
                /// See also `try_par_new_with_builder` (in this impl block)
                /// for parallel construction and signing from slices.
                pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
                    keys: impl FallibleRewindableLender<
                        RewindError: Error + Send + Sync + 'static,
                        Error: Error + Send + Sync + 'static,
                    > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
                    n: usize,
                    hash_width: usize,
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self> {
                    validate_hash_width::<H>(hash_width)?;
                    let (func, keys) = $bytes::try_new_inner(keys, n, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
                    sign(
                        &func,
                        keys.rewind()?,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| {
                            let h = hashes.truncate_hash(h);
                            hashes.set_value(i, h)
                        },
                        pl,
                    )?;
                    Ok(Self { func, hashes })
                }

                #[doc = concat!("Creates a new signed ", $name, " for byte-sequence keys with")]
                /// sub-word-width hashes from a slice, parallelizing
                /// construction and signing, using default [`VBuilder`]
                /// settings.
                ///
                /// `hash_width`-bit hashes are stored for verification,
                /// giving a false-positive rate of 2<sup>−`hash_width`</sup>.
                ///
                /// The keys must be in strictly increasing lexicographic
                /// order (byte-level comparison), and must either contain no
                /// zero bytes or be prefix-free (a virtual zero byte is
                /// appended internally).
                ///
                /// This is a convenience wrapper around
                /// `try_par_new_with_builder` (in this impl block) with
                /// `VBuilder::default()`.
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($str), "};")]
                /// # use sux::bits::BitFieldVec;
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys = vec!["alpha", "beta", "delta", "gamma"];
                #[doc = concat!("type BSFunc = SignedFunc<", stringify!($str), ", BitFieldVec<Box<[usize]>>>;")]
                /// let func = BSFunc::try_par_new(&keys, 12, no_logging![])?.try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
                    keys: &[B],
                    hash_width: usize,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self>
                where
                    K: Sync,
                {
                    Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
                }

                /// Like `try_par_new` (in this impl block), but uses the given
                #[doc = concat!("[`VBuilder`] to configure ", $inner, ".")]
                ///
                /// If keys are produced sequentially (e.g., from a file), use
                /// `try_new_with_builder` (in this impl block) instead.
                ///
                /// # Examples
                ///
                /// ```rust
                /// # #[cfg(feature = "rayon")]
                /// # fn main() -> anyhow::Result<()> {
                #[doc = concat!("# use sux::func::{SignedFunc, ", stringify!($str), ", VBuilder};")]
                /// # use sux::bits::BitFieldVec;
                /// # use sux::traits::TryIntoUnaligned;
                /// # use dsi_progress_logger::no_logging;
                /// let keys = vec!["alpha", "beta", "delta", "gamma"];
                #[doc = concat!("type BSFunc = SignedFunc<", stringify!($str), ", BitFieldVec<Box<[usize]>>>;")]
                /// let func = BSFunc::try_par_new_with_builder(&keys, 12, VBuilder::default(), no_logging![])?
                ///     .try_into_unaligned()?;
                ///
                /// for (i, &key) in keys.iter().enumerate() {
                ///     assert_eq!(func.get(key), Some(i));
                /// }
                /// # Ok(())
                /// # }
                /// # #[cfg(not(feature = "rayon"))]
                /// # fn main() {}
                /// ```
                pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
                    keys: &[B],
                    hash_width: usize,
                    builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
                    pl: &mut (impl ProgressLog + Clone + Send + Sync),
                ) -> Result<Self>
                where
                    K: Sync,
                {
                    validate_hash_width::<H>(hash_width)?;
                    let func = $bytes::try_par_new_with_builder(keys, builder, pl)?;
                    let seed = func.seed();
                    let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
                    let mask = hashes.mask();
                    let atomic_hashes = AtomicBitFieldVec::<&mut [H::Atomic]>::try_from(
                        BitFieldVec::<&mut [H]>::from(&mut hashes),
                    )?;
                    // The indices of the keys are distinct, but the fields of
                    // different indices may share a word: we write atomically
                    par_sign(
                        &func,
                        keys,
                        |i, key: &K| (i, K::to_sig(key, seed)),
                        |i, h| {
                            atomic_hashes.set_atomic(i, h.as_to::<H>() & mask, Ordering::Relaxed)
                        },
                        pl,
                    );
                    Ok(Self { func, hashes })
                }
            }
        };
    }

    impl_signed_lcp!(
        LcpMmphfInt, LcpMmphf, LcpMmphfStr,
        name: "LCP-based MMPHF",
        inner: "the internal `lcp_len_offset` VFunc",
        params: [
            S0: Sig + Send + Sync,
            E0: ShardEdge<S0, 3> + MemSize + FlatType,
            S1: Sig + Send + Sync,
            E1: ShardEdge<S1, 3> + MemSize + FlatType,
        ],
        args: [S0, E0, S1, E1],
        bounds: [
            SigVal<S0, usize>: RadixKey,
            SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
            SigVal<S1, usize>: RadixKey,
            SigVal<E1::LocalSig, usize>: BitXor + BitXorAssign,
        ]
    );

    impl_signed_lcp!(
        Lcp2MmphfInt, Lcp2Mmphf, Lcp2MmphfStr,
        name: "two-step LCP-based MMPHF",
        inner: "the internal VFuncs",
        params: [
            S0: Sig + Send + Sync,
            E0: ShardEdge<S0, 3> + MemSize + FlatType,
            F0: ShardEdge<S0, 3> + MemSize + FlatType,
            S1: Sig + Send + Sync,
            E1: ShardEdge<S1, 3> + MemSize + FlatType,
        ],
        args: [S0, E0, F0, S1, E1],
        bounds: [
            SigVal<S0, usize>: RadixKey,
            SigVal<S0, u64>: RadixKey,
            SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
            SigVal<E0::LocalSig, u64>: BitXor + BitXorAssign,
            SigVal<F0::LocalSig, usize>: BitXor + BitXorAssign,
            SigVal<F0::LocalSig, u64>: BitXor + BitXorAssign,
            SigVal<S1, usize>: RadixKey,
            SigVal<E1::LocalSig, usize>: BitXor + BitXorAssign,
        ]
    );

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<PHastR<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<K: ?Sized + ToSig<[u64; 1]>, D: SeedStoreBuild, W: Word> SignedFunc<PHastR<K, D>, Box<[W]>>
    where
        u64: PrimitiveNumberAs<W>,
    {
        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from keys using
        /// default [`PHastRBuilder`] settings.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// This is a convenience wrapper around `try_new_with_builder` (in
        /// this impl block) with `PHastRBuilder::default()`.
        ///
        /// If keys are available as a slice, `try_par_new` (in this impl
        /// block) parallelizes construction and signing.
        ///
        /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
        ///   module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// use sux::func::{PHastR, SignedFunc};
        /// type SFunc = SignedFunc<PHastR<usize>, Box<[u16]>>;
        /// let func = SFunc::try_new(FromCloneableIntoIterator::new(0..100_usize), no_logging![])?;
        ///
        /// let mut seen = vec![false; 100];
        /// for i in 0..100 {
        ///     let v = func.get(i).unwrap();
        ///     assert!(!seen[v]);
        ///     seen[v] = true;
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`lenders`]: crate::utils::lenders
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, PHastRBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from keys using the
        /// given [`PHastRBuilder`] configuration.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// After building the function, the lender is rewound, and the keys
        /// are read again to compute their hashes. The builder controls
        /// construction parameters such as [offline mode].
        ///
        /// See also `try_par_new_with_builder` (in this impl block) for
        /// parallel construction and signing from slices.
        ///
        /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
        ///   module provides easy ways to build such lenders.
        ///
        /// [offline mode]: PHastRBuilder::offline
        /// [`lenders`]: crate::utils::lenders
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            mut keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            let func = <PHastR<K, D>>::try_new_with_builder(&mut keys, builder, pl)?;
            let seed = func.seed;
            let mut hashes = vec![W::ZERO; func.len()].into_boxed_slice();
            sign(
                &func,
                keys.rewind()?,
                |_, key: &K| {
                    let sig = K::to_sig(key, seed);
                    (func.get_by_sig(key, sig), sig)
                },
                |i, h| hashes[i] = h.as_to::<W>(),
                pl,
            )?;
            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// default [`PHastRBuilder`] settings.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// This is a convenience wrapper around `try_par_new_with_builder`
        /// (in this impl block) with `PHastRBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// `try_new` (in this impl block) instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// use sux::func::{PHastR, SignedFunc};
        /// type SFunc = SignedFunc<PHastR<usize>, Box<[u16]>>;
        /// let keys: Vec<usize> = (0..100).collect();
        /// let func = SFunc::try_par_new(&keys, no_logging![])?;
        ///
        /// let mut seen = vec![false; 100];
        /// for i in 0..100 {
        ///     let v = func.get(i).unwrap();
        ///     assert!(!seen[v]);
        ///     seen[v] = true;
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
            D: Sync,
        {
            Self::try_par_new_with_builder(keys, PHastRBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// the given [`PHastRBuilder`] configuration.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// `try_new_with_builder` (in this impl block) instead.
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
            D: Sync,
        {
            let func = <PHastR<K, D>>::try_par_new_with_builder(keys, builder, pl)?;
            let seed = func.seed;
            let mut hashes = vec![W::ZERO; func.len()];
            let sync_hashes = hashes.as_sync_slice();
            // SAFETY: the outputs of the keys are distinct, so each cell is
            // written by a single thread
            par_sign(
                &func,
                keys,
                |_, key: &K| {
                    let sig = K::to_sig(key, seed);
                    (func.get_by_sig(key, sig), sig)
                },
                |i, h| unsafe { sync_hashes[i].set(h.as_to::<W>()) },
                pl,
            );
            Ok(SignedFunc {
                func,
                hashes: hashes.into_boxed_slice(),
            })
        }
    }

    impl<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStoreBuild,
        H: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned<Value = H>>,
    > SignedFunc<PHastR<K, D>, BitFieldVec<Box<[H]>>>
    where
        u64: PrimitiveNumberAs<H>,
    {
        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from keys using
        /// default [`PHastRBuilder`] settings.
        ///
        /// `hash_width`-bit hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around `try_new_with_builder` (in
        /// this impl block) with `PHastRBuilder::default()`.
        ///
        /// If keys are available as a slice, `try_par_new` (in this impl
        /// block) parallelizes construction and signing.
        ///
        /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
        ///   module provides easy ways to build such lenders.
        /// * `hash_width` - the number of hash bits per key
        ///   (`1..=min(H::BITS, 64)`).
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// use sux::bits::BitFieldVec;
        /// use sux::func::{PHastR, SignedFunc};
        /// type BSFunc = SignedFunc<PHastR<usize>, BitFieldVec<Box<[usize]>>>;
        /// let func = BSFunc::try_new(FromCloneableIntoIterator::new(0..100_usize), 12, no_logging![])?;
        ///
        /// let mut seen = vec![false; 100];
        /// for i in 0..100 {
        ///     let v = func.get(i).unwrap();
        ///     assert!(!seen[v]);
        ///     seen[v] = true;
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`lenders`]: crate::utils::lenders
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            hash_width: usize,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, hash_width, PHastRBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from keys using the
        /// given [`PHastRBuilder`] configuration.
        ///
        /// `hash_width`-bit hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// After building the function, the lender is rewound, and the keys
        /// are read again to compute their hashes. The builder controls
        /// construction parameters such as [offline mode].
        ///
        /// See also `try_par_new_with_builder` (in this impl block) for
        /// parallel construction and signing from slices.
        ///
        /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
        ///   module provides easy ways to build such lenders.
        /// * `hash_width` - the number of hash bits per key
        ///   (`1..=min(H::BITS, 64)`).
        ///
        /// [offline mode]: PHastRBuilder::offline
        /// [`lenders`]: crate::utils::lenders
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            mut keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            hash_width: usize,
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            validate_hash_width::<H>(hash_width)?;
            let func = <PHastR<K, D>>::try_new_with_builder(&mut keys, builder, pl)?;
            let seed = func.seed;
            let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
            sign(
                &func,
                keys.rewind()?,
                |_, key: &K| {
                    let sig = K::to_sig(key, seed);
                    (func.get_by_sig(key, sig), sig)
                },
                |i, h| {
                    let h = hashes.truncate_hash(h);
                    hashes.set_value(i, h)
                },
                pl,
            )?;
            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// default [`PHastRBuilder`] settings.
        ///
        /// `hash_width`-bit hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around `try_par_new_with_builder`
        /// (in this impl block) with `PHastRBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// `try_new` (in this impl block) instead.
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
            D: Sync,
        {
            Self::try_par_new_with_builder(keys, hash_width, PHastRBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// the given [`PHastRBuilder`] configuration.
        ///
        /// `hash_width`-bit hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// `try_new_with_builder` (in this impl block) instead.
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
            D: Sync,
        {
            validate_hash_width::<H>(hash_width)?;
            let func = <PHastR<K, D>>::try_par_new_with_builder(keys, builder, pl)?;
            let seed = func.seed;
            let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
            let mask = hashes.mask();
            let atomic_hashes =
                AtomicBitFieldVec::<&mut [H::Atomic]>::try_from(BitFieldVec::<&mut [H]>::from(
                    &mut hashes,
                ))?;
            // The outputs of the keys are distinct, but the fields of
            // different outputs may share a word: we write atomically
            par_sign(
                &func,
                keys,
                |_, key: &K| {
                    let sig = K::to_sig(key, seed);
                    (func.get_by_sig(key, sig), sig)
                },
                |i, h| atomic_hashes.set_atomic(i, h.as_to::<H>() & mask, Ordering::Relaxed),
                pl,
            );
            Ok(SignedFunc { func, hashes })
        }
    }
} // mod build

// ═══════════════════════════════════════════════════════════════════
// Aligned <-> Unaligned conversions
// ═══════════════════════════════════════════════════════════════════

use crate::traits::{TryIntoUnaligned, Unaligned, Word};

impl<F: TryIntoUnaligned, H: TryIntoUnaligned> TryIntoUnaligned for SignedFunc<F, H> {
    type Unaligned = SignedFunc<F::Unaligned, H::Unaligned>;
    fn try_into_unaligned(
        self,
    ) -> std::result::Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(SignedFunc {
            func: self.func.try_into_unaligned()?,
            hashes: self.hashes.try_into_unaligned()?,
        })
    }
}

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, H: TryIntoUnaligned>
    From<Unaligned<SignedFunc<VFunc<K, BitFieldVec<Box<[W]>>, S, E>, H>>>
    for SignedFunc<VFunc<K, BitFieldVec<Box<[W]>>, S, E>, H>
where
    H: From<H::Unaligned>,
{
    fn from(f: Unaligned<SignedFunc<VFunc<K, BitFieldVec<Box<[W]>>, S, E>, H>>) -> Self {
        SignedFunc {
            func: f.func.into(),
            hashes: f.hashes.into(),
        }
    }
}

impl<
    K: ?Sized,
    D: crate::func::phast_r::SeedStoreBuild + TryIntoUnaligned + From<Unaligned<D>>,
    P,
    H: TryIntoUnaligned,
> From<Unaligned<SignedFunc<PHastR<K, D, P, crate::func::phast_r::Remap>, H>>>
    for SignedFunc<PHastR<K, D, P, crate::func::phast_r::Remap>, H>
where
    H: From<H::Unaligned>,
{
    fn from(f: Unaligned<SignedFunc<PHastR<K, D, P, crate::func::phast_r::Remap>, H>>) -> Self {
        SignedFunc {
            func: f.func.into(),
            hashes: f.hashes.into(),
        }
    }
}
