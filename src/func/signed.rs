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
//! `SignedFunc<PHastR<str>, Box<[u16]>>`. For each such concrete type, this
//! module provides `try_new`/`try_par_new` constructors (and their
//! `_with_builder` variants), plus `get` and (except for [`PHastR`])
//! `get_by_sig` methods;
//! [`SignedFunc::from_parts`] assembles an instance from an inner function
//! and a hash store. All `try_*` constructors require the `rayon` feature.

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
        crate::func::mix64(sig[0])
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
/// stays 64-bit either way). For a [`PHastR`], the verification hash is
/// `mix64` of the 64-bit signature that determines the output of the keys of
/// the first level, so the same irreducible false positive happens for keys
/// sharing the signature.
///
/// Instances of this structure are immutable; they are built using [`try_new`]
/// or one of its variants, and can be serialized using [ε-serde] or [`serde`].
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Examples
///
/// See the various [`try_new`] implementations.
///
/// [`try_new`]: SignedFunc::try_new
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
// (type aliases section removed: use SignedFunc<LcpMmphfStr, Box<[u64]>> etc. directly)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::func::VBuilder;
    use crate::func::phast_r::{PHastRBuilder, SeedStoreBuild};
    use crate::utils::FallibleRewindableLender;
    use anyhow::Result;
    use atomic_primitive::{AtomicPrimitive, PrimitiveAtomicUnsigned};
    use core::error::Error;
    use dsi_progress_logger::ProgressLog;
    use lender::*;
    use rdst::RadixKey;
    use std::ops::{BitXor, BitXorAssign};
    use value_traits::slices::SliceByValueMut;

    /// Fills a `Box<[W]>` hash array from a key lender.
    ///
    /// Iterates the first `n` elements of `keys`, converting each borrowed
    /// key to a signature via the `to_sig` closure and then computing the
    /// [verification hash](SignableFunc::remixed_hash) through `func`. The
    /// resulting full-width hashes are stored in a newly allocated boxed
    /// slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the lender yields fewer than `n` elements.
    fn fill_hashes<W, S, F, L>(
        func: &F,
        seed: u64,
        n: usize,
        mut keys: L,
        to_sig: impl Fn(&<L as FallibleLending<'_>>::Lend, u64) -> S,
        pl: &mut impl ProgressLog,
    ) -> Result<Box<[W]>>
    where
        W: Word,
        S: Sig,
        F: SignableFunc<Sig = S>,
        u64: PrimitiveNumberAs<W>,
        L: FallibleLender<Error: std::error::Error + Send + Sync + 'static>,
        for<'lend> L: FallibleLending<'lend>,
    {
        pl.item_name("key");
        pl.expected_updates(Some(n));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        let mut hashes = vec![W::MIN; n];
        for (index, hash) in hashes.iter_mut().enumerate() {
            let key = keys.next()?.ok_or_else(|| {
                anyhow::anyhow!("expected {n} keys while signing, but got {index}")
            })?;
            *hash = func.remixed_hash(to_sig(&key, seed)).as_to::<W>();
            pl.light_update();
        }
        pl.done();
        Ok(hashes.into_boxed_slice())
    }

    /// Fills a [`BitFieldVec<Box<[H]>>`] hash array from a key
    /// lender.
    ///
    /// Iterates the first `n` elements of `keys`, converting each borrowed
    /// key to a signature via the `to_sig` closure and then computing the
    /// [verification hash](SignableFunc::remixed_hash) through `func`. The
    /// hash is truncated using the [`BitFieldVec`]'s stored mask and stored
    /// in a [`BitFieldVec`] of the given `hash_width`.
    ///
    /// # Errors
    ///
    /// Returns an error if the lender yields fewer than `n` elements.
    ///
    /// [`BitFieldVec<Box<[H]>>`]: BitFieldVec
    fn fill_bit_hashes<H, S, F, L>(
        func: &F,
        seed: u64,
        n: usize,
        hash_width: usize,
        mut keys: L,
        to_sig: impl Fn(&<L as FallibleLending<'_>>::Lend, u64) -> S,
        pl: &mut impl ProgressLog,
    ) -> Result<BitFieldVec<Box<[H]>>>
    where
        H: Word,
        S: Sig,
        F: SignableFunc<Sig = S>,
        u64: PrimitiveNumberAs<H>,
        L: FallibleLender<Error: std::error::Error + Send + Sync + 'static>,
        for<'lend> L: FallibleLending<'lend>,
    {
        pl.item_name("key");
        pl.expected_updates(Some(n));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, n);
        for index in 0..n {
            let key = keys.next()?.ok_or_else(|| {
                anyhow::anyhow!("expected {n} keys while signing, but got {index}")
            })?;
            let h = hashes.truncate_hash(func.remixed_hash(to_sig(&key, seed)));
            hashes.set_value(index, h);
            pl.light_update();
        }
        pl.done();
        Ok(hashes)
    }

    /// Fills a `Box<[W]>` hash array from a key slice.
    ///
    /// Iterates the first `n` elements of `keys`, converting each key to a
    /// signature via [`ToSig::to_sig`] and then computing the [verification
    /// hash](SignableFunc::remixed_hash) through `func`. The resulting
    /// full-width hashes are stored in a newly allocated boxed slice.
    ///
    /// # Panics
    ///
    /// Panics if `keys.len() < n`.
    fn fill_hashes_from_slice<B, K, W, S, F>(
        func: &F,
        seed: u64,
        n: usize,
        keys: &[B],
        pl: &mut (impl ProgressLog + Send),
    ) -> Box<[W]>
    where
        B: Borrow<K> + Sync,
        K: ?Sized + ToSig<S>,
        W: Word + Send + Sync,
        S: Sig,
        F: SignableFunc<Sig = S> + Sync,
        u64: PrimitiveNumberAs<W>,
    {
        use rayon::prelude::*;
        use sync_cell_slice::SyncSlice;
        pl.item_name("key");
        pl.expected_updates(Some(n));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        let mut hashes = vec![W::MIN; n];
        let sync_hashes = hashes.as_sync_slice();
        keys[..n]
            .par_iter()
            .enumerate()
            .for_each(|(i, key)| unsafe {
                sync_hashes[i].set(
                    func.remixed_hash(K::to_sig(key.borrow(), seed))
                        .as_to::<W>(),
                );
            });
        pl.update_with_count(n);
        pl.done();
        hashes.into_boxed_slice()
    }

    /// Fills a [`BitFieldVec<Box<[H]>>`] hash array from a key
    /// slice.
    ///
    /// Iterates the first `n` elements of `keys`, converting each key to a
    /// signature via [`ToSig::to_sig`] and then computing the [verification
    /// hash](SignableFunc::remixed_hash) through `func`. The hash is
    /// truncated using the [`BitFieldVec`]'s stored mask and stored in a
    /// [`BitFieldVec`] of the given `hash_width`.
    ///
    /// # Panics
    ///
    /// Panics if `keys.len() < n`.
    ///
    /// [`BitFieldVec<Box<[H]>>`]: BitFieldVec
    fn fill_bit_hashes_from_slice<B, K, H, S, F>(
        func: &F,
        seed: u64,
        n: usize,
        hash_width: usize,
        keys: &[B],
        pl: &mut impl ProgressLog,
    ) -> BitFieldVec<Box<[H]>>
    where
        B: Borrow<K>,
        K: ?Sized + ToSig<S>,
        H: Word,
        S: Sig,
        F: SignableFunc<Sig = S>,
        u64: PrimitiveNumberAs<H>,
    {
        pl.item_name("key");
        pl.expected_updates(Some(n));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, n);
        for (i, key) in keys[..n].iter().enumerate() {
            let h = hashes.truncate_hash(func.remixed_hash(K::to_sig(key.borrow(), seed)));
            hashes.set_value(i, h);
            pl.light_update();
        }
        pl.done();
        hashes
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<VFunc<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3> + MemSize + FlatType,
        H: crate::traits::Word,
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
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
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

            let num_keys = func.len();

            pl.push_log_target(" ▸ hashes");
            let mut hashes = vec![H::ZERO; num_keys].into_boxed_slice();
            {
                use rayon::prelude::*;
                use sync_cell_slice::SyncSlice;
                let shards: Vec<_> = store.iter().collect();
                let num_shards = shards.len();
                let mut shard_pl = pl.clone();
                shard_pl.item_name("shard");
                shard_pl.expected_updates(Some(shards.len()));
                shard_pl.log_level(log::Level::Info);
                shard_pl.start("Signing shards...");
                let sync_hashes = hashes.as_sync_slice();
                let shard_edge = &func.shard_edge;
                shards.into_par_iter().for_each(|shard| {
                    for sig_val in shard.iter() {
                        unsafe {
                            let hash = H::as_from(shard_edge.remixed_hash(sig_val.sig));
                            sync_hashes[sig_val.val].set(hash);
                        }
                    }
                });
                shard_pl.update_with_count(num_shards);
                shard_pl.done();
            }
            pl.pop_log_target();

            Ok(SignedFunc { func, hashes })
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
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_hashes_from_slice(&func, func.seed(), func.len(), keys, pl);
            pl.pop_log_target();
            Ok(SignedFunc { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<VFunc<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3> + MemSize + FlatType,
        H: crate::traits::Word,
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
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// inner-function construction.
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
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
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
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// See also [`try_par_new_with_builder`] for parallel inner-function
        /// construction from slices.
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
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
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

            let num_keys = func.len();

            pl.push_log_target(" ▸ hashes");
            let num_shards = 1usize << func.shard_edge.shard_high_bits();
            pl.item_name("shard");
            pl.expected_updates(Some(num_shards));
            pl.log_level(log::Level::Info);
            pl.start("Signing shards...");
            let mut hashes: BitFieldVec<Box<[H]>> =
                BitFieldVec::<Box<[H]>>::new_padded(hash_width, num_keys);
            for shard in store.iter() {
                for sig_val in shard.iter() {
                    let pos = sig_val.val;
                    let hash = hashes.truncate_hash(func.shard_edge.remixed_hash(sig_val.sig));
                    hashes.set_value(pos, hash);
                }
                pl.update();
            }
            pl.done();
            pl.pop_log_target();

            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing inner-function construction with rayon and
        /// using default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
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
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
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
        /// slices, parallelizing inner-function construction with rayon and
        /// using the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
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
        /// [`try_new_with_builder`]: Self::try_new_with_builder
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
            pl.push_log_target(" ▸ hashes");
            let hashes =
                fill_bit_hashes_from_slice(&func, func.seed(), func.len(), hash_width, keys, pl);
            pl.pop_log_target();
            Ok(SignedFunc { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<LcpMmphfInt<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>, Box<[W]>>
    where
        IntBitPrefix<K>: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<W>,
    {
        /// Creates a new signed LCP-based MMPHF for integers.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// The keys must be in strictly increasing order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfInt};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<LcpMmphfInt<u64>, Box<[u16]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal `lcp_len_offset` VFunc.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let (func, keys) = LcpMmphfInt::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_hashes(
                &func,
                func.seed(),
                n,
                &mut keys,
                |key, seed| K::to_sig(*key, seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for integers from a slice,
        /// using parallel hash computation and default [`VBuilder`] settings.
        ///
        /// The keys must be in strictly increasing order.
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
        /// # use sux::func::{SignedFunc, LcpMmphfInt};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<LcpMmphfInt<u64>, Box<[u16]>>>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new(
            keys: &[K],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal `lcp_len_offset` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfInt, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<LcpMmphfInt<u64>, Box<[u16]>>>::try_par_new_with_builder(
        ///         &keys, VBuilder::default(), no_logging![],
        ///     )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let func = LcpMmphfInt::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes =
                fill_hashes_from_slice::<K, K, W, S0, _>(&func, func.seed(), keys.len(), keys, pl);
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<LcpMmphf<K, ...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>, Box<[W]>>
    where
        BitPrefix: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<W>,
    {
        /// Creates a new signed LCP-based MMPHF for byte-sequence keys.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// The keys must be in strictly increasing lexicographic order
        /// (byte-level comparison), and must either contain no zero bytes or be
        /// prefix-free (a virtual zero byte is appended internally).
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfStr};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <SignedFunc<LcpMmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal `lcp_len_offset` VFunc.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let (func, keys) = LcpMmphf::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_hashes(
                &func,
                func.seed(),
                n,
                &mut keys,
                |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for byte-sequence keys from
        /// a slice, using parallel hash computation and default [`VBuilder`]
        /// settings.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
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
        /// # use sux::func::{SignedFunc, LcpMmphfStr};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <SignedFunc<LcpMmphfStr, Box<[u64]>>>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal `lcp_len_offset` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfStr, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <SignedFunc<LcpMmphfStr, Box<[u64]>>>::try_par_new_with_builder(
        ///         &keys, VBuilder::default(), no_logging![],
        ///     )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            let func = LcpMmphf::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes =
                fill_hashes_from_slice::<B, K, W, S0, _>(&func, func.seed(), keys.len(), keys, pl);
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<Lcp2MmphfInt<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        F0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>, Box<[W]>>
    where
        IntBitPrefix<K>: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<E0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<W>,
    {
        /// Creates a new signed two-step LCP-based MMPHF for integers.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// The keys must be in strictly increasing order.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfInt};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfInt<u64>, Box<[u16]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let (func, keys) = Lcp2MmphfInt::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_hashes(
                &func,
                func.seed(),
                n,
                &mut keys,
                |key, seed| K::to_sig(*key, seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for integers from
        /// a slice, using parallel hash computation and default [`VBuilder`]
        /// settings.
        ///
        /// The keys must be in strictly increasing order.
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
        /// # use sux::func::{SignedFunc, Lcp2MmphfInt};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfInt<u64>, Box<[u16]>>>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new(
            keys: &[K],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfInt, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfInt<u64>, Box<[u16]>>>::try_par_new_with_builder(
        ///         &keys, VBuilder::default(), no_logging![],
        ///     )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let func = Lcp2MmphfInt::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes =
                fill_hashes_from_slice::<K, K, W, S0, _>(&func, func.seed(), keys.len(), keys, pl);
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<Lcp2Mmphf<K, ...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        F0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>, Box<[W]>>
    where
        BitPrefix: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<E0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<W>,
    {
        /// Creates a new signed two-step LCP-based MMPHF for byte-sequence keys.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfStr};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let (func, keys) = Lcp2Mmphf::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_hashes(
                &func,
                func.seed(),
                n,
                &mut keys,
                |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for byte-sequence
        /// keys from a slice, using parallel hash computation and default
        /// [`VBuilder`] settings.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
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
        /// # use sux::func::{SignedFunc, Lcp2MmphfStr};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfStr, Box<[u64]>>>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfStr, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfStr, Box<[u64]>>>::try_par_new_with_builder(
        ///         &keys, VBuilder::default(), no_logging![],
        ///     )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            let func = Lcp2Mmphf::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes =
                fill_hashes_from_slice::<B, K, W, S0, _>(&func, func.seed(), keys.len(), keys, pl);
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<LcpMmphfInt<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>, BitFieldVec<Box<[H]>>>
    where
        IntBitPrefix<K>: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<H>,
    {
        /// Creates a new signed LCP-based MMPHF for integers with
        /// sub-word-width hashes.
        ///
        /// `hash_width` is the number of hash bits stored per key (must be in
        /// `1..=min(H::BITS, 64)`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// inner-function construction.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfInt};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// type BSFunc = SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal `lcp_len_offset` VFunc.
        ///
        /// See also [`try_par_new_with_builder`] for parallel inner-function
        /// construction from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let (func, keys) = LcpMmphfInt::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes(
                &func,
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(*key, seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for integers with
        /// sub-word-width hashes from a slice, using parallel inner-function
        /// construction and default [`VBuilder`] settings.
        ///
        /// `hash_width` is the number of hash bits stored per key (must be in
        /// `1..=min(H::BITS, 64)`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
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
        /// # use sux::func::{SignedFunc, LcpMmphfInt};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// type BSFunc = SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new(&keys, 8, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new(
            keys: &[K],
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal `lcp_len_offset` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfInt, VBuilder};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// type BSFunc = SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder(
            keys: &[K],
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let func = LcpMmphfInt::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes_from_slice::<K, K, H, S0, _>(
                &func,
                func.seed(),
                keys.len(),
                hash_width,
                keys,
                pl,
            );
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<LcpMmphf<K, ...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>, BitFieldVec<Box<[H]>>>
    where
        BitPrefix: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<H>,
    {
        /// Creates a new signed LCP-based MMPHF for byte-sequence keys with
        /// sub-word-width hashes.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
        ///
        /// `hash_width` is the number of hash bits stored per key (must be in
        /// `1..=min(H::BITS, 64)`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// inner-function construction.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfStr};
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// # use sux::traits::TryIntoUnaligned;
        /// let keys = vec!["alpha", "beta", "delta", "gamma"];
        /// type BSFunc = SignedFunc<LcpMmphfStr, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_new(FromSlice::new(&keys), keys.len(), 12, no_logging![])?.try_into_unaligned()?;
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
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal `lcp_len_offset` VFunc.
        ///
        /// See also [`try_par_new_with_builder`] for parallel inner-function
        /// construction from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let (func, keys) = LcpMmphf::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes(
                &func,
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for byte-sequence keys with
        /// sub-word-width hashes from a slice, using parallel inner-function
        /// construction and default [`VBuilder`] settings.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
        ///
        /// `hash_width` is the number of hash bits stored per key (must be in
        /// `1..=min(H::BITS, 64)`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
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
        /// # use sux::func::{SignedFunc, LcpMmphfStr};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["alpha", "beta", "delta", "gamma"];
        /// type BSFunc = SignedFunc<LcpMmphfStr, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new(&keys, 12, no_logging![])?.try_into_unaligned()?;
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
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
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

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal `lcp_len_offset` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, LcpMmphfStr, VBuilder};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["alpha", "beta", "delta", "gamma"];
        /// type BSFunc = SignedFunc<LcpMmphfStr, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new_with_builder(&keys, 12, VBuilder::default(), no_logging![])?.try_into_unaligned()?;
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
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let func = LcpMmphf::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes_from_slice::<B, K, H, S0, _>(
                &func,
                func.seed(),
                keys.len(),
                hash_width,
                keys,
                pl,
            );
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<Lcp2MmphfInt<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        F0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    >
        SignedFunc<
            Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>,
            BitFieldVec<Box<[H]>>,
        >
    where
        IntBitPrefix<K>: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<E0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<H>,
    {
        /// Creates a new signed two-step LCP-based MMPHF for integers with
        /// sub-word-width hashes.
        ///
        /// `hash_width` is the number of hash bits stored per key (must be in
        /// `1..=min(H::BITS, 64)`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// inner-function construction.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfInt};
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// # use sux::traits::TryIntoUnaligned;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfInt<u64>, BitFieldVec<Box<[usize]>>>>::try_new(
        ///         FromSlice::new(&keys), 5, 8, no_logging![])?.try_into_unaligned()?;
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`] for parallel inner-function
        /// construction from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let (func, keys) = Lcp2MmphfInt::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes(
                &func,
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(*key, seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for integers with
        /// sub-word-width hashes from a slice, using parallel inner-function
        /// construction and default [`VBuilder`] settings.
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
        /// # use sux::func::{SignedFunc, Lcp2MmphfInt};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// type BSFunc = SignedFunc<Lcp2MmphfInt<u64>, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new(&keys, 8, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new(
            keys: &[K],
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfInt, VBuilder};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// type BSFunc = SignedFunc<Lcp2MmphfInt<u64>, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder(
            keys: &[K],
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let func = Lcp2MmphfInt::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes_from_slice::<K, K, H, S0, _>(
                &func,
                func.seed(),
                keys.len(),
                hash_width,
                keys,
                pl,
            );
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<Lcp2Mmphf<K, ...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + FlatType,
        F0: ShardEdge<S0, 3> + MemSize + FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + FlatType,
    > SignedFunc<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>, BitFieldVec<Box<[H]>>>
    where
        BitPrefix: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<E0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<F0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
        u64: PrimitiveNumberAs<H>,
    {
        /// Creates a new signed two-step LCP-based MMPHF for byte-sequence keys
        /// with sub-word-width hashes.
        ///
        /// `hash_width` is the number of hash bits stored per key (must be in
        /// `1..=min(H::BITS, 64)`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`] with `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// inner-function construction.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfStr};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["alpha", "beta", "delta", "gamma"];
        /// let func =
        ///     <SignedFunc<Lcp2MmphfStr, BitFieldVec<Box<[usize]>>>>::try_new(
        ///         FromSlice::new(&keys), 4, 8, no_logging![])?.try_into_unaligned()?;
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_new`], but uses the given [`VBuilder`] to configure
        /// the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`] for parallel inner-function
        /// construction from slices.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let (func, keys) = Lcp2Mmphf::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes(
                &func,
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
                pl,
            )?;
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for byte-sequence
        /// keys with sub-word-width hashes from a slice, using parallel
        /// inner-function construction and default [`VBuilder`] settings.
        ///
        /// The keys must be in strictly increasing lexicographic order, and must
        /// either contain no zero bytes or be prefix-free (a virtual zero byte is
        /// appended internally).
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
        /// # use sux::func::{SignedFunc, Lcp2MmphfStr};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// type BSFunc = SignedFunc<Lcp2MmphfStr, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new(&keys, 8, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
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

        /// Like [`try_par_new`], but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{SignedFunc, Lcp2MmphfStr, VBuilder};
        /// # use sux::bits::BitFieldVec;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// type BSFunc = SignedFunc<Lcp2MmphfStr, BitFieldVec<Box<[usize]>>>;
        /// let func =
        ///     BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), Some(i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_par_new`]: Self::try_par_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            // Verification hashes come from a 64-bit remixed hash.
            validate_hash_width::<H>(hash_width)?;
            let func = Lcp2Mmphf::try_par_new_inner(keys, builder, pl)?;
            pl.push_log_target(" ▸ hashes");
            let hashes = fill_bit_hashes_from_slice::<B, K, H, S0, _>(
                &func,
                func.seed(),
                keys.len(),
                hash_width,
                keys,
                pl,
            );
            pl.pop_log_target();
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: SignedFunc<PHastR<...>>
    // ═══════════════════════════════════════════════════════════════════

    /// Signs the keys of a lender for a [`PHastR`], passing to `set` the
    /// output of each key and the verification hash of its signature.
    ///
    /// # Errors
    ///
    /// Returns an error if the lender yields fewer keys than the function
    /// contains.
    fn sign_phast_r<K, D, B, L>(
        func: &PHastR<K, D>,
        mut keys: L,
        mut set: impl FnMut(usize, u64),
        pl: &mut impl ProgressLog,
    ) -> Result<()>
    where
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStore,
        B: ?Sized + Borrow<K>,
        L: FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    {
        let n = func.len();
        pl.item_name("key");
        pl.expected_updates(Some(n));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        for index in 0..n {
            let key = keys.next()?.ok_or_else(|| {
                anyhow::anyhow!("expected {n} keys while signing, but got {index}")
            })?;
            let key: &K = key.borrow();
            let sig = K::to_sig(key, func.seed());
            set(func.get_by_sig(key, sig), func.remixed_hash(sig));
            pl.light_update();
        }
        pl.done();
        Ok(())
    }

    /// Signs the keys of a slice for a [`PHastR`] in parallel, passing to
    /// `set` the output of each key and the verification hash of its
    /// signature.
    ///
    /// The outputs of the keys are distinct, so `set` can write them
    /// concurrently.
    fn par_sign_phast_r<K, D, B>(
        func: &PHastR<K, D>,
        keys: &[B],
        set: impl Fn(usize, u64) + Sync,
        pl: &mut impl ProgressLog,
    ) where
        K: ?Sized + ToSig<[u64; 1]> + Sync,
        D: SeedStore + Sync,
        B: Borrow<K> + Sync,
    {
        use rayon::prelude::*;
        pl.item_name("key");
        pl.expected_updates(Some(keys.len()));
        pl.log_level(log::Level::Info);
        pl.start("Signing keys...");
        keys.par_iter().for_each(|key| {
            let key: &K = key.borrow();
            let sig = K::to_sig(key, func.seed());
            set(func.get_by_sig(key, sig), func.remixed_hash(sig));
        });
        pl.update_with_count(keys.len());
        pl.done();
    }

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
        /// This is a convenience wrapper around [`try_new_with_builder`]
        /// with `PHastRBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the construction.
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
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
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
        /// See also [`try_par_new_with_builder`] for parallel construction
        /// from slices.
        ///
        /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
        ///   module provides easy ways to build such lenders.
        ///
        /// [offline mode]: PHastRBuilder::offline
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
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
            let keys = keys.rewind()?;
            let mut hashes = vec![W::ZERO; func.len()].into_boxed_slice();
            pl.push_log_target(" ▸ hashes");
            sign_phast_r(&func, keys, |i, h| hashes[i] = h.as_to::<W>(), pl)?;
            pl.pop_log_target();
            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// default [`PHastRBuilder`] settings.
        ///
        /// Full-word hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−min(`W::BITS`, 64)</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`] with `PHastRBuilder::default()`.
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
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
            D: Sync,
            W: Send + Sync,
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
        /// [`try_new_with_builder`] instead.
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
            D: Sync,
            W: Send + Sync,
        {
            use sync_cell_slice::SyncSlice;
            let func = <PHastR<K, D>>::try_par_new_with_builder(keys, builder, pl)?;
            let mut hashes = vec![W::ZERO; func.len()];
            let sync_hashes = hashes.as_sync_slice();
            pl.push_log_target(" ▸ hashes");
            // SAFETY: the outputs of the keys are distinct, so each cell is
            // written by a single thread
            par_sign_phast_r(
                &func,
                keys,
                |i, h| unsafe { sync_hashes[i].set(h.as_to::<W>()) },
                pl,
            );
            pl.pop_log_target();
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
        /// This is a convenience wrapper around [`try_new_with_builder`]
        /// with `PHastRBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the construction.
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
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
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
        /// See also [`try_par_new_with_builder`] for parallel construction
        /// from slices.
        ///
        /// * `keys` - a [`FallibleRewindableLender`]. The [`lenders`]
        ///   module provides easy ways to build such lenders.
        /// * `hash_width` - the number of hash bits per key
        ///   (`1..=min(H::BITS, 64)`).
        ///
        /// [offline mode]: PHastRBuilder::offline
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
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
            let keys = keys.rewind()?;
            let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
            pl.push_log_target(" ▸ hashes");
            sign_phast_r(
                &func,
                keys,
                |i, h| {
                    let h = hashes.truncate_hash(h);
                    hashes.set_value(i, h)
                },
                pl,
            )?;
            pl.pop_log_target();
            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`PHastR`] from in-memory key
        /// slices, parallelizing construction and signing with rayon, using
        /// default [`PHastRBuilder`] settings.
        ///
        /// `hash_width`-bit hashes are stored for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`] with `PHastRBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`] instead.
        ///
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        /// [`try_new`]: Self::try_new
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
        /// [`try_new_with_builder`] instead.
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
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
            use crate::bits::AtomicBitFieldVec;
            use crate::traits::AtomicBitFieldSlice;
            use std::sync::atomic::Ordering;
            validate_hash_width::<H>(hash_width)?;
            let func = <PHastR<K, D>>::try_par_new_with_builder(keys, builder, pl)?;
            let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, func.len());
            let mask = hashes.mask();
            {
                // The outputs of the keys are distinct, but the fields of
                // different outputs may share a word: we write atomically
                let atomic_hashes = AtomicBitFieldVec::<&mut [H::Atomic]>::try_from(
                    BitFieldVec::<&mut [H]>::from(&mut hashes),
                )?;
                pl.push_log_target(" ▸ hashes");
                par_sign_phast_r(
                    &func,
                    keys,
                    |i, h| atomic_hashes.set_atomic(i, h.as_to::<H>() & mask, Ordering::Relaxed),
                    pl,
                );
                pl.pop_log_target();
            }
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
