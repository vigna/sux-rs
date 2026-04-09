/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
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
//! `Box<[u64]>`, giving false-positive probability 2<sup>−64</sup>) or a
//! [`BitFieldVec`] with a caller-chosen bit width (e.g., 8 bits for ≈0.4%
//! false positives). Per-inner-type `get` methods are provided via
//! monomorphized `impl` blocks.
//!
//! Use concrete types directly, like `SignedFunc<LcpMmphfStr, Box<[u64]>>` or
//! `SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>`. For each such
//! concrete type, this module provides `try_new`, `try_new_with_builder`, and
//! `get` methods.

use std::borrow::Borrow;
use std::mem::size_of;

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::VFunc;
use crate::func::lcp_mmphf::{BitPrefix, IntBitPrefix, LcpMmphf, LcpMmphfInt};
use crate::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use crate::func::shard_edge::ShardEdge;
use crate::traits::Backend;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::{PrimitiveInteger, PrimitiveNumber, PrimitiveNumberAs};
use value_traits::slices::SliceByValue;

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

impl<W: PrimitiveNumber> TruncateHash<W> for Box<[W]>
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
/// functions. It provides access to the seed, [signature type](Sig),
/// [`ShardEdge`], and key count, so that [`SignedFunc`] can verify hashes
/// without knowing which specific type of function it wraps.
pub trait SignableFunc {
    /// The signature type used by the inner function (e.g., `[u64; 2]`).
    type Sig: Sig;
    /// The [`ShardEdge`] used by the inner function.
    type Edge: ShardEdge<Self::Sig, 3>;

    /// Returns the seed used to hash keys into signatures.
    fn seed(&self) -> u64;
    /// Returns a reference to the [`ShardEdge`] used by the inner function.
    fn shard_edge(&self) -> &Self::Edge;
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
    type Edge = E0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.offset_lcp_length.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E0 {
        &self.offset_lcp_length.shard_edge
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
    type Edge = E0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.offset_lcp_length.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E0 {
        &self.offset_lcp_length.shard_edge
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
    type Edge = E0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.fused.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E0 {
        &self.fused.shard_edge
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
    type Edge = E0;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.fused.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E0 {
        &self.fused.shard_edge
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<K: ?Sized, D: SliceByValue, S: Sig, E: ShardEdge<S, 3>> SignableFunc for VFunc<K, D, S, E> {
    type Sig = S;
    type Edge = E;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E {
        &self.shard_edge
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.num_keys
    }
}

/// A signed function using a [`TruncateHash`] to store verification hashes.
///
/// Wraps an inner function `F` (any type implementing [`SignableFunc`]) and adds
/// per-key verification hashes so that queries for keys outside the original
/// set return `None`. The false-positive probability depends on the hash
/// storage: for `Box<[W]>` it is 2<sup>−`W::BITS`</sup>; for [`BitFieldVec`]
/// it is 2<sup>−`hash_width`</sup>.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Examples
///
/// Wrapping a [`VFunc`] with full-width hashes:
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use sux::func::{SignedFunc, VFunc};
/// # use sux::bits::BitFieldVec;
/// # use dsi_progress_logger::no_logging;
/// # use sux::utils::FromCloneableIntoIterator;
/// let func = <SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>>>::try_new(
///     FromCloneableIntoIterator::new(0..100_usize),
///     100,
///     no_logging![],
/// )?;
/// assert_eq!(func.get(42_usize), Some(42));
/// assert_eq!(func.get(999_usize), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// Wrapping an [`LcpMmphfStr`](crate::func::LcpMmphfStr) for sorted string keys:
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use sux::func::{SignedFunc, LcpMmphfStr};
/// # use dsi_progress_logger::no_logging;
/// # use sux::utils::FromSlice;
/// let keys = vec!["alpha", "beta", "gamma"];
/// let func = <SignedFunc<LcpMmphfStr, Box<[u64]>>>::try_new(
///     FromSlice::new(&keys), keys.len(), no_logging![],
/// )?;
/// assert_eq!(func.get("beta"), Some(1));
/// assert_eq!(func.get("missing"), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SignedFunc<F, H> {
    pub(crate) func: F,
    pub(crate) hashes: H,
}

impl<F, H> SignedFunc<F, H> {
    /// Creates a new `SignedFunc` from a function and a hash slice.
    ///
    /// This is a low-level constructor; prefer
    /// [`try_new`](Self::try_new)/[`try_new_with_builder`](Self::try_new_with_builder)
    /// when possible.
    pub fn from_parts(func: F, hashes: H) -> Self {
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
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let expected = self
            .hashes
            .truncate_hash(self.func.shard_edge().remixed_hash(sig));
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

// ── VFunc `get` ──────────────────────────────────────────────────

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

// ── LcpMmphfInt `get` ────────────────────────────────────────────

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
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: K) -> Option<usize> {
        let rank = self.func.get(key);
        self.verify(rank, K::to_sig(key, self.func.seed()))
    }
}

// ── LcpMmphf `get` ──────────────────────────────────────────────

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
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let rank = self.func.get(key);
        self.verify(rank, K::to_sig(key, self.func.seed()))
    }
}

// ── Lcp2MmphfInt `get` ──────────────────────────────────────────

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
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: K) -> Option<usize> {
        let rank = self.func.get(key);
        self.verify(rank, K::to_sig(key, self.func.seed()))
    }
}

// ── Lcp2Mmphf `get` ─────────────────────────────────────────────

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
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let rank = self.func.get(key);
        self.verify(rank, K::to_sig(key, self.func.seed()))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — helper functions
// (type aliases section removed: use SignedFunc<LcpMmphfStr, Box<[u64]>> etc. directly)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::func::VBuilder;
    use crate::utils::FallibleRewindableLender;
    use anyhow::Result;
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
    /// remixed hash through `shard_edge`. The resulting full-width hashes
    /// are stored in a newly allocated boxed slice.
    ///
    /// # Panics
    ///
    /// Panics if the lender yields fewer than `n` elements.
    fn fill_hashes<W, S, E, L>(
        shard_edge: &E,
        seed: u64,
        n: usize,
        mut keys: L,
        to_sig: impl Fn(&<L as FallibleLending<'_>>::Lend, u64) -> S,
    ) -> Result<Box<[W]>>
    where
        W: Word,
        S: Sig,
        E: ShardEdge<S, 3>,
        u64: PrimitiveNumberAs<W>,
        L: FallibleLender<Error: std::error::Error + Send + Sync + 'static>,
        for<'lend> L: FallibleLending<'lend>,
    {
        let mut hashes = vec![W::MIN; n];
        for hash in hashes.iter_mut() {
            let key = keys.next()?.expect("Not enough keys for hashes");
            *hash = shard_edge.remixed_hash(to_sig(&key, seed)).as_to::<W>();
        }
        Ok(hashes.into_boxed_slice())
    }

    /// Fills a [`BitFieldVec<Box<[H]>>`](BitFieldVec) hash array from a key
    /// lender.
    ///
    /// Iterates the first `n` elements of `keys`, converting each borrowed
    /// key to a signature via the `to_sig` closure and then computing the
    /// remixed hash through `shard_edge`. The hash is truncated using the
    /// [`BitFieldVec`]'s stored mask and stored in a [`BitFieldVec`] of the
    /// given `hash_width`.
    ///
    /// # Panics
    ///
    /// Panics if the lender yields fewer than `n` elements.
    fn fill_bit_hashes<H, S, E, L>(
        shard_edge: &E,
        seed: u64,
        n: usize,
        hash_width: usize,
        mut keys: L,
        to_sig: impl Fn(&<L as FallibleLending<'_>>::Lend, u64) -> S,
    ) -> Result<BitFieldVec<Box<[H]>>>
    where
        H: Word,
        S: Sig,
        E: ShardEdge<S, 3>,
        u64: PrimitiveNumberAs<H>,
        L: FallibleLender<Error: std::error::Error + Send + Sync + 'static>,
        for<'lend> L: FallibleLending<'lend>,
    {
        let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, n);
        for i in 0..n {
            let key = keys.next()?.expect("Not enough keys for hashes");
            let h = hashes.truncate_hash(shard_edge.remixed_hash(to_sig(&key, seed)));
            hashes.set_value(i, h);
        }
        Ok(hashes)
    }

    /// Fills a `Box<[W]>` hash array from a key slice.
    ///
    /// Iterates the first `n` elements of `keys`, converting each key to a
    /// signature via [`ToSig::to_sig`] and then computing the remixed hash
    /// through `shard_edge`. The resulting full-width hashes are stored in a
    /// newly allocated boxed slice.
    ///
    /// # Panics
    ///
    /// Panics if `keys.len() < n`.
    fn fill_hashes_from_slice<B, K, W, S, E>(
        shard_edge: &E,
        seed: u64,
        n: usize,
        keys: &[B],
    ) -> Box<[W]>
    where
        B: Borrow<K>,
        K: ?Sized + ToSig<S>,
        W: Word,
        S: Sig,
        E: ShardEdge<S, 3>,
        u64: PrimitiveNumberAs<W>,
    {
        let mut hashes = vec![W::MIN; n];
        for (hash, key) in hashes.iter_mut().zip(keys.iter()) {
            *hash = shard_edge
                .remixed_hash(K::to_sig(key.borrow(), seed))
                .as_to::<W>();
        }
        hashes.into_boxed_slice()
    }

    /// Fills a [`BitFieldVec<Box<[H]>>`](BitFieldVec) hash array from a key
    /// slice.
    ///
    /// Iterates the first `n` elements of `keys`, converting each key to a
    /// signature via [`ToSig::to_sig`] and then computing the remixed hash
    /// through `shard_edge`. The hash is truncated using the [`BitFieldVec`]'s
    /// stored mask and stored in a [`BitFieldVec`] of the given `hash_width`.
    ///
    /// # Panics
    ///
    /// Panics if `keys.len() < n`.
    fn fill_bit_hashes_from_slice<B, K, H, S, E>(
        shard_edge: &E,
        seed: u64,
        n: usize,
        hash_width: usize,
        keys: &[B],
    ) -> BitFieldVec<Box<[H]>>
    where
        B: Borrow<K>,
        K: ?Sized + ToSig<S>,
        H: Word,
        S: Sig,
        E: ShardEdge<S, 3>,
        u64: PrimitiveNumberAs<H>,
    {
        let mut hashes = BitFieldVec::<Box<[H]>>::new_padded(hash_width, n);
        for (i, key) in keys.iter().enumerate().take(n) {
            let h = hashes.truncate_hash(shard_edge.remixed_hash(K::to_sig(key.borrow(), seed)));
            hashes.set_value(i, h);
        }
        hashes
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<VFunc<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<K, S, E, H> SignedFunc<VFunc<K, BitFieldVec<Box<[usize]>>, S, E>, Box<[H]>>
    where
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        H: crate::traits::Word,
        SigVal<S, usize>: RadixKey,
        SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using
        /// default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `H::BITS`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>-`H::BITS`</sup>.
        ///
        /// * `keys` must be rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys; a significantly wrong
        ///   value may degrade performance or cause extra retries.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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
        ///     100,
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
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using the
        /// given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `H::BITS`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>-`H::BITS`</sup>.
        ///
        /// * `keys` must be rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
        ///     100,
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
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let (func, mut store, _) = builder.expected_num_keys(n).try_build_func_and_store(
                keys,
                FromCloneableIntoIterator::from(0..),
                BitFieldVec::new_padded,
                pl,
            )?;

            let num_keys = func.len();

            // Create the hash vector
            let mut hashes = vec![H::ZERO; num_keys].into_boxed_slice();

            // Enumerate the store and extract hashes using the same method as filters
            pl.item_name("hash");
            pl.expected_updates(Some(num_keys));
            pl.start("Storing hashes...");

            for shard in store.iter() {
                for sig_val in shard.iter() {
                    let pos = sig_val.val;
                    let hash = H::as_from(func.shard_edge().remixed_hash(sig_val.sig));
                    hashes.set_value(pos, hash);
                    pl.light_update();
                }
            }

            pl.done();

            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing hash computation and store population with
        /// rayon, using default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `H::BITS`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>-`H::BITS`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            u64: PrimitiveNumberAs<H>,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing hash computation and store population with
        /// rayon, using the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `H::BITS`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>-`H::BITS`</sup>.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            u64: PrimitiveNumberAs<H>,
        {
            let values: Vec<usize> = (0..keys.len()).collect();
            let func = <VFunc<K, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                keys, &values, builder, pl,
            )?;
            let hashes = fill_hashes_from_slice(func.shard_edge(), func.seed(), func.len(), keys);
            Ok(SignedFunc { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<VFunc<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<K, S, E, H> SignedFunc<VFunc<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
    where
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        H: crate::traits::Word,
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
        /// * `keys` must be rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys; a significantly wrong
        ///   value may degrade performance or cause extra retries.
        /// * `hash_width` is the number of hash bits per key (at most
        ///   `H::BITS`).
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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
        ///     100,
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
            n: usize,
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            u64: PrimitiveNumberAs<H>,
        {
            Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from keys using
        /// the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// * `keys` must be rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys.
        /// * `hash_width` is the number of hash bits per key (at most
        ///   `H::BITS`).
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
        ///     100,
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
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            u64: PrimitiveNumberAs<H>,
        {
            assert!(hash_width > 0);
            assert!(hash_width <= H::BITS as usize);

            let (func, mut store, _) = builder.expected_num_keys(n).try_build_func_and_store(
                keys,
                FromCloneableIntoIterator::from(0..),
                BitFieldVec::<Box<[usize]>>::new_padded,
                pl,
            )?;

            let num_keys = func.len();

            // Create the hash vector
            let mut hashes: BitFieldVec<Box<[H]>> =
                BitFieldVec::<Box<[H]>>::new_padded(hash_width, num_keys);

            // Enumerate the store and extract hashes
            pl.item_name("hash");
            pl.expected_updates(Some(num_keys));
            pl.start("Storing hashes...");

            for shard in store.iter() {
                for sig_val in shard.iter() {
                    let pos = sig_val.val;
                    let hash = hashes.truncate_hash(func.shard_edge().remixed_hash(sig_val.sig));
                    hashes.set_value(pos, hash);
                    pl.light_update();
                }
            }

            pl.done();

            Ok(SignedFunc { func, hashes })
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing hash computation and store population with
        /// rayon, using default [`VBuilder`] settings.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
            S: Send,
            u64: PrimitiveNumberAs<H>,
        {
            Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Builds a [`SignedFunc`] wrapping a [`VFunc`] from in-memory key
        /// slices, parallelizing hash computation and store population with
        /// rayon, using the given [`VBuilder`] configuration.
        ///
        /// The function maps each key to its index in the input sequence
        /// and stores `hash_width`-bit hashes for verification, giving a
        /// false-positive rate of 2<sup>−`hash_width`</sup>.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            u64: PrimitiveNumberAs<H>,
        {
            assert!(hash_width > 0);
            assert!(hash_width <= H::BITS as usize);
            let values: Vec<usize> = (0..keys.len()).collect();
            let func = <VFunc<K, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                keys, &values, builder, pl,
            )?;
            let hashes = fill_bit_hashes_from_slice(
                func.shard_edge(),
                func.seed(),
                func.len(),
                hash_width,
                keys,
            );
            Ok(SignedFunc { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<LcpMmphfInt<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// The keys must be in strictly increasing order.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal `offset_lcp_length` VFunc.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
                K::to_sig(*key, seed)
            })?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for integers from a slice,
        /// using parallel hash computation and default [`VBuilder`] settings.
        ///
        /// The keys must be in strictly increasing order.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new(
            keys: &[K],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal `offset_lcp_length` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///         &keys, VBuilder::default().offline(true), no_logging![],
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
        pub fn try_par_new_with_builder(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let func = LcpMmphfInt::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_hashes_from_slice::<K, K, W, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<LcpMmphf<K, ...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// The keys must be in strictly increasing lexicographic order
        /// (byte-level comparison).
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal `offset_lcp_length` VFunc.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
                K::to_sig(<B as Borrow<K>>::borrow(key), seed)
            })?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for byte-sequence keys from
        /// a slice, using parallel hash computation and default [`VBuilder`]
        /// settings.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal `offset_lcp_length` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///         &keys, VBuilder::default().offline(true), no_logging![],
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
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            let func = LcpMmphf::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_hashes_from_slice::<B, K, W, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<Lcp2MmphfInt<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
                K::to_sig(*key, seed)
            })?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for integers from
        /// a slice, using parallel hash computation and default [`VBuilder`]
        /// settings.
        ///
        /// The keys must be in strictly increasing order.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new(
            keys: &[K],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///         &keys, VBuilder::default().offline(true), no_logging![],
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
        pub fn try_par_new_with_builder(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let func = Lcp2MmphfInt::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_hashes_from_slice::<K, K, W, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<Lcp2Mmphf<K, ...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        W: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
                K::to_sig(<B as Borrow<K>>::borrow(key), seed)
            })?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for byte-sequence
        /// keys from a slice, using parallel hash computation and default
        /// [`VBuilder`] settings.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///         &keys, VBuilder::default().offline(true), no_logging![],
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
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            let func = Lcp2Mmphf::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_hashes_from_slice::<B, K, W, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<LcpMmphfInt<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// `hash_width` is the number of hash bits stored per key (must be
        /// in `1 . . H::BITS`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal `offset_lcp_length` VFunc.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let (func, keys) = LcpMmphfInt::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            let hashes = fill_bit_hashes(
                func.shard_edge(),
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(*key, seed),
            )?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for integers with
        /// sub-word-width hashes from a slice, using parallel hash
        /// computation and default [`VBuilder`] settings.
        ///
        /// `hash_width` is the number of hash bits stored per key (must be
        /// in `1 . . H::BITS`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new(
            keys: &[K],
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal `offset_lcp_length` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///     BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default().offline(true), no_logging![])?.try_into_unaligned()?;
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let func = LcpMmphfInt::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_bit_hashes_from_slice::<K, K, H, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                hash_width,
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<LcpMmphf<K, ...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// `hash_width` is the number of hash bits stored per key (must be
        /// in `1 . . H::BITS`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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
        /// # use crate::sux::traits::TryIntoUnaligned;
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal `offset_lcp_length` VFunc.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let (func, keys) = LcpMmphf::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            let hashes = fill_bit_hashes(
                func.shard_edge(),
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
            )?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed LCP-based MMPHF for byte-sequence keys with
        /// sub-word-width hashes from a slice, using parallel hash computation
        /// and default [`VBuilder`] settings.
        ///
        /// `hash_width` is the number of hash bits stored per key (must be
        /// in `1 . . H::BITS`). False-positive probability is
        /// 2<sup>−`hash_width`</sup>.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal `offset_lcp_length` VFunc.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///     BSFunc::try_par_new_with_builder(&keys, 12, VBuilder::default().offline(true), no_logging![])?.try_into_unaligned()?;
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
            hash_width: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let func = LcpMmphf::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_bit_hashes_from_slice::<B, K, H, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                hash_width,
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<Lcp2MmphfInt<...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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
        /// # use crate::sux::traits::TryIntoUnaligned;
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let (func, keys) = Lcp2MmphfInt::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            let hashes = fill_bit_hashes(
                func.shard_edge(),
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(*key, seed),
            )?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for integers with
        /// sub-word-width hashes from a slice, using parallel hash computation
        /// and default [`VBuilder`] settings.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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
        pub fn try_par_new(
            keys: &[K],
            hash_width: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, hash_width, VBuilder::default(), pl)
        }

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///     BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default().offline(true), no_logging![])?.try_into_unaligned()?;
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let func = Lcp2MmphfInt::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_bit_hashes_from_slice::<K, K, H, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                hash_width,
                keys,
            );
            Ok(Self { func, hashes })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors — SignedFunc<Lcp2Mmphf<K, ...>, BitFieldVec<...>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        H: Word,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
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
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
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

        /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
        /// configure the internal VFuncs.
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let (func, keys) = Lcp2Mmphf::try_new_inner(keys, n, builder, pl)?;
            let mut keys = keys.rewind()?;
            let hashes = fill_bit_hashes(
                func.shard_edge(),
                func.seed(),
                n,
                hash_width,
                &mut keys,
                |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
            )?;
            Ok(Self { func, hashes })
        }

        /// Creates a new signed two-step LCP-based MMPHF for byte-sequence
        /// keys with sub-word-width hashes from a slice, using parallel hash
        /// computation and default [`VBuilder`] settings.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
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

        /// Like [`try_par_new`](Self::try_par_new), but uses the given
        /// [`VBuilder`] to configure the internal VFuncs.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
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
        ///     BSFunc::try_par_new_with_builder(&keys, 8, VBuilder::default().offline(true), no_logging![])?.try_into_unaligned()?;
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
            assert!(hash_width > 0 && hash_width <= H::BITS as usize);
            let func = Lcp2Mmphf::try_par_new_inner(keys, builder, pl)?;
            let hashes = fill_bit_hashes_from_slice::<B, K, H, S0, E0>(
                func.shard_edge(),
                func.seed(),
                keys.len(),
                hash_width,
                keys,
            );
            Ok(Self { func, hashes })
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
