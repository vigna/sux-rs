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
//! [`BitFieldVec`] with a caller-chosen bit width (e.g., 8 bits for ~0.4%
//! false positives). Per-inner-type `get` methods are provided via
//! monomorphized `impl` blocks.
//!
//! Use concrete types directly, like `SignedFunc<LcpMmphfStr, Box<[u64]>>` or
//! `SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>`. For each such
//! concrete type, this module provides `try_new`, `try_new_with_builder`, and
//! `get` methods.

use std::borrow::Borrow;
use std::mem::size_of;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    crate::utils::FallibleRewindableLender,
    anyhow::Result,
    core::error::Error,
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
    value_traits::slices::SliceByValueMut,
};

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::VFunc;
use crate::func::lcp_mmphf::{LcpMmphf, LcpMmphfInt};
use crate::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use crate::func::shard_edge::{Fuse3Shards, ShardEdge};
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
/// functions. It provides access to the seed, shard edge, and key count, so
/// that [`SignedFunc`] can verify hashes without knowing which specific type
/// of function it wraps.
pub trait SignableFunc {
    type Sig: Sig;
    type Edge: ShardEdge<Self::Sig, 3>;

    fn seed(&self) -> u64;
    fn shard_edge(&self) -> &Self::Edge;
    fn len(&self) -> usize;
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: PrimitiveInteger, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableFunc
    for LcpMmphfInt<T, D, S, E>
{
    type Sig = S;
    type Edge = E;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.offset_lcp_length.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E {
        &self.offset_lcp_length.shard_edge
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<K: ?Sized, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableFunc
    for LcpMmphf<K, D, S, E>
{
    type Sig = S;
    type Edge = E;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.offset_lcp_length.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E {
        &self.offset_lcp_length.shard_edge
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<T: PrimitiveInteger, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableFunc
    for Lcp2MmphfInt<T, D, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    type Sig = S;
    type Edge = E;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.fused.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E {
        &self.fused.shard_edge
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<K: ?Sized, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableFunc
    for Lcp2Mmphf<K, D, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    type Sig = S;
    type Edge = E;

    #[inline(always)]
    fn seed(&self) -> u64 {
        self.fused.seed
    }
    #[inline(always)]
    fn shard_edge(&self) -> &E {
        &self.fused.shard_edge
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }
}

impl<T: ?Sized, D: SliceByValue, S: Sig, E: ShardEdge<S, 3>> SignableFunc for VFunc<T, D, S, E> {
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
/// Wrapping an [`LcpMmphfStr`] for sorted string keys:
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
    T: ?Sized + ToSig<S>,
    D: SliceByValue<Value: Word + BinSafe>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
> SignedFunc<VFunc<T, D, S, E>, H>
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
    pub fn get(&self, key: impl Borrow<T>) -> Option<D::Value> {
        self.get_by_sig(T::to_sig(key.borrow(), self.func.seed()))
    }
}

// ── LcpMmphfInt `get` ────────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedFunc<LcpMmphfInt<T, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        let rank = self.func.get(key);
        self.verify(rank, T::to_sig(key, self.func.seed()))
    }
}

// ── LcpMmphf `get` ──────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedFunc<LcpMmphf<K, D, S, E>, H>
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
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedFunc<Lcp2MmphfInt<T, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        let rank = self.func.get(key);
        self.verify(rank, T::to_sig(key, self.func.seed()))
    }
}

// ── Lcp2Mmphf `get` ─────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber> + TruncateHash<H::Value>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedFunc<Lcp2Mmphf<K, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
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
#[cfg(feature = "rayon")]
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
#[cfg(feature = "rayon")]
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
    let mut hashes = BitFieldVec::<Box<[H]>>::new_unaligned(hash_width, n);
    for i in 0..n {
        let key = keys.next()?.expect("Not enough keys for hashes");
        let h = hashes.truncate_hash(shard_edge.remixed_hash(to_sig(&key, seed)));
        hashes.set_value(i, h);
    }
    Ok(hashes)
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<VFunc<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, S, E, H> SignedFunc<VFunc<T, BitFieldVec<Box<[usize]>>, S, E>, Box<[H]>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
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
    pub fn try_new<B: ?Sized + Borrow<T>>(
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
    /// The builder controls construction parameters such as [offline
    /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
    /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
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
    pub fn try_new_with_builder<B: ?Sized + Borrow<T>>(
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
            BitFieldVec::new_unaligned,
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
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<VFunc<...>, BitFieldVec<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, S, E, H> SignedFunc<VFunc<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
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
    pub fn try_new<B: ?Sized + Borrow<T>>(
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
    pub fn try_new_with_builder<B: ?Sized + Borrow<T>>(
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
            BitFieldVec::<Box<[usize]>>::new_unaligned,
            pl,
        )?;

        let num_keys = func.len();

        // Create the hash vector
        let mut hashes: BitFieldVec<Box<[H]>> =
            BitFieldVec::<Box<[H]>>::new_unaligned(hash_width, num_keys);

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
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<LcpMmphfInt<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, W, S, E> SignedFunc<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    W: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
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
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{SignedFunc, LcpMmphfInt};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: SignedFunc<LcpMmphfInt<u64>, Box<[u16]>> =
    ///     <SignedFunc<LcpMmphfInt<u64>, Box<[u16]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
    /// configure the internal `offset_lcp_length` VFunc.
    pub fn try_new_with_builder(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let (func, keys) = LcpMmphfInt::try_new_inner(keys, n, builder, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
            T::to_sig(*key, seed)
        })?;
        Ok(Self { func, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<LcpMmphf<K, ...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<K, W, S, E> SignedFunc<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    W: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
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
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{SignedFunc, LcpMmphfStr};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: SignedFunc<LcpMmphfStr, Box<[u64]>> =
    ///     <SignedFunc<LcpMmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
    pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let (func, keys) = LcpMmphf::try_new_inner(keys, n, builder, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
            K::to_sig(<B as Borrow<K>>::borrow(key), seed)
        })?;
        Ok(Self { func, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<Lcp2MmphfInt<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, W, S, E> SignedFunc<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    W: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, u64>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<E::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, u64>:
        std::ops::BitXor + std::ops::BitXorAssign,
    u64: PrimitiveNumberAs<W>,
{
    /// Creates a new signed two-step LCP-based MMPHF for integers.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder) with
    /// `VBuilder::default()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{SignedFunc, Lcp2MmphfInt};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: SignedFunc<Lcp2MmphfInt<u64>, Box<[u16]>> =
    ///     <SignedFunc<Lcp2MmphfInt<u64>, Box<[u16]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
    /// configure the internal VFuncs.
    pub fn try_new_with_builder(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let (func, keys) = Lcp2MmphfInt::try_new_inner(keys, n, builder, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
            T::to_sig(*key, seed)
        })?;
        Ok(Self { func, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<Lcp2Mmphf<K, ...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<K, W, S, E> SignedFunc<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    W: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, u64>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<E::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, u64>:
        std::ops::BitXor + std::ops::BitXorAssign,
    u64: PrimitiveNumberAs<W>,
{
    /// Creates a new signed two-step LCP-based MMPHF for byte-sequence keys.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder) with
    /// `VBuilder::default()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{SignedFunc, Lcp2MmphfStr};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: SignedFunc<Lcp2MmphfStr, Box<[u64]>> =
    ///     <SignedFunc<Lcp2MmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
    pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let (func, keys) = Lcp2Mmphf::try_new_inner(keys, n, builder, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_hashes(func.shard_edge(), func.seed(), n, &mut keys, |key, seed| {
            K::to_sig(<B as Borrow<K>>::borrow(key), seed)
        })?;
        Ok(Self { func, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<LcpMmphfInt<...>, BitFieldVec<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, H, S, E> SignedFunc<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    H: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
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
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{SignedFunc, LcpMmphfInt};
    /// # use sux::bits::BitFieldVec;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// type BSFunc = SignedFunc<LcpMmphfInt<u64>, BitFieldVec<Box<[usize]>>>;
    /// let func: BSFunc =
    ///     BSFunc::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
    }

    /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
    /// configure the internal `offset_lcp_length` VFunc.
    pub fn try_new_with_builder(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        hash_width: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
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
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self { func, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<LcpMmphf<K, ...>, BitFieldVec<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<K, H, S, E> SignedFunc<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    H: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
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
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{SignedFunc, LcpMmphfStr};
    /// # use sux::bits::BitFieldVec;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["alpha", "beta", "delta", "gamma"];
    /// type BSFunc = SignedFunc<LcpMmphfStr, BitFieldVec<Box<[usize]>>>;
    /// let func: BSFunc =
    ///     BSFunc::try_new(FromSlice::new(&keys), keys.len(), 12, no_logging![])?;
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
    pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        hash_width: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
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
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<Lcp2MmphfInt<...>, BitFieldVec<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, H, S, E> SignedFunc<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    H: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, u64>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<E::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, u64>:
        std::ops::BitXor + std::ops::BitXorAssign,
    u64: PrimitiveNumberAs<H>,
{
    /// Creates a new signed two-step LCP-based MMPHF for integers with
    /// sub-word-width hashes.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder) with
    /// `VBuilder::default()`.
    pub fn try_new(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
    }

    /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`] to
    /// configure the internal VFuncs.
    pub fn try_new_with_builder(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        hash_width: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
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
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self { func, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — SignedFunc<Lcp2Mmphf<K, ...>, BitFieldVec<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<K, H, S, E> SignedFunc<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    H: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, u64>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<E::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, u64>:
        std::ops::BitXor + std::ops::BitXorAssign,
    u64: PrimitiveNumberAs<H>,
{
    /// Creates a new signed two-step LCP-based MMPHF for byte-sequence keys
    /// with sub-word-width hashes.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder) with
    /// `VBuilder::default()`.
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
    pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        hash_width: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
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
}

// ═══════════════════════════════════════════════════════════════════
// Aligned <-> Unaligned conversions
// ═══════════════════════════════════════════════════════════════════

use crate::traits::{TryIntoUnaligned, Word};

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

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, H>
    From<SignedFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, H>>
    for SignedFunc<VFunc<T, BitFieldVec<Box<[W]>>, S, E>, H>
{
    fn from(f: SignedFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, H>) -> Self {
        SignedFunc {
            func: f.func.into(),
            hashes: f.hashes,
        }
    }
}

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, W2: Word>
    From<SignedFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, BitFieldVecU<Box<[W2]>>>>
    for SignedFunc<VFunc<T, BitFieldVec<Box<[W]>>, S, E>, BitFieldVec<Box<[W2]>>>
{
    fn from(
        f: SignedFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, BitFieldVecU<Box<[W2]>>>,
    ) -> Self {
        SignedFunc {
            func: f.func.into(),
            hashes: f.hashes.into(),
        }
    }
}
