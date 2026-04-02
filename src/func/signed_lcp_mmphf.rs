/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity, private_bounds)]

//! Signed LCP-based monotone minimal perfect hash functions.
//!
//! Two wrappers are provided, following the same pattern as
//! [`SignedVFunc`](crate::dict::SignedVFunc) /
//! [`BitSignedVFunc`](crate::dict::BitSignedVFunc):
//!
//! - [`SignedLcpMmphf`] stores full-width hashes (e.g., `Box<[u64]>`).
//!   False-positive probability is 2<sup>−`H::Value::BITS`</sup>.
//! - [`BitSignedLcpMmphf`] stores hashes in a [`BitFieldVec`] with a
//!   caller-chosen bit width. False-positive probability is
//!   2<sup>−`hash_width`</sup>.
//!
//! See [`SignedLcpMmphfInt`], [`SignedLcpMmphfStr`],
//! [`BitSignedLcpMmphfInt`], [`BitSignedLcpMmphfStr`], etc. for type
//! aliases.

use crate::bits::BitFieldVec;
use crate::dict::SignableMphf;
use crate::func::VFunc;
use crate::func::lcp_mmphf::{LcpMmphf, LcpMmphfInt};
use crate::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use crate::func::shard_edge::{Fuse3Shards, FuseLge3Shards, ShardEdge};
use crate::utils::*;
use mem_dbg::*;
use num_primitive::{PrimitiveInteger, PrimitiveNumber};
use value_traits::slices::SliceByValue;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder, crate::traits::Word, anyhow::Result, dsi_progress_logger::ProgressLog,
    lender::*, num_primitive::PrimitiveNumberAs, rdst::RadixKey, std::borrow::Borrow,
    value_traits::slices::SliceByValueMut,
};

// ═══════════════════════════════════════════════════════════════════
// Mmphf trait — abstracts over the four inner MMPHF types
// ═══════════════════════════════════════════════════════════════════

impl<T: PrimitiveInteger, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableMphf
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

impl<K: ?Sized, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableMphf
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

impl<T: PrimitiveInteger, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableMphf
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

impl<K: ?Sized, D: SliceByValue<Value = usize>, S: Sig, E: ShardEdge<S, 3>> SignableMphf
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

impl<T: ?Sized, D: SliceByValue, S: Sig, E: ShardEdge<S, 3>> SignableMphf for VFunc<T, D, S, E> {
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

// ═══════════════════════════════════════════════════════════════════
// SignedLcpMmphf — full-width hashes
// ═══════════════════════════════════════════════════════════════════

/// A signed LCP-based monotone minimal perfect hash function.
///
/// Wraps an inner MMPHF (either [`LcpMmphfInt`] or [`LcpMmphf`]) and adds
/// per-key verification hashes so that queries for keys outside the original
/// set return `None` (with false-positive probability
/// 2<sup>−`H::Value::BITS`</sup>).
///
/// This type is generic over the inner function `F` and the hash storage
/// `H`, following the same pattern as
/// [`SignedVFunc`](crate::dict::SignedVFunc). See [`SignedLcpMmphfInt`],
/// [`SignedLcpMmphfStr`], and [`SignedLcpMmphfSliceU8`] for convenient type
/// aliases.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SignedLcpMmphf<F, H> {
    pub(crate) inner: F,
    pub(crate) hashes: H,
}

// ── Unified query methods ──────────────────────────────────────────

impl<F: SignableMphf, H: SliceByValue<Value: PrimitiveNumber>> SignedLcpMmphf<F, H> {
    /// Verifies that the stored hash matches the remixed hash for the
    /// given rank and signature.
    #[inline(always)]
    fn verify(&self, rank: usize, sig: F::Sig) -> Option<usize> {
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let expected = self.inner.shard_edge().remixed_hash(sig);
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == <H::Value>::as_from(expected).as_to::<u64>() {
            Some(rank)
        } else {
            None
        }
    }

    /// Returns the number of keys.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the function contains no keys.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ── Integer `get` ──────────────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<LcpMmphfInt<T, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    ///
    /// False positives happen with probability
    /// 2<sup>−`H::Value::BITS`</sup>.
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, T::to_sig(key, self.inner.seed()))
    }
}

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<Lcp2MmphfInt<T, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, T::to_sig(key, self.inner.seed()))
    }
}

// ── Byte-sequence `get` ────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<LcpMmphf<K, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, K::to_sig(key, self.inner.seed()))
    }
}

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<Lcp2Mmphf<K, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, K::to_sig(key, self.inner.seed()))
    }
}

// ── Type aliases ───────────────────────────────────────────────────

/// A [`SignedLcpMmphf`] wrapping a [`LcpMmphfInt`].
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::SignedLcpMmphfInt;
/// # use sux::utils::FromSlice;
/// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
///
/// let func: SignedLcpMmphfInt<u64> = SignedLcpMmphfInt::try_new(
///     FromSlice::new(&keys),
///     keys.len(),
///     no_logging![],
/// )?;
///
/// for (i, &key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key), Some(i));
/// }
/// assert_eq!(func.get(999), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type SignedLcpMmphfInt<T, H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`SignedLcpMmphf`] wrapping a [`LcpMmphf`] for `str` keys.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::SignedLcpMmphfStr;
/// # use sux::utils::FromSlice;
/// let keys = vec![
///     "alpha".to_owned(),
///     "beta".to_owned(),
///     "delta".to_owned(),
///     "gamma".to_owned(),
/// ];
///
/// let func: SignedLcpMmphfStr =
///     SignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_str()), Some(i));
/// }
/// assert_eq!(func.get("not_a_key"), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type SignedLcpMmphfStr<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<LcpMmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`SignedLcpMmphf`] wrapping a [`LcpMmphf`] for `[u8]` keys.
pub type SignedLcpMmphfSliceU8<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<LcpMmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`SignedLcpMmphf`] wrapping a [`Lcp2MmphfInt`].
pub type SignedLcp2MmphfInt<T, H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;
/// A [`SignedLcpMmphf`] wrapping a [`Lcp2Mmphf`] for `str` keys.
pub type SignedLcp2MmphfStr<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<Lcp2Mmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;
/// A [`SignedLcpMmphf`] wrapping a [`Lcp2Mmphf`] for `[u8]` keys.
pub type SignedLcp2MmphfSliceU8<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<Lcp2Mmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── Constructors ───────────────────────────────────────────────────

/// Helper: fills a `Box<[W]>` hash array from a cloned key lender.
///
/// `to_sig` converts each borrowed key to a signature.
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

/// Helper: fills a `BitFieldVec<Box<[H]>>` hash array from a cloned key lender.
#[cfg(feature = "rayon")]
fn fill_bit_hashes<H, S, E, L>(
    shard_edge: &E,
    seed: u64,
    n: usize,
    hash_width: usize,
    hash_mask: u64,
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
        let h = (shard_edge.remixed_hash(to_sig(&key, seed)) & hash_mask).as_to::<H>();
        hashes.set_value(i, h);
    }
    Ok(hashes)
}

// ── SignedLcpMmphf constructors (LcpMmphfInt) ─────────────────────

#[cfg(feature = "rayon")]
impl<T, W, S, E> SignedLcpMmphf<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
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
    /// The keys must be in strictly increasing order. The lender must be
    /// [`Clone`] so that an additional pass can compute verification
    /// hashes after building the inner MMPHF.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::SignedLcpMmphfInt;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: SignedLcpMmphfInt<u64, Box<[u16]>> =
    ///     SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>
        + Clone,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let keys_for_hashes = keys.clone();
        let inner = LcpMmphfInt::try_new(keys, n, pl)?;
        let hashes = fill_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            keys_for_hashes,
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self { inner, hashes })
    }
}

// ── SignedLcpMmphf constructors (LcpMmphf) ────────────────────────

#[cfg(feature = "rayon")]
impl<K, W, S, E> SignedLcpMmphf<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
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
    /// (byte-level comparison). The lender must be [`Clone`] so that an
    /// additional pass can compute verification hashes after building the
    /// inner MMPHF.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::SignedLcpMmphfStr;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: SignedLcpMmphfStr<Box<[u16]>> =
    ///     SignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>
        + Clone,
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>
        + Clone,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let keys_for_hashes = keys.clone();
        let inner = LcpMmphf::try_new_with_builder(keys, n, builder, pl)?;
        let hashes = fill_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            keys_for_hashes,
            |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
        )?;
        Ok(Self { inner, hashes })
    }
}

// ── SignedLcpMmphf constructors (Lcp2MmphfInt) ────────────────────

#[cfg(feature = "rayon")]
impl<T, W, S, E> SignedLcpMmphf<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
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
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::SignedLcp2MmphfInt;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: SignedLcp2MmphfInt<u64, Box<[u16]>> =
    ///     SignedLcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>
        + Clone,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let keys_for_hashes = keys.clone();
        let inner = Lcp2MmphfInt::try_new(keys, n, pl)?;
        let hashes = fill_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            keys_for_hashes,
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self { inner, hashes })
    }
}

// ── SignedLcpMmphf constructors (Lcp2Mmphf) ───────────────────────

#[cfg(feature = "rayon")]
impl<K, W, S, E> SignedLcpMmphf<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S, E>, Box<[W]>>
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
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::SignedLcp2MmphfStr;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: SignedLcp2MmphfStr<Box<[u16]>> =
    ///     SignedLcp2MmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>
        + Clone,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let keys_for_hashes = keys.clone();
        let inner = Lcp2Mmphf::try_new(keys, n, pl)?;
        let hashes = fill_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            keys_for_hashes,
            |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
        )?;
        Ok(Self { inner, hashes })
    }
}

// ═══════════════════════════════════════════════════════════════════
// BitSignedLcpMmphf — sub-word-width hashes in a BitFieldVec
// ═══════════════════════════════════════════════════════════════════

/// A bit-signed LCP-based monotone minimal perfect hash function.
///
/// Like [`SignedLcpMmphf`], but stores hashes in a [`BitFieldVec`] with a
/// caller-chosen bit width, and compares only the masked bits. This is
/// useful when you want to trade space for a higher false-positive rate
/// (e.g., 8-bit hashes give 2⁻⁸ ≈ 0.4 % false positives).
///
/// If you are using all the bits of the hash type (e.g., 64-bit hashes on
/// `u64`), use [`SignedLcpMmphf`] instead — hash comparison will be faster.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitSignedLcpMmphf<F, H> {
    pub(crate) inner: F,
    pub(crate) hashes: H,
    pub(crate) hash_mask: u64,
}

// ── Unified query methods ──────────────────────────────────────────

impl<F: SignableMphf, H: SliceByValue<Value: PrimitiveNumber>> BitSignedLcpMmphf<F, H> {
    /// Verifies that the stored hash matches the masked remixed hash for
    /// the given rank and signature.
    #[inline(always)]
    fn verify(&self, rank: usize, sig: F::Sig) -> Option<usize> {
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let expected = self.inner.shard_edge().remixed_hash(sig) & self.hash_mask;
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == expected { Some(rank) } else { None }
    }

    /// Returns the number of keys.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the function contains no keys.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ── Integer `get` ──────────────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedLcpMmphf<LcpMmphfInt<T, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, T::to_sig(key, self.inner.seed()))
    }
}

// ── Byte-sequence `get` ────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedLcpMmphf<LcpMmphf<K, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, K::to_sig(key, self.inner.seed()))
    }
}

// ── Integer `get` (Lcp2) ───────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedLcpMmphf<Lcp2MmphfInt<T, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, T::to_sig(key, self.inner.seed()))
    }
}

// ── Byte-sequence `get` (Lcp2) ────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedLcpMmphf<Lcp2Mmphf<K, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        let rank = self.inner.get(key);
        self.verify(rank, K::to_sig(key, self.inner.seed()))
    }
}

// ── Type aliases ───────────────────────────────────────────────────

/// A [`BitSignedLcpMmphf`] wrapping a [`LcpMmphfInt`].
pub type BitSignedLcpMmphfInt<T, H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`LcpMmphf`] for `str` keys.
pub type BitSignedLcpMmphfStr<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<LcpMmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`LcpMmphf`] for `[u8]` keys.
pub type BitSignedLcpMmphfSliceU8<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<LcpMmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`Lcp2MmphfInt`].
pub type BitSignedLcp2MmphfInt<T, H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`Lcp2Mmphf`] for `str` keys.
pub type BitSignedLcp2MmphfStr<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<Lcp2Mmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`Lcp2Mmphf`] for `[u8]` keys.
pub type BitSignedLcp2MmphfSliceU8<
    H = BitFieldVec<Box<[usize]>>,
    S = [u64; 2],
    E = FuseLge3Shards,
> = BitSignedLcpMmphf<Lcp2Mmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── BitSignedLcpMmphf constructors (LcpMmphfInt) ──────────────────

#[cfg(feature = "rayon")]
impl<T, H, S, E>
    BitSignedLcpMmphf<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    H: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    u64: PrimitiveNumberAs<H>,
{
    /// Creates a new bit-signed LCP-based MMPHF for integers.
    ///
    /// `hash_width` is the number of hash bits stored per key (must be
    /// in `1..=H::BITS`). False-positive probability is
    /// 2<sup>−`hash_width`</sup>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::BitSignedLcpMmphfInt;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: BitSignedLcpMmphfInt<u64> =
    ///     BitSignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>
        + Clone,
        n: usize,
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        assert!(hash_width > 0 && hash_width <= H::BITS as usize);
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let keys_for_hashes = keys.clone();
        let inner = LcpMmphfInt::try_new(keys, n, pl)?;
        let hashes = fill_bit_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            hash_width,
            hash_mask,
            keys_for_hashes,
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self {
            inner,
            hashes,
            hash_mask,
        })
    }
}

// ── BitSignedLcpMmphf constructors (LcpMmphf) ────────────────────

#[cfg(feature = "rayon")]
impl<K, H, S, E>
    BitSignedLcpMmphf<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    H: Word,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    u64: PrimitiveNumberAs<H>,
{
    /// Creates a new bit-signed LCP-based MMPHF for byte-sequence keys.
    ///
    /// `hash_width` is the number of hash bits stored per key (must be
    /// in `1..=H::BITS`). False-positive probability is
    /// 2<sup>−`hash_width`</sup>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::BitSignedLcpMmphfStr;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: BitSignedLcpMmphfStr =
    ///     BitSignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?;
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>
        + Clone,
        n: usize,
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        assert!(hash_width > 0 && hash_width <= H::BITS as usize);
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let keys_for_hashes = keys.clone();
        let inner = LcpMmphf::try_new(keys, n, pl)?;
        let hashes = fill_bit_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            hash_width,
            hash_mask,
            keys_for_hashes,
            |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
        )?;
        Ok(Self {
            inner,
            hashes,
            hash_mask,
        })
    }
}

// ── BitSignedLcpMmphf constructors (Lcp2MmphfInt) ─────────────────

#[cfg(feature = "rayon")]
impl<T, H, S, E>
    BitSignedLcpMmphf<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
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
    /// Creates a new bit-signed two-step LCP-based MMPHF for integers.
    ///
    /// `hash_width` is the number of hash bits stored per key (must be
    /// in `1..=H::BITS`). False-positive probability is
    /// 2<sup>−`hash_width`</sup>.
    pub fn try_new(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>
        + Clone,
        n: usize,
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        assert!(hash_width > 0 && hash_width <= H::BITS as usize);
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let keys_for_hashes = keys.clone();
        let inner = Lcp2MmphfInt::try_new(keys, n, pl)?;
        let hashes = fill_bit_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            hash_width,
            hash_mask,
            keys_for_hashes,
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self {
            inner,
            hashes,
            hash_mask,
        })
    }
}

// ── BitSignedLcpMmphf constructors (Lcp2Mmphf) ───────────────────

#[cfg(feature = "rayon")]
impl<K, H, S, E>
    BitSignedLcpMmphf<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
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
    /// Creates a new bit-signed two-step LCP-based MMPHF for byte-sequence keys.
    ///
    /// `hash_width` is the number of hash bits stored per key (must be
    /// in `1..=H::BITS`). False-positive probability is
    /// 2<sup>−`hash_width`</sup>.
    pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>
        + Clone,
        n: usize,
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        assert!(hash_width > 0 && hash_width <= H::BITS as usize);
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let keys_for_hashes = keys.clone();
        let inner = Lcp2Mmphf::try_new(keys, n, pl)?;
        let hashes = fill_bit_hashes(
            inner.shard_edge(),
            inner.seed(),
            n,
            hash_width,
            hash_mask,
            keys_for_hashes,
            |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
        )?;
        Ok(Self {
            inner,
            hashes,
            hash_mask,
        })
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

use crate::traits::TryIntoUnaligned;

// -- SignedLcpMmphf: only inner needs converting, hashes stay as-is --

impl<F: TryIntoUnaligned, H> TryIntoUnaligned for SignedLcpMmphf<F, H> {
    type Unaligned = SignedLcpMmphf<F::Unaligned, H>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(SignedLcpMmphf {
            inner: self.inner.try_into_unaligned()?,
            hashes: self.hashes,
        })
    }
}

// -- BitSignedLcpMmphf: both inner and hashes are converted --

impl<F: TryIntoUnaligned, H: TryIntoUnaligned> TryIntoUnaligned for BitSignedLcpMmphf<F, H> {
    type Unaligned = BitSignedLcpMmphf<F::Unaligned, H::Unaligned>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(BitSignedLcpMmphf {
            inner: self.inner.try_into_unaligned()?,
            hashes: self.hashes.try_into_unaligned()?,
            hash_mask: self.hash_mask,
        })
    }
}
