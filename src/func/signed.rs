/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity, private_bounds)]

//! Signed static functions and monotone minimal perfect hash functions.
//!
//! A signed function stores for each key a hash. When querying a key, the
//! function first computes the hash of the key and compares it with the stored
//! hash for the returned index. If the hashes match, the index is returned;
//! otherwise, [`None`] is returned. This allows the function to reject queries
//! for keys outside the original set, with a false-positive probability
//! depending on the size of the stored hashes.
//!
//! Two wrappers are provided:
//!
//! - [`SignedFunc`] stores full-width hashes (e.g., `Box<[u64]>`).
//!   False-positive probability is 2<sup>-`H::Value::BITS`</sup>.
//! - [`BitSignedFunc`] stores hashes in a [`BitFieldVec`] with a
//!   caller-chosen bit width. False-positive probability is
//!   2<sup>-`hash_width`</sup>.
//!
//! Both wrappers are generic over the inner function `F` (which must implement
//! [`SignableFunc`]) and the hash storage `H`. Per-inner-type `get` methods are
//! provided via monomorphized `impl` blocks.
//!
//! See [`SignedLcpMmphfInt`], [`SignedLcpMmphfStr`], [`BitSignedLcpMmphfInt`],
//! [`BitSignedLcpMmphfStr`], etc., for convenience type aliases.

use std::borrow::Borrow;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    crate::utils::FallibleRewindableLender,
    anyhow::Result,
    core::error::Error,
    dsi_progress_logger::ProgressLog,
    lender::*,
    num_primitive::PrimitiveNumberAs,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
    value_traits::slices::SliceByValueMut,
};

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::VFunc;
use crate::func::lcp_mmphf::{LcpMmphf, LcpMmphfInt};
use crate::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use crate::func::shard_edge::{Fuse3Shards, FuseLge3Shards, ShardEdge};
use crate::utils::*;
use mem_dbg::*;
use num_primitive::{PrimitiveInteger, PrimitiveNumber};
use value_traits::slices::SliceByValue;

/// Common interface for inner functions used by signed wrappers.
///
/// This trait is not intended to be implemented by users; it iss an internal
/// abstraction to allow the signed wrappers to work with different static
/// functions. It provides access to the seed, shard edge, and key count, so
/// that [`SignedFunc`] and [`BitSignedFunc`] can verify hashes without knowing
/// which specific MMPHF variant they wrap.
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

// ═══════════════════════════════════════════════════════════════════
// SignedFunc — full-width hashes
// ═══════════════════════════════════════════════════════════════════

/// A signed function using a [`SliceByValue`] to store full-width hashes.
///
/// Wraps an inner function `F` (any type implementing [`SignableFunc`]) and adds
/// per-key verification hashes so that queries for keys outside the original
/// set return `None` (with false-positive probability
/// 2<sup>-`H::Value::BITS`</sup>).
///
/// Usually, the [`SliceByValue`] will be a boxed slice. Note that the result of
/// the [`SliceByValue`] is assumed to be a hash of size
/// `SliceByValue::Value::BITS`. If you are using implementations returning less
/// hash bits (such as a [`BitFieldVec<Box<[W]>>`](BitFieldVec)), you will need to use
/// [`BitSignedFunc`] instead.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
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

impl<F: SignableFunc, H: SliceByValue<Value: PrimitiveNumber>> SignedFunc<F, H> {
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
        let expected = self.func.shard_edge().remixed_hash(sig);
        let stored = self
            .hashes
            .get_value(index.as_to::<usize>())?
            .as_to::<u64>();
        if stored == <H::Value>::as_from(expected).as_to::<u64>() {
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
    W: Word + BinSafe,
    D: SliceByValue<Value = W>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue<Value: PrimitiveNumber>,
> SignedFunc<VFunc<T, D, S, E>, H>
{
    /// Returns the index of a key associated with the given signature, if there
    /// was such a key in the list provided at construction time; otherwise,
    /// returns `None`.
    ///
    /// False positives happen with probability
    /// 2<sup>-`SliceByValue::Value::BITS`</sup>.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> Option<W> {
        self.verify(self.func.get_by_sig(sig), sig)
    }

    /// Returns the index of the given key, if the key was in the list provided at
    /// construction time; otherwise, returns `None`.
    ///
    /// False positives happen with probability
    /// 2<sup>-`SliceByValue::Value::BITS`</sup>.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> Option<W> {
        self.get_by_sig(T::to_sig(key.borrow(), self.func.seed()))
    }
}

// ── LcpMmphfInt `get` ────────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: SliceByValue<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedFunc<LcpMmphfInt<T, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    ///
    /// False positives happen with probability
    /// 2<sup>-`H::Value::BITS`</sup>.
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
    H: SliceByValue<Value: PrimitiveNumber>,
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
    H: SliceByValue<Value: PrimitiveNumber>,
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
    H: SliceByValue<Value: PrimitiveNumber>,
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
// BitSignedFunc — sub-word-width hashes in a BitFieldVec
// ═══════════════════════════════════════════════════════════════════

/// A bit-signed function using a [`SliceByValue`] to store hashes.
///
/// This structure contains a `hash_mask`, and values returned by the
/// [`SliceByValue`] are compared only on the masked bits. This approach makes
/// it possible to have, for example, signatures stored in a [`BitFieldVec`]
/// using fewer bits than the integer type supporting the [`BitFieldVec`]. If you
/// are using all the bits of the type (e.g., 16-bit signatures on `u16`),
/// please consider using a [`SignedFunc`] as hash comparison will be faster.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitSignedFunc<F, H> {
    pub(crate) func: F,
    pub(crate) hashes: H,
    pub(crate) hash_mask: u64,
}

impl<F, H> BitSignedFunc<F, H> {
    /// Creates a new `BitSignedFunc` from a function, a hash slice, and
    /// a hash mask.
    ///
    /// This is a low-level constructor; prefer
    /// [`try_new`](Self::try_new)/[`try_new_with_builder`](Self::try_new_with_builder)
    /// when possible.
    pub fn from_parts(func: F, hashes: H, hash_mask: u64) -> Self {
        Self {
            func,
            hashes,
            hash_mask,
        }
    }
}

// ── Unified query helpers ────────────────────────────────────────

impl<F: SignableFunc, H: SliceByValue<Value: PrimitiveNumber>> BitSignedFunc<F, H> {
    /// Verifies that the stored hash matches the masked remixed hash for
    /// the given index and signature.
    #[inline(always)]
    fn verify<V: PrimitiveNumber>(&self, index: V, sig: F::Sig) -> Option<V> {
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let expected = self.func.shard_edge().remixed_hash(sig) & self.hash_mask;
        let stored = self
            .hashes
            .get_value(index.as_to::<usize>())?
            .as_to::<u64>();
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
    H: SliceByValue<Value: PrimitiveNumber>,
> BitSignedFunc<VFunc<T, D, S, E>, H>
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
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedFunc<LcpMmphfInt<T, D, S, E>, H>
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
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedFunc<LcpMmphf<K, D, S, E>, H>
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
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedFunc<Lcp2MmphfInt<T, D, S, E>, H>
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
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedFunc<Lcp2Mmphf<K, D, S, E>, H>
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
// Backward-compatible type aliases
// ═══════════════════════════════════════════════════════════════════

// ── SignedFunc convenience aliases (LcpMmphf) ────────────────────

/// A [`SignedFunc`] wrapping a [`LcpMmphfInt`].
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
    SignedFunc<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`SignedFunc`] wrapping a [`LcpMmphf`] for `str` keys.
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
    SignedFunc<LcpMmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`SignedFunc`] wrapping a [`LcpMmphf`] for `[u8]` keys.
pub type SignedLcpMmphfSliceU8<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedFunc<LcpMmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── SignedFunc convenience aliases (Lcp2Mmphf) ───────────────────

/// A [`SignedFunc`] wrapping a [`Lcp2MmphfInt`].
pub type SignedLcp2MmphfInt<T, H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedFunc<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;
/// A [`SignedFunc`] wrapping a [`Lcp2Mmphf`] for `str` keys.
pub type SignedLcp2MmphfStr<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedFunc<Lcp2Mmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;
/// A [`SignedFunc`] wrapping a [`Lcp2Mmphf`] for `[u8]` keys.
pub type SignedLcp2MmphfSliceU8<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedFunc<Lcp2Mmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── BitSignedFunc convenience aliases (LcpMmphf) ────────────────

/// A [`BitSignedFunc`] wrapping a [`LcpMmphfInt`].
pub type BitSignedLcpMmphfInt<T, H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedFunc<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedFunc`] wrapping a [`LcpMmphf`] for `str` keys.
pub type BitSignedLcpMmphfStr<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedFunc<LcpMmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedFunc`] wrapping a [`LcpMmphf`] for `[u8]` keys.
pub type BitSignedLcpMmphfSliceU8<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedFunc<LcpMmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── BitSignedFunc convenience aliases (Lcp2Mmphf) ───────────────

/// A [`BitSignedFunc`] wrapping a [`Lcp2MmphfInt`].
pub type BitSignedLcp2MmphfInt<T, H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedFunc<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedFunc`] wrapping a [`Lcp2Mmphf`] for `str` keys.
pub type BitSignedLcp2MmphfStr<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedFunc<Lcp2Mmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedFunc`] wrapping a [`Lcp2Mmphf`] for `[u8]` keys.
pub type BitSignedLcp2MmphfSliceU8<
    H = BitFieldVec<Box<[usize]>>,
    S = [u64; 2],
    E = FuseLge3Shards,
> = BitSignedFunc<Lcp2Mmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ═══════════════════════════════════════════════════════════════════
// Constructors — helper functions
// ═══════════════════════════════════════════════════════════════════

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
            false,
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
// Constructors — BitSignedFunc<VFunc<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, S, E, H> BitSignedFunc<VFunc<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    H: crate::traits::Word,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
{
    /// Builds a [`BitSignedFunc`] wrapping a [`VFunc`] from keys using
    /// default [`VBuilder`] settings.
    ///
    /// The function maps each key to its index in the input sequence
    /// and stores `hash_width`-bit hashes for verification, giving a
    /// false-positive rate of 2<sup>-`hash_width`</sup>.
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
    /// use sux::func::{BitSignedFunc, VFunc};
    /// use sux::bits::BitFieldVec;
    /// type BSFunc = BitSignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;
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

    /// Builds a [`BitSignedFunc`] wrapping a [`VFunc`] from keys using
    /// the given [`VBuilder`] configuration.
    ///
    /// The function maps each key to its index in the input sequence
    /// and stores `hash_width`-bit hashes for verification, giving a
    /// false-positive rate of 2<sup>-`hash_width`</sup>.
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
    /// use sux::func::{BitSignedFunc, VBuilder, VFunc};
    /// use sux::bits::BitFieldVec;
    /// type BSFunc = BitSignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;
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
            false,
            pl,
        )?;

        let num_keys = func.len();
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        // Create the signature vector
        let mut hashes: BitFieldVec<Box<[H]>> =
            BitFieldVec::<Box<[H]>>::new_unaligned(hash_width, num_keys);

        // Enumerate the store and extract hashes
        pl.item_name("hash");
        pl.expected_updates(Some(num_keys));
        pl.start("Storing hashes...");

        for shard in store.iter() {
            for sig_val in shard.iter() {
                let pos = sig_val.val;
                let hash = (func.shard_edge().remixed_hash(sig_val.sig) & hash_mask).as_to::<H>();
                hashes.set_value(pos, hash);
                pl.light_update();
            }
        }

        pl.done();

        Ok(BitSignedFunc {
            func,
            hashes,
            hash_mask,
        })
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
        let (func, _, keys) = LcpMmphfInt::try_new_inner(keys, n, builder, true, pl)?;
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
        let (func, _, keys) = LcpMmphf::try_new_inner(keys, n, builder, true, pl)?;
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
// Constructors — BitSignedFunc<LcpMmphfInt<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, H, S, E>
    BitSignedFunc<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
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
    /// 2<sup>-`hash_width`</sup>.
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
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let (func, _, keys) = LcpMmphfInt::try_new_inner(keys, n, builder, true, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_bit_hashes(
            func.shard_edge(),
            func.seed(),
            n,
            hash_width,
            hash_mask,
            &mut keys,
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self {
            func,
            hashes,
            hash_mask,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — BitSignedFunc<LcpMmphf<K, ...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<K, H, S, E> BitSignedFunc<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
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
    /// 2<sup>-`hash_width`</sup>.
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
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let (func, _, keys) = LcpMmphf::try_new_inner(keys, n, builder, true, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_bit_hashes(
            func.shard_edge(),
            func.seed(),
            n,
            hash_width,
            hash_mask,
            &mut keys,
            |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
        )?;
        Ok(Self {
            func,
            hashes,
            hash_mask,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — BitSignedFunc<Lcp2MmphfInt<...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<T, H, S, E>
    BitSignedFunc<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
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
    /// # use sux::func::BitSignedLcp2MmphfInt;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func: BitSignedLcp2MmphfInt<u64> =
    ///     BitSignedLcp2MmphfInt::try_new(
    ///         FromSlice::new(&keys), keys.len(), 8, no_logging![],
    ///     )?;
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
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let (func, keys) = Lcp2MmphfInt::try_new_inner(keys, n, builder, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_bit_hashes(
            func.shard_edge(),
            func.seed(),
            n,
            hash_width,
            hash_mask,
            &mut keys,
            |key, seed| T::to_sig(*key, seed),
        )?;
        Ok(Self {
            func,
            hashes,
            hash_mask,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors — BitSignedFunc<Lcp2Mmphf<K, ...>>
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
impl<K, H, S, E> BitSignedFunc<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
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
    /// # use sux::func::BitSignedLcp2MmphfStr;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromSlice;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func: BitSignedLcp2MmphfStr =
    ///     BitSignedLcp2MmphfStr::try_new(
    ///         FromSlice::new(&keys), keys.len(), 12, no_logging![],
    ///     )?;
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
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        let (func, keys) = Lcp2Mmphf::try_new_inner(keys, n, builder, pl)?;
        let mut keys = keys.rewind()?;
        let hashes = fill_bit_hashes(
            func.shard_edge(),
            func.seed(),
            n,
            hash_width,
            hash_mask,
            &mut keys,
            |key, seed| K::to_sig(<B as Borrow<K>>::borrow(key), seed),
        )?;
        Ok(Self {
            func,
            hashes,
            hash_mask,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Aligned <-> Unaligned conversions
// ═══════════════════════════════════════════════════════════════════

use crate::traits::{TryIntoUnaligned, Word};

// -- SignedFunc: only func needs converting, hashes stay as-is --

impl<F: TryIntoUnaligned, H> TryIntoUnaligned for SignedFunc<F, H> {
    type Unaligned = SignedFunc<F::Unaligned, H>;
    fn try_into_unaligned(
        self,
    ) -> std::result::Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(SignedFunc {
            func: self.func.try_into_unaligned()?,
            hashes: self.hashes,
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

// -- BitSignedFunc: both func and hashes are converted --

impl<F: TryIntoUnaligned, H: TryIntoUnaligned> TryIntoUnaligned for BitSignedFunc<F, H> {
    type Unaligned = BitSignedFunc<F::Unaligned, H::Unaligned>;
    fn try_into_unaligned(
        self,
    ) -> std::result::Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(BitSignedFunc {
            func: self.func.try_into_unaligned()?,
            hashes: self.hashes.try_into_unaligned()?,
            hash_mask: self.hash_mask,
        })
    }
}

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>>
    From<BitSignedFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, BitFieldVecU<Box<[W]>>>>
    for BitSignedFunc<VFunc<T, BitFieldVec<Box<[W]>>, S, E>, BitFieldVec<Box<[W]>>>
{
    fn from(
        f: BitSignedFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, BitFieldVecU<Box<[W]>>>,
    ) -> Self {
        BitSignedFunc {
            func: f.func.into(),
            hashes: f.hashes.into(),
            hash_mask: f.hash_mask,
        }
    }
}
