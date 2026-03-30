/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

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
use crate::func::lcp_mmphf::{LcpMmphf, LcpMmphfInt};
use crate::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use crate::func::mix64;
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

// ── Type aliases ────────────────────────────────────────────────────

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
/// # use sux::func::SignedLcpMmphfSliceU8;
/// # use sux::utils::FromSlice;
/// let keys: Vec<Vec<u8>> = vec![
///     b"alpha".to_vec(),
///     b"beta".to_vec(),
///     b"delta".to_vec(),
///     b"gamma".to_vec(),
/// ];
///
/// let func: SignedLcpMmphfSliceU8 = SignedLcpMmphfSliceU8::try_new(
///     FromSlice::new(&keys),
///     keys.len(),
///     no_logging![],
/// )?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_slice()), Some(i));
/// }
/// assert_eq!(func.get(b"not_a_key".as_slice()), None);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type SignedLcpMmphfSliceU8<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<LcpMmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── Integer impl ────────────────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: BitFieldSlice<Value = usize>,
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
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let rank = self.inner.get(key);
        let sig = T::to_sig(key, self.inner.offset_lcp_length.seed);
        let shard_edge = &self.inner.offset_lcp_length.shard_edge;
        let expected = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)));
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == <H::Value>::as_from(expected).as_to::<u64>() {
            Some(rank)
        } else {
            None
        }
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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

        let seed = inner.offset_lcp_length.seed;
        let shard_edge: &E = &inner.offset_lcp_length.shard_edge;
        let mut hashes = vec![W::MIN; n];
        let mut keys = keys_for_hashes;
        for hash in hashes.iter_mut() {
            let key: T = *keys.next()?.unwrap();
            let sig = T::to_sig(key, seed);
            *hash = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))).as_to::<W>();
        }

        Ok(Self {
            inner,
            hashes: hashes.into_boxed_slice(),
        })
    }
}

// ── Byte-sequence impl ──────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: BitFieldSlice<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<LcpMmphf<K, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    ///
    /// False positives happen with probability
    /// 2<sup>−`H::Value::BITS`</sup>.
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let rank = self.inner.get(key);
        let sig = K::to_sig(key, self.inner.offset_lcp_length.seed);
        let shard_edge = &self.inner.offset_lcp_length.shard_edge;
        let expected = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)));
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == <H::Value>::as_from(expected).as_to::<u64>() {
            Some(rank)
        } else {
            None
        }
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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

        let seed = inner.offset_lcp_length.seed;
        let shard_edge: &E = &inner.offset_lcp_length.shard_edge;
        let mut hashes = vec![W::MIN; n];
        let mut keys = keys_for_hashes;
        for hash in hashes.iter_mut() {
            let key = keys.next()?.unwrap();
            let k_ref: &K = <B as Borrow<K>>::borrow(key);
            let sig = K::to_sig(k_ref, seed);
            *hash = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))).as_to::<W>();
        }

        Ok(Self {
            inner,
            hashes: hashes.into_boxed_slice(),
        })
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

// ── Type aliases ────────────────────────────────────────────────────

/// A [`BitSignedLcpMmphf`] wrapping a [`LcpMmphfInt`].
pub type BitSignedLcpMmphfInt<T, H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<LcpMmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`LcpMmphf`] for `str` keys.
pub type BitSignedLcpMmphfStr<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<LcpMmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;

/// A [`BitSignedLcpMmphf`] wrapping a [`LcpMmphf`] for `[u8]` keys.
pub type BitSignedLcpMmphfSliceU8<H = BitFieldVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> =
    BitSignedLcpMmphf<LcpMmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

// ── Integer impl ────────────────────────────────────────────────────

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: BitFieldSlice<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedLcpMmphf<LcpMmphfInt<T, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    ///
    /// False positives happen with probability 2<sup>−`hash_width`</sup>
    /// (defined at construction time).
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let rank = self.inner.get(key);
        let sig = T::to_sig(key, self.inner.offset_lcp_length.seed);
        let shard_edge = &self.inner.offset_lcp_length.shard_edge;
        let expected = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))) & self.hash_mask;
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == expected { Some(rank) } else { None }
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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

        let seed = inner.offset_lcp_length.seed;
        let shard_edge: &E = &inner.offset_lcp_length.shard_edge;
        let mut hashes: BitFieldVec<Box<[H]>> =
            BitFieldVec::<Box<[H]>>::new_unaligned(hash_width, n);
        let mut keys = keys_for_hashes;
        for i in 0..n {
            let key: T = *keys.next()?.unwrap();
            let sig = T::to_sig(key, seed);
            let h =
                (mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))) & hash_mask).as_to::<H>();
            hashes.set_value(i, h);
        }

        Ok(Self {
            inner,
            hashes,
            hash_mask,
        })
    }
}

// ── Byte-sequence impl ──────────────────────────────────────────────

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: BitFieldSlice<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> BitSignedLcpMmphf<LcpMmphf<K, D, S, E>, H>
{
    /// Returns the rank of the given key if it was in the original set,
    /// or `None` if the verification hash does not match.
    ///
    /// False positives happen with probability 2<sup>−`hash_width`</sup>
    /// (defined at construction time).
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let rank = self.inner.get(key);
        let sig = K::to_sig(key, self.inner.offset_lcp_length.seed);
        let shard_edge = &self.inner.offset_lcp_length.shard_edge;
        let expected = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))) & self.hash_mask;
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == expected { Some(rank) } else { None }
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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

        let seed = inner.offset_lcp_length.seed;
        let shard_edge: &E = &inner.offset_lcp_length.shard_edge;
        let mut hashes: BitFieldVec<Box<[H]>> =
            BitFieldVec::<Box<[H]>>::new_unaligned(hash_width, n);
        let mut keys = keys_for_hashes;
        for i in 0..n {
            let key = keys.next()?.unwrap();
            let k_ref: &K = <B as Borrow<K>>::borrow(key);
            let sig = K::to_sig(k_ref, seed);
            let h =
                (mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))) & hash_mask).as_to::<H>();
            hashes.set_value(i, h);
        }

        Ok(Self {
            inner,
            hashes,
            hash_mask,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Type aliases and impls for Lcp2Mmphf (two-step variant)
// ═══════════════════════════════════════════════════════════════════

/// A [`SignedLcpMmphf`] wrapping a [`Lcp2MmphfInt`].
pub type SignedLcp2MmphfInt<T, H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<Lcp2MmphfInt<T, BitFieldVec<Box<[usize]>>, S, E>, H>;
/// A [`SignedLcpMmphf`] wrapping a [`Lcp2Mmphf`] for `str` keys.
pub type SignedLcp2MmphfStr<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<Lcp2Mmphf<str, BitFieldVec<Box<[usize]>>, S, E>, H>;
/// A [`SignedLcpMmphf`] wrapping a [`Lcp2Mmphf`] for `[u8]` keys.
pub type SignedLcp2MmphfSliceU8<H = Box<[u64]>, S = [u64; 2], E = FuseLge3Shards> =
    SignedLcpMmphf<Lcp2Mmphf<[u8], BitFieldVec<Box<[usize]>>, S, E>, H>;

impl<
    T: PrimitiveInteger + ToSig<S> + Copy,
    D: BitFieldSlice<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<Lcp2MmphfInt<T, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    #[inline]
    pub fn get(&self, key: T) -> Option<usize> {
        const {
            assert!(size_of::<H::Value>() <= size_of::<u64>());
        }
        let rank = self.inner.get(key);
        let sig = T::to_sig(key, self.inner.offsets.seed);
        let shard_edge = &self.inner.offsets.shard_edge;
        let expected = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)));
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == <H::Value>::as_from(expected).as_to::<u64>() {
            Some(rank)
        } else {
            None
        }
    }
    pub const fn len(&self) -> usize {
        self.inner.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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
        let seed = inner.offsets.seed;
        let shard_edge: &E = &inner.offsets.shard_edge;
        let mut hashes = vec![W::MIN; n];
        let mut keys = keys_for_hashes;
        for hash in hashes.iter_mut() {
            let key: T = *keys.next()?.unwrap();
            *hash = mix64(shard_edge.edge_hash(shard_edge.local_sig(T::to_sig(key, seed))))
                .as_to::<W>();
        }
        Ok(Self {
            inner,
            hashes: hashes.into_boxed_slice(),
        })
    }
}

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S>,
    D: BitFieldSlice<Value = usize>,
    H: SliceByValue<Value: PrimitiveNumber>,
    S: Sig,
    E: ShardEdge<S, 3>,
> SignedLcpMmphf<Lcp2Mmphf<K, D, S, E>, H>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    #[inline]
    pub fn get(&self, key: &K) -> Option<usize> {
        const {
            assert!(size_of::<H::Value>() <= size_of::<u64>());
        }
        let rank = self.inner.get(key);
        let sig = K::to_sig(key, self.inner.offsets.seed);
        let shard_edge = &self.inner.offsets.shard_edge;
        let expected = mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)));
        let stored = self.hashes.get_value(rank)?.as_to::<u64>();
        if stored == <H::Value>::as_from(expected).as_to::<u64>() {
            Some(rank)
        } else {
            None
        }
    }
    pub const fn len(&self) -> usize {
        self.inner.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

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
        let seed = inner.offsets.seed;
        let shard_edge: &E = &inner.offsets.shard_edge;
        let mut hashes = vec![W::MIN; n];
        let mut keys = keys_for_hashes;
        for hash in hashes.iter_mut() {
            let key = keys.next()?.unwrap();
            let k_ref: &K = <B as Borrow<K>>::borrow(key);
            *hash = mix64(shard_edge.edge_hash(shard_edge.local_sig(K::to_sig(k_ref, seed))))
                .as_to::<W>();
        }
        Ok(Self {
            inner,
            hashes: hashes.into_boxed_slice(),
        })
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

use crate::traits::{BitFieldSlice, TryIntoUnaligned};

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
