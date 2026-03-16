/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Index functions with signed values.

use std::borrow::Borrow;

use crate::func::shard_edge::ShardEdge;
use crate::traits::bit_field_slice::*;
use crate::utils::*;
use crate::{bits::BitFieldVec, func::VFunc};
use mem_dbg::*;
use num_primitive::PrimitiveNumber;
use value_traits::slices::SliceByValue;

/// A signed index function using a [`SliceByValue`] to store hashes.
///
/// Usually, the [`SliceByValue`] will be a boxed slice. Note that the result of
/// the [`SliceByValue`] is assumed to be a hash of size
/// `SliceByValue::Value::BITS`. If you are using implementations returning less
/// hash bits (such as a [`BitFieldVec`]), you will need to use
/// [`BitSignedVFunc`] instead.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SignedVFunc<F, H: SliceByValue> {
    pub(crate) func: F,
    pub(crate) hashes: H,
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    D: SliceByValue<Value = W>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue,
> SignedVFunc<VFunc<T, W, D, S, E>, H>
where
    H::Value: PrimitiveNumber,
{
    /// Returns the index of a key associated with the given signature, if there
    /// was such a key in the list provided at construction time; otherwise,
    /// returns `None`.
    ///
    /// False positives happen with probability
    /// 2<sup>–`SliceByValue::Value::BITS`</sup>.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> Option<W> {
        // Static check that H::Value → u64 conversion is lossless
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let index = self.func.get_by_sig(sig);
        let shard_edge = &self.func.shard_edge;
        // as_to is safe: index is bounded by num_keys, which is a usize
        if self
            .hashes
            .get_value(index.as_to::<usize>())?
            .as_to::<u64>()
            == <H::Value>::as_from(
                crate::func::mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))),
            )
            .as_to::<u64>()
        {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the index of given key, if the key was in the list provided at
    /// construction time; otherwise, returns `None`.
    ///
    /// False positives happen with probability
    /// 2<sup>–`SliceByValue::Value::BITS`</sup>.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> Option<W> {
        self.get_by_sig(T::to_sig(key.borrow(), self.func.seed))
    }

    /// Returns the number of keys in the function.
    pub const fn len(&self) -> usize {
        self.func.num_keys
    }

    /// Returns whether the function has no keys.
    pub const fn is_empty(&self) -> bool {
        self.func.num_keys == 0
    }
}

/// A bit-signed index function using a [`SliceByValue`] to store hashes.
///
/// This structure contains a `hash_mask`, and values returned by the
/// [`SliceByValue`] are compared only on the masked bits. This approach makes
/// it possible to have, for example, signature stored in a [`BitFieldVec`]
/// using fewer bits than the integer type supporting the [`BitFieldVec`]. If you
/// are using all the bits of the type (e.g., 16-bit signatures on `u16`),
/// please consider using a [`SignedVFunc`] as hash comparison will be faster.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitSignedVFunc<F, H: SliceByValue> {
    pub(crate) func: F,
    pub(crate) hashes: H,
    pub(crate) hash_mask: u64,
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    D: SliceByValue<Value = W>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue,
> BitSignedVFunc<VFunc<T, W, D, S, E>, H>
where
    H::Value: PrimitiveNumber,
{
    /// Returns the index of a key associated with the given signature, if there
    /// was such a key in the list provided at construction time; otherwise,
    /// returns `None`.
    ///
    /// False positives happen with probability defined at [construction
    /// time](crate::func::VBuilder::try_build_bit_sig_index).
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> Option<W> {
        // Static check that H::Value → u64 conversion is lossless
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let index = self.func.get_by_sig(sig);
        let shard_edge = &self.func.shard_edge;
        // as_to is safe: index is bounded by num_keys, which is a usize
        if self
            .hashes
            .get_value(index.as_to::<usize>())?
            .as_to::<u64>()
            == (crate::func::mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)))
                & self.hash_mask)
        {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the index of given key, if the key was in the list provided at
    /// construction time; otherwise, returns `None`.
    ///
    /// False positives happen with probability defined at [construction
    /// time](crate::func::VBuilder::try_build_bit_sig_index).
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> Option<W> {
        self.get_by_sig(T::to_sig(key.borrow(), self.func.seed))
    }

    /// Returns the number of keys in the function.
    pub const fn len(&self) -> usize {
        self.func.num_keys
    }

    /// Returns whether the function has no keys.
    pub const fn is_empty(&self) -> bool {
        self.func.num_keys == 0
    }
}

impl<T: ?Sized + ToSig<S>, W: Word + BinSafe, S: Sig, E: ShardEdge<S, 3>, H: SliceByValue>
    BitSignedVFunc<VFunc<T, W, BitFieldVec<W>, S, E>, H>
where
    H::Value: PrimitiveNumber,
{
    /// Returns the index of a key associated with the given signature, if there
    /// was such a key in the list provided at construction time; otherwise,
    /// returns `None`, using [unaligned reads](BitFieldVec::get_unaligned).
    ///
    /// False positives happen with probability defined at [construction
    /// time](crate::func::VBuilder::try_build_bit_sig_index).
    ///
    /// This method uses [`BitFieldVec::get_unaligned`], and has
    /// the same constraints.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig_unaligned(&self, sig: S) -> Option<W> {
        // Static check that H::Value → u64 conversion is lossless
        const {
            assert!(
                size_of::<H::Value>() <= size_of::<u64>(),
                "Hash value type must fit in u64 without truncation"
            );
        }
        let index = self.func.get_by_sig_unaligned(sig);
        let shard_edge = &self.func.shard_edge;
        // as_to is safe: index is bounded by num_keys, which is a usize
        if self
            .hashes
            .get_value(index.as_to::<usize>())?
            .as_to::<u64>()
            == (crate::func::mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)))
                & self.hash_mask)
        {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the index of given key, if the key was in the list provided at
    /// construction time; otherwise, returns `None`, using [unaligned
    /// reads](BitFieldVec::get_unaligned).
    ///
    /// False positives happen with probability defined at [construction
    /// time](crate::func::VBuilder::try_build_bit_sig_index).
    ///
    /// This method uses [`BitFieldVec::get_unaligned`], and has
    /// the same constraints.
    #[inline(always)]
    pub fn get_unaligned(&self, key: impl Borrow<T>) -> Option<W> {
        self.get_by_sig_unaligned(T::to_sig(key.borrow(), self.func.seed))
    }
}
