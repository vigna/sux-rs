/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::borrow::Borrow;

use crate::func::shard_edge::ShardEdge;
use crate::traits::bit_field_slice::*;
use crate::utils::*;
use crate::{bits::BitFieldVec, func::VFunc};
use common_traits::{UpcastableFrom, UpcastableInto};
use mem_dbg::*;
use value_traits::slices::SliceByValue;

#[derive(Debug, MemDbg, MemSize)]
//#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SignedVFunc<F, H: SliceByValue<Value: Copy>> {
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
    H: SliceByValue<Value: Copy>,
> SignedVFunc<VFunc<T, W, D, S, E>, H>
where
    H::Value: UpcastableInto<u64>,
    usize: UpcastableFrom<W>,
{
    /// Returns the value associated with the given signature, or a random value
    /// if the signature is not the signature of a key.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> Option<W> {
        let index = self.func.get_by_sig(sig);
        let shard_edge = &self.func.shard_edge;
        if self.hashes.get_value(index.upcast())?.upcast()
            == (crate::func::mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)))
                & self.hash_mask)
        {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the value associated with the given key, or a random value if the
    /// key is not present.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> Option<W> {
        self.get_by_sig(T::to_sig(key.borrow(), self.func.seed))
    }

    /// Returns the number of keys in the function.
    pub fn len(&self) -> usize {
        self.func.num_keys
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue<Value: Copy>,
> SignedVFunc<VFunc<T, W, BitFieldVec<W>, S, E>, H>
where
    H::Value: UpcastableInto<u64>,
    usize: UpcastableFrom<W>,
{
    /// Returns the value associated with the given signature, or a random value
    /// if the signature is not the signature of a key, using [unaligned
    /// reads](BitFieldVec::get_unaligned).
    ///
    /// This method uses [`BitFieldVec::get_unaligned`], and has
    /// the same constraints.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig_unaligned(&self, sig: S) -> Option<W> {
        let index = self.func.get_by_sig_unaligned(sig);
        let shard_edge = &self.func.shard_edge;
        if self.hashes.get_value(index.upcast())?.upcast()
            == (crate::func::mix64(shard_edge.edge_hash(shard_edge.local_sig(sig)))
                & self.hash_mask)
        {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the value associated with the given key, or a random value if
    /// the key is not present, using [unaligned
    /// reads](BitFieldVec::get_unaligned).
    ///
    /// This method uses [`BitFieldVec::get_unaligned`], and has
    /// the same constraints.
    #[inline(always)]
    pub fn get_unaligned(&self, key: impl Borrow<T>) -> Option<W> {
        self.get_by_sig_unaligned(T::to_sig(key.borrow(), self.func.seed))
    }
}
