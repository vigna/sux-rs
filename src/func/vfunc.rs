/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::borrow::Borrow;

use super::shard_edge::FuseLge3Shards;
use crate::bits::BitFieldVec;
use crate::func::shard_edge::ShardEdge;
use crate::traits::bit_field_slice::*;
use crate::utils::*;
use epserde::prelude::*;
use mem_dbg::*;

/// Static functions with low space overhead, fast parallel construction, and
/// fast queries.
///
/// *Static functions* map keys to values, but they do not store the keys:
/// querying a static function with a key outside of the original set will lead
/// to an arbitrary result. Another name for static functions *retrieval data
/// structure*. Values are retrieved using the [`get`](VFunc::get) method. On
/// some architectures, and with some constraints,
/// [`get_unaligned`](VFunc::get_unaligned) might be faster.
///
/// In exchange, static functions have a very low space overhead, and make it
/// possible to store the association between keys and values just in the space
/// required by the values (with a small overhead).
///
/// This structure is based on “[ε-Cost Sharding: Scaling Hypergraph-Based
/// Static Functions and Filters to Trillions of
/// Keys](https://arxiv.org/abs/2503.18397)”. Space overhead with respect to the
/// optimum depends on the [`ShardEdge`] type. The default is
/// [`FuseLge3Shards`], which provides 10.5% space overhead for large key sets
/// (above a few million keys), which grow up to 12% going towards smaller key
/// sets. Details on other possible [`ShardEdge`] implementations can be found
/// in the [`shard_edge`](crate::func::shard_edge) module documentation.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`). Please see the documentation of
/// [`VBuilder`](crate::func::VBuilder) for examples.
///
/// # Generics
///
/// * `T`: The type of the keys.
/// * `W`: The word used to store the data, which is also the output type. It
///   can be any unsigned type.
/// * `D`: The backend storing the function data. It can be a
///   [`BitFieldVec<W>`](crate::bits::BitFieldVec) or a `Box<[W]>`. In the first
///   case, the data is stored using exactly the number of bits needed, but
///   access is slightly slower, while in the second case the data is stored in
///   a boxed slice of `W`, thus forcing the number of bits to the number of
///   bits of `W`, but access will be faster. Note that for most bit sizes in
///   the first case on some architectures you can use [unaligned
///   reads](VFunc::get_unaligned) to get faster queries.
/// * `S`: The signature type. The default is `[u64; 2]`. You can switch to
///   `[u64; 1]` (and possibly
///   [`FuseLge3NoShards`](crate::func::shard_edge::FuseLge3NoShards)) for
///   slightly faster construction and queries, but the construction will not
///   scale beyond 3.8 billion keys.
/// * `E`: The sharding and edge logic type. The default is [`FuseLge3Shards`].
///   For small sets of keys you might try
///   [`FuseLge3NoShards`](crate::func::shard_edge::FuseLge3NoShards), possibly
///   coupled with `[u64; 1]` signatures. For functions with more than a few
///   dozen billion keys, you might try
///   `FuseLge3FullSigs`](crate::func::shard_edge::FuseLge3FullSigs).
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFunc<
    T: ?Sized + ToSig<S>,
    W: ZeroCopy + Word = usize,
    D: BitFieldSlice<W> = Box<[W]>,
    S: Sig = [u64; 2],
    E: ShardEdge<S, 3> = FuseLge3Shards,
> {
    pub(crate) shard_edge: E,
    pub(crate) seed: u64,
    pub(crate) num_keys: usize,
    pub(crate) data: D,
    pub(crate) _marker_t: std::marker::PhantomData<T>,
    pub(crate) _marker_w: std::marker::PhantomData<W>,
    pub(crate) _marker_s: std::marker::PhantomData<S>,
}

impl<T: ?Sized + ToSig<S>, W: ZeroCopy + Word, D: BitFieldSlice<W>, S: Sig, E: ShardEdge<S, 3>>
    VFunc<T, W, D, S, E>
{
    /// Returns the value associated with the given signature, or a random value
    /// if the signature is not the signature of a key.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> W {
        let edge = self.shard_edge.edge(sig);
        unsafe {
            self.data.get_unchecked(edge[0])
                ^ self.data.get_unchecked(edge[1])
                ^ self.data.get_unchecked(edge[2])
        }
    }

    /// Returns the value associated with the given key, or a random value if the
    /// key is not present.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> W {
        self.get_by_sig(T::to_sig(key.borrow(), self.seed))
    }

    /// Returns the number of keys in the function.
    pub fn len(&self) -> usize {
        self.num_keys
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ?Sized + ToSig<S>, W: ZeroCopy + Word, S: Sig, E: ShardEdge<S, 3>>
    VFunc<T, W, BitFieldVec<W>, S, E>
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
    pub fn get_by_sig_unaligned(&self, sig: S) -> W {
        let edge = self.shard_edge.edge(sig);
        unsafe {
            self.data.get_unaligned_unchecked(edge[0])
                ^ self.data.get_unaligned_unchecked(edge[1])
                ^ self.data.get_unaligned_unchecked(edge[2])
        }
    }

    /// Returns the value associated with the given key, or a random value if
    /// the key is not present, using [unaligned
    /// reads](BitFieldVec::get_unaligned).
    ///
    /// This method uses [`BitFieldVec::get_unaligned`], and has
    /// the same constraints.
    #[inline(always)]
    pub fn get_unaligned(&self, key: impl Borrow<T>) -> W {
        self.get_by_sig_unaligned(T::to_sig(key.borrow(), self.seed))
    }
}
