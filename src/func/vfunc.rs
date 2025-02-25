/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use crate::bits::*;
use crate::traits::bit_field_slice::*;
use crate::utils::*;
use common_traits::CastableInto;
use epserde::prelude::*;
use mem_dbg::*;
use std::borrow::Borrow;
use std::ops::Index;

/// Static functions with 10%-11% space overhead for large key sets, fast
/// parallel construction, and fast queries.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`).
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFunc<
    T: ?Sized + ToSig<S>,
    W: ZeroCopy + Word = usize,
    D: BitFieldSlice<W> = BitFieldVec<W>,
    S: Sig = [u64; 2],
    const SHARDED: bool = false,
> {
    pub(in crate::func) shard_high_bits: u32,
    pub(in crate::func) log2_seg_size: u32,
    pub(in crate::func) shard_mask: u32,
    pub(in crate::func) seed: u64,
    pub(in crate::func) l: usize,
    pub(in crate::func) num_keys: usize,
    pub(in crate::func) data: D,
    pub(in crate::func) _marker_t: std::marker::PhantomData<T>,
    pub(in crate::func) _marker_w: std::marker::PhantomData<W>,
    pub(in crate::func) _marker_s: std::marker::PhantomData<S>,
}

/// Filters with 10%-11% space overhead for large key sets, fast parallel
/// construction, and fast queries.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`).
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFilter<W: ZeroCopy + Word, F> {
    pub(in crate::func) func: F,
    pub(in crate::func) filter_mask: W,
}

/// Shard and edge information.
///
/// This trait is used to derive shards and edges from key signatures. The
/// `shard` method returns the shard; the `edge` method is used to derive the
/// three vertices of the edge associated with the key; the `shard_edge` method
/// is used to derive the three vertices of the edge associated with a key
/// inside the shard the key belongs to (its default implementation just calls
/// `edge` with shard zero).
///
/// Edges must follow the generation method of fuse graphs (see ”[Dense Peelable
/// Random Uniform Hypergraphs](https://doi.org/10.4230/LIPIcs.ESA.2019.38)”.
/// Using the signature bits as a seed, one extracts a pseudorandom first
/// segment between 0 and `l`, and then chooses a pseudorandom position in the
/// first segment and in the following two (or more, in the case of higher
/// degree).
pub trait ShardEdge: Send + Sync {
    fn shard(&self, shard_high_bits: u32, shard_mask: u32) -> usize;

    #[inline(always)]
    #[must_use]
    fn shard_edge(&self, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        self.edge(0, shard_high_bits, l, log2_seg_size)
    }

    fn edge(&self, shard: usize, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3];
}
/// In this implementation:
/// - the `shard_high_bits` most significant bits of the first component are
///   used to select a shard;
/// - the following 32 bits of the first component are used to select the first
///   segment using fixed-point arithmetic;
/// - the lower 32 bits of first component are used to select the first vertex;
/// - the upper 32 bits of the second component are used to select the second
///   vertex;
/// - the lower 32 bits of the second component are used to select the third
///   vertex.
///
/// Note that the lower `shard_high_bits` of the bits used to select the first
/// segment are the same as the upper `shard_high_bits` of the bits used to
/// select the first segment, but being the result mostly sensitive to the high
/// bits, this is not a problem.
impl ShardEdge for [u64; 2] {
    #[inline(always)]
    #[must_use]
    fn shard(&self, shard_high_bits: u32, shard_mask: u32) -> usize {
        // This must work even when shard_high_bits is zero
        (self[0].rotate_left(shard_high_bits) & shard_mask as u64) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(&self, shard: usize, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        let first_segment = (((self[0] << shard_high_bits >> 32) * l as u64) >> 32) as usize;
        let shard_offset = shard * ((l + 2) << log2_seg_size);
        let start = shard_offset + (first_segment << log2_seg_size);
        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        [
            (self[0] as usize & segment_mask) + start,
            ((self[1] >> 32) as usize & segment_mask) + start + segment_size,
            (self[1] as usize & segment_mask) + start + 2 * segment_size,
        ]
    }
}

/// In this implementation:
/// - the `shard_high_bits` most significant bits of the two 32-bit halves XOR'd
///   together are used to select a shard;
/// - the two 32-bit halves XOR'd together and rotated to the left by
///   `shard_high_bits` are used to select the first segment using fixed-point
///   arithmetic;
/// - the lower 21 bits are used to select the first vertex;
/// - the next 21 bits are used to select the second vertex;
/// - the next 21 bits are used to select the third vertex.
///
/// Note that the lower `shard_high_bits` of the bits used to select the first
/// segment are the same as the bits used to select the shard, but being the
/// result mostly sensitive to the high bits, this is not a problem.
impl ShardEdge for [u64; 1] {
    fn shard(&self, shard_high_bits: u32, shard_mask: u32) -> usize {
        // This must work even when shard_high_bits is zero
        let xor = self[0] as u32 ^ (self[0] >> 32) as u32;
        (xor.rotate_left(shard_high_bits) & shard_mask) as usize
    }

    fn edge(&self, shard: usize, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        let xor = self[0] as u32 ^ (self[0] >> 32) as u32;
        let first_segment = ((xor.rotate_left(shard_high_bits) as u64 * l as u64) >> 32) as usize;
        let shard_offset = shard * ((l + 2) << log2_seg_size);
        let start = shard_offset + (first_segment << log2_seg_size);
        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        [
            (self[0] as usize & segment_mask) + start,
            ((self[0] >> 21) as usize & segment_mask) + start + segment_size,
            ((self[0] >> 42) as usize & segment_mask) + start + 2 * segment_size,
        ]
    }
}

impl<
        T: ?Sized + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W>,
        S: Sig + ShardEdge,
        const SHARDED: bool,
    > VFunc<T, W, D, S, SHARDED>
{
    /// Return the value associated with the given signature, or a random value
    /// if the signature is not the signature of a key .
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: &S) -> W {
        if SHARDED {
            let shard = sig.shard(self.shard_high_bits, self.shard_mask);
            // shard * self.segment_size * (2^log2_l + 2)
            let edge = sig.edge(shard, self.shard_high_bits, self.l, self.log2_seg_size);

            unsafe {
                self.data.get_unchecked(edge[0])
                    ^ self.data.get_unchecked(edge[1])
                    ^ self.data.get_unchecked(edge[2])
            }
        } else {
            let edge = sig.shard_edge(0, self.l, self.log2_seg_size);
            unsafe {
                self.data.get_unchecked(edge[0])
                    ^ self.data.get_unchecked(edge[1])
                    ^ self.data.get_unchecked(edge[2])
            }
        }
    }

    /// Return the value associated with the given key, or a random value if the
    /// key is not present.
    #[inline(always)]
    pub fn get(&self, key: &T) -> W {
        self.get_by_sig(&T::to_sig(key, self.seed))
    }

    /// Return the number of keys in the function.
    pub fn len(&self) -> usize {
        self.num_keys
    }

    /// Return whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<
        T: ?Sized + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W>,
        S: Sig + ShardEdge,
        const SHARDED: bool,
    > VFilter<W, VFunc<T, W, D, S, SHARDED>>
where
    u64: CastableInto<W>,
{
    /// Return the value associated with the given signature by the underlying
    /// function, or a random value if the signature is not the signature of a
    /// key .
    ///
    /// The user should not normally call this method, but rather
    /// [`contains_by_sig`](VFilter::contains_by_sig).
    #[inline(always)]
    pub fn get_by_sig(&self, sig: &S) -> W {
        self.func.get_by_sig(sig)
    }

    /// Return the value associated with the given key by the underlying
    /// function, or a random value if the key is not present.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline]
    pub fn get(&self, key: &T) -> W {
        self.func.get(key)
    }

    /// Return whether a signature is contained in the filter.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline(always)]
    pub fn contains_by_sig(&self, sig: &S) -> bool {
        self.func.get_by_sig(sig) == sig.sig_u64().cast() & self.filter_mask
    }

    /// Return whether a key is contained in the filter.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline]
    pub fn contains(&self, key: &T) -> bool {
        self.contains_by_sig(&T::to_sig(key, self.func.seed))
    }

    /// Return the number of keys in the filter.
    pub fn len(&self) -> usize {
        self.func.len()
    }

    /// Return whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty()
    }
}

impl<
        T: ?Sized + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W>,
        S: Sig + ShardEdge,
        const SHARDED: bool,
        B: Borrow<T>,
    > Index<B> for VFilter<W, VFunc<T, W, D, S, SHARDED>>
where
    u64: CastableInto<W>,
{
    type Output = bool;

    #[inline(always)]
    fn index(&self, key: B) -> &Self::Output {
        if self.contains(key.borrow()) {
            &true
        } else {
            &false
        }
    }
}

#[cfg(test)]
mod tests {
    use dsi_progress_logger::no_logging;
    use epserde::{deser::Deserialize, ser::Serialize, utils::AlignedCursor};
    use rdst::RadixKey;

    use crate::{
        func::VBuilder,
        utils::{FromIntoIterator, Sig, SigVal, ToSig},
    };

    use super::{ShardEdge, VFilter, VFunc};

    #[test]
    fn test_filter_func() -> anyhow::Result<()> {
        _test_filter_func::<[u64; 1]>()?;
        _test_filter_func::<[u64; 2]>()?;
        Ok(())
    }

    fn _test_filter_func<S: Sig + ShardEdge + Send + Sync>() -> anyhow::Result<()>
    where
        usize: ToSig<S>,
        SigVal<S, ()>: RadixKey,
        VFilter<u8, VFunc<usize, u8, Vec<u8>, S, true>>: Serialize + Deserialize,
    {
        for n in [0_usize, 10, 1000, 100_000, 3_000_000] {
            let filter = VBuilder::<_, _, Vec<u8>, S, true, ()>::default()
                .log2_buckets(4)
                .offline(false)
                .try_build_filter(FromIntoIterator::from(0..n), no_logging![])?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            filter.serialize(&mut cursor).unwrap();
            cursor.set_position(0);
            // TODO: This does not work with deserialize_eps
            let filter =
                VFilter::<u8, VFunc<_, _, Vec<u8>, S, true>>::deserialize_full(&mut cursor)?;
            for i in 0..n {
                let sig = ToSig::<S>::to_sig(&i, filter.func.seed);
                assert_eq!(sig.sig_u64() & 0xFF, filter.get(&i) as u64);
            }
        }

        Ok(())
    }
}
