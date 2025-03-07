/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use crate::func::{shard_edge::ShardEdge, VFunc};
use crate::traits::bit_field_slice::*;
use crate::utils::{Sig, ToSig};
use common_traits::CastableInto;
use epserde::prelude::*;
use mem_dbg::*;
use std::borrow::Borrow;
use std::ops::Index;

/// Static filters (i.e., static probabilistic dictionaries) with low space
/// overhead, fast parallel construction, and fast queries.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`).
///
/// Note that this structure implements the [`Index`] trait, which provides a
/// convenient access to the filter. Please see the documentation of
/// [`VBuilder`](crate::func::VBuilder) for examples.
///
/// # Generics
///
/// * `W`: The output type. See the discussion about the generic `D` of
///        [`VFunc`].
/// * `F`: The type of [`VFunc`] used to store the mapping from keys to
///        signatures. This type will also imply the type of the keys.
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFilter<W: ZeroCopy + Word, F> {
    pub(in crate) func: F,
    pub(in crate) filter_mask: W,
    pub(in crate) sig_bits: u32,
}

impl<T: ?Sized + ToSig<S>, W: ZeroCopy + Word, D: BitFieldSlice<W>, S: Sig, E: ShardEdge<S, 3>>
    VFilter<W, VFunc<T, W, D, S, E>>
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
    pub fn get_by_sig(&self, sig: S) -> W {
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
    pub fn contains_by_sig(&self, sig: S) -> bool {
        self.func.get_by_sig(sig) == sig.sig_u64().cast() & self.filter_mask
    }

    /// Return whether a key is contained in the filter.
    #[inline]
    pub fn contains(&self, key: &T) -> bool {
        self.contains_by_sig(T::to_sig(key, self.func.seed))
    }

    /// Return the number of keys in the filter.
    pub fn len(&self) -> usize {
        self.func.len()
    }

    /// Return whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty()
    }

    /// Return the number of signature bits.
    ///
    /// contained in the filter. The filter precision is
    /// thus 2<sup>-`sig_bits`</sup>.
    pub fn sig_bits(&self) -> u32 {
        self.sig_bits
    }
}

impl<
        T: ?Sized + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W>,
        S: Sig,
        E: ShardEdge<S, 3>,
        B: Borrow<T>,
    > Index<B> for VFilter<W, VFunc<T, W, D, S, E>>
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
    use std::ops::{BitXor, BitXorAssign};

    use dsi_progress_logger::no_logging;
    use epserde::{deser::Deserialize, ser::Serialize, traits::TypeHash, utils::AlignedCursor};
    use rdst::RadixKey;

    use crate::{
        func::{shard_edge::FuseLge3Shards, VBuilder},
        utils::{EmptyVal, FromIntoIterator, Sig, SigVal, ToSig},
    };

    use super::{ShardEdge, VFilter, VFunc};

    #[test]
    fn test_filter_func() -> anyhow::Result<()> {
        _test_filter_func::<[u64; 2]>()?;
        _test_filter_func::<[u64; 2]>()?;
        Ok(())
    }

    fn _test_filter_func<S: Sig + Send + Sync>() -> anyhow::Result<()>
    where
        usize: ToSig<S>,
        SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
        FuseLge3Shards: ShardEdge<S, 3>,
        VFunc<usize, u8, Box<[u8]>, S, FuseLge3Shards>: Serialize + TypeHash, // Weird
        VFilter<u8, VFunc<usize, u8, Box<[u8]>, S, FuseLge3Shards>>: Serialize,
    {
        for n in [0_usize, 10, 1000, 100_000, 1_000_000] {
            let filter = VBuilder::<u8, Box<[_]>, S, FuseLge3Shards>::default()
                .log2_buckets(4)
                .offline(false)
                .try_build_filter(FromIntoIterator::from(0..n), no_logging![])?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            filter.serialize(&mut cursor).unwrap();
            cursor.set_position(0);
            let filter =
                VFilter::<u8, VFunc<usize, _, Box<[_]>, S, FuseLge3Shards>>::deserialize_eps(
                    cursor.as_bytes(),
                )?;
            for i in 0..n {
                let sig = ToSig::<S>::to_sig(&i, filter.func.seed);
                assert_eq!(sig.sig_u64() & 0xFF, filter.get(&i) as u64);
            }
        }

        Ok(())
    }
}
