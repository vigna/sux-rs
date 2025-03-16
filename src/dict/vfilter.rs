/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use crate::func::mix64;
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
/// [ε-serde](`epserde`). They contain a mapping from keys to hashes stored in a
/// [`VFunc`]; [`contains`](VFilter::contains) checks that the hash of a key is
/// equal of the hash stored by the function for the same key.
///
/// Please read the [`VFunc`] documentation for more information about the space
/// usage and the ways in which a filter can be built. A construction time you
/// have to choose a number *b* of hash bits per key, and the filter precision
/// (false-positive rate) will be 2⁻*ᵇ*.
///
/// Note that this structure implements the [`Index`] trait, which provides a
/// convenient access to the filter. Please see the documentation of
/// [`VBuilder`](crate::func::VBuilder) for examples.
///
/// # Generics
///
/// * `W`: The type of the hashes associated to keys. See the discussion about
///        the generic `D` of [`VFunc`].
/// * `F`: The type of [`VFunc`] used to store the mapping from keys to hashes.
///        This type will also imply the type of the keys.
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFilter<W: ZeroCopy + Word, F> {
    pub(crate) func: F,
    pub(crate) filter_mask: W,
    pub(crate) hash_bits: u32,
}

impl<T: ?Sized + ToSig<S>, W: ZeroCopy + Word, D: BitFieldSlice<W>, S: Sig, E: ShardEdge<S, 3>>
    VFilter<W, VFunc<T, W, D, S, E>>
where
    u64: CastableInto<W>,
{
    /// Returns the hash associated with the given signature by the underlying
    /// function, or a random hash if the signature is not the signature of a
    /// key .
    ///
    /// The user should not normally call this method, but rather
    /// [`contains_by_sig`](VFilter::contains_by_sig).
    #[inline(always)]
    pub fn get_by_sig(&self, sig: S) -> W {
        self.func.get_by_sig(sig)
    }

    /// Returns the hash associated with the given key by the underlying
    /// function, or a random hash if the key is not present.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline]
    pub fn get(&self, key: impl Borrow<T>) -> W {
        self.func.get(key)
    }

    /// Returns whether a signature is contained in the filter.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline(always)]
    pub fn contains_by_sig(&self, sig: S) -> bool {
        self.func.get_by_sig(sig)
            == mix64(self.func.shard_edge.local_sig(sig).sig_u64()).cast() & self.filter_mask
    }

    /// Returns whether a key is contained in the filter.
    #[inline]
    pub fn contains(&self, key: impl Borrow<T>) -> bool {
        self.contains_by_sig(T::to_sig(key.borrow(), self.func.seed))
    }

    /// Returns the number of keys in the filter.
    pub fn len(&self) -> usize {
        self.func.len()
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty()
    }

    /// Returns the number bits of the hash associated with keys.
    ///
    /// The filter precision (false-positive rate) is 2<sup>-`hash_bits`</sup>.
    pub fn hash_bits(&self) -> u32 {
        self.hash_bits
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
        if self.contains(key) {
            &true
        } else {
            &false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{BitXor, BitXorAssign};

    use common_traits::UpcastableFrom;
    use dsi_progress_logger::no_logging;
    use rdst::RadixKey;

    use crate::{
        func::{
            mix64,
            shard_edge::{FuseLge3NoShards, FuseLge3Shards},
            VBuilder,
        },
        utils::{EmptyVal, FromIntoIterator, Sig, SigVal, ToSig},
    };

    use super::ShardEdge;

    #[test]
    fn test_filter_func() -> anyhow::Result<()> {
        _test_filter_func::<[u64; 1], FuseLge3NoShards>()?;
        _test_filter_func::<[u64; 2], FuseLge3Shards>()?;
        Ok(())
    }

    fn _test_filter_func<S: Sig + Send + Sync, E: ShardEdge<S, 3, LocalSig = [u64; 1]>>(
    ) -> anyhow::Result<()>
    where
        usize: ToSig<S>,
        u128: UpcastableFrom<usize>,
        SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
        SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    {
        for n in [0_usize, 10, 1000, 100_000, 1_000_000] {
            let filter = VBuilder::<u8, Box<[_]>, S, E>::default()
                .log2_buckets(4)
                .offline(false)
                .try_build_filter(FromIntoIterator::from(0..n), no_logging![])?;
            for i in 0..n {
                let sig = ToSig::<S>::to_sig(i, filter.func.seed);
                assert_eq!(
                    mix64(filter.func.shard_edge.local_sig(sig)[0]) & 0xFF,
                    filter.get(i) as u64
                );
            }
        }

        Ok(())
    }
}
