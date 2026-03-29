/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

//! Static filters (approximate membership structures with false positives).

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::mix64;
use crate::func::{VFunc, shard_edge::ShardEdge};
use crate::traits::{Word, bit_field_slice::*};
use crate::utils::{BinSafe, Sig, ToSig};
use mem_dbg::*;
use num_primitive::{PrimitiveNumber, PrimitiveNumberAs};
use std::borrow::Borrow;
use std::ops::Index;

/// A static filter (approximate membership data structure) with
/// controllable false-positive rate.
///
/// A `VFilter` wraps a [`VFunc`] that maps each key to a *b*-bit hash. A
/// membership query recomputes the hash from the key's signature and compares
/// it against the stored value: if they match, the key is probably in the set;
/// if they differ, the key is not in the set. The false-positive rate is 2⁻*ᵇ*.
/// For values of *b* that correspond to the size of an unsigned type, you can
/// use a boxed slice as a backend.
///
/// Instances are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized with
/// [ε-serde](https://crates.io/crates/epserde).
///
/// This structure implements the [`Index`] trait for convenient
/// `filter[key]` syntax (returning `&bool`).
///
/// Please see the documentation of [`VBuilder`](crate::func::VBuilder)
/// for construction examples.
///
/// # Generics
///
/// * `W`: The unsigned integer type used to store hashes.
/// * `F`: The underlying [`VFunc`] type (determines key type, signature
///   type, sharding, and backend).
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VFilter<W, F> {
    /// The underlying static function mapping keys to hashes.
    pub(crate) func: F,
    /// Bit mask applied to the derived hash before comparison.
    ///
    /// Equal to `W::MAX >> (W::BITS - hash_bits)`.
    pub(crate) filter_mask: W,
    /// Number of hash bits per key (determines the false-positive rate).
    pub(crate) hash_bits: u32,
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    D: BitFieldSlice<Value = W>,
    S: Sig,
    E: ShardEdge<S, 3>,
> VFilter<W, VFunc<T, W, D, S, E>>
where
    u64: PrimitiveNumberAs<W>,
{
    /// Returns whether the key with the given pre-computed signature is
    /// likely in the set.
    ///
    /// Derives a *b*-bit hash from `sig` and compares it against the
    /// hash stored by the underlying [`VFunc`]. Returns `true` on
    /// match (key probably present, with false-positive rate 2⁻*ᵇ*),
    /// `false` on mismatch (key definitely absent).
    ///
    /// This is the signature-level entry point; most callers should use
    /// [`contains`](Self::contains) instead.
    #[inline(always)]
    pub fn contains_by_sig(&self, sig: S) -> bool {
        let shard_edge = &self.func.shard_edge;
        // Derive the expected hash from the signature's edge hash,
        // mix it with mix64 for avalanche, and mask to hash_bits.
        let expected =
            mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))).as_to::<W>() & self.filter_mask;
        // Compare against the hash stored by the VFunc.
        self.func.get_by_sig(sig) == expected
    }

    /// Returns whether `key` is likely in the set.
    ///
    /// Computes the key's signature and delegates to
    /// [`contains_by_sig`](Self::contains_by_sig). Returns `true`
    /// on match (false-positive rate 2⁻*ᵇ*), `false` on mismatch
    /// (definitely absent).
    #[inline]
    pub fn contains(&self, key: impl Borrow<T>) -> bool {
        self.contains_by_sig(T::to_sig(key.borrow(), self.func.seed))
    }

    /// Returns the number of keys in the filter.
    pub const fn len(&self) -> usize {
        self.func.num_keys
    }

    /// Returns `true` if the filter contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.func.num_keys == 0
    }

    /// Returns the number of hash bits per key.
    ///
    /// The filter's false-positive rate is 2<sup>−`hash_bits`</sup>.
    pub const fn hash_bits(&self) -> u32 {
        self.hash_bits
    }
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    D: BitFieldSlice<Value = W>,
    S: Sig,
    E: ShardEdge<S, 3>,
    B: Borrow<T>,
> Index<B> for VFilter<W, VFunc<T, W, D, S, E>>
where
    u64: PrimitiveNumberAs<W>,
{
    type Output = bool;

    /// Indexes the filter by key, returning `&true` or `&false`.
    ///
    /// Equivalent to [`contains`](VFilter::contains) but satisfying
    /// the [`Index`] trait for `filter[key]` syntax.
    #[inline(always)]
    fn index(&self, key: B) -> &Self::Output {
        // Return references to static bools — this is the standard
        // pattern for Index<_> -> &bool.
        if self.contains(key) { &true } else { &false }
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

impl<T: ?Sized, W: Word + BinSafe, S: Sig, E: ShardEdge<S, 3>> crate::traits::TryIntoUnaligned
    for VFilter<W, VFunc<T, W, BitFieldVec<Box<[W]>>, S, E>>
{
    type Unaligned = VFilter<W, VFunc<T, W, BitFieldVecU<Box<[W]>>, S, E>>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(VFilter {
            func: self.func.try_into_unaligned()?,
            filter_mask: self.filter_mask,
            hash_bits: self.hash_bits,
        })
    }
}

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>>
    From<VFilter<W, VFunc<T, W, BitFieldVecU<Box<[W]>>, S, E>>>
    for VFilter<W, VFunc<T, W, BitFieldVec<Box<[W]>>, S, E>>
{
    fn from(f: VFilter<W, VFunc<T, W, BitFieldVecU<Box<[W]>>, S, E>>) -> Self {
        VFilter {
            func: f.func.into(),
            filter_mask: f.filter_mask,
            hash_bits: f.hash_bits,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use std::ops::{BitXor, BitXorAssign};

    use dsi_progress_logger::no_logging;
    use num_primitive::PrimitiveNumberAs;
    use rdst::RadixKey;

    use crate::{
        func::{
            VBuilder, mix64,
            shard_edge::{FuseLge3NoShards, FuseLge3Shards},
        },
        utils::{EmptyVal, FromCloneableIntoIterator, Sig, SigVal, ToSig},
    };

    use super::ShardEdge;

    #[test]
    fn test_filter_func() -> anyhow::Result<()> {
        _test_filter_func::<[u64; 1], FuseLge3NoShards>()?;
        _test_filter_func::<[u64; 2], FuseLge3Shards>()?;
        Ok(())
    }

    fn _test_filter_func<S: Sig + Send + Sync, E: ShardEdge<S, 3, LocalSig = [u64; 1]>>()
    -> anyhow::Result<()>
    where
        usize: ToSig<S>,
        u128: PrimitiveNumberAs<usize>,
        SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
        SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    {
        for n in [0_usize, 10, 1000, 100_000, 1_000_000] {
            let filter = VBuilder::<u8, Box<[_]>, S, E>::default()
                .log2_buckets(4)
                .offline(false)
                .try_build_filter(FromCloneableIntoIterator::from(0..n), no_logging![])?;
            // Verify that the stored hash matches the expected derivation
            // for every key in the set.
            let shard_edge = &filter.func.shard_edge;
            for i in 0..n {
                let sig = ToSig::<S>::to_sig(i, filter.func.seed);
                assert_eq!(
                    mix64(shard_edge.edge_hash(shard_edge.local_sig(sig))) & 0xFF,
                    filter.func.get_by_sig(sig) as u64,
                    "Hash mismatch for key {i}"
                );
            }
        }

        Ok(())
    }
}
