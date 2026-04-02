/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

//! Static filters (approximate membership structures with false positives).

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::{VFunc, shard_edge::ShardEdge};
use crate::traits::{Backend, Word};
use crate::utils::{BinSafe, Sig, ToSig};
use mem_dbg::*;
use num_primitive::{PrimitiveInteger, PrimitiveNumber, PrimitiveNumberAs};
use std::borrow::Borrow;
use std::ops::Index;
use value_traits::slices::SliceByValue;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    crate::utils::{EmptyVal, FallibleRewindableLender, SigVal},
    anyhow::Result,
    core::error::Error,
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
    value_traits::slices::SliceByValueMut,
};

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
/// Instances are immutable; they are built using a [`VBuilder`] and can be
/// serialized with [ε-serde](https://crates.io/crates/epserde).
///
/// This structure implements the [`Index`] trait for convenient
/// `filter[key]` syntax (returning `&bool`).
///
/// Please see the documentation of [`VBuilder`] for construction examples.
///
/// # Generics
///
/// * `W`: The unsigned integer type used to store hashes.
/// * `F`: The underlying [`VFunc`] type (determines key type, signature
///   type, sharding, and backend).
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    epserde(bound(
        deser = "for<'a> <F as epserde::deser::DeserInner>::DeserType<'a>: Backend<Word = F::Word>"
    ))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VFilter<F: Backend> {
    /// The underlying static function mapping keys to hashes.
    pub(crate) func: F,
    /// Bit mask applied to the derived hash before comparison.
    ///
    /// Equal to `W::MAX >> (W::BITS - hash_bits)`, where `hash_bits`
    /// is the number of hash bits per key.
    pub(crate) filter_mask: F::Word,
}

impl<F: Backend> VFilter<F> {
    /// Creates a new `VFilter` from a function, a filter mask, and hash
    /// bit count.
    ///
    /// This is a low-level constructor; prefer
    /// [`try_new`](VFilter::try_new)/[`try_new_with_builder`](VFilter::try_new_with_builder)
    /// when possible.
    pub fn from_parts(func: F, filter_mask: F::Word) -> Self {
        Self { func, filter_mask }
    }
}

impl<T: ?Sized + ToSig<S>, D: SliceByValue<Value: Word + BinSafe>, S: Sig, E: ShardEdge<S, 3>>
    VFilter<VFunc<T, D, S, E>>
where
    u64: PrimitiveNumberAs<D::Value>,
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
        // Derive the expected hash from the signature via the canonical
        // remixed hash, and mask to hash_bits.
        let expected =
            self.func.shard_edge.remixed_hash(sig).as_to::<D::Value>() & self.filter_mask;
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
    pub fn hash_bits(&self) -> u32 {
        D::Value::BITS - self.filter_mask.leading_zeros()
    }
}

impl<
    T: ?Sized + ToSig<S>,
    D: SliceByValue<Value: Word + BinSafe>,
    S: Sig,
    E: ShardEdge<S, 3>,
    B: Borrow<T>,
> Index<B> for VFilter<VFunc<T, D, S, E>>
where
    u64: PrimitiveNumberAs<D::Value>,
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
    for VFilter<VFunc<T, BitFieldVec<Box<[W]>>, S, E>>
{
    type Unaligned = VFilter<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(VFilter {
            func: self.func.try_into_unaligned()?,
            filter_mask: self.filter_mask,
        })
    }
}

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>>
    From<VFilter<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>>>
    for VFilter<VFunc<T, BitFieldVec<Box<[W]>>, S, E>>
{
    fn from(f: VFilter<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>>) -> Self {
        VFilter {
            func: f.func.into(),
            filter_mask: f.filter_mask,
        }
    }
}

// ── Convenience constructors ───────────────────────────────────────

#[cfg(feature = "rayon")]
impl<T, W, S, E> VFilter<VFunc<T, Box<[W]>, S, E>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, EmptyVal>: RadixKey,
    SigVal<E::LocalSig, EmptyVal>: BitXor + BitXorAssign,
{
    /// Builds a [`VFilter`] with a `Box<[W]>` backend from keys using
    /// default [`VBuilder`] settings.
    ///
    /// The number of hash bits per key equals `W::BITS`, giving a
    /// false-positive rate of 2<sup>−`W::BITS`</sup>. To use fewer
    /// bits per key (trading space for a higher false-positive rate),
    /// use the [`BitFieldVec`] variant with an explicit `filter_bits`
    /// parameter.
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
    /// # use sux::dict::VFilter;
    /// # use sux::func::VFunc;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let filter = <VFilter<VFunc<usize, Box<[u8]>>>>::try_new(
    ///     FromCloneableIntoIterator::new(0..100),
    ///     100,
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert!(filter[i]);
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
    ) -> Result<Self>
    where
        for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
            crate::traits::bit_field_slice::BitFieldSliceMut,
        for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
        for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
    {
        Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Builds a [`VFilter`] with a `Box<[W]>` backend from keys using
    /// the given [`VBuilder`] configuration.
    ///
    /// The number of hash bits per key equals `W::BITS`, giving a
    /// false-positive rate of 2<sup>−`W::BITS`</sup>.
    ///
    /// * `keys` must be rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys.
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
    /// # use sux::dict::VFilter;
    /// # use sux::func::{VBuilder, VFunc};
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let filter = <VFilter<VFunc<usize, Box<[u8]>>>>::try_new_with_builder(
    ///     FromCloneableIntoIterator::new(0..100),
    ///     100,
    ///     VBuilder::default().offline(true),
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert!(filter[i]);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new_with_builder<B: ?Sized + Borrow<T>, P: ProgressLog + Clone + Send + Sync>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<Box<[W]>, S, E>,
        pl: &mut P,
    ) -> Result<Self>
    where
        for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
            crate::traits::BitFieldSliceMut,
        for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
        for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
    {
        let filter_mask = W::MAX;
        let func = builder.expected_num_keys(n).try_build_filter(
            keys,
            W::BITS as usize,
            |_, len| vec![W::ZERO; len].into(),
            &|shard_edge, sig_val| {
                W::as_from(crate::func::mix64(shard_edge.edge_hash(sig_val.sig)))
            },
            pl,
        )?;

        Ok(VFilter { func, filter_mask })
    }
}

#[cfg(feature = "rayon")]
impl<T, W, S, E> VFilter<VFunc<T, BitFieldVec<Box<[W]>>, S, E>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, EmptyVal>: RadixKey,
    SigVal<E::LocalSig, EmptyVal>: BitXor + BitXorAssign,
{
    /// Builds a [`VFilter`] with a [`BitFieldVec`] backend from keys
    /// using default [`VBuilder`] settings.
    ///
    /// * `keys` must be rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys; a significantly wrong
    ///   value may degrade performance or cause extra retries.
    /// * `filter_bits` is the number of hash bits per key; the
    ///   false-positive rate is 2<sup>−`filter_bits`</sup>.
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
    /// # use sux::dict::VFilter;
    /// # use sux::func::VFunc;
    /// # use sux::bits::BitFieldVec;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let filter = <VFilter<VFunc<usize, BitFieldVec<Box<[usize]>>>>>::try_new(
    ///     FromCloneableIntoIterator::new(0..100),
    ///     100,
    ///     5,
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert!(filter[i]);
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
        filter_bits: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self>
    where
        for<'a> <BitFieldVec<Box<[W]>> as SliceByValueMut>::ChunksMut<'a>: Send,
        for<'a> <<BitFieldVec<Box<[W]>> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
    {
        Self::try_new_with_builder(keys, n, filter_bits, VBuilder::default(), pl)
    }

    /// Builds a [`VFilter`] with a [`BitFieldVec`] backend from keys
    /// using the given [`VBuilder`] configuration.
    ///
    /// * `keys` must be rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys.
    /// * `filter_bits` is the number of hash bits per key; the
    ///   false-positive rate is 2<sup>−`filter_bits`</sup>.
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
    /// # use sux::dict::VFilter;
    /// # use sux::func::{VBuilder, VFunc};
    /// # use sux::bits::BitFieldVec;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let filter = <VFilter<VFunc<usize, BitFieldVec<Box<[usize]>>>>>::try_new_with_builder(
    ///     FromCloneableIntoIterator::new(0..100),
    ///     100,
    ///     5,
    ///     VBuilder::default().offline(true),
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert!(filter[i]);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new_with_builder<B: ?Sized + Borrow<T>, P: ProgressLog + Clone + Send + Sync>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        filter_bits: usize,
        builder: VBuilder<BitFieldVec<Box<[W]>>, S, E>,
        pl: &mut P,
    ) -> Result<Self>
    where
        for<'a> <BitFieldVec<Box<[W]>> as SliceByValueMut>::ChunksMut<'a>: Send,
        for<'a> <<BitFieldVec<Box<[W]>> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
    {
        assert!(filter_bits > 0);
        assert!(filter_bits <= W::BITS as usize);
        let filter_mask = W::MAX >> (W::BITS - filter_bits as u32);
        let func = builder.expected_num_keys(n).try_build_filter(
            keys,
            filter_bits,
            BitFieldVec::new_unaligned,
            &|shard_edge, sig_val| {
                W::as_from(crate::func::mix64(shard_edge.edge_hash(sig_val.sig))) & filter_mask
            },
            pl,
        )?;

        Ok(VFilter { func, filter_mask })
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
            VBuilder, VFunc,
            shard_edge::{FuseLge3NoShards, FuseLge3Shards},
        },
        utils::{EmptyVal, FromCloneableIntoIterator, Sig, SigVal, ToSig},
    };

    use super::{ShardEdge, VFilter};

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
            let filter = <VFilter<VFunc<usize, Box<[u8]>, S, E>>>::try_new_with_builder(
                FromCloneableIntoIterator::from(0..n),
                n,
                VBuilder::default().log2_buckets(4).offline(false),
                no_logging![],
            )?;
            // Verify that the stored hash matches the expected derivation
            // for every key in the set.
            for i in 0..n {
                let sig = ToSig::<S>::to_sig(i, filter.func.seed);
                assert_eq!(
                    filter.func.shard_edge.remixed_hash(sig) & 0xFF,
                    filter.func.get_by_sig(sig) as u64,
                    "Hash mismatch for key {i}"
                );
            }
        }

        Ok(())
    }
}
