/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::Backend;
use value_traits::slices::SliceByValue;

use super::shard_edge::FuseLge3Shards;
use crate::bits::BitFieldVec;
use crate::func::shard_edge::ShardEdge;
use crate::traits::Word;
use crate::utils::*;
use mem_dbg::*;
use std::borrow::Borrow;

/// Static functions with low space overhead, fast parallel construction, and
/// fast queries.
///
/// *Static functions* map keys to values, but they do not store the keys:
/// querying a static function with a key outside of the original set will lead
/// to an arbitrary result. Another name for static functions is *retrieval data
/// structure*. Values are retrieved using the [`get`](VFunc::get) method.
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
/// Instances of this structure are immutable; they are built
/// using a [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](https://crates.io/crates/epserde). Please see the documentation of
/// [`VBuilder`](crate::func::VBuilder) for examples.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Generics
///
/// * `K`: The type of the keys.
/// * `W`: The word used to store the data, which is also the output type. It
///   can be any unsigned type.
/// * `D`: The backend storing the function data. It can be a
///   [`BitFieldVec<Box<[W]>>`](crate::bits::BitFieldVec) or a `Box<[W]>`. In the first
///   case, the data is stored using exactly the number of bits needed, but
///   access is slightly slower, while in the second case the data is stored in
///   a boxed slice of `W`, thus forcing the number of bits to the number of
///   bits of `W`, but access will be faster. Note that for most bit sizes in
///   the first case on some architectures you can use
///   [`TryIntoUnaligned`] to convert the function into one using [unaligned
///   reads](BitFieldVec::get_unaligned) for faster queries.
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
///   [`FuseLge3FullSigs`](crate::func::shard_edge::FuseLge3FullSigs).
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VFunc<K: ?Sized, D, S = [u64; 2], E = FuseLge3Shards> {
    pub(crate) shard_edge: E,
    pub(crate) seed: u64,
    pub(crate) num_keys: usize,
    pub(crate) data: D,
    pub(crate) _marker: std::marker::PhantomData<(*const K, S)>,
}

impl<K: ?Sized, D: SliceByValue, S, E> Backend for VFunc<K, D, S, E> {
    type Word = D::Value;
}

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>> VFunc<K, BitFieldVec<Box<[W]>>, S, E> {
    /// Creates a VFunc with zero keys.
    ///
    /// The internal data has bit width 0, so `get` always returns zero.
    /// This is safe because [`BitFieldVec::new(0, 0)`] allocates one
    /// word and `get_value_unchecked` with bit width 0 always reads
    /// index 0 and masks with 0.
    pub fn empty() -> Self {
        Self {
            shard_edge: E::default(),
            seed: 0,
            num_keys: 0,
            data: BitFieldVec::<Vec<W>>::new(0, 0).into(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K: ?Sized + ToSig<S>, D: SliceByValue<Value: Word + BinSafe>, S: Sig, E: ShardEdge<S, 3>>
    VFunc<K, D, S, E>
{
    /// Returns the value associated with the given signature, or a random value
    /// if the signature is not the signature of a key.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> D::Value {
        let edge = self.shard_edge.edge(sig);
        // SAFETY: The ShardEdge implementation guarantees that all indices
        // returned by `edge()` are within bounds of `self.data`. This invariant
        // is established during construction by VBuilder, which ensures the
        // data array is sized according to the ShardEdge's `num_vertices()`
        unsafe {
            self.data.get_value_unchecked(edge[0])
                ^ self.data.get_value_unchecked(edge[1])
                ^ self.data.get_value_unchecked(edge[2])
        }
    }

    /// Returns the value associated with the given key, or a random value if the
    /// key is not present.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> D::Value {
        self.get_by_sig(K::to_sig(key.borrow(), self.seed))
    }

    /// Returns the number of keys in the function.
    pub const fn len(&self) -> usize {
        self.num_keys
    }

    /// Returns whether the function has no keys.
    pub const fn is_empty(&self) -> bool {
        self.num_keys == 0
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

use crate::traits::{TryIntoUnaligned, Unaligned};

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>> TryIntoUnaligned
    for VFunc<K, BitFieldVec<Box<[W]>>, S, E>
{
    type Unaligned = VFunc<K, Unaligned<BitFieldVec<Box<[W]>>>, S, E>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(VFunc {
            shard_edge: self.shard_edge,
            seed: self.seed,
            num_keys: self.num_keys,
            data: self.data.try_into_unaligned()?,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>>
    From<Unaligned<VFunc<K, BitFieldVec<Box<[W]>>, S, E>>>
    for VFunc<K, BitFieldVec<Box<[W]>>, S, E>
{
    fn from(vf: Unaligned<VFunc<K, BitFieldVec<Box<[W]>>, S, E>>) -> Self {
        VFunc {
            shard_edge: vf.shard_edge,
            seed: vf.seed,
            num_keys: vf.num_keys,
            data: BitFieldVec::from(vf.data),
            _marker: std::marker::PhantomData,
        }
    }
}

// ── Constructors (require rayon) ──────────────────────────────────

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::func::VBuilder;
    use crate::traits::bit_field_slice::BitFieldSliceMut;
    use anyhow::Result;
    use core::error::Error;
    use dsi_progress_logger::ProgressLog;
    use lender::*;
    use rdst::RadixKey;
    use std::ops::{BitXor, BitXorAssign};
    use value_traits::slices::SliceByValueMut;

    impl<K, W, S, E> VFunc<K, Box<[W]>, S, E>
    where
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        W: Word + BinSafe,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        SigVal<S, W>: RadixKey,
        SigVal<E::LocalSig, W>: BitXor + BitXorAssign,
    {
        /// Builds a [`VFunc`] with a `Box<[W]>` backend from keys and values
        /// using default [`VBuilder`] settings.
        ///
        /// * `keys` and `values` must be aligned (one value per key, same
        ///   order) and rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys; a significantly wrong
        ///   value may degrade performance or cause extra retries.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// # Examples
        ///
        /// If keys and values are available as slices, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::VFunc;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// let func = <VFunc<usize, Box<[u8]>>>::try_new(
        ///     FromCloneableIntoIterator::new(0..100),
        ///     FromCloneableIntoIterator::new(0..100_u8),
        ///     100,
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100_u8 {
        ///     assert_eq!(i, func.get(&(i as usize)));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
                BitFieldSliceMut,
            for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
        {
            Self::try_new_with_builder(keys, values, n, VBuilder::default(), pl)
        }

        /// Builds a [`VFunc`] with a `Box<[W]>` backend from keys and values
        /// using the given [`VBuilder`] configuration.
        ///
        /// * `keys` and `values` must be aligned (one value per key, same
        ///   order) and rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// # Examples
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{VBuilder, VFunc};
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// let func = <VFunc<usize, Box<[u8]>>>::try_new_with_builder(
        ///     FromCloneableIntoIterator::new(0..100),
        ///     FromCloneableIntoIterator::new(0..100_u8),
        ///     100,
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100_u8 {
        ///     assert_eq!(i, func.get(&(i as usize)));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            n: usize,
            builder: VBuilder<Box<[W]>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
                BitFieldSliceMut,
            for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
        {
            Ok(builder
                .expected_num_keys(n)
                .try_build_func(
                    keys,
                    values,
                    |_bit_width, len| vec![W::ZERO; len].into(),
                    pl,
                )?
                .0)
        }

        /// Builds a [`VFunc`] with a `Box<[W]>` backend from in-memory key
        /// and value slices, parallelizing hash computation and store
        /// population with rayon, using default [`VBuilder`] settings.
        ///
        /// Each key is hashed on a rayon worker thread and deposited directly
        /// into its SigStore bucket. This is faster than
        /// [`try_new`](Self::try_new) for large in-memory key sets.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// # Examples
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::VFunc;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = (0..1000).collect();
        /// let values: Vec<u8> = (0..1000).map(|x| x as u8).collect();
        /// let func =
        ///     <VFunc<u64, Box<[u8]>>>::try_par_new(&keys, &values, no_logging![])?;
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i as u8);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_par_new(
            keys: &[impl Borrow<K> + Sync],
            values: &[W],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            W: Copy,
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
                BitFieldSliceMut,
            for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
        {
            Self::try_par_new_with_builder(keys, values, VBuilder::default(), pl)
        }

        /// Builds a [`VFunc`] with a `Box<[W]>` backend from in-memory key
        /// and value slices, parallelizing hash computation and store
        /// population with rayon, using the given [`VBuilder`] configuration.
        ///
        /// Each key is hashed on a rayon worker thread and deposited directly
        /// into its SigStore bucket. This is faster than
        /// [`try_new`](Self::try_new) for large in-memory key sets.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// # Examples
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{VFunc, VBuilder};
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = (0..1000).collect();
        /// let values: Vec<u8> = (0..1000).map(|x| x as u8).collect();
        /// let func =
        ///     <VFunc<u64, Box<[u8]>>>::try_par_new_with_builder(&keys, &values, VBuilder::default(), no_logging![])?;
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i as u8);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_par_new_with_builder(
            keys: &[impl Borrow<K> + Sync],
            values: &[W],
            builder: VBuilder<Box<[W]>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            W: Copy,
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
                BitFieldSliceMut,
            for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
            for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item: Send,
        {
            let n = keys.len();
            builder.expected_num_keys(n).try_par_populate_and_build(
                keys,
                &|i| values[i],
                &mut |builder, seed, mut store, _max_value, _num_keys, pl, _state: &mut ()| {
                    let data: Box<[W]> = vec![
                        W::ZERO;
                        builder.shard_edge.num_vertices()
                            * builder.shard_edge.num_shards()
                    ]
                    .into();
                    let func = builder.try_build_from_shard_iter(
                        seed,
                        data,
                        store.drain(),
                        &|_, sv| sv.val,
                        &|_| {},
                        pl,
                    )?;
                    Ok(func)
                },
                pl,
                (),
            )
        }
    }

    impl<K, W, S, E> VFunc<K, BitFieldVec<Box<[W]>>, S, E>
    where
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        W: Word + BinSafe,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        SigVal<S, W>: RadixKey,
        SigVal<E::LocalSig, W>: BitXor + BitXorAssign,
    {
        /// Builds a [`VFunc`] with a [`BitFieldVec`] backend from keys and
        /// values using default [`VBuilder`] settings.
        ///
        /// * `keys` and `values` must be aligned (one value per key, same
        ///   order) and rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys; a significantly wrong
        ///   value may degrade performance or cause extra retries.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`](Self::try_new_with_builder) with
        /// `VBuilder::default()`.
        ///
        /// # Examples
        ///
        /// If keys and values are available as slices, [`try_par_new`](Self::try_par_new)
        /// parallelizes the hash computation for faster construction.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::VFunc;
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// let func = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new(
        ///     FromCloneableIntoIterator::new(0..100),
        ///     100,
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(i, func.get(&i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, values, n, VBuilder::default(), pl)
        }

        /// Builds a [`VFunc`] with a [`BitFieldVec`] backend from keys and
        /// values using the given [`VBuilder`] configuration.
        ///
        /// * `keys` and `values` must be aligned (one value per key, same
        ///   order) and rewindable (they may be rewound on retry).
        /// * `n` is the expected number of keys.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// # Examples
        ///
        /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// for parallel hash computation from slices.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{VBuilder, VFunc};
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// let func = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
        ///     FromCloneableIntoIterator::new(0..100),
        ///     100,
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(i, func.get(&i));
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[W]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            builder
                .expected_num_keys(n)
                .try_build_func(
                    keys,
                    values,
                    |bit_width, len| BitFieldVec::<Box<[W]>>::new_padded(bit_width, len),
                    pl,
                )
                .map(|res| res.0)
        }

        /// Builds a [`VFunc`] with a [`BitFieldVec`] backend from in-memory
        /// key and value slices, parallelizing hash computation and store
        /// population with rayon, using default [`VBuilder`] settings.
        ///
        /// Each key is hashed on a rayon worker thread and deposited directly
        /// into its SigStore bucket. This is faster than
        /// [`try_new`](Self::try_new) for large in-memory key sets.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
        /// with `VBuilder::default()`.
        ///
        /// # Examples
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`](Self::try_new) instead.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::VFunc;
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = (0..1000).collect();
        /// let values: Vec<usize> = (0..1000).collect();
        /// let func =
        ///     <VFunc<u64, BitFieldVec<Box<[usize]>>>>::try_par_new(&keys, &values, no_logging![])?;
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_par_new(
            keys: &[impl Borrow<K> + Sync],
            values: &[W],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            W: Copy,
        {
            Self::try_par_new_with_builder(keys, values, VBuilder::default(), pl)
        }

        /// Builds a [`VFunc`] with a [`BitFieldVec`] backend from in-memory
        /// key and value slices, parallelizing hash computation and store
        /// population with rayon, using the given [`VBuilder`] configuration.
        ///
        /// Each key is hashed on a rayon worker thread and deposited directly
        /// into its SigStore bucket. This is faster than
        /// [`try_new`](Self::try_new) for large in-memory key sets.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
        /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
        ///
        /// # Examples
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{VFunc, VBuilder};
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = (0..1000).collect();
        /// let values: Vec<usize> = (0..1000).collect();
        /// let func =
        ///     <VFunc<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
        ///         &keys, &values, VBuilder::default(), no_logging![],
        ///     )?;
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        pub fn try_par_new_with_builder(
            keys: &[impl Borrow<K> + Sync],
            values: &[W],
            builder: VBuilder<BitFieldVec<Box<[W]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
            S: Send,
            W: Copy,
        {
            let n = keys.len();
            builder.expected_num_keys(n).try_par_populate_and_build(
                keys,
                &|i| values[i],
                &mut |builder, seed, mut store, max_value, _num_keys, pl, _state: &mut ()| {
                    builder.bit_width = max_value.bit_len() as usize;
                    let data = BitFieldVec::<Box<[W]>>::new_padded(
                        builder.bit_width,
                        builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                    );
                    let func = builder.try_build_from_shard_iter(
                        seed,
                        data,
                        store.drain(),
                        &|_, sv| sv.val,
                        &|_| {},
                        pl,
                    )?;
                    Ok(func)
                },
                pl,
                (),
            )
        }
    }
} // mod build
