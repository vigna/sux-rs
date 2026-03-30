/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Index functions with signed values.

use std::borrow::Borrow;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    crate::func::mix64,
    crate::utils::FallibleRewindableLender,
    anyhow::Result,
    core::error::Error,
    dsi_progress_logger::ProgressLog,
    lender::*,
    num_primitive::PrimitiveNumberAs,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
    value_traits::slices::SliceByValueMut,
};

use crate::bits::{BitFieldVec, BitFieldVecU};
use crate::func::VFunc;
use crate::func::shard_edge::ShardEdge;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveNumber;
use value_traits::slices::SliceByValue;

/// A signed index function using a [`SliceByValue`] to store hashes.
///
/// Usually, the [`SliceByValue`] will be a boxed slice. Note that the result of
/// the [`SliceByValue`] is assumed to be a hash of size
/// `SliceByValue::Value::BITS`. If you are using implementations returning less
/// hash bits (such as a [`BitFieldVec<Box<[W]>>`](BitFieldVec)), you will need to use
/// [`BitSignedVFunc`] instead.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SignedVFunc<F, H> {
    pub(crate) func: F,
    pub(crate) hashes: H,
}

impl<F, H> SignedVFunc<F, H> {
    /// Creates a new `SignedVFunc` from a function and a hash slice.
    ///
    /// This is a low-level constructor; prefer
    /// [`try_new`](Self::try_new)/[`try_new_with_builder`](Self::try_new_with_builder)
    /// when possible.
    pub fn from_parts(func: F, hashes: H) -> Self {
        Self { func, hashes }
    }
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    D: SliceByValue<Value = W>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue<Value: PrimitiveNumber>,
> SignedVFunc<VFunc<T, D, S, E>, H>
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
            == <H::Value>::as_from(crate::func::mix64(
                shard_edge.edge_hash(shard_edge.local_sig(sig)),
            ))
            .as_to::<u64>()
        {
            Some(index)
        } else {
            None
        }
    }

    /// Returns the index of the given key, if the key was in the list provided at
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
pub struct BitSignedVFunc<F, H> {
    pub(crate) func: F,
    pub(crate) hashes: H,
    pub(crate) hash_mask: u64,
}

impl<
    T: ?Sized + ToSig<S>,
    D: SliceByValue<Value: Word + BinSafe>,
    S: Sig,
    E: ShardEdge<S, 3>,
    H: SliceByValue<Value: PrimitiveNumber>,
> BitSignedVFunc<VFunc<T, D, S, E>, H>
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
    pub fn get_by_sig(&self, sig: S) -> Option<D::Value> {
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

    /// Returns the index of the given key, if the key was in the list provided at
    /// construction time; otherwise, returns `None`.
    ///
    /// False positives happen with probability defined at [construction
    /// time](crate::func::VBuilder::try_build_bit_sig_index).
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> Option<D::Value> {
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

// ── Convenience constructors ───────────────────────────────────────

#[cfg(feature = "rayon")]
impl<T, S, E, H> SignedVFunc<VFunc<T, BitFieldVec<Box<[usize]>>, S, E>, Box<[H]>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    H: crate::traits::Word,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
{
    /// Builds a [`SignedVFunc`] from keys using default [`VBuilder`]
    /// settings.
    ///
    /// The function maps each key to its index in the input sequence
    /// and stores `H::BITS`-bit hashes for verification, giving a
    /// false-positive rate of 2<sup>−`H::BITS`</sup>.
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
    /// # use sux::dict::SignedVFunc;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// use sux::func::VFunc;
    /// use sux::bits::BitFieldVec;
    /// let func: SignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>> =
    ///     SignedVFunc::try_new(
    ///     FromCloneableIntoIterator::new(0..100_usize),
    ///     100,
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert_eq!(func.get(i), Some(i));
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
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
    }

    /// Builds a [`SignedVFunc`] from keys using the given [`VBuilder`]
    /// configuration.
    ///
    /// The function maps each key to its index in the input sequence
    /// and stores `H::BITS`-bit hashes for verification, giving a
    /// false-positive rate of 2<sup>−`H::BITS`</sup>.
    ///
    /// * `keys` must be rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys.
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::dict::SignedVFunc;
    /// # use sux::func::VBuilder;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// use sux::func::VFunc;
    /// use sux::bits::BitFieldVec;
    /// let func: SignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u16]>> =
    ///     SignedVFunc::try_new_with_builder(
    ///     FromCloneableIntoIterator::new(0..100_usize),
    ///     100,
    ///     VBuilder::default().offline(true),
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert_eq!(func.get(i), Some(i));
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new_with_builder<B: ?Sized + Borrow<T>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let (func, mut store) = builder.expected_num_keys(n).try_build_func_and_store(
            keys,
            FromCloneableIntoIterator::from(0..),
            BitFieldVec::new_unaligned,
            false,
            pl,
        )?;

        let num_keys = func.num_keys;
        let shard_edge = &func.shard_edge;

        // Create the hash vector
        let mut hashes = vec![H::ZERO; num_keys].into_boxed_slice();

        // Enumerate the store and extract hashes using the same method as filters
        pl.item_name("hash");
        pl.expected_updates(Some(num_keys));
        pl.start("Storing hashes...");

        for shard in store.iter() {
            for sig_val in shard.iter() {
                let pos = sig_val.val;
                let local_sig = shard_edge.local_sig(sig_val.sig);
                let hash = H::as_from(mix64(shard_edge.edge_hash(local_sig)));
                hashes.set_value(pos, hash);
                pl.light_update();
            }
        }

        pl.done();

        Ok(SignedVFunc { func, hashes })
    }
}

#[cfg(feature = "rayon")]
impl<T, S, E, H> BitSignedVFunc<VFunc<T, BitFieldVec<Box<[usize]>>, S, E>, BitFieldVec<Box<[H]>>>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    H: crate::traits::Word,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
{
    /// Builds a [`BitSignedVFunc`] from keys using default [`VBuilder`]
    /// settings.
    ///
    /// The function maps each key to its index in the input sequence
    /// and stores `hash_width`-bit hashes for verification, giving a
    /// false-positive rate of 2<sup>−`hash_width`</sup>.
    ///
    /// * `keys` must be rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys; a significantly wrong
    ///   value may degrade performance or cause extra retries.
    /// * `hash_width` is the number of hash bits per key (at most
    ///   `H::BITS`).
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
    /// # use sux::dict::BitSignedVFunc;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// use sux::func::VFunc;
    /// use sux::bits::BitFieldVec;
    /// let func: BitSignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>> =
    ///     BitSignedVFunc::try_new(
    ///     FromCloneableIntoIterator::new(0..100_usize),
    ///     100,
    ///     8,
    ///     no_logging![],
    /// )?;
    ///
    /// for i in 0..100 {
    ///     assert_eq!(func.get(i), Some(i));
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
        hash_width: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self>
    where
        u64: PrimitiveNumberAs<H>,
    {
        Self::try_new_with_builder(keys, n, hash_width, VBuilder::default(), pl)
    }

    /// Builds a [`BitSignedVFunc`] from keys using the given
    /// [`VBuilder`] configuration.
    ///
    /// The function maps each key to its index in the input sequence
    /// and stores `hash_width`-bit hashes for verification, giving a
    /// false-positive rate of 2<sup>−`hash_width`</sup>.
    ///
    /// * `keys` must be rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys.
    /// * `hash_width` is the number of hash bits per key (at most
    ///   `H::BITS`).
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::dict::BitSignedVFunc;
    /// # use sux::func::{VBuilder, VFunc};
    /// # use sux::bits::BitFieldVec;
    /// # use dsi_progress_logger::no_logging;
    /// # use sux::utils::FromCloneableIntoIterator;
    /// let func: BitSignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>> =
    ///     BitSignedVFunc::try_new_with_builder(
    ///         FromCloneableIntoIterator::new(0..100_usize),
    ///         100,
    ///         8,
    ///         VBuilder::default().offline(true),
    ///         no_logging![],
    ///     )?;
    ///
    /// for i in 0..100 {
    ///     assert_eq!(func.get(i), Some(i));
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    pub fn try_new_with_builder<B: ?Sized + Borrow<T>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        hash_width: usize,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self>
    where
        u64: PrimitiveNumberAs<H>,
    {
        assert!(hash_width > 0);
        assert!(hash_width <= H::BITS as usize);

        let (func, mut store) = builder.expected_num_keys(n).try_build_func_and_store(
            keys,
            FromCloneableIntoIterator::from(0..),
            BitFieldVec::<Box<[usize]>>::new_unaligned,
            false,
            pl,
        )?;

        let num_keys = func.num_keys;
        let shard_edge = &func.shard_edge;
        let hash_mask = if hash_width == 64 {
            u64::MAX
        } else {
            (1u64 << hash_width) - 1
        };

        // Create the signature vector
        let mut hashes: BitFieldVec<Box<[H]>> =
            BitFieldVec::<Box<[H]>>::new_unaligned(hash_width, num_keys);

        // Enumerate the store and extract signatures using the same method as filters
        pl.item_name("hash");
        pl.expected_updates(Some(num_keys));
        pl.start("Storing hashes...");

        for shard in store.iter() {
            for sig_val in shard.iter() {
                let pos = sig_val.val;
                let local_sig = shard_edge.local_sig(sig_val.sig);
                let hash = (mix64(shard_edge.edge_hash(local_sig)) & hash_mask).as_to::<H>();
                hashes.set_value(pos, hash);
                pl.light_update();
            }
        }

        pl.done();

        Ok(BitSignedVFunc {
            func,
            hashes,
            hash_mask,
        })
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

// -- SignedVFunc: only func needs converting, hashes stay as-is --

use crate::traits::{TryIntoUnaligned, Word};

impl<F: TryIntoUnaligned, H> TryIntoUnaligned for SignedVFunc<F, H> {
    type Unaligned = SignedVFunc<F::Unaligned, H>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(SignedVFunc {
            func: self.func.try_into_unaligned()?,
            hashes: self.hashes,
        })
    }
}

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, H>
    From<SignedVFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, H>>
    for SignedVFunc<VFunc<T, BitFieldVec<Box<[W]>>, S, E>, H>
{
    fn from(f: SignedVFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, H>) -> Self {
        SignedVFunc {
            func: f.func.into(),
            hashes: f.hashes,
        }
    }
}

// -- BitSignedVFunc: both func and hashes are converted --

impl<F: TryIntoUnaligned, H: TryIntoUnaligned> TryIntoUnaligned for BitSignedVFunc<F, H> {
    type Unaligned = BitSignedVFunc<F::Unaligned, H::Unaligned>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(BitSignedVFunc {
            func: self.func.try_into_unaligned()?,
            hashes: self.hashes.try_into_unaligned()?,
            hash_mask: self.hash_mask,
        })
    }
}

impl<T: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>>
    From<BitSignedVFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, BitFieldVecU<Box<[W]>>>>
    for BitSignedVFunc<VFunc<T, BitFieldVec<Box<[W]>>, S, E>, BitFieldVec<Box<[W]>>>
{
    fn from(
        f: BitSignedVFunc<VFunc<T, BitFieldVecU<Box<[W]>>, S, E>, BitFieldVecU<Box<[W]>>>,
    ) -> Self {
        BitSignedVFunc {
            func: f.func.into(),
            hashes: f.hashes.into(),
            hash_mask: f.hash_mask,
        }
    }
}
