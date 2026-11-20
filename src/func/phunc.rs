/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Static functions storing their values in an array indexed by a minimal
//! perfect hash function.

use std::borrow::Borrow;

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::func::PHastR;
use crate::func::phastr::{LevelParams, Remap, SeedStore, SeedStoreBuild};
use crate::traits::{TryIntoUnaligned, Unaligned, UnalignedConversionError};
use crate::utils::ToSig;

/// A minimal perfect hash function, which maps bijectively a set of *n* keys
/// to the integers in [0 . . *n*).
///
/// This trait is the interface through which a [`PHunc`] finds the position
/// of the value of a key. It is implemented by [`PHastR`].
///
/// # Safety
///
/// The value returned by [`index`] must be smaller than [`len`] for every
/// key, including the keys outside the original set; if the function is
/// empty, it must be zero. A [`PHunc`] relies on this property to access its
/// values without bounds checks.
///
/// [`index`]: Mphf::index
/// [`len`]: Mphf::len
pub unsafe trait Mphf {
    /// The type of the keys.
    type Key: ?Sized;

    /// Returns the number of keys.
    fn len(&self) -> usize;

    /// Returns whether the function has no keys.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the output of the given key.
    ///
    /// Different keys of the original set have different outputs. If the
    /// key was not in the original set, the result is arbitrary, but it is
    /// still smaller than [`len`] (zero, if the function is empty).
    ///
    /// [`len`]: Mphf::len
    fn index(&self, key: &Self::Key) -> usize;
}

/// A static function storing its values in an array indexed by a minimal
/// perfect hash function.
///
/// A [`PHunc`] is made of a [minimal perfect hash function] `F`, which maps
/// the *n* keys to distinct positions in [0 . . *n*), and of an array `D` of
/// *n* values: the value of a key is stored at the position of the key.
/// Like for all static functions, querying a key outside of the original set
/// will lead to an arbitrary result (one of the values).
///
/// With respect to a [`VFunc`], which is the other generic static function
/// of this crate:
///
/// - The space overhead is that of the minimal perfect hash function
///   (about 1.94 bits per key for a [`PHastR`] with default parameters),
///   rather than a fraction of the size of the values (10.5% for a
///   [`VFunc`] with default parameters): thus, a [`PHunc`] uses less space
///   when values have more than about twenty bits.
///
/// - Values are stored as they are, so they can be of any type that can be
///   stored in the array, and not just integers.
///
/// - A query needs two dependent memory accesses (one for the minimal
///   perfect hash function, and one for the array), whereas the three
///   accesses of a [`VFunc`] are independent, so on large key sets a
///   [`PHunc`] is usually slower when queries are issued one at a time.
///
/// - Construction from lenders reads keys and values just once.
///
/// The array can be any implementation of [`SliceByValue`]: a boxed slice
/// provides the fastest queries, whereas a [`BitFieldVec`] uses for each
/// value just the bits that are necessary for the largest one.
///
/// [`PHunc`] implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access: this
/// is useful for values stored in a [`BitFieldVec`], which are then accessed
/// with [unaligned reads].
///
/// Instances of this structure are immutable; they are built using the
/// constructors listed below, or assembled using [`from_parts`], and can be
/// serialized using [ε-serde] or [`serde`].
///
/// # Construction
///
/// Constructors are available for a [`PHunc`] based on a [`PHastR`], with
/// values stored in a `Box<[V]>` or in a `BitFieldVec<Box<[W]>>`; in the
/// second case, the bit width is that of the largest value. They require
/// the `rayon` feature.
///
/// To build a function from a slice of keys and a slice of values, use
/// `try_par_new`, which is parallel in all its phases; to read keys and
/// values from two [lenders] (for example, from files), use `try_new`,
/// which reads them just once: values are kept next to the signatures of
/// their keys, in memory or, in [offline mode], on disk, and they are
/// written into the array in parallel after the minimal perfect hash
/// function has been built, without reading the keys again. The variants
/// `try_par_new_with_builder` and `try_new_with_builder` make it possible
/// to configure the construction of the [`PHastR`] using a
/// [`PHastRBuilder`]. The memory used by the construction is that
/// [used by `PHastR`], plus the array of values; construction from lenders
/// stores in addition a copy of the values, in memory or on disk.
///
/// For other minimal perfect hash functions and other arrays, build the two
/// components, store the value of each key at the position returned by
/// [`Mphf::index`], and use [`from_parts`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logs;
/// use sux::bits::BitFieldVec;
/// use sux::func::{PHastR, PHunc};
///
/// let keys: Vec<String> = (0..10_000).map(|i| format!("key{i}")).collect();
/// let values: Vec<usize> = (0..10_000).map(|i| i * 3).collect();
///
/// // Values in a boxed slice
/// let func =
///     <PHunc<PHastR<str>, Box<[usize]>>>::try_par_new(&keys, &values, no_logs![])?;
/// for (key, &value) in keys.iter().zip(&values) {
///     assert_eq!(func.get(key.as_str()), value);
/// }
///
/// // Values in a bit-field vector, using 15 bits per value
/// let func = <PHunc<PHastR<str>, BitFieldVec<Box<[usize]>>>>::try_par_new(
///     &keys,
///     &values,
///     no_logs![],
/// )?;
/// for (key, &value) in keys.iter().zip(&values) {
///     assert_eq!(func.get(key.as_str()), value);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// [minimal perfect hash function]: Mphf
/// [`VFunc`]: crate::func::VFunc
/// [`BitFieldVec`]: crate::bits::BitFieldVec
/// [unaligned reads]: crate::bits::BitFieldVec::get_unaligned
/// [`from_parts`]: PHunc::from_parts
/// [ε-serde]: https://crates.io/crates/epserde
/// [`serde`]: https://crates.io/crates/serde
/// [lenders]: crate::utils::lenders
/// [offline mode]: crate::func::PHastRBuilder::offline
/// [`PHastRBuilder`]: crate::func::PHastRBuilder
/// [used by `PHastR`]: PHastR#construction
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PHunc<F, D> {
    /// The minimal perfect hash function.
    func: F,
    /// The values, indexed by the outputs of the keys; if there are no keys,
    /// a single value, which is what queries return.
    data: D,
}

impl<F: Mphf, D: SliceByValue> PHunc<F, D> {
    /// Creates a new [`PHunc`] from a minimal perfect hash function and an
    /// array containing the value of each key at the position returned by
    /// [`Mphf::index`].
    ///
    /// If the function is empty, the array must contain a single value,
    /// which is what queries return.
    ///
    /// # Panics
    ///
    /// Panics if the number of values is different from the number of keys
    /// or, if there are no keys, from one.
    pub fn from_parts(func: F, data: D) -> Self {
        assert_eq!(
            data.len(),
            func.len().max(1),
            "the number of values must be the number of keys (one, if there are no keys)"
        );
        Self { func, data }
    }

    /// Returns the value associated with the given key.
    ///
    /// If the key was not in the original set, the result is one of the
    /// values (the padding value, if the function is empty).
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<F::Key>) -> D::Value {
        // SAFETY: the output of a key is smaller than the number of keys, or
        // zero if there are no keys, and there are as many values as keys,
        // or one if there are no keys
        unsafe { self.data.get_value_unchecked(self.func.index(key.borrow())) }
    }

    /// Returns the number of keys.
    pub fn len(&self) -> usize {
        self.func.len()
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty()
    }
}

impl<F, D> PHunc<F, D> {
    /// Returns the minimal perfect hash function and the array of values.
    pub fn into_parts(self) -> (F, D) {
        (self.func, self.data)
    }
}

// SAFETY: PHastR::get returns a value smaller than the number of keys for
// every key, and zero if there are no keys
unsafe impl<
    K: ?Sized + ToSig<[u64; 1]>,
    S: SeedStore,
    P: AsRef<[LevelParams]>,
    R: SliceByValue<Value = usize>,
> Mphf for PHastR<K, S, P, R>
{
    type Key = K;

    #[inline(always)]
    fn len(&self) -> usize {
        PHastR::len(self)
    }

    #[inline(always)]
    fn index(&self, key: &K) -> usize {
        self.get(key)
    }
}

impl<F: TryIntoUnaligned, D: TryIntoUnaligned> TryIntoUnaligned for PHunc<F, D> {
    type Unaligned = PHunc<F::Unaligned, D::Unaligned>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        Ok(PHunc {
            func: self.func.try_into_unaligned()?,
            data: self.data.try_into_unaligned()?,
        })
    }
}

impl<
    K: ?Sized,
    S: SeedStoreBuild + TryIntoUnaligned + From<Unaligned<S>>,
    P,
    D: TryIntoUnaligned + From<Unaligned<D>>,
> From<Unaligned<PHunc<PHastR<K, S, P, Remap>, D>>> for PHunc<PHastR<K, S, P, Remap>, D>
{
    fn from(f: Unaligned<PHunc<PHastR<K, S, P, Remap>, D>>) -> Self {
        PHunc {
            func: f.func.into(),
            data: f.data.into(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::bits::BitFieldVec;
    use crate::func::phastr::PHastRBuilder;
    use crate::func::signed::build::par_fill_hashes;
    use crate::traits::{AtomicBitFieldSlice, Word};
    use crate::utils::{BinSafe, PrimitiveUnsignedExt};
    use anyhow::{Result, ensure};
    use atomic_primitive::{AtomicPrimitive, PrimitiveAtomicUnsigned};
    use core::error::Error;
    use dsi_progress_logger::ProgressLog;
    use lender::{FallibleLender, FallibleLending};
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;
    use sync_cell_slice::SyncSlice;

    /// Passes to `set` the output and the value of each key, in parallel;
    /// the outputs of the keys are distinct, so `set` can write them
    /// concurrently.
    fn par_store<K: ?Sized + ToSig<[u64; 1]> + Sync, S: SeedStore + Sync, V: Copy + Sync>(
        func: &PHastR<K, S>,
        keys: &[impl Borrow<K> + Sync],
        values: &[V],
        set: impl Fn(usize, V) + Sync,
        pl: &mut impl ProgressLog,
    ) {
        debug_assert_eq!(keys.len(), values.len());
        pl.item_name("key");
        pl.expected_updates(Some(keys.len()));
        pl.start("Storing the values of the keys...");
        keys.par_iter()
            .zip(values)
            .for_each(|(key, &value)| set(func.get(key.borrow()), value));
        pl.update_with_count(keys.len());
        pl.done();
    }

    /// Checks that there are as many values as keys.
    fn check_lens(num_keys: usize, num_values: usize) -> Result<()> {
        ensure!(
            num_keys == num_values,
            "There are {num_keys} keys, but {num_values} values"
        );
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: PHunc<PHastR<...>, Box<[V]>>
    // ═══════════════════════════════════════════════════════════════════

    impl<K: ?Sized + ToSig<[u64; 1]>, S: SeedStoreBuild + MemSize + FlatType + Sync, V: BinSafe>
        PHunc<PHastR<K, S>, Box<[V]>>
    {
        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a boxed slice, from keys and values using default
        /// [`PHastRBuilder`] settings.
        ///
        /// This is a convenience wrapper around `try_new_with_builder` (in
        /// this impl block) with `PHastRBuilder::default()`.
        ///
        /// If you have a slice of keys and a slice of values, `try_par_new`
        /// (in this impl block) is faster.
        ///
        /// * `keys` - a [`FallibleLender`] returning the keys, which is read
        ///   just once. The [`lenders`] module provides easy ways to build
        ///   such lenders.
        ///
        /// * `values` - a [`FallibleLender`] returning the values, in the
        ///   order of their keys, which is read just once.
        ///
        /// * `pl` - a progress logger, which logs the parameters and the
        ///   phases of the construction.
        ///
        /// # Errors
        ///
        /// Returns an error if there are duplicate keys, if the two lenders
        /// return a different number of items, if a lender returns an
        /// error, or if there are too many keys for the platform.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logs;
        /// use sux::func::{PHastR, PHunc};
        /// use sux::utils::FromCloneableIntoIterator;
        ///
        /// let func = <PHunc<PHastR<usize>, Box<[u32]>>>::try_new(
        ///     FromCloneableIntoIterator::new(0..10_000_usize),
        ///     FromCloneableIntoIterator::new((0..10_000_u32).map(|i| i * i)),
        ///     no_logs![],
        /// )?;
        ///
        /// for key in 0..10_000_usize {
        ///     assert_eq!(func.get(key), (key * key) as u32);
        /// }
        /// # Ok(())
        /// # }
        /// ```
        ///
        /// [`lenders`]: crate::utils::lenders
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend V>,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, values, PHastRBuilder::default(), pl)
        }

        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a boxed slice, from keys and values using the given
        /// [`PHastRBuilder`] configuration.
        ///
        /// The two lenders are read just once: each value is kept next to
        /// the two signatures of its key, in memory or, in [offline mode],
        /// on disk, and, once the [`PHastR`] has been built, the values are
        /// written in parallel at the outputs of their keys, which are
        /// computed from the signatures.
        ///
        /// See `try_new` (in this impl block) for the arguments and the
        /// errors.
        ///
        /// [offline mode]: PHastRBuilder::offline
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend V>,
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            let (func, stored) = builder.try_build_from_lender_with_values::<K, S, B, V>(
                keys,
                values,
                |_| {},
                pl,
            )?;
            let mut data = vec![V::new_zeroed(); func.len().max(1)];
            if let Some(stored) = stored {
                let sync_data = data.as_sync_slice();
                // SAFETY: the outputs of the keys are distinct, so each cell
                // is written by a single thread
                stored.for_each(&func, |i, v| unsafe { sync_data[i].set(v) }, pl)?;
            }
            Ok(PHunc {
                func,
                data: data.into_boxed_slice(),
            })
        }

        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a boxed slice, from a slice of keys and a slice of values, using
        /// default [`PHastRBuilder`] settings.
        ///
        /// This is a convenience wrapper around `try_par_new_with_builder`
        /// (in this impl block) with `PHastRBuilder::default()`.
        ///
        /// If keys and values are produced sequentially (e.g., from a
        /// file), use `try_new` (in this impl block) instead.
        ///
        /// # Errors
        ///
        /// Returns an error if there are duplicate keys, if the two slices
        /// have different lengths, or if there are too many keys for the
        /// platform.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logs;
        /// use sux::func::{PHastR, PHunc};
        ///
        /// let keys: Vec<u64> = (0..10_000).collect();
        /// let values: Vec<[u8; 3]> = (0..10_000_u32)
        ///     .map(|i| [i as u8, (i >> 8) as u8, 7])
        ///     .collect();
        /// let func =
        ///     <PHunc<PHastR<u64>, Box<[[u8; 3]]>>>::try_par_new(&keys, &values, no_logs![])?;
        ///
        /// for (key, &value) in keys.iter().zip(&values) {
        ///     assert_eq!(func.get(key), value);
        /// }
        /// # Ok(())
        /// # }
        /// ```
        pub fn try_par_new(
            keys: &[impl Borrow<K> + Sync],
            values: &[V],
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, values, PHastRBuilder::default(), pl)
        }

        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a boxed slice, from a slice of keys and a slice of values, using
        /// the given [`PHastRBuilder`] configuration.
        ///
        /// Signatures are computed, and values are written at the outputs
        /// of their keys, in parallel. [Offline mode] is not supported.
        ///
        /// See `try_par_new` (in this impl block) for the errors.
        ///
        /// [Offline mode]: PHastRBuilder::offline
        pub fn try_par_new_with_builder(
            keys: &[impl Borrow<K> + Sync],
            values: &[V],
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
        {
            check_lens(keys.len(), values.len())?;
            let func = <PHastR<K, S>>::try_par_new_with_builder(keys, builder, pl)?;
            let mut data = vec![V::new_zeroed(); func.len().max(1)];
            let sync_data = data.as_sync_slice();
            // SAFETY: the outputs of the keys are distinct, so each cell is
            // written by a single thread
            par_store(
                &func,
                keys,
                values,
                |i, v| unsafe { sync_data[i].set(v) },
                pl,
            );
            Ok(PHunc {
                func,
                data: data.into_boxed_slice(),
            })
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Constructors: PHunc<PHastR<...>, BitFieldVec<Box<[W]>>>
    // ═══════════════════════════════════════════════════════════════════

    impl<
        K: ?Sized + ToSig<[u64; 1]>,
        S: SeedStoreBuild + MemSize + FlatType + Sync,
        W: Word + BinSafe + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned<Value = W>>,
    > PHunc<PHastR<K, S>, BitFieldVec<Box<[W]>>>
    {
        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a [`BitFieldVec`], from keys and values using default
        /// [`PHastRBuilder`] settings.
        ///
        /// This is a convenience wrapper around `try_new_with_builder` (in
        /// this impl block) with `PHastRBuilder::default()`.
        ///
        /// If you have a slice of keys and a slice of values, `try_par_new`
        /// (in this impl block) is faster.
        ///
        /// * `keys` - a [`FallibleLender`] returning the keys, which is read
        ///   just once. The [`lenders`] module provides easy ways to build
        ///   such lenders.
        ///
        /// * `values` - a [`FallibleLender`] returning the values, in the
        ///   order of their keys, which is read just once.
        ///
        /// * `pl` - a progress logger, which logs the parameters and the
        ///   phases of the construction.
        ///
        /// # Errors
        ///
        /// Returns an error if there are duplicate keys, if the two lenders
        /// return a different number of items, if a lender returns an
        /// error, or if there are too many keys for the platform.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logs;
        /// use sux::bits::BitFieldVec;
        /// use sux::func::{PHastR, PHunc};
        /// use sux::utils::FromCloneableIntoIterator;
        ///
        /// // Each value uses ten bits
        /// let func = <PHunc<PHastR<usize>, BitFieldVec<Box<[usize]>>>>::try_new(
        ///     FromCloneableIntoIterator::new(0..10_000_usize),
        ///     FromCloneableIntoIterator::new((0..10_000_usize).map(|i| i % 1000)),
        ///     no_logs![],
        /// )?;
        ///
        /// for key in 0..10_000_usize {
        ///     assert_eq!(func.get(key), key % 1000);
        /// }
        /// # Ok(())
        /// # }
        /// ```
        ///
        /// [`lenders`]: crate::utils::lenders
        pub fn try_new<B: ?Sized + Borrow<K>>(
            keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, values, PHastRBuilder::default(), pl)
        }

        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a [`BitFieldVec`], from keys and values using the given
        /// [`PHastRBuilder`] configuration.
        ///
        /// The two lenders are read just once: each value is kept next to
        /// the two signatures of its key, in memory or, in [offline mode],
        /// on disk, and, once the [`PHastR`] has been built, the values are
        /// written in parallel at the outputs of their keys, which are
        /// computed from the signatures. The bit width is that of the
        /// largest value, which is found while reading the values.
        ///
        /// See `try_new` (in this impl block) for the arguments and the
        /// errors.
        ///
        /// [offline mode]: PHastRBuilder::offline
        pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
            keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleLender<Error: Error + Send + Sync + 'static>
            + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self> {
            let mut max = W::ZERO;
            let (func, stored) = builder.try_build_from_lender_with_values::<K, S, B, W>(
                keys,
                values,
                |&v| max = max.max(v),
                pl,
            )?;
            let mut result = Ok(());
            let data = par_fill_hashes::<W>(max.bit_len() as usize, func.len().max(1), |data| {
                if let Some(stored) = &stored {
                    // The outputs of the keys are distinct, but the fields
                    // of different outputs may share a word: we write
                    // atomically
                    result =
                        stored.for_each(&func, |i, v| data.set_atomic(i, v, Ordering::Relaxed), pl);
                }
            });
            result?;
            Ok(PHunc { func, data })
        }

        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a [`BitFieldVec`], from a slice of keys and a slice of values,
        /// using default [`PHastRBuilder`] settings.
        ///
        /// This is a convenience wrapper around `try_par_new_with_builder`
        /// (in this impl block) with `PHastRBuilder::default()`.
        ///
        /// If keys and values are produced sequentially (e.g., from a
        /// file), use `try_new` (in this impl block) instead.
        ///
        /// # Errors
        ///
        /// Returns an error if there are duplicate keys, if the two slices
        /// have different lengths, or if there are too many keys for the
        /// platform.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # fn main() -> anyhow::Result<()> {
        /// # use dsi_progress_logger::no_logs;
        /// use sux::bits::BitFieldVec;
        /// use sux::func::{PHastR, PHunc};
        /// use sux::traits::TryIntoUnaligned;
        ///
        /// let keys: Vec<u64> = (0..10_000).collect();
        /// let values: Vec<usize> = (0..10_000).map(|i| i % 1000).collect();
        /// let func = <PHunc<PHastR<u64>, BitFieldVec<Box<[usize]>>>>::try_par_new(
        ///     &keys,
        ///     &values,
        ///     no_logs![],
        /// )?;
        /// // Unaligned reads are usually faster
        /// let func = func.try_into_unaligned()?;
        ///
        /// for (key, &value) in keys.iter().zip(&values) {
        ///     assert_eq!(func.get(key), value);
        /// }
        /// # Ok(())
        /// # }
        /// ```
        pub fn try_par_new(
            keys: &[impl Borrow<K> + Sync],
            values: &[W],
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, values, PHastRBuilder::default(), pl)
        }

        /// Builds a [`PHunc`] based on a [`PHastR`], with values stored in
        /// a [`BitFieldVec`], from a slice of keys and a slice of values,
        /// using the given [`PHastRBuilder`] configuration.
        ///
        /// Signatures are computed, and values are written at the outputs
        /// of their keys, in parallel. The bit width is that of the largest
        /// value. [Offline mode] is not supported.
        ///
        /// See `try_par_new` (in this impl block) for the errors.
        ///
        /// [Offline mode]: PHastRBuilder::offline
        pub fn try_par_new_with_builder(
            keys: &[impl Borrow<K> + Sync],
            values: &[W],
            builder: PHastRBuilder,
            pl: &mut impl ProgressLog,
        ) -> Result<Self>
        where
            K: Sync,
        {
            check_lens(keys.len(), values.len())?;
            let func = <PHastR<K, S>>::try_par_new_with_builder(keys, builder, pl)?;
            let max = values.par_iter().copied().max().unwrap_or(W::ZERO);
            let data = par_fill_hashes::<W>(max.bit_len() as usize, func.len().max(1), |data| {
                // The outputs of the keys are distinct, but the fields of
                // different outputs may share a word: we write atomically
                par_store(
                    &func,
                    keys,
                    values,
                    |i, v| data.set_atomic(i, v, Ordering::Relaxed),
                    pl,
                );
            });
            Ok(PHunc { func, data })
        }
    }
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests {
    use super::*;
    use crate::bits::BitFieldVec;
    use crate::func::PHastRBuilder;
    use crate::utils::{FromSlice, PrimitiveUnsignedExt};
    use anyhow::Result;
    use dsi_progress_logger::no_logs;

    // In tests, the stores of a construction from a lender have 32
    // partitions, with blocks of 16 to 64 items on disk: the sizes below go
    // from stores with empty partitions to stores with many blocks per
    // partition, whose length grows during the construction.
    const SIZES: [usize; 9] = [0, 1, 2, 31, 100, 1000, 4097, 10_000, 30_000];

    /// Builds from lenders, with the given builder, functions with values
    /// of different sizes on the first `n` integers, and checks their
    /// values; checks also that the minimal perfect hash function is the
    /// one built without values.
    fn check(n: usize, builder: impl Fn() -> PHastRBuilder) -> Result<()> {
        let keys: Vec<u64> = (0..n as u64).collect();
        let plain =
            <PHastR<u64>>::try_new_with_builder(FromSlice::new(&keys), builder(), no_logs![])?;

        // Items as large as half a record
        let values: Vec<u64> = keys
            .iter()
            .map(|&k| k.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let func = <PHunc<PHastR<u64>, Box<[u64]>>>::try_new_with_builder(
            FromSlice::new(&keys),
            FromSlice::new(&values),
            builder(),
            no_logs![],
        )?;
        assert_eq!(func.len(), n);
        for (key, &value) in keys.iter().zip(&values) {
            assert_eq!(func.get(key), value, "key {key} of {n}");
            assert_eq!(func.func.get(key), plain.get(key), "key {key} of {n}");
        }

        // Items of one byte
        let values: Vec<u8> = keys.iter().map(|&k| (k * 7) as u8).collect();
        let func = <PHunc<PHastR<u64>, Box<[u8]>>>::try_new_with_builder(
            FromSlice::new(&keys),
            FromSlice::new(&values),
            builder(),
            no_logs![],
        )?;
        for (key, &value) in keys.iter().zip(&values) {
            assert_eq!(func.get(key), value, "key {key} of {n}");
        }

        // Items larger than a record, with odd alignment
        let values: Vec<[u8; 19]> = keys
            .iter()
            .map(|&k| std::array::from_fn(|i| (k >> (i % 8)) as u8))
            .collect();
        let func = <PHunc<PHastR<u64>, Box<[[u8; 19]]>>>::try_new_with_builder(
            FromSlice::new(&keys),
            FromSlice::new(&values),
            builder(),
            no_logs![],
        )?;
        for (key, &value) in keys.iter().zip(&values) {
            assert_eq!(func.get(key), value, "key {key} of {n}");
        }

        // Fields sharing words
        let values: Vec<usize> = keys.iter().map(|&k| (k % 1000) as usize).collect();
        let func = <PHunc<PHastR<u64>, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
            FromSlice::new(&keys),
            FromSlice::new(&values),
            builder(),
            no_logs![],
        )?;
        assert_eq!(
            func.data.bit_width(),
            values.iter().copied().max().unwrap_or(0).bit_len() as usize
        );
        for (key, &value) in keys.iter().zip(&values) {
            assert_eq!(func.get(key), value, "key {key} of {n}");
        }
        Ok(())
    }

    #[test]
    fn test_lenders() -> Result<()> {
        for n in SIZES {
            check(n, PHastRBuilder::default)?;
            check(n, || PHastRBuilder::default().offline(true))?;
        }
        Ok(())
    }

    /// The values are written by a thread for each range of partitions.
    #[test]
    fn test_threads() -> Result<()> {
        for threads in [1, 2, 5, 40] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?
                .install(|| {
                    for n in [100, 4097, 20_000] {
                        check(n, PHastRBuilder::default)?;
                        check(n, || PHastRBuilder::default().offline(true))?;
                    }
                    Ok::<_, anyhow::Error>(())
                })?;
        }
        Ok(())
    }

    #[test]
    fn test_zero_values() -> Result<()> {
        let keys: Vec<u64> = (0..1000).collect();
        let values = vec![0_usize; 1000];
        let func = <PHunc<PHastR<u64>, BitFieldVec<Box<[usize]>>>>::try_new(
            FromSlice::new(&keys),
            FromSlice::new(&values),
            no_logs![],
        )?;
        assert_eq!(func.data.bit_width(), 1);
        for key in &keys {
            assert_eq!(func.get(key), 0);
        }
        Ok(())
    }

    #[test]
    fn test_different_lengths() {
        let keys: Vec<u64> = (0..1000).collect();
        for num_values in [0, 999, 1001] {
            let values = vec![0_u64; num_values];
            for offline in [false, true] {
                let result = <PHunc<PHastR<u64>, Box<[u64]>>>::try_new_with_builder(
                    FromSlice::new(&keys),
                    FromSlice::new(&values),
                    PHastRBuilder::default().offline(offline),
                    no_logs![],
                );
                assert!(result.is_err(), "{num_values} values");
            }
            assert!(
                <PHunc<PHastR<u64>, Box<[u64]>>>::try_par_new(&keys, &values, no_logs![]).is_err()
            );
        }
    }

    #[test]
    fn test_duplicate_keys() {
        let mut keys: Vec<u64> = (0..1000).collect();
        keys[999] = 0;
        let values = vec![0_u64; 1000];
        assert!(
            <PHunc<PHastR<u64>, Box<[u64]>>>::try_new(
                FromSlice::new(&keys),
                FromSlice::new(&values),
                no_logs![],
            )
            .is_err()
        );
        assert!(<PHunc<PHastR<u64>, Box<[u64]>>>::try_par_new(&keys, &values, no_logs![]).is_err());
    }
}
