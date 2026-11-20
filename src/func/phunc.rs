/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Static functions storing their values in an array indexed by a minimal
//! perfect hash function.

use std::borrow::Borrow;

use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use value_traits::slices::SliceByValue;

use crate::func::lcp_mmphf::{BitPrefix, IntBitPrefix};
use crate::func::phastr::{LevelParams, Remap, SeedStore, SeedStoreBuild};
use crate::func::shard_edge::ShardEdge;
use crate::func::{Lcp2Mmphf, Lcp2MmphfInt, LcpMmphf, LcpMmphfInt, PHastR};
use crate::traits::{TryIntoUnaligned, Unaligned, UnalignedConversionError};
use crate::utils::{Sig, ToSig};

/// A minimal perfect hash function, which maps bijectively a set of *n* keys
/// to the integers in [0 . . *n*).
///
/// This trait is the interface through which a [`PHunc`] finds the position
/// of the value of a key. It is implemented by [`PHastR`], and by the
/// monotone minimal perfect hash functions [`LcpMmphfInt`], [`LcpMmphf`],
/// [`Lcp2MmphfInt`], and [`Lcp2Mmphf`], which map each key to its rank.
///
/// The constant [`CLAMPED`] tells whether the output of the keys outside
/// the original set is still in [0 . . *n*): if it is not, a [`PHunc`]
/// clamps the output before accessing its values.
///
/// The trait is implemented for references and boxes, so several
/// [`PHunc`]s can share the same function.
///
/// # Safety
///
/// If [`CLAMPED`] is true, the value returned by [`index`] must be smaller
/// than [`len`] for every key, including the keys outside the original set;
/// if the function is empty, it must be zero. A [`PHunc`] relies on this
/// property to access its values without bounds checks. Implementations
/// for which [`CLAMPED`] is false have no obligation.
///
/// [`CLAMPED`]: Mphf::CLAMPED
/// [`index`]: Mphf::index
/// [`len`]: Mphf::len
pub unsafe trait Mphf {
    /// The type of the keys.
    type Key: ?Sized;

    /// Whether the value returned by [`index`] is smaller than [`len`] also
    /// for the keys outside the original set (and zero if the function is
    /// empty).
    ///
    /// [`index`]: Mphf::index
    /// [`len`]: Mphf::len
    const CLAMPED: bool;

    /// Returns the number of keys.
    fn len(&self) -> usize;

    /// Returns whether the function has no keys.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the output of the given key.
    ///
    /// Different keys of the original set have different outputs, which are
    /// smaller than [`len`]. If the key was not in the original set, the
    /// result is arbitrary; if [`CLAMPED`] is true, it is still smaller than
    /// [`len`] (zero, if the function is empty).
    ///
    /// [`len`]: Mphf::len
    /// [`CLAMPED`]: Mphf::CLAMPED
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
/// The space overhead is that of the minimal perfect hash function (about
/// 1.94 bits per key for a [`PHastR`] with default parameters), whatever the
/// size of the values; moreover, values are stored as they are, so they can
/// be of any type that can be stored in the array, and not just integers.
/// A query needs two dependent memory accesses: one for the minimal perfect
/// hash function, and one for the array.
///
/// The position of the value of a key is available, too: [`index`] returns
/// it, and [`get_with_index`] returns it together with the value, computing
/// the output of the minimal perfect hash function just once. Since the
/// positions of the keys are distinct and smaller than *n*, they can be used
/// to access other arrays of data associated with the keys; with a monotone
/// minimal perfect hash function such as [`LcpMmphf`], the position of a key
/// is its rank.
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
/// [constructors] listed below, or assembled using [`from_parts`], and can
/// be serialized using [ε-serde] or [`serde`].
///
/// The `phunc` binary (`cli` feature) builds and serializes instances of
/// this structure.
///
/// # Alternatives
///
/// - A [`VFunc`] stores integer values with a space overhead that is a
///   fraction of their size (10.5% with default parameters), rather than a
///   fixed number of bits per key: it uses less space when values have
///   fewer than about twenty bits. A query needs three memory accesses, but
///   they are independent. A [`VFunc`] does not associate a position with
///   each key, so it cannot be used to access other data: each array of
///   data needs its own function.
///
/// - [`VFunc2`] and [`CompVFunc`] use less space when the distribution of
///   the values is skewed.
///
/// - A [`PHastR`] alone is sufficient if the values are any bijection
///   between the keys and [0 . . *n*), and an [`LcpMmphf`] if they are the
///   ranks of the keys.
///
/// - Contrarily to a [`VFunc`] mapping each key to its index, a [`PHunc`]
///   cannot be wrapped in a [`SignedFunc`].
///
/// # Construction
///
/// Constructors are available for a [`PHunc`] based on a [`PHastR`], with
/// values stored in a `Box<[V]>` or in a `BitFieldVec<Box<[W]>>`; in the
/// second case, the bit width is that of the largest value. They require
/// the `rayon` feature.
///
/// To build a function from a slice of keys and a slice of values, use
/// [`try_par_new`], which is parallel in all its phases; to read keys and
/// values from two [lenders] (for example, from files), use [`try_new`],
/// which reads them just once: values are kept next to the signatures of
/// their keys, in memory or, in [offline mode], on disk, and they are
/// written into the array in parallel after the minimal perfect hash
/// function has been built, without reading the keys again. The variants
/// [`try_par_new_with_builder`] and [`try_new_with_builder`] make it
/// possible to configure the construction of the [`PHastR`] using a
/// [`PHastRBuilder`]. The memory used by the construction is that
/// [used by `PHastR`], plus the array of values; construction from lenders
/// stores in addition a copy of the values, in memory or on disk.
///
/// For other minimal perfect hash functions and other arrays, build the two
/// components, store the value of each key at the position returned by
/// [`Mphf::index`], and use [`from_parts`]. In particular, with a monotone
/// minimal perfect hash function such as [`LcpMmphf`], which maps each key
/// to its rank, the array is just the list of the values in the order of
/// the keys.
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
/// let func = <PHunc<PHastR<str>, Box<[usize]>>>::try_par_new(
///     &keys,
///     &values,
///     no_logs![],
/// )?;
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
/// [`VFunc2`]: crate::func::VFunc2
/// [`CompVFunc`]: crate::func::CompVFunc
/// [`SignedFunc`]: crate::func::SignedFunc
/// [`BitFieldVec`]: crate::bits::BitFieldVec
/// [unaligned reads]: crate::bits::BitFieldVec::get_unaligned
/// [`from_parts`]: PHunc::from_parts
/// [`index`]: PHunc::index
/// [`get_with_index`]: PHunc::get_with_index
/// [constructors]: PHunc#construction
/// [`try_new`]: PHunc::try_new
/// [`try_new_with_builder`]: PHunc::try_new_with_builder
/// [`try_par_new`]: PHunc::try_par_new
/// [`try_par_new_with_builder`]: PHunc::try_par_new_with_builder
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
    mphf: F,
    /// The values, indexed by the outputs of the keys; if there are no keys,
    /// a single value, which is what queries return.
    values: D,
}

impl<F: Mphf, D: SliceByValue> PHunc<F, D> {
    /// Creates a new [`PHunc`] from a minimal perfect hash function and an
    /// array containing the value of each key at the position returned by
    /// [`Mphf::index`].
    ///
    /// If the function is empty, the array must contain a single value,
    /// which is what queries return.
    ///
    /// Both parts can be references: for example, several functions can
    /// share the same minimal perfect hash function.
    ///
    /// # Panics
    ///
    /// Panics if the number of values is different from the number of keys
    /// or, if there are no keys, from one.
    ///
    /// # Examples
    ///
    /// Two functions sharing a minimal perfect hash function:
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// # use dsi_progress_logger::no_logs;
    /// use sux::func::{Mphf, PHastR, PHunc};
    ///
    /// let keys = ["apple", "banana", "cherry"];
    /// let phf = <PHastR<str>>::try_par_new(&keys, no_logs![])?;
    ///
    /// let mut lens = vec![0; 3];
    /// let mut initials = vec![' '; 3];
    /// for key in keys {
    ///     let index = phf.index(key);
    ///     lens[index] = key.len();
    ///     initials[index] = key.chars().next().unwrap();
    /// }
    ///
    /// let lens = PHunc::from_parts(&phf, lens);
    /// let initials = PHunc::from_parts(&phf, initials);
    /// assert_eq!(lens.get("banana"), 6);
    /// assert_eq!(initials.get("cherry"), 'c');
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_parts(mphf: F, values: D) -> Self {
        assert_eq!(
            values.len(),
            mphf.len().max(1),
            "the number of values must be the number of keys (one, if there are no keys)"
        );
        Self { mphf, values }
    }

    /// Returns the value associated with the given key.
    ///
    /// If the key was not in the original set, the result is one of the
    /// values (the padding value, if the function is empty).
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<F::Key>) -> D::Value {
        // SAFETY: see index
        unsafe { self.values.get_value_unchecked(self.index(key)) }
    }

    /// Returns the position of the value of the given key, that is, the
    /// output of the minimal perfect hash function.
    ///
    /// Different keys of the original set have different positions in
    /// [0 . . *n*), where *n* is the number of keys, so the position can be
    /// used to access other arrays of data associated with the keys; with a
    /// monotone minimal perfect hash function, it is the rank of the key.
    ///
    /// If the key was not in the original set, the result is arbitrary, but
    /// it is still in [0 . . *n*) (zero, if the function is empty), even if
    /// the minimal perfect hash function does not [clamp its output].
    ///
    /// [clamp its output]: Mphf::CLAMPED
    #[inline(always)]
    pub fn index(&self, key: impl Borrow<F::Key>) -> usize {
        let index = self.mphf.index(key.borrow());
        // The test is on a constant, so functions that clamp their output
        // pay nothing. In both cases the result is smaller than the number
        // of values (which is the number of keys, or one if there are no
        // keys): by the contract of Mphf, or because we clamp it.
        if F::CLAMPED {
            index
        } else {
            index.min(self.values.len() - 1)
        }
    }

    /// Returns the position of the value of the given key (see [`index`])
    /// and the value, computing the output of the minimal perfect hash
    /// function just once.
    ///
    /// [`index`]: Self::index
    #[inline(always)]
    pub fn get_with_index(&self, key: impl Borrow<F::Key>) -> (usize, D::Value) {
        let index = self.index(key);
        // SAFETY: see index
        (index, unsafe { self.values.get_value_unchecked(index) })
    }

    /// Returns the number of keys.
    pub fn len(&self) -> usize {
        self.mphf.len()
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.mphf.is_empty()
    }
}

impl<F, D> PHunc<F, D> {
    /// Returns a reference to the minimal perfect hash function.
    pub const fn mphf(&self) -> &F {
        &self.mphf
    }

    /// Returns a reference to the array of values, which are indexed by
    /// the positions returned by [`index`].
    ///
    /// [`index`]: Self::index
    pub const fn values(&self) -> &D {
        &self.values
    }

    /// Returns the minimal perfect hash function and the array of values.
    pub fn into_parts(self) -> (F, D) {
        (self.mphf, self.values)
    }
}

// Mphf is an unsafe trait, which autoimpl cannot implement, so we forward it
// to references and boxes explicitly.
macro_rules! forward_mphf {
    ($($ptr:ty),+) => {$(
        // SAFETY: all items, including CLAMPED, are those of the function
        // the pointer refers to
        unsafe impl<T: Mphf + ?Sized> Mphf for $ptr {
            type Key = T::Key;
            const CLAMPED: bool = T::CLAMPED;

            #[inline(always)]
            fn len(&self) -> usize {
                (**self).len()
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                (**self).is_empty()
            }

            #[inline(always)]
            fn index(&self, key: &Self::Key) -> usize {
                (**self).index(key)
            }
        }
    )+};
}

forward_mphf!(&T, &mut T, Box<T>);

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
    const CLAMPED: bool = true;

    #[inline(always)]
    fn len(&self) -> usize {
        PHastR::len(self)
    }

    #[inline(always)]
    fn index(&self, key: &K) -> usize {
        self.get(key)
    }
}

// SAFETY: the output is not clamped, so there is no obligation
unsafe impl<
    K: PrimitiveInteger + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Mphf for LcpMmphfInt<K, D, S0, E0, S1, E1>
where
    IntBitPrefix<K>: ToSig<S1>,
{
    type Key = K;
    const CLAMPED: bool = false;

    #[inline(always)]
    fn len(&self) -> usize {
        LcpMmphfInt::len(self)
    }

    #[inline(always)]
    fn index(&self, key: &K) -> usize {
        self.get(*key)
    }
}

// SAFETY: the output is not clamped, so there is no obligation
unsafe impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Mphf for LcpMmphf<K, D, S0, E0, S1, E1>
where
    BitPrefix: ToSig<S1>,
{
    type Key = K;
    const CLAMPED: bool = false;

    #[inline(always)]
    fn len(&self) -> usize {
        LcpMmphf::len(self)
    }

    #[inline(always)]
    fn index(&self, key: &K) -> usize {
        self.get(key)
    }
}

// SAFETY: the output is not clamped, so there is no obligation
unsafe impl<
    K: PrimitiveInteger + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Mphf for Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1>
where
    IntBitPrefix<K>: ToSig<S1>,
{
    type Key = K;
    const CLAMPED: bool = false;

    #[inline(always)]
    fn len(&self) -> usize {
        Lcp2MmphfInt::len(self)
    }

    #[inline(always)]
    fn index(&self, key: &K) -> usize {
        self.get(*key)
    }
}

// SAFETY: the output is not clamped, so there is no obligation
unsafe impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Mphf for Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>
where
    BitPrefix: ToSig<S1>,
{
    type Key = K;
    const CLAMPED: bool = false;

    #[inline(always)]
    fn len(&self) -> usize {
        Lcp2Mmphf::len(self)
    }

    #[inline(always)]
    fn index(&self, key: &K) -> usize {
        self.get(key)
    }
}

impl<F: TryIntoUnaligned<Unaligned: Mphf>, D: TryIntoUnaligned<Unaligned: SliceByValue>>
    TryIntoUnaligned for PHunc<F, D>
{
    type Unaligned = PHunc<F::Unaligned, D::Unaligned>;

    /// Converts the two parts.
    ///
    /// # Panics
    ///
    /// Panics if the converted parts have a different number of keys and
    /// values (see [`from_parts`]): the conversions are safe code, but
    /// queries access the values without bounds checks.
    ///
    /// [`from_parts`]: PHunc::from_parts
    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        Ok(PHunc::from_parts(
            self.mphf.try_into_unaligned()?,
            self.values.try_into_unaligned()?,
        ))
    }
}

impl<
    K: ?Sized + ToSig<[u64; 1]>,
    S: SeedStoreBuild + TryIntoUnaligned<Unaligned: SeedStore> + From<Unaligned<S>>,
    P: AsRef<[LevelParams]>,
    D: SliceByValue + TryIntoUnaligned<Unaligned: SliceByValue> + From<Unaligned<D>>,
> From<Unaligned<PHunc<PHastR<K, S, P, Remap>, D>>> for PHunc<PHastR<K, S, P, Remap>, D>
{
    /// Converts the two parts.
    ///
    /// # Panics
    ///
    /// Panics if the converted parts have a different number of keys and
    /// values (see [`from_parts`]).
    ///
    /// [`from_parts`]: PHunc::from_parts
    fn from(f: Unaligned<PHunc<PHastR<K, S, P, Remap>, D>>) -> Self {
        PHunc::from_parts(f.mphf.into(), f.values.into())
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
    use crate::func::vbuilder::check_num_values;
    use crate::traits::{AtomicBitFieldSlice, Word};
    use crate::utils::{BinSafe, PrimitiveUnsignedExt};
    use anyhow::Result;
    use atomic_primitive::{AtomicPrimitive, PrimitiveAtomicUnsigned};
    use core::error::Error;
    use dsi_progress_logger::ProgressLog;
    use lender::{FallibleLender, FallibleLending};
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;
    use std::time::Instant;
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
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|(key, &value)| set(func.get(key.borrow()), value));
        pl.update_with_count(keys.len());
        pl.done();
    }

    /// Logs the maximum value and the bit width of the values of a function
    /// with `n` keys, in the format used by [`VFunc`].
    ///
    /// [`VFunc`]: crate::func::VFunc
    fn log_values<W: Word>(n: usize, max: W, pl: &mut impl ProgressLog) {
        pl.info(format_args!("Max value: {}", max));
        pl.info(format_args!(
            "Number of keys: {}; bit width: {}",
            n,
            max.bit_len()
        ));
    }

    /// Logs the number of bits per key of a function with the given
    /// minimal perfect hash function and `data_bytes` bytes of values, and
    /// the time taken by its construction, in the format used by the other
    /// functions (the lines logged by the construction of the [`PHastR`] are
    /// about the minimal perfect hash function only).
    fn log_summary<K: ?Sized, S: SeedStoreBuild + MemSize + FlatType>(
        func: &PHastR<K, S>,
        data_bytes: usize,
        bit_width: Option<usize>,
        start: Instant,
        pl: &mut impl ProgressLog,
    ) {
        let n = func.len().max(1) as f64;
        // In floating point, as the number of bits can overflow on 32-bit
        // platforms
        let bits = (func.mem_size(SizeFlags::default()) + data_bytes) as f64 * 8.0 / n;
        match bit_width {
            Some(bit_width) => pl.info(format_args!(
                "Bits/key: {:.3} ({:+.3}% with respect to bit width)",
                bits,
                100.0 * (bits / bit_width as f64 - 1.0)
            )),
            None => pl.info(format_args!("Bits/key: {:.3}", bits)),
        }
        let elapsed = start.elapsed();
        pl.info(format_args!(
            "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
            elapsed.as_secs_f64(),
            func.len(),
            elapsed.as_nanos() as f64 / n
        ));
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
        ///   order of their keys, which is read just once. It may return
        ///   more values than there are keys (in particular, it may be
        ///   infinite); the extra values are ignored.
        ///
        /// * `pl` - a progress logger, which logs the parameters and the
        ///   phases of the construction.
        ///
        /// # Errors
        ///
        /// Returns an error if there are duplicate keys, if there are
        /// fewer values than keys ([`MismatchedKeysAndValues`]), if a lender
        /// returns an error, or if there are too many keys for the platform.
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
        /// [`MismatchedKeysAndValues`]: crate::func::BuildError::MismatchedKeysAndValues
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
            let start = Instant::now();
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
            log_summary(&func, size_of_val(&*data), None, start, pl);
            Ok(PHunc {
                mphf: func,
                values: data.into_boxed_slice(),
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
        /// have different lengths ([`MismatchedKeysAndValues`]), or if
        /// there are too many keys for the platform.
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
        /// let func = <PHunc<PHastR<u64>, Box<[[u8; 3]]>>>::try_par_new(
        ///     &keys,
        ///     &values,
        ///     no_logs![],
        /// )?;
        ///
        /// for (key, &value) in keys.iter().zip(&values) {
        ///     assert_eq!(func.get(key), value);
        /// }
        /// # Ok(())
        /// # }
        /// ```
        ///
        /// [`MismatchedKeysAndValues`]: crate::func::BuildError::MismatchedKeysAndValues
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
            check_num_values(keys.len(), values.len())?;
            let start = Instant::now();
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
            log_summary(&func, size_of_val(&*data), None, start, pl);
            Ok(PHunc {
                mphf: func,
                values: data.into_boxed_slice(),
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
        ///   order of their keys, which is read just once. It may return
        ///   more values than there are keys (in particular, it may be
        ///   infinite); the extra values are ignored.
        ///
        /// * `pl` - a progress logger, which logs the parameters and the
        ///   phases of the construction.
        ///
        /// # Errors
        ///
        /// Returns an error if there are duplicate keys, if there are
        /// fewer values than keys ([`MismatchedKeysAndValues`]), if a lender
        /// returns an error, or if there are too many keys for the platform.
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
        /// [`MismatchedKeysAndValues`]: crate::func::BuildError::MismatchedKeysAndValues
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
            let start = Instant::now();
            let mut max = W::ZERO;
            let (func, stored) = builder.try_build_from_lender_with_values::<K, S, B, W>(
                keys,
                values,
                |&v| max = max.max(v),
                pl,
            )?;
            log_values(func.len(), max, pl);
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
            log_summary(
                &func,
                size_of_val(data.as_slice()),
                Some(data.bit_width()),
                start,
                pl,
            );
            Ok(PHunc {
                mphf: func,
                values: data,
            })
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
        /// have different lengths ([`MismatchedKeysAndValues`]), or if
        /// there are too many keys for the platform.
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
        ///
        /// [`MismatchedKeysAndValues`]: crate::func::BuildError::MismatchedKeysAndValues
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
            check_num_values(keys.len(), values.len())?;
            let start = Instant::now();
            let func = <PHastR<K, S>>::try_par_new_with_builder(keys, builder, pl)?;
            let max = values
                .par_iter()
                .with_min_len(crate::RAYON_MIN_LEN)
                .copied()
                .max()
                .unwrap_or(W::ZERO);
            log_values(func.len(), max, pl);
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
            log_summary(
                &func,
                size_of_val(data.as_slice()),
                Some(data.bit_width()),
                start,
                pl,
            );
            Ok(PHunc {
                mphf: func,
                values: data,
            })
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
            assert_eq!(func.mphf.get(key), plain.get(key), "key {key} of {n}");
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
            func.values.bit_width(),
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
        assert_eq!(func.values.bit_width(), 1);
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
                // A lender can return more values than there are keys
                assert_eq!(result.is_err(), num_values < 1000, "{num_values} values");
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
