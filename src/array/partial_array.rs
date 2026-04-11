/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Immutable [partial array] implementations.
//!
//! [partial array]: PartialArray

use std::marker::PhantomData;

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::bits::{BitFieldVec, BitVec};
use crate::dict::EliasFanoBuilder;
use crate::dict::elias_fano::EliasFano;
use crate::panic_if_out_of_bounds;
use crate::rank_sel::{Rank9, SelectZeroAdaptConst};
use crate::traits::Backend;
use crate::traits::TryIntoUnaligned;
use crate::traits::Unaligned;
use crate::traits::{BitVecOps, BitVecOpsMut};
use crate::traits::{RankUnchecked, SuccUnchecked};

// Rank9 is inherently u64-based, so the dense index must use u64 backing
// regardless of usize.
type DenseIndex = Rank9<BitVec<Box<[u64]>>>;

/// An internal index for sparse partial arrays.
///
/// We cannot use directly an [Elias–Fano] structure because we need to
/// keep track of the first invalid position; and we need to keep track of
/// the first invalid position because we want to implement just
/// [`SuccUnchecked`] on the Elias–Fano structure, because it requires just
/// [`SelectZeroUnchecked`], whereas [`Succ`] would require
/// [`SelectUnchecked`] as well.
///
/// [Elias–Fano]: crate::dict::EliasFano
/// [`SuccUnchecked`]: crate::traits::SuccUnchecked
/// [`SelectZeroUnchecked`]: crate::traits::SelectZeroUnchecked
/// [`Succ`]: crate::traits::Succ
/// [`SelectUnchecked`]: crate::traits::SelectUnchecked
#[doc(hidden)]
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseIndex<D, L = BitFieldVec<Box<[u64]>>> {
    ef: EliasFano<u64, SelectZeroAdaptConst<BitVec<D>, Box<[usize]>, 12, 3>, L>,
    /// self.ef should not be queried for values >= self.first_invalid_position
    first_invalid_pos: usize,
}

/// Builder for creating an immutable partial array.
///
/// The builder allows you to specify the array length and then add
/// (position, value) pairs. Positions must be added in strictly
/// increasing order.
///
/// To get a builder you can use either [new_dense] or [new_sparse].
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct PartialArrayBuilder<T, B> {
    builder: B,
    values: Vec<T>,
    len: usize,
    min_next_pos: usize,
}

/// Creates a new builder for a dense partial array of the given length.
///
/// A dense partial array stores a bit vector of the given length to mark
/// which positions contain values, and uses ranking on this bit vector to
/// map positions to indices in a contiguous value array.
///
/// If your set of values is really sparse, consider using a
/// [sparse partial array].
///
/// # Examples
///
/// ```rust
/// # use sux::array::partial_array;
/// let mut builder = partial_array::new_dense(10);
/// builder.set(1, "foo");
/// builder.set(2, "hello");
/// builder.set(7, "world");
///
/// let array = builder.build();
/// assert_eq!(array.get(1), Some(&"foo"));
/// assert_eq!(array.get(2), Some(&"hello"));
/// assert_eq!(array.get(3), None);
/// assert_eq!(array.get(7), Some(&"world"));
/// ```
///
/// [sparse partial array]: new_sparse
pub fn new_dense<T>(len: usize) -> PartialArrayBuilder<T, BitVec<Box<[u64]>>> {
    let n_of_words = len.div_ceil(64);
    // SAFETY: the backing has exactly enough words for len bits
    let bit_vec = unsafe { BitVec::from_raw_parts(vec![0u64; n_of_words].into_boxed_slice(), len) };
    PartialArrayBuilder {
        builder: bit_vec,
        values: vec![],
        len,
        min_next_pos: 0,
    }
}

impl<T> PartialArrayBuilder<T, BitVec<Box<[u64]>>> {
    /// Sets a value at the given position.
    ///
    /// The provided position must be greater than the last position set.
    pub fn set(&mut self, position: usize, value: T) {
        if position < self.min_next_pos {
            panic!(
                "Positions must be set in increasing order: got {} after {}",
                position,
                self.min_next_pos - 1
            );
        }
        panic_if_out_of_bounds!(position, self.len);

        // SAFETY: position < len
        unsafe {
            self.builder.set_unchecked(position, true);
        }
        self.values.push(value);
        self.min_next_pos = position + 1;
    }

    /// Builds the immutable dense partial array.
    pub fn build(self) -> PartialArray<T, Rank9<BitVec<Box<[u64]>>>> {
        let (bit_vec, values) = (self.builder, self.values);
        let rank9 = Rank9::new(bit_vec);
        let values = values.into_boxed_slice();

        PartialArray {
            index: rank9,
            values,
            _marker: PhantomData,
        }
    }
}

/// Creates a new builder for a sparse partial array of the given length.
///
/// A sparse partial array stores the non-empty positions of the array in an
/// [Elias-Fano] structure.
///
/// [Elias-Fano]: crate::dict::EliasFano
///
/// If your set of values is really dense, consider using a [dense partial
/// array](new_dense).
///
/// # Examples
///
/// ```rust
/// # use sux::array::partial_array;
/// let mut builder = partial_array::new_sparse(10, 3);
/// builder.set(1, "foo");
/// builder.set(2, "hello");
/// builder.set(7, "world");
///
/// let array = builder.build();
/// assert_eq!(array.get(1), Some(&"foo"));
/// assert_eq!(array.get(2), Some(&"hello"));
/// assert_eq!(array.get(3), None);
/// assert_eq!(array.get(7), Some(&"world"));
/// ```
///
/// Note that you must specify the number of values in advance.
pub fn new_sparse<T>(
    len: usize,
    num_values: usize,
) -> PartialArrayBuilder<T, EliasFanoBuilder<u64>> {
    PartialArrayBuilder {
        builder: EliasFanoBuilder::<u64>::new(num_values, len as u64),
        values: vec![],
        len,
        min_next_pos: 0,
    }
}

impl<T> PartialArrayBuilder<T, EliasFanoBuilder<u64>> {
    /// Sets a value at the given position.
    ///
    /// The provided position must be greater than the last position
    /// set.
    pub fn set(&mut self, position: usize, value: T) {
        if position < self.min_next_pos {
            panic!(
                "Positions must be set in increasing order: got {} after {}",
                position,
                self.min_next_pos - 1
            );
        }
        panic_if_out_of_bounds!(position, self.len);
        // SAFETY: conditions have been just checked
        unsafe { self.builder.push_unchecked(position as u64) };
        self.values.push(value);
        self.min_next_pos = position + 1;
    }

    /// Builds the immutable sparse partial array.
    pub fn build(self) -> PartialArray<T, SparseIndex<Box<[usize]>>> {
        let (builder, values) = (self.builder, self.values);
        let ef_dict = builder.build_with_dict();
        let values = values.into_boxed_slice();

        PartialArray {
            index: SparseIndex {
                ef: ef_dict,
                first_invalid_pos: self.min_next_pos,
            },
            values,
            _marker: PhantomData,
        }
    }
}

/// Extends the builder with an iterator of (position, value) pairs.
///
/// Position must be in strictly increasing order. The first returned
/// position must be greater than the last position set.
impl<T> Extend<(usize, T)> for PartialArrayBuilder<T, BitVec<Box<[u64]>>> {
    fn extend<I: IntoIterator<Item = (usize, T)>>(&mut self, iter: I) {
        for (pos, val) in iter {
            self.set(pos, val);
        }
    }
}

/// Extends the builder with an iterator of (position, value) pairs.
///
/// Position must be in strictly increasing order. The first returned
/// position must be greater than the last position set.
impl<T> Extend<(usize, T)> for PartialArrayBuilder<T, EliasFanoBuilder<u64>> {
    fn extend<I: IntoIterator<Item = (usize, T)>>(&mut self, iter: I) {
        for (pos, val) in iter {
            self.set(pos, val);
        }
    }
}

/// An immutable partial array that supports efficient queries
/// in compacted storage.
///
/// This structure stores a *partial array*—an array in which only
/// some positions contain values. There is a [dense] and a [sparse]
/// implementation with different space/time trade-offs.
///
/// For convenience, this structure implements [`SliceByValue`].
///
/// When the index structure `P` supports it, this structure implements the
/// [`TryIntoUnaligned`] trait, allowing it
/// to be converted into (usually faster) structures using unaligned access.
///
/// See [`PartialArrayBuilder`] for details on how to create a partial array.
///
/// [dense]: new_dense
/// [sparse]: new_sparse
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartialArray<T, P, V = Box<[T]>> {
    index: P,
    values: V,
    _marker: PhantomData<T>,
}

impl<T, P, V: AsRef<[T]>> PartialArray<T, P, V> {
    /// Returns the number of values stored in the array.
    #[inline(always)]
    pub fn num_values(&self) -> usize {
        self.values.as_ref().len()
    }
}

impl<T, V: AsRef<[T]>> PartialArray<T, DenseIndex, V> {
    /// Returns the total length of the array.
    ///
    /// This is the length that was specified when creating the builder,
    /// not the number of values actually stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if the array has no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a reference to the value at the given position.
    ///
    /// Returns `Some(&value)` if a value is present at the position,
    /// or `None` if no value was stored there.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::partial_array;
    /// let mut builder = partial_array::new_dense(10);
    /// builder.set(5, 42);
    ///
    /// let array = builder.build();
    /// assert_eq!(array.get(5), Some(&42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<&T> {
        panic_if_out_of_bounds!(position, self.len());

        // SAFETY: we just checked
        unsafe { self.get_unchecked(position) }
    }

    /// # Safety
    ///
    /// position < len()
    pub unsafe fn get_unchecked(&self, position: usize) -> Option<&T> {
        // Check if there's a value at this position
        // SAFETY: position < len() guaranteed by caller
        if !unsafe { self.index.get_unchecked(position) } {
            return None;
        }

        // Use ranking to find the index in the values array
        // SAFETY: position < len() guaranteed by caller
        let value_index = unsafe { self.index.rank_unchecked(position) };

        // SAFETY: necessarily value_index < num_values().
        Some(unsafe { self.values.as_ref().get_unchecked(value_index) })
    }
}

impl<T, D: Backend<Word = usize> + AsRef<[usize]>, L: SliceByValue<Value = u64>, V: AsRef<[T]>>
    PartialArray<T, SparseIndex<D, L>, V>
where
    for<'b> &'b L: crate::traits::IntoUncheckedIterator<Item = u64>,
{
    /// Returns the total length of the array.
    ///
    /// This is the length that was specified when creating the builder,
    /// not the number of values actually stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index.ef.upper_bound() as usize
    }

    /// Returns `true` if the array has no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index.ef.upper_bound() == 0
    }

    /// Gets a reference to the value at the given position.
    ///
    /// Returns `Some(&value)` if a value is present at the position,
    /// or `None` if no value was stored there.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::partial_array;
    /// let mut builder = partial_array::new_sparse(10, 1);
    /// builder.set(5, 42);
    ///
    /// let array = builder.build();
    /// assert_eq!(array.get(5), Some(&42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<&T> {
        panic_if_out_of_bounds!(position, self.len());

        // SAFETY: we just checked
        unsafe { self.get_unchecked(position) }
    }

    /// # Safety
    ///
    /// position < len()
    pub unsafe fn get_unchecked(&self, position: usize) -> Option<&T> {
        if position >= self.index.first_invalid_pos {
            return None;
        }
        // Check if there's a value at this position
        // SAFETY: position <= last set position
        let (index, pos) = unsafe { self.index.ef.succ_unchecked::<false>(position as u64) };

        if pos != position as u64 {
            None
        } else {
            // SAFETY: necessarily value_index < num values.
            Some(unsafe { self.values.as_ref().get_unchecked(index) })
        }
    }
}

/// Returns an option even when using `get_value_unchecked` because it should be safe to call
/// whenever `position < len()`.
impl<T: Clone, V: AsRef<[T]>> SliceByValue for PartialArray<T, DenseIndex, V> {
    type Value = Option<T>;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn get_value_unchecked(&self, position: usize) -> Self::Value {
        // SAFETY: position < len() guaranteed by caller
        unsafe { self.get_unchecked(position) }.cloned()
    }
}

/// Returns an option even when using `get_value_unchecked` because it should be safe to call
/// whenever `position < len()`.
impl<
    T: Clone,
    D: Backend<Word = usize> + AsRef<[usize]>,
    L: SliceByValue<Value = u64>,
    V: AsRef<[T]>,
> SliceByValue for PartialArray<T, SparseIndex<D, L>, V>
where
    for<'b> &'b L: crate::traits::IntoUncheckedIterator<Item = u64>,
{
    type Value = Option<T>;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn get_value_unchecked(&self, position: usize) -> Self::Value {
        // SAFETY: position < len() guaranteed by caller
        unsafe { self.get_unchecked(position) }.cloned()
    }
}

// ── Aligned ↔ Unaligned conversion ──────────────────────────────────

impl<D> TryIntoUnaligned for SparseIndex<D> {
    type Unaligned = SparseIndex<D, Unaligned<BitFieldVec<Box<[u64]>>>>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(SparseIndex {
            ef: self.ef.try_into_unaligned()?,
            first_invalid_pos: self.first_invalid_pos,
        })
    }
}

impl<T, P: TryIntoUnaligned, V> TryIntoUnaligned for PartialArray<T, P, V> {
    type Unaligned = PartialArray<T, P::Unaligned, V>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(PartialArray {
            index: self.index.try_into_unaligned()?,
            values: self.values,
            _marker: PhantomData,
        })
    }
}
