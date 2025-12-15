/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Immutable [partial array](PartialArray) implementations.

use std::marker::PhantomData;

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::bits::BitVec;
use crate::dict::EliasFanoBuilder;
use crate::dict::elias_fano::EliasFano;
use crate::panic_if_out_of_bounds;
use crate::rank_sel::{Rank9, SelectZeroAdaptConst};
use crate::traits::{BitVecOps, BitVecOpsMut};
use crate::traits::{RankUnchecked, SuccUnchecked};

type DenseIndex = Rank9<BitVec<Box<[usize]>>>;

/// An internal index for sparse partial arrays.
///
/// We cannot use directly an [Elias–Fano](crate::dict::EliasFano) structure
/// because we need to keep track of the first invalid position; and we need to
/// keep track of the first invalid position because we want to implement just
/// [`SuccUnchecked`](crate::traits::SuccUnchecked) on the Elias–Fano structure,
/// because it requires just
/// [`SelectZeroUnchecked`](crate::traits::SelectZeroUnchecked), whereas
/// [`Succ`](crate::traits::Succ) would require
/// [`SelectUnchecked`](crate::traits::SelectUnchecked) as well.
#[doc(hidden)]
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseIndex<D> {
    ef: EliasFano<SelectZeroAdaptConst<BitVec<D>, D, 12, 3>>,
    /// self.ef should be not be queried for values >= self.first_invalid_position
    first_invalid_pos: usize,
}

/// Builder for creating an immutable partial array.
///
/// The builder allows you to specify the array length and then add
/// (position, value) pairs. Positions must be added in strictly
/// increasing order.
///
/// To get a builder you can use either [new_dense] or [new_sparse].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct PartialArrayBuilder<T, B> {
    builder: B,
    values: Vec<T>,
    len: usize,
    min_next_pos: usize,
}

/// Creates a new builder for a dense partial array of the given length.
///
/// A dense partial array stores a bit vector of the given length to mark
/// which positions contain values, and use ranking on this bit vector to
/// map positions to indices in a contiguous value array.
///
/// If your set of values is really sparse, consider using a
/// [sparse partial array](new_sparse).
///
/// # Examples
///
/// ```rust
/// use sux::array::partial_array;
///
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
pub fn new_dense<T>(len: usize) -> PartialArrayBuilder<T, BitVec<Box<[usize]>>> {
    PartialArrayBuilder {
        builder: BitVec::new(len).into(),
        values: vec![],
        len,
        min_next_pos: 0,
    }
}

impl<T> PartialArrayBuilder<T, BitVec<Box<[usize]>>> {
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
    pub fn build(self) -> PartialArray<T, Rank9<BitVec<Box<[usize]>>>> {
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
/// [Elias-Fano](crate::dict::EliasFano) structure.
///
/// If your set of values is really dense, consider using a [dense partial
/// array](new_dense).
///
/// # Examples
///
/// ```rust
/// use sux::array::partial_array;
///
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
pub fn new_sparse<T>(len: usize, num_values: usize) -> PartialArrayBuilder<T, EliasFanoBuilder> {
    PartialArrayBuilder {
        builder: EliasFanoBuilder::new(num_values, len),
        values: vec![],
        len,
        min_next_pos: 0,
    }
}

impl<T> PartialArrayBuilder<T, EliasFanoBuilder> {
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
        // SAFETY: conditions have been just checked.
        unsafe { self.builder.push_unchecked(position) };
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
impl<T> Extend<(usize, T)> for PartialArrayBuilder<T, BitVec<Box<[usize]>>> {
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
impl<T> Extend<(usize, T)> for PartialArrayBuilder<T, EliasFanoBuilder> {
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
/// some positions contain values. There is a [dense](new_dense)
/// and a [sparse](new_sparse) implementation with different
/// space/time trade-offs.
///
/// For convenience, this structure implements
/// [`SliceByValue`](crate::traits::slices::SliceByValue).
///
/// See [`PartialArrayBuilder`] for details on how to create a partial array.
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartialArray<T, P, V: AsRef<[T]> = Box<[T]>> {
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

    /// Returns true if the array length is 0.
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
    /// use sux::array::partial_array;
    /// let mut builder = partial_array::new_dense(10);
    /// builder.set(5, 42);
    ///
    /// let array = builder.build();
    /// assert_eq!(array.get(5), Some(&42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<&T> {
        panic_if_out_of_bounds!(position, self.len());
        // Check if there's a value at this position
        // SAFETY: position < len()
        if !unsafe { self.index.get_unchecked(position) } {
            return None;
        }

        // Use ranking to find the index in the values array
        // SAFETY: position < len()
        let value_index = unsafe { self.index.rank_unchecked(position) };

        // SAFETY: necessarily value_index < num_values().
        Some(unsafe { self.values.as_ref().get_unchecked(value_index) })
    }
}

impl<T, D: AsRef<[usize]>, V: AsRef<[T]>> PartialArray<T, SparseIndex<D>, V> {
    /// Returns the total length of the array.
    ///
    /// This is the length that was specified when creating the builder,
    /// not the number of values actually stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index.ef.upper_bound()
    }

    /// Returns true if the array len is 0.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index.ef.len() == 0
    }

    /// Gets a reference to the value at the given position.
    ///
    /// Returns `Some(&value)` if a value is present at the position,
    /// or `None` if no value was stored there.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sux::array::partial_array;
    /// let mut builder = partial_array::new_sparse(10, 1);
    /// builder.set(5, 42);
    ///
    /// let array = builder.build();
    /// assert_eq!(array.get(5), Some(&42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<&T> {
        if position >= self.index.first_invalid_pos {
            panic_if_out_of_bounds!(position, self.len());
            return None;
        }
        // Check if there's a value at this position
        // SAFETY: position <= last set position
        let (index, pos) = unsafe { self.index.ef.succ_unchecked::<false>(position) };

        if pos != position {
            None
        } else {
            // SAFETY: necessarily value_index < num values.
            Some(unsafe { self.values.as_ref().get_unchecked(index) })
        }
    }
}

impl<T: Clone, P, V: AsRef<[T]>> SliceByValue for PartialArray<T, P, V> {
    type Value = T;

    fn len(&self) -> usize {
        self.values.as_ref().len()
    }

    unsafe fn get_value_unchecked(&self, index: usize) -> Self::Value {
        // SAFETY: the caller guarantees that index < len()
        unsafe { self.values.as_ref().get_unchecked(index) }.clone()
    }
}
