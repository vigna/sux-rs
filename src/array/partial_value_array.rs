/*
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Immutable partial array implementations for cheaply-clonable values
//!
//! This supports storing values in [`BitFieldVec`] to save memory.

use std::marker::PhantomData;

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::bits::BitFieldVec;
use crate::bits::BitVec;
use crate::dict::EliasFanoBuilder;
use crate::dict::elias_fano::EliasFano;
use crate::rank_sel::{Rank9, SelectZeroAdaptConst};
use crate::traits::Word;
use crate::traits::{BitVecOps, BitVecOpsMut};
use crate::traits::{RankUnchecked, SuccUnchecked};

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

type DenseIndex = Rank9<BitVec<Box<[usize]>>>;

/// Builder for creating an immutable partial array.
///
/// The builder allows you to specify the array length and then add
/// (position, value) pairs. Positions must be added in strictly
/// increasing order.
///
/// To get a builder you can use either [new_dense] or [new_sparse].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct PartialValueArrayBuilder<T, B> {
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
/// use sux::array::partial_value_array;
///
/// let mut builder = partial_value_array::new_dense(10);
/// builder.set(1, 123u32);
/// builder.set(2, 45678);
/// builder.set(7, 90);
///
/// let array = builder.build_bitfieldvec();
/// assert_eq!(array.get(1), Some(123));
/// assert_eq!(array.get(2), Some(45678));
/// assert_eq!(array.get(3), None);
/// assert_eq!(array.get(7), Some(90));
/// ```
pub fn new_dense<T>(len: usize) -> PartialValueArrayBuilder<T, BitVec<Box<[usize]>>> {
    PartialValueArrayBuilder {
        builder: BitVec::new(len).into(),
        values: vec![],
        len,
        min_next_pos: 0,
    }
}

fn build_bitfieldvec<T: Word>(values: Vec<T>) -> BitFieldVec<T> {
    let bit_width = values
        .iter()
        .map(|value| value.len())
        .max()
        .unwrap_or(1)
        .try_into()
        .expect("bit_width overflowed usize");
    let mut bfv = BitFieldVec::with_capacity(bit_width, values.len());
    for value in values {
        bfv.push(value);
    }
    bfv
}

impl<T: Clone> PartialValueArrayBuilder<T, BitVec<Box<[usize]>>> {
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
        if position >= self.len {
            panic!(
                "Position {} is out of bounds for array of len {}",
                position, self.len
            );
        }
        // SAFETY: position < len
        unsafe {
            self.builder.set_unchecked(position, true);
        }
        self.values.push(value);
        self.min_next_pos = position + 1;
    }

    /// Builds the immutable dense partial array using [`BitFieldVec`] as backend.
    pub fn build_bitfieldvec(self) -> PartialValueArray<T, DenseIndex, BitFieldVec<T>>
    where
        T: Word,
    {
        let (bit_vec, values) = (self.builder, self.values);
        let rank9 = Rank9::new(bit_vec);
        let values = build_bitfieldvec(values);

        PartialValueArray {
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
/// use sux::array::partial_value_array;
///
/// let mut builder = partial_value_array::new_sparse(10, 3);
/// builder.set(1, 123u32);
/// builder.set(2, 45678);
/// builder.set(7, 90);
///
/// let array = builder.build_bitfieldvec();
/// assert_eq!(array.get(1), Some(123));
/// assert_eq!(array.get(2), Some(45678));
/// assert_eq!(array.get(3), None);
/// assert_eq!(array.get(7), Some(90));
/// ```
///
/// Note that you must specify the number of values in advance.
pub fn new_sparse<T>(
    len: usize,
    num_values: usize,
) -> PartialValueArrayBuilder<T, EliasFanoBuilder> {
    PartialValueArrayBuilder {
        builder: EliasFanoBuilder::new(num_values, len),
        values: vec![],
        len,
        min_next_pos: 0,
    }
}

impl<T: Clone> PartialValueArrayBuilder<T, EliasFanoBuilder> {
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
        if position >= self.len {
            panic!(
                "Position {} is out of bounds for array of len {}",
                position, self.len
            );
        }
        // SAFETY: conditions have been just checked.
        unsafe { self.builder.push_unchecked(position) };
        self.values.push(value);
        self.min_next_pos = position + 1;
    }

    /// Builds the immutable sparse partial array using [`BitFieldVec`] as backend.
    pub fn build_bitfieldvec(
        self,
    ) -> PartialValueArray<T, SparseIndex<Box<[usize]>>, BitFieldVec<T>>
    where
        T: Word,
    {
        let (builder, values) = (self.builder, self.values);
        let ef_dict = builder.build_with_dict();
        let values = build_bitfieldvec(values);

        PartialValueArray {
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
impl<T: Clone> Extend<(usize, T)> for PartialValueArrayBuilder<T, BitVec<Box<[usize]>>> {
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
impl<T: Clone> Extend<(usize, T)> for PartialValueArrayBuilder<T, EliasFanoBuilder> {
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
/// See [`PartialValueArrayBuilder`] for details on how to create a partial array.
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartialValueArray<T: Clone, P, V: SliceByValue<Value = T>> {
    index: P,
    values: V,
    _marker: PhantomData<T>,
}

impl<T: Clone, P, V: SliceByValue<Value = T>> PartialValueArray<T, P, V> {
    /// Returns the number of values stored in the array.
    #[inline(always)]
    pub fn num_values(&self) -> usize {
        self.values.len()
    }
}

impl<T: Clone, V: SliceByValue<Value = T>> PartialValueArray<T, DenseIndex, V> {
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
    /// use sux::array::partial_value_array;
    /// let mut builder = partial_value_array::new_dense(10);
    /// builder.set(5, 42u32);
    ///
    /// let array = builder.build_bitfieldvec();
    /// assert_eq!(array.get(5), Some(42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<T> {
        if position >= self.len() {
            panic!(
                "Position {} is out of bounds for array of len {}",
                position,
                self.len()
            );
        }
        // Check if there's a value at this position
        // SAFETY: position < len()
        if !unsafe { self.index.get_unchecked(position) } {
            return None;
        }

        // Use ranking to find the index in the values array
        // SAFETY: position < len()
        let value_index = unsafe { self.index.rank_unchecked(position) };

        // SAFETY: necessarily value_index < num_values().
        Some(unsafe { self.values.get_value_unchecked(value_index) })
    }
}

impl<T: Clone, D: AsRef<[usize]>, V: SliceByValue<Value = T>>
    PartialValueArray<T, SparseIndex<D>, V>
{
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
    /// use sux::array::partial_value_array;
    /// let mut builder = partial_value_array::new_sparse(10, 1);
    /// builder.set(5, 42u32);
    ///
    /// let array = builder.build_bitfieldvec();
    /// assert_eq!(array.get(5), Some(42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<T> {
        if position >= self.index.first_invalid_pos {
            if position >= self.len() {
                panic!(
                    "Position {} is out of bounds for array of len {}",
                    position,
                    self.len()
                );
            }
            return None;
        }
        // Check if there's a value at this position
        // SAFETY: position <= last set position
        let (index, pos) = unsafe { self.index.ef.succ_unchecked::<false>(position) };

        if pos != position {
            None
        } else {
            // SAFETY: necessarily value_index < num values.
            Some(unsafe { self.values.get_value_unchecked(index) })
        }
    }
}
