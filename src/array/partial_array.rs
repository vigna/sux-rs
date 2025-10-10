/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Immutable partial array implementation using bit vectors and ranking.
//!
//! A partial array is an array where only some positions contain values.
//! This implementation uses a bit vector to mark which positions have values
//! and stores the actual values in a contiguous array. Queries use ranking
//! on the bit vector to efficiently map positions to value indices.

use crate::bits::{BitVec, BitVecOpsMut};
use crate::rank_sel::Rank9;
use crate::traits::Rank;
use mem_dbg::*;

/// Builder for creating an immutable partial array.
///
/// The builder allows you to specify the array size and then add
/// (position, value) pairs. Positions must be added in strictly
/// increasing order.
///
/// # Examples
///
/// ```rust
/// use sux::array::PartialArrayBuilder;
///
/// let mut builder = PartialArrayBuilder::new(10);
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
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct PartialArrayBuilder<T> {
    bit_vec: BitVec<Box<[usize]>>,
    values: Vec<T>,
    size: usize,
    min_next_pos: usize,
}

impl<T> PartialArrayBuilder<T> {
    /// Creates a new builder for a partial array of the given size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::PartialArrayBuilder;
    /// let builder = PartialArrayBuilder::<i32>::new(100);
    /// ```
    pub fn new(size: usize) -> Self {
        Self {
            bit_vec: BitVec::new(size).into(),
            values: vec![],
            size,
            min_next_pos: 0,
        }
    }

    /// Sets a value at the given position.
    ///
    /// The provided position must be greater than the last position
    /// set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::PartialArrayBuilder;
    /// let mut builder = PartialArrayBuilder::new(10);
    /// builder.set(2, "test");
    /// builder.set(5, "test");
    /// ```
    pub fn set(&mut self, position: usize, value: T) {
        if position < self.min_next_pos {
            panic!(
                "Positions must be set in increasing order: got {} after {}",
                position,
                self.min_next_pos - 1
            );
        }
        if position >= self.size {
            panic!(
                "Position {} is out of bounds for array of size {}",
                position, self.size
            );
        }
        // SAFETY: position < size
        unsafe {
            self.bit_vec.set_unchecked(position, true);
        }
        self.values.push(value);
        self.min_next_pos = position + 1;
    }

    /// Builds the immutable partial array.
    ///
    /// This method consumes the builder and creates the final partial array
    /// with an optimized representation using bit vectors and ranking.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::PartialArrayBuilder;
    /// let mut builder = PartialArrayBuilder::new(10);
    /// builder.extend(std::iter::zip(0..5, ["zero", "one", "two", "three", "four"]));
    /// builder.set(7, "seven");
    ///
    /// let array = builder.build();
    /// assert_eq!(array.len(), 10);
    /// ```
    pub fn build(self) -> PartialArray<T> {
        let (bit_vec, values) = (self.bit_vec, self.values);
        let rank9 = Rank9::new(bit_vec);
        let values = values.into_boxed_slice();

        PartialArray { rank9, values }
    }

    /// Returns the size of the array being built.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the array size is 0.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of values currently stored in the builder.
    pub fn num_values(&self) -> usize {
        self.values.len()
    }
}

/// Extends the builder with an iterator of (position, value) pairs.
///
/// Position must be in strictly increasing order. The first returned
/// position must be greater than the last position set.
impl<T> Extend<(usize, T)> for PartialArrayBuilder<T> {
    fn extend<I: IntoIterator<Item = (usize, T)>>(&mut self, iter: I) {
        for (pos, val) in iter {
            self.set(pos, val);
        }
    }
}

/// An immutable partial array that supports efficient queries.
///
/// This structure uses a bit vector to mark which positions contain values
/// and a ranking structure ([`Rank9`]) to efficiently map positions to
/// indices in a contiguous value array.
///
/// # Examples
///
/// ```rust
/// # use sux::array::PartialArrayBuilder;
/// let mut builder = PartialArrayBuilder::new(8);
/// builder.extend(std::iter::zip(0..5, ["zero", "one", "two", "three", "four"]));
/// builder.set(7, "seven");
/// let array = builder.build();
///
/// assert_eq!(array.get(1), Some(&"one"));
/// assert_eq!(array.get(4), Some(&"four"));
/// assert_eq!(array.get(5), None);
/// assert_eq!(array.get(6), None);
/// assert_eq!(array.get(7), Some(&"seven"));
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartialArray<T> {
    rank9: Rank9<BitVec<Box<[usize]>>>,
    values: Box<[T]>,
}

impl<T> PartialArray<T> {
    /// Returns the total size of the array.
    ///
    /// This is the size that was specified when creating the builder,
    /// not the number of values actually stored.
    pub fn len(&self) -> usize {
        self.rank9.len()
    }

    /// Returns true if the array size is 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of values stored in the array.
    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    /// Gets a reference to the value at the given position.
    ///
    /// Returns `Some(&value)` if a value is present at the position,
    /// or `None` if no value was stored there.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::PartialArrayBuilder;
    /// let mut builder = PartialArrayBuilder::new(10);
    /// builder.set(5, 42);
    ///
    /// let array = builder.build();
    /// assert_eq!(array.get(5), Some(&42));
    /// assert_eq!(array.get(6), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<&T> {
        // Check if there's a value at this position
        if !self.rank9[position] {
            return None;
        }

        // Use ranking to find the index in the values array
        let value_index = self.rank9.rank(position);

        // SAFETY: necessarily value_index < num_values().
        Some(unsafe { self.values.get_unchecked(value_index) })
    }

    /// Returns an iterator over all (position, value) pairs in the array.
    ///
    /// The pairs are yielded in increasing order of position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::array::PartialArrayBuilder;
    /// let mut builder = PartialArrayBuilder::new(10);
    /// builder.set(1, "one");
    /// builder.set(2, "two");
    /// builder.set(5, "five");
    ///
    /// let array = builder.build();
    /// let pairs: Vec<_> = array.iter().collect();
    /// assert_eq!(pairs, vec![(1, &"one"), (2, &"two"), (5, &"five")]);
    /// ```
    pub fn iter(&self) -> PartialArrayIter<'_, T> {
        PartialArrayIter {
            array: self,
            position: 0,
            value_index: 0,
        }
    }

    /// Returns an iterator over all values in the array.
    ///
    /// Values are yielded in increasing order of their positions.
    pub fn values(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    /// Returns an iterator over all positions that contain values.
    ///
    /// Positions are yielded in increasing order.
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(move |&pos| self.rank9[pos])
    }
}

/// Iterator over (position, value) pairs in a partial array.
#[derive(Debug)]
pub struct PartialArrayIter<'a, T> {
    array: &'a PartialArray<T>,
    position: usize,
    value_index: usize,
}

impl<'a, T> Iterator for PartialArrayIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        // Find the next position with a value
        // TODO: use BitVec traits when available
        while self.position < self.array.len() {
            if self.array.rank9[self.position] {
                let position = self.position;
                let value = &self.array.values[self.value_index];

                self.position += 1;
                self.value_index += 1;

                return Some((position, value));
            }
            self.position += 1;
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.num_values() - self.value_index;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for PartialArrayIter<'a, T> {}

/// Returns an iterator over (position, value) pairs in a partial array.
impl<'a, T> IntoIterator for &'a PartialArray<T> {
    type Item = (usize, &'a T);
    type IntoIter = PartialArrayIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
