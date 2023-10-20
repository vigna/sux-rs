/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Traits for indexed dictionaries, possibly with support for additional
operations such as predecessor and successor.

*/

/**

A dictionary of values indexed by a `usize`.

The input and output values may be different, to make it easier to implement
compressed structures (see, e.g., [rear-coded lists](crate::dict::rear_coded_list::RearCodedList)).

It is suggested that any implementation of this trait also implements
[`IntoIterator`] with `Item = Self::Output` on a reference. This property can be tested
on a type `D` with the clause `where for<'a> &'a D: IntoIterator<Item = Self::Output>`.
Many implementations offer also a method `into_iter_from` that returns an iterator
starting at a given position in the dictionary.

*/
pub trait IndexedDict {
    type Input: PartialEq<Self::Output> + PartialEq + ?Sized;
    type Output: PartialEq<Self::Input> + PartialEq;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`IndexedDict::len`)).
    fn get(&self, index: usize) -> Self::Output {
        if index >= self.len() {
            panic!("Index out of bounds: {} >= {}", index, self.len())
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }

    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`IndexedDict::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output;

    /// Return true if the dictionary contains the given value.
    ///
    /// The default implementations just checks iteratively
    /// if the value is equal to any of the values in the dictionary.
    fn contains(&self, value: &Self::Input) -> bool {
        for i in 0..self.len() {
            if self.get(i) == *value {
                return true;
            }
        }
        false
    }

    /// Return the length (number of items) of the dictionary.
    fn len(&self) -> usize;

    /// Return true if [`len`](`IndexedDict::len`) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Successor computation for dictionaries whose values are monotonically increasing.
pub trait Succ: IndexedDict
where
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
{
    /// Return the index of the successor and the successor
    /// of the given value, or `None` if there is no successor.
    /// The successor is the first value in the dictionary
    /// that is greater than or equal to the given value.
    fn succ(&self, value: &Self::Input) -> Option<(usize, Self::Output)> {
        if self.is_empty() || value > &self.get(self.len() - 1) {
            None
        } else {
            Some(unsafe { self.succ_unchecked(value) })
        }
    }

    /// Return the index of the successor and the successor
    /// of the given value, or `None` if there is no successor.
    /// The successor is the first value in the dictionary
    /// that is greater than or equal to the given value.
    ///
    /// # Safety
    /// The successors must exist.
    unsafe fn succ_unchecked(&self, value: &Self::Input) -> (usize, Self::Output);
}

/// Predecessor computation for dictionaries whoses value are monotonically increasing.
pub trait Pred: IndexedDict
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Return the index of the predecessor and the predecessor
    /// of the given value, or `None` if there is no predecessor.
    /// The predecessor is the last value in the dictionary
    /// that is less than the given value.
    fn pred(&self, value: &Self::Input) -> Option<(usize, Self::Output)> {
        if self.is_empty() || value <= &self.get(0) {
            None
        } else {
            Some(unsafe { self.pred_unchecked(value) })
        }
    }
    /// Return the index of the predecessor and the predecessor
    /// of the given value, or `None` if there is no predecessor.
    /// The predecessor is the last value in the dictionary
    /// that is less than the given value.
    ///
    /// # Safety
    /// The predecessor must exist.
    unsafe fn pred_unchecked(&self, value: &Self::Input) -> (usize, Self::Output);
}
