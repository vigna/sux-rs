/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/**

A dictionary of values indexed by a `usize`.

The input and output values may be different, to make it easier to implement
compressed structures (see, e.g., [rear-coded lists](crate::dict::rear_coded_list::RearCodedList)).

*/
pub trait IndexedDict {
    type OutputValue: PartialEq<Self::InputValue> + PartialEq;
    type InputValue: PartialEq<Self::OutputValue> + PartialEq + ?Sized;

    /// The type of the iterator returned by [`iter`](`IndexedDict::iter`).
    /// and [`iter_from`](`IndexedDict::iter_from`).
    type Iterator<'a>: ExactSizeIterator<Item = Self::OutputValue> + 'a
    where
        Self: 'a;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`IndexedDict::len`)).
    fn get(&self, index: usize) -> Self::OutputValue {
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
    unsafe fn get_unchecked(&self, index: usize) -> Self::OutputValue;

    /// Return true if the dictionary contains the given value.
    ///
    /// The default implementations just checks iteratively
    /// if the value is equal to any of the values in the dictionary.
    fn contains(&self, value: &Self::InputValue) -> bool {
        for i in 0..self.len() {
            if self.get(i) == *value {
                return true;
            }
        }
        false
    }

    /// Return the length (number of items) of the dictionary.
    fn len(&self) -> usize;

    /// Return true of [`len`](`IndexedDict::len`) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return an iterator on the whole dictionary.
    fn iter(&self) -> Self::Iterator<'_>;

    /// Return an iterator on the dictionary starting from the specified index.
    fn iter_from(&self, from: usize) -> Self::Iterator<'_>;
}

/// Successor computation for dictionaries whose values are monotonically increasing.
pub trait Succ: IndexedDict
where
    Self::OutputValue: PartialOrd<Self::InputValue> + PartialOrd,
    Self::InputValue: PartialOrd<Self::OutputValue> + PartialOrd,
{
    /// Return the index of the successor and the successor
    /// of the given value, or `None` if there is no successor.
    /// The successor is the first value in the dictionary
    /// that is greater than or equal to the given value.
    fn succ(&self, value: &Self::InputValue) -> Option<(usize, Self::OutputValue)> {
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
    unsafe fn succ_unchecked(&self, value: &Self::InputValue) -> (usize, Self::OutputValue);
}

/// Predecessor computation for dictionaries whoses value are monotonically increasing.
pub trait Pred: IndexedDict
where
    Self::OutputValue: PartialOrd<Self::InputValue> + PartialOrd,
    Self::InputValue: PartialOrd<Self::OutputValue> + PartialOrd,
{
    /// Return the index of the predecessor and the predecessor
    /// of the given value, or `None` if there is no predecessor.
    /// The predecessor is the last value in the dictionary
    /// that is less than the given value.
    fn pred(&self, value: &Self::InputValue) -> Option<(usize, Self::OutputValue)> {
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
    unsafe fn pred_unchecked(&self, value: &Self::InputValue) -> (usize, Self::OutputValue);
}
