/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/// A dictionary of monotonically increasing values that can be indexed by a `usize`.
pub trait IndexedDict {
    /// The type of the values stored in the dictionary.
    type Value;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`IndexedDict::len`)).

    fn get(&self, index: usize) -> Self::Value {
        if index >= self.len() {
            panic!("Index out of bounds: {} >= {}", index, self.len())
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }

    /// Return the value at the specified index.
    ///
    /// # Safety
    ///
    /// `index` must be in [0..[len](`IndexedDict::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> Self::Value;

    /// Return the length (number of items) of the dictionary.
    fn len(&self) -> usize;

    /// Return true of [`len`](`IndexedDict::len`) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Successor: IndexedDict {
    /// Return the index of the successor and the successor
    /// of the given value, or `None` if there is no successor.
    /// The successor is the first value in the dictionary
    /// that is greater than or equal to the given value.
    fn successor(&self, value: Self::Value) -> Option<(usize, Self::Value)>;
}

pub trait Predecessor: IndexedDict {
    /// Return the index of the predecessor and the predecessor
    /// of the given value, or `None` if there is no predecessor.
    /// The predecessor is the last value in the dictionary
    /// that is less than the given value.
    fn predecessor(&self, value: Self::Value) -> Option<Self::Value>;
}
