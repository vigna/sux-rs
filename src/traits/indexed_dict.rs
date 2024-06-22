/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Traits for indexed dictionaries.
//!
//! An indexed dictionary is a dictionary of values indexed by a `usize`. When
//! the values are monotonically increasing, such a dictionary might provide
//! additional operations such as [predecessor](Pred) and [successor](Succ).
//!
//! There are seven traits:
//! - [`Types`] defines the type of the values in the dictionary. The type of
//!   input and output values may be different: for example, in a dictionary of
//!   strings (see, e.g.,
//!   [`RearCodedList`](crate::dict::rear_coded_list::RearCodedList)), one
//!   usually accepts as inputs references to [`str`] but returns owned
//!   [`String`]s.
//! - [`IndexedSeq`] provide positional access to the values in the dictionary.
//! - [`IndexedDict`] provides access by value, that is, given a value, its
//!   position in the dictionary.
//! - [`SuccUnchecked`]/[`Succ`] provide the successor of a value in a sorted
//!   dictionary.
//! - [`PredUnchecked`]/[`Pred`] provide the predecessor of a value in a sorted
//!   dictionary.
//!
//! [`IndexedSeq`], [`IndexedDict`], [`SuccUnchecked`], and [`PredUnchecked`]
//! are independent. A structure may implement any combination of them, provided
//! it implements [`Types`].
//!
//! All method accepting values have arguments of type `impl
//! Borrow<Self::Input>`. This makes it possible to pass a value both by
//! reference and by value, which is particularly convenient in the case of
//! primitive types (see, e.g.,
//! [`EliasFano`](crate::dict::elias_fano::EliasFano)). Note that this goal
//! cannot be achieved using [`AsRef`] because there is no blanket
//! implementation of `AsRef<T>` for `T`, as it happens in the case of
//! [`Borrow`].
//!
//!
//! It is suggested that any implementation of this trait also implements
//! [`IntoIterator`] with `Item = Self::Output` on a reference. This property
//! can be tested on a type `D` with the clause `where for<'a> &'a D:
//! IntoIterator<Item = Self::Output>`. Many implementations offer also a
//! convenience method `iter` that is equivalent to [`IntoIterator::into_iter`],
//! and a method `iter_from` that returns an iterator starting at a given
//! position in the dictionary.

use impl_tools::autoimpl;
use std::borrow::Borrow;

/// The types of the dictionary.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait Types {
    type Input: PartialEq<Self::Output> + PartialEq + ?Sized;
    type Output: PartialEq<Self::Input> + PartialEq;
}

/// Positional access to the dictionary.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait IndexedSeq: Types {
    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`IndexedSeq::len`)).
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
    /// `index` must be in [0..[len](`IndexedSeq::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output;

    /// Return the length (number of items) of the dictionary.
    fn len(&self) -> usize;

    /// Return true if [`len`](`IndexedSeq::len`) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Access by value to the dictionary.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait IndexedDict: Types {
    /// Return the index of the given value if the dictionary contains it and
    /// `None` otherwise.
    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize>;

    /// Return true if the dictionary contains the given value.
    ///
    /// The default implementations just calls
    /// [`index_of`](`IndexedDict::index_of`).
    fn contains(&self, value: impl Borrow<Self::Input>) -> bool {
        self.index_of(value).is_some()
    }
}

/// Unchecked successor computation for dictionaries whose values are monotonically increasing.
pub trait SuccUnchecked: Types
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Return the index of the successor and the successor of the given value.
    ///
    /// The successor is the least value in the dictionary that is greater than
    /// or equal to the given value, if `STRICT` is `false`, or the least value
    /// in the dictionary that is greater than the given value, if `STRICT` is
    /// `true`.
    ///
    /// If there are repeated values, the index of the one returned depends on
    /// the implementation.
    ///
    /// # Safety
    ///
    /// The successors must exist.
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output);
}

/// Successor computation for dictionaries whose values are monotonically increasing.
pub trait Succ: SuccUnchecked + IndexedSeq
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Return the index of the successor and the successor
    /// of the given value, or `None` if there is no successor.
    ///
    /// The successor is the least value in the dictionary
    /// that is greater than or equal to the given value.
    ///
    /// If there are repeated values, the index of the one returned
    /// depends on the implementation.
    fn succ(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        if self.is_empty() || *value.borrow() > self.get(self.len() - 1) {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<false>(value) })
        }
    }

    /// Return the index of the strict successor and the strict successor
    /// of the given value, or `None` if there is no strict successor.
    ///
    /// The strict successor is the least value in the dictionary
    /// that is greater than the given value.
    ///
    /// If there are repeated values, the index of the one returned
    /// depends on the implementation.
    fn succ_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        if self.is_empty() || *value.borrow() >= self.get(self.len() - 1) {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<true>(value) })
        }
    }
}

/// Unchecked predecessor computation for dictionaries whose values are monotonically increasing.
pub trait PredUnchecked: Types
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Return the index of the predecessor and the predecessor of the given
    /// value, or `None` if there is no predecessor.
    ///
    /// The predecessor is the greatest value in the dictionary that is less
    /// than or equal to the given value, if `STRICT` is `false`, or the
    /// greatest value in the dictionary that is less than the given value, if
    /// `STRICT` is `true`.
    ///
    /// If there are repeated values, the index of the one returned depends on
    /// the implementation.
    ///
    /// # Safety
    ///
    /// The predecessor must exist.
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output);
}

/// Predecessor computation for dictionaries whose values are monotonically increasing.
pub trait Pred: PredUnchecked + IndexedSeq
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Return the index of the predecessor and the predecessor
    /// of the given value, or `None` if there is no predecessor.
    ///
    /// The predecessor is the greatest value in the dictionary
    /// that is less than or equal to the given value.
    ///
    /// If there are repeated values, the index of the one returned
    /// depends on the implementation.
    fn pred(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        if self.is_empty() || *value.borrow() < self.get(0) {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<false>(value) })
        }
    }

    /// Return the index of the strict predecessor and the strict predecessor
    /// of the given value, or `None` if there is no strict predecessor.
    ///
    /// The strict predecessor is the greatest value in the dictionary
    /// that is less than the given value.
    ///
    /// If there are repeated values, the index of the one returned
    /// depends on the implementation.
    fn pred_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        if self.is_empty() || *value.borrow() <= self.get(0) {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<true>(value) })
        }
    }
}
