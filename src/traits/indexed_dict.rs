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
//! We suggest that every implementation of [`IndexedSeq`] also implements
//! [`IntoIterator`]/[`IntoIteratoFrom`](crate::traits::iter::IntoIteratorFrom)
//! with `Item = Self::Output` on a reference. This property can be tested on a
//! type `T` with the clause `where for<'a> &'a T: IntoIteratorFrom<Item =
//! Self::Output>` (or `where for<'a> &'a T: IntoIterator<Item = Self::Output>`,
//! if you don't need to select the starting position). Many implementations
//! offer also equivalent `iter`/`iter_from` convenience methods.

use ambassador::delegatable_trait;
use impl_tools::autoimpl;
use std::borrow::Borrow;

/// The types of the dictionary.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait Types {
    type Input: PartialEq<Self::Output> + PartialEq + ?Sized;
    type Output: PartialEq<Self::Input> + PartialEq;
}

/// Access by position to the dictionary.
///
/// # Notes
///
/// This trait does not include an `iter` iteration method with a default
/// implementation, even it would be convenient, because it would cause
/// significant problems with structures that have their own implementation of
/// the method, and in which the implementation is dependent on additional trait
/// bounds (see, e.g., [`EliasFano`](crate::dict::elias_fano::EliasFano)).
///
/// More precisely, the inherent implementation could not be used to override
/// the default implementation, due to the additional trait bounds, and thus the
/// selection of the inherent vs. default trait implementation would depend on
/// the type of the variable, which might lead to efficiency bugs difficult to
/// diagnose. Having a different name for the trait and inherent iteration
/// method would make the call predictable, but it would be less ergonomic.
///
/// The (pretty standard) strategy outlined in the [module
/// documentation](crate::traits::indexed_dict) is more flexible, as it allows
/// to write methods that use the inherent implementation only if available.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait IndexedSeq: Types {
    /// Returns the value at the specified index.
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

    /// Returns the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`IndexedSeq::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output;

    /// Returns the length (number of items) of the dictionary.
    fn len(&self) -> usize;

    /// Returns true if [`len`](`IndexedSeq::len`) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Access by value to the dictionary.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait IndexedDict: Types {
    /// Returns the index of the given value if the dictionary contains it and
    /// `None` otherwise.
    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize>;

    /// Returns true if the dictionary contains the given value.
    ///
    /// The default implementations just calls
    /// [`index_of`](`IndexedDict::index_of`).
    fn contains(&self, value: impl Borrow<Self::Input>) -> bool {
        self.index_of(value).is_some()
    }
}

/// Unchecked successor computation for dictionaries whose values are monotonically increasing.
#[delegatable_trait]
pub trait SuccUnchecked: Types
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Returns the index of the successor and the successor of the given value.
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

impl<I, O, T: SuccUnchecked<Input = I, Output = O> + ?Sized> SuccUnchecked for &T
where
    I: PartialOrd<O> + PartialOrd,
    O: PartialOrd<I> + PartialOrd,
{
    #[inline(always)]
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output) {
        (*self).succ_unchecked::<STRICT>(value)
    }
}

/// Successor computation for dictionaries whose values are monotonically increasing.
#[delegatable_trait]
pub trait Succ: SuccUnchecked + IndexedSeq
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Returns the index of the successor and the successor
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

    /// Returns the index of the strict successor and the strict successor
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

impl<I, O, T: Succ<Input = I, Output = O> + ?Sized> Succ for &T
where
    I: PartialOrd<O> + PartialOrd,
    O: PartialOrd<I> + PartialOrd,
{
    #[inline(always)]
    fn succ(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        (*self).succ(value)
    }

    #[inline(always)]
    fn succ_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        (*self).succ_strict(value)
    }
}

/// Unchecked predecessor computation for dictionaries whose values are monotonically increasing.
#[delegatable_trait]
pub trait PredUnchecked: Types
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Returns the index of the predecessor and the predecessor of the given
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

impl<I, O, T: PredUnchecked<Input = I, Output = O> + ?Sized> PredUnchecked for &T
where
    I: PartialOrd<O> + PartialOrd,
    O: PartialOrd<I> + PartialOrd,
{
    #[inline(always)]
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output) {
        (*self).pred_unchecked::<STRICT>(value)
    }
}

/// Predecessor computation for dictionaries whose values are monotonically increasing.
#[delegatable_trait]
pub trait Pred: PredUnchecked + IndexedSeq
where
    Self::Input: PartialOrd<Self::Output> + PartialOrd,
    Self::Output: PartialOrd<Self::Input> + PartialOrd,
{
    /// Returns the index of the predecessor and the predecessor
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

    /// Returns the index of the strict predecessor and the strict predecessor
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

impl<I, O, T: Pred<Input = I, Output = O> + ?Sized> Pred for &T
where
    I: PartialOrd<O> + PartialOrd,
    O: PartialOrd<I> + PartialOrd,
{
    #[inline(always)]
    fn pred(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        (*self).pred(value)
    }

    #[inline(always)]
    fn pred_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output)> {
        (*self).pred_strict(value)
    }
}
