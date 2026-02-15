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
//! - [`IndexedSeq`] provides positional access to the values in the dictionary.
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
//! All methods accepting values have arguments of type `impl
//! Borrow<Self::Input>`. This makes it possible to pass a value both by
//! reference and by value, which is particularly convenient in the case of
//! primitive types (see, e.g.,
//! [`EliasFano`](crate::dict::elias_fano::EliasFano)).
//!
//! We suggest that every implementation of [`IndexedSeq`] also implements
//! [`IntoIterator`]/[`IntoIteratorFrom`](super::iter::IntoIteratorFrom) with `Item = Self::Output` on a
//! reference. This property can be tested on a type `T` with the clause `where
//! for<'a> &'a T: IntoIteratorFrom<Item = Self::Output>` (or `where for<'a> &'a
//! T: IntoIterator<Item = Self::Output>`, if you don't need to select the
//! starting position). Many implementations also offer equivalent
//! `iter`/`iter_from` convenience methods.

use ambassador::delegatable_trait;
use impl_tools::autoimpl;
use std::borrow::Borrow;

use super::iter::BidiIterator;
use crate::{debug_assert_bounds, panic_if_out_of_bounds};

/// The types of the dictionary.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait Types {
    type Input: for<'a> PartialEq<Self::Output<'a>> + PartialEq + ?Sized;
    type Output<'a>: PartialEq<Self::Input> + PartialEq;
}

/// Access by position to the dictionary.
///
/// # Notes
///
/// This trait does not include an `iter` iteration method with a default
/// implementation, although it would be convenient, because it would cause
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
/// documentation](crate::traits::indexed_dict) is more flexible, as it makes it
/// possible to write methods that use the inherent implementation only if
/// available.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait IndexedSeq: Types {
    /// Returns the value at the specified index.
    ///
    /// # Panics
    ///
    /// May panic if the index is not in [0..[len](`IndexedSeq::len`)).
    fn get(&self, index: usize) -> Self::Output<'_> {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index) }
    }

    /// Returns the value at the specified index.
    ///
    /// # Safety
    ///
    /// `index` must be in [0..[len](`IndexedSeq::len`)). No bounds checking
    /// is performed.
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_>;

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
    /// The default implementation just calls
    /// [`index_of`](`IndexedDict::index_of`).
    fn contains(&self, value: impl Borrow<Self::Input>) -> bool {
        self.index_of(value).is_some()
    }
}

/// Unchecked successor computation for dictionaries whose values are monotonically increasing.
pub trait SuccUnchecked: Types
where
    Self::Input: for<'a> PartialOrd<Self::Output<'a>> + PartialOrd,
    for<'a> Self::Output<'a>: PartialOrd<Self::Input> + PartialOrd,
{
    /// The forward iterator type returned by
    /// [`iter_from_succ_unchecked`](SuccUnchecked::iter_from_succ_unchecked).
    type Iter<'a>: Iterator<Item = Self::Output<'a>>
    where
        Self: 'a;
    /// The bidirectional iterator type returned by
    /// [`iter_bidi_from_succ_unchecked`](SuccUnchecked::iter_bidi_from_succ_unchecked).
    type BidiIter<'a>: BidiIterator<Item = Self::Output<'a>>
    where
        Self: 'a;

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
    /// The successor must exist.
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output<'_>);

    /// Returns the index of the successor and an iterator starting at
    /// the successor position.
    ///
    /// The iterator's first [`next()`](Iterator::next) call returns the
    /// successor value itself. The index returned is the position of the
    /// successor in the sequence.
    ///
    /// # Safety
    ///
    /// The successor must exist.
    unsafe fn iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Iter<'_>);

    /// Returns the index of the successor and a bidirectional iterator
    /// positioned at the successor.
    ///
    /// The iterator's first [`next()`](Iterator::next) call returns the
    /// successor value itself; the first
    /// [`prev()`](crate::traits::BidiIterator::prev) call returns the element
    /// before the successor.
    ///
    /// # Safety
    ///
    /// The successor must exist.
    unsafe fn iter_bidi_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::BidiIter<'_>);
}

impl<T: SuccUnchecked + ?Sized> SuccUnchecked for &T
where
    T::Input: for<'a> PartialOrd<T::Output<'a>> + PartialOrd,
    for<'a> T::Output<'a>: PartialOrd<T::Input> + PartialOrd,
{
    type Iter<'a>
        = T::Iter<'a>
    where
        Self: 'a;
    type BidiIter<'a>
        = T::BidiIter<'a>
    where
        Self: 'a;

    #[inline(always)]
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output<'_>) {
        unsafe { (*self).succ_unchecked::<STRICT>(value) }
    }

    #[inline(always)]
    unsafe fn iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Iter<'_>) {
        unsafe { (*self).iter_from_succ_unchecked::<STRICT>(value) }
    }

    #[inline(always)]
    unsafe fn iter_bidi_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::BidiIter<'_>) {
        unsafe { (*self).iter_bidi_from_succ_unchecked::<STRICT>(value) }
    }
}

/// Successor computation for dictionaries whose values are monotonically increasing.
#[delegatable_trait]
pub trait Succ: SuccUnchecked + IndexedSeq
where
    Self::Input: for<'a> PartialOrd<Self::Output<'a>> + PartialOrd,
    for<'a> Self::Output<'a>: PartialOrd<Self::Input> + PartialOrd,
{
    /// Returns the index of the successor and the successor
    /// of the given value, or `None` if there is no successor.
    ///
    /// The successor is the least value in the dictionary
    /// that is greater than or equal to the given value.
    ///
    /// If there are repeated values, the index of the one returned
    /// depends on the implementation.
    fn succ(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        if self.is_empty() || *value.borrow() > unsafe { self.get_unchecked(self.len() - 1) } {
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
    fn succ_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        if self.is_empty() || *value.borrow() >= unsafe { self.get_unchecked(self.len() - 1) } {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<true>(value) })
        }
    }

    /// Returns the index of the successor and an iterator starting at
    /// the successor position, or `None` if there is no successor.
    fn iter_from_succ(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Iter<'_>)> {
        if self.is_empty() || *value.borrow() > unsafe { self.get_unchecked(self.len() - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict successor and an iterator starting
    /// at the strict successor position, or `None` if there is no strict
    /// successor.
    fn iter_from_succ_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::Iter<'_>)> {
        if self.is_empty() || *value.borrow() >= unsafe { self.get_unchecked(self.len() - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<true>(value) })
        }
    }

    /// Returns the index of the successor and a bidirectional iterator
    /// positioned at the successor, or `None` if there is no successor.
    fn iter_bidi_from_succ(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        if self.is_empty() || *value.borrow() > unsafe { self.get_unchecked(self.len() - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict successor and a bidirectional iterator
    /// positioned at the strict successor, or `None` if there is no strict
    /// successor.
    fn iter_bidi_from_succ_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        if self.is_empty() || *value.borrow() >= unsafe { self.get_unchecked(self.len() - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<true>(value) })
        }
    }
}

impl<T: Succ + ?Sized> Succ for &T
where
    T::Input: for<'a> PartialOrd<T::Output<'a>> + PartialOrd,
    for<'a> T::Output<'a>: PartialOrd<T::Input> + PartialOrd,
{
    #[inline(always)]
    fn succ(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        (*self).succ(value)
    }

    #[inline(always)]
    fn succ_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        (*self).succ_strict(value)
    }

    #[inline(always)]
    fn iter_from_succ(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Iter<'_>)> {
        (*self).iter_from_succ(value)
    }

    #[inline(always)]
    fn iter_from_succ_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::Iter<'_>)> {
        (*self).iter_from_succ_strict(value)
    }

    #[inline(always)]
    fn iter_bidi_from_succ(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        (*self).iter_bidi_from_succ(value)
    }

    #[inline(always)]
    fn iter_bidi_from_succ_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        (*self).iter_bidi_from_succ_strict(value)
    }
}

/// Unchecked predecessor computation for dictionaries whose values are monotonically increasing.
pub trait PredUnchecked: Types
where
    Self::Input: for<'a> PartialOrd<Self::Output<'a>> + PartialOrd,
    for<'a> Self::Output<'a>: PartialOrd<Self::Input> + PartialOrd,
{
    /// The backward iterator type returned by
    /// [`iter_back_from_pred_unchecked`](PredUnchecked::iter_back_from_pred_unchecked).
    type BackIter<'a>: Iterator<Item = Self::Output<'a>>
    where
        Self: 'a;
    /// The bidirectional iterator type returned by
    /// [`iter_bidi_from_pred_unchecked`](PredUnchecked::iter_bidi_from_pred_unchecked).
    type BidiIter<'a>: BidiIterator<Item = Self::Output<'a>>
    where
        Self: 'a;

    /// Returns the index of the predecessor and the predecessor of the given
    /// value.
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
    ) -> (usize, Self::Output<'_>);

    /// Returns the index of the predecessor and a backward iterator starting
    /// at the predecessor position.
    ///
    /// The iterator's first [`next()`](Iterator::next) call returns the
    /// predecessor value itself, and subsequent calls return preceding elements
    /// in decreasing order.
    ///
    /// # Safety
    ///
    /// The predecessor must exist.
    unsafe fn iter_back_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::BackIter<'_>);

    /// Returns the index of the predecessor and a bidirectional iterator
    /// positioned at the predecessor.
    ///
    /// The iterator's first [`next()`](Iterator::next) call returns the
    /// predecessor value itself; the first
    /// [`prev()`](crate::traits::BidiIterator::prev) call returns the element
    /// before the predecessor.
    ///
    /// # Safety
    ///
    /// The predecessor must exist.
    unsafe fn iter_bidi_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::BidiIter<'_>);
}

impl<T: PredUnchecked + ?Sized> PredUnchecked for &T
where
    T::Input: for<'a> PartialOrd<T::Output<'a>> + PartialOrd,
    for<'a> T::Output<'a>: PartialOrd<T::Input> + PartialOrd,
{
    type BackIter<'a>
        = T::BackIter<'a>
    where
        Self: 'a;
    type BidiIter<'a>
        = T::BidiIter<'a>
    where
        Self: 'a;

    #[inline(always)]
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output<'_>) {
        unsafe { (*self).pred_unchecked::<STRICT>(value) }
    }

    #[inline(always)]
    unsafe fn iter_back_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::BackIter<'_>) {
        unsafe { (*self).iter_back_from_pred_unchecked::<STRICT>(value) }
    }

    #[inline(always)]
    unsafe fn iter_bidi_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::BidiIter<'_>) {
        unsafe { (*self).iter_bidi_from_pred_unchecked::<STRICT>(value) }
    }
}

/// Predecessor computation for dictionaries whose values are monotonically increasing.
#[delegatable_trait]
pub trait Pred: PredUnchecked + IndexedSeq
where
    Self::Input: for<'a> PartialOrd<Self::Output<'a>> + PartialOrd,
    for<'a> Self::Output<'a>: PartialOrd<Self::Input> + PartialOrd,
{
    /// Returns the index of the predecessor and the predecessor
    /// of the given value, or `None` if there is no predecessor.
    ///
    /// The predecessor is the greatest value in the dictionary
    /// that is less than or equal to the given value.
    ///
    /// If there are repeated values, the index of the one returned
    /// depends on the implementation.
    fn pred(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        if self.is_empty() || *value.borrow() < unsafe { self.get_unchecked(0) } {
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
    fn pred_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        if self.is_empty() || *value.borrow() <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<true>(value) })
        }
    }

    /// Returns the index of the predecessor and a backward iterator starting
    /// at the predecessor, or [`None`] if there is no predecessor.
    fn iter_back_from_pred(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
        if self.is_empty() || *value.borrow() < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict predecessor and a backward iterator
    /// starting at the strict predecessor, or [`None`] if there is no strict
    /// predecessor.
    fn iter_back_from_pred_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
        if self.is_empty() || *value.borrow() <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<true>(value) })
        }
    }

    /// Returns the index of the predecessor and a bidirectional iterator
    /// positioned at the predecessor, or [`None`] if there is no predecessor.
    fn iter_bidi_from_pred(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        if self.is_empty() || *value.borrow() < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict predecessor and a bidirectional
    /// iterator positioned at the strict predecessor, or [`None`] if there is
    /// no strict predecessor.
    fn iter_bidi_from_pred_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        if self.is_empty() || *value.borrow() <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<true>(value) })
        }
    }
}

impl<T: Pred + ?Sized> Pred for &T
where
    T::Input: for<'a> PartialOrd<T::Output<'a>> + PartialOrd,
    for<'a> T::Output<'a>: PartialOrd<T::Input> + PartialOrd,
{
    #[inline(always)]
    fn pred(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        (*self).pred(value)
    }

    #[inline(always)]
    fn pred_strict(&self, value: impl Borrow<Self::Input>) -> Option<(usize, Self::Output<'_>)> {
        (*self).pred_strict(value)
    }

    #[inline(always)]
    fn iter_back_from_pred(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
        (*self).iter_back_from_pred(value)
    }

    #[inline(always)]
    fn iter_back_from_pred_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
        (*self).iter_back_from_pred_strict(value)
    }

    #[inline(always)]
    fn iter_bidi_from_pred(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        (*self).iter_bidi_from_pred(value)
    }

    #[inline(always)]
    fn iter_bidi_from_pred_strict(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        (*self).iter_bidi_from_pred_strict(value)
    }
}

// Delegations for slices, vectors, and boxed slices

macro_rules! impl_types {
    ($($ty:ty),*) => {$(
        impl Types for [$ty] {
            type Input = $ty;
            type Output<'a> = $ty;
        }

        impl Types for Vec<$ty> {
            type Input = $ty;
            type Output<'a> = $ty;
        }

        impl<const N: usize> Types for [$ty; N] {
            type Input = $ty;
            type Output<'a> = $ty;
        }
    )*};
}

impl_types!(u8, u16, u32, u64, u128, usize);
impl_types!(i8, i16, i32, i64, i128, isize);

macro_rules! impl_indexed_seq {
    ($($ty:ty),*) => {$(
        impl IndexedSeq for [$ty] {
            fn get(&self, index: usize) -> Self::Output<'_> {
                self[index]
            }

            unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
                debug_assert_bounds!(index, self.len());
                // SAFETY: the caller must ensure index < self.len()
                unsafe { *self.get_unchecked(index) }
            }

            fn len(&self) -> usize {
                self.len()
            }

            fn is_empty(&self) -> bool {
                self.is_empty()
            }
        }

        impl IndexedSeq for Vec<$ty> {
            fn get(&self, index: usize) -> Self::Output<'_> {
                self[index]
            }

            unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
                use std::ops::Deref;
                debug_assert_bounds!(index, self.len());
                // SAFETY: the caller must ensure index < self.len()
                unsafe { *self.deref().get_unchecked(index) }
            }

            fn len(&self) -> usize {
                self.len()
            }

            fn is_empty(&self) -> bool {
                self.is_empty()
            }
        }

        impl<const N: usize> IndexedSeq for [$ty; N] {
            fn get(&self, index: usize) -> Self::Output<'_> {
                self[index]
            }

            unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
                debug_assert_bounds!(index, self.len());
                // SAFETY: the caller must ensure index < self.len()
                unsafe { *self.as_slice().get_unchecked(index) }
            }

            fn len(&self) -> usize {
                self.as_slice().len()
            }

            fn is_empty(&self) -> bool {
                self.as_slice().is_empty()
            }
        }
    )*};
}

impl_indexed_seq!(u8, u16, u32, u64, u128, usize);
impl_indexed_seq!(i8, i16, i32, i64, i128, isize);
