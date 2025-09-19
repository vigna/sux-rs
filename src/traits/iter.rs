/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Additional iteration-related traits.

use epserde::deser::{DeserType, DeserializeInner, MemCase};
use impl_tools::autoimpl;

/// Conversion into an [`Iterator`] starting from a given position.
///
/// This trait is similar to [`IntoIterator`], but it allows to specify a
/// starting position for the iteration. Calling
/// [`into_iter`](IntoIterator::into_iter) followed by a call to
/// [`skip`](Iterator::skip) would be sufficient, but for many compressed and
/// succinct data structures the setup of an iterator can be expensive, and that
/// setup is usually required again after a skip.
///
/// We provide implementations for (references to) slices, vectors, and boxed
/// slices, and delegations for [`MemCase`].
pub trait IntoIteratorFrom: IntoIterator {
    /// Which kind of iterator are we turning this into?
    type IntoIterFrom: Iterator<Item = <Self as IntoIterator>::Item>;

    /// Creates an iterator from a value and a starting position.
    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom;
}

impl<'a, T> IntoIteratorFrom for &'a [T] {
    type IntoIterFrom = std::iter::Skip<std::slice::Iter<'a, T>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.iter().skip(from)
    }
}

impl<T> IntoIteratorFrom for Vec<T> {
    type IntoIterFrom = std::iter::Skip<std::vec::IntoIter<T>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.into_iter().skip(from)
    }
}

impl<'a, T> IntoIteratorFrom for &'a Vec<T> {
    type IntoIterFrom = std::iter::Skip<std::slice::Iter<'a, T>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.iter().skip(from)
    }
}

impl<T> IntoIteratorFrom for Box<[T]> {
    type IntoIterFrom = std::iter::Skip<std::vec::IntoIter<T>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        IntoIterator::into_iter(self).skip(from)
    }
}

impl<'a, T> IntoIteratorFrom for &'a Box<[T]> {
    type IntoIterFrom = std::iter::Skip<std::slice::Iter<'a, T>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.into_iter().skip(from)
    }
}

/// A trait for iterating on values very quickly and very unsafely.
///
/// The purpose of this trait is to allow cheap parallel iteration over
/// multiple structures of the same size. The hosting code can take care
/// that the iteration is safe, and can use this unsafe
/// trait to iterate very cheaply over each structure. See the implementation
/// of [`EliasFanoIterator`](crate::dict::elias_fano::EliasFanoIterator) for an example.
#[autoimpl(for<T: trait + ?Sized> &mut T, Box<T>)]
pub trait UncheckedIterator {
    type Item;
    /// Returns the next item in the iterator. If there is no next item,
    /// the result is undefined.
    /// # Safety
    /// The caller must ensure that there is a next item.
    unsafe fn next_unchecked(&mut self) -> Self::Item;
}

/// A trait for types that can turn into an [`UncheckedIterator`].
///
/// Differently from [`IntoIterator`], this trait provides a way
/// to obtain an iterator starting from a given position.
pub trait IntoUncheckedIterator: Sized {
    type Item;
    type IntoUncheckedIter: UncheckedIterator<Item = Self::Item>;

    /// Creates an unchecked iterator starting from the first position.
    fn into_unchecked_iter(self) -> Self::IntoUncheckedIter {
        self.into_unchecked_iter_from(0)
    }

    /// Creates an unchecked iterator starting from the given position.
    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter;
}

/// A trait for types that can turn into an [`UncheckedIterator`] moving backwards.
///
/// Differently from [`IntoIterator`], this trait provides a way
/// to obtain an iterator starting from a given position.
///
/// Note that
/// [`into_rev_unchecked_iter`](IntoReverseUncheckedIterator::into_rev_unchecked_iter_from)
/// cannot be implemented in terms of
/// [`into_rev_unchecked_iter_from`](IntoReverseUncheckedIterator::into_rev_unchecked_iter_from)
/// because we cannot know which is the last position.
pub trait IntoReverseUncheckedIterator: Sized {
    type Item;
    type IntoRevUncheckedIter: UncheckedIterator<Item = Self::Item>;

    /// Creates a reverse unchecked iterator starting from the end.
    fn into_rev_unchecked_iter(self) -> Self::IntoRevUncheckedIter;
    /// Creates a reverse unchecked iterator starting from the given position.
    fn into_rev_unchecked_iter_from(self, from: usize) -> Self::IntoRevUncheckedIter;
}

// Delegations for MemCase

impl<'a, S: DeserializeInner> IntoIteratorFrom for &'a MemCase<S>
where
    &'a DeserType<'a, S>: IntoIteratorFrom,
{
    type IntoIterFrom = <&'a DeserType<'a, S> as IntoIteratorFrom>::IntoIterFrom;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.uncase().into_iter_from(from)
    }
}

impl<'a, S: DeserializeInner> IntoUncheckedIterator for &'a MemCase<S>
where
    for<'b> &'b DeserType<'b, S>: IntoUncheckedIterator,
{
    type Item = <&'a DeserType<'a, S> as IntoUncheckedIterator>::Item;
    type IntoUncheckedIter = <&'a DeserType<'a, S> as IntoUncheckedIterator>::IntoUncheckedIter;

    fn into_unchecked_iter(self) -> Self::IntoUncheckedIter {
        self.uncase().into_unchecked_iter()
    }

    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter {
        self.uncase().into_unchecked_iter_from(from)
    }
}

impl<'a, S: DeserializeInner> IntoReverseUncheckedIterator for &'a MemCase<S>
where
    for<'b> &'b DeserType<'b, S>: IntoReverseUncheckedIterator,
{
    type Item = <&'a DeserType<'a, S> as IntoReverseUncheckedIterator>::Item;
    type IntoRevUncheckedIter =
        <&'a DeserType<'a, S> as IntoReverseUncheckedIterator>::IntoRevUncheckedIter;

    fn into_rev_unchecked_iter(self) -> Self::IntoRevUncheckedIter {
        self.uncase().into_rev_unchecked_iter()
    }

    fn into_rev_unchecked_iter_from(self, from: usize) -> Self::IntoRevUncheckedIter {
        self.uncase().into_rev_unchecked_iter_from(from)
    }
}
