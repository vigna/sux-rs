/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Additional iteration-related traits.
//!
//! - [`IntoIteratorFrom`] makes it possible to start iterators from
//!   a given position, which is useful for compressed and succinct data
//!   structures where the setup of an iterator can be expensive.
//!   It is implemented for basic containers (arrays, vectors, etc.).
//!
//! - [`UncheckedIterator`]/[`IntoUncheckedIterator`]/[`IntoUncheckedBackIterator`]
//!   are traits providing unsafe cheap iteration. Their main purpose is to
//!   abstract fast, unchecked iteration over implementations of
//!   [`BitFieldSlice`](crate::traits::BitFieldSlice).
//!
//! - [`IntoBackIterator`]/[`IntoBackIteratorFrom`] are counterparts of
//!   [`IntoIterator`]/[`IntoIteratorFrom`] for backward iteration.
//!
//! - [`BidiIterator`]/[`IntoBidiIterator`]/[`IntoBidiIteratorFrom`] are traits
//!   for bidirectional iteration. These will usually be slower than their unidirectional
//!   counterparts, but they provide more flexibility. [`ExactSizeBidiIterator`]
//!   and [`FusedBidiIterator`] are the bidirectional counterparts of
//!   [`ExactSizeIterator`] and [`FusedIterator`](core::iter::FusedIterator).
//!   [`SwappedIter`] wraps a [`BidiIterator`] swapping the roles of
//!   [`next`](Iterator::next) and [`prev`](BidiIterator::prev).

use impl_tools::autoimpl;
use mem_dbg::{MemDbg, MemSize};

/// Conversion into an [`Iterator`] starting from a given position.
///
/// This trait is similar to [`IntoIterator`], but it allows specifying a
/// starting position for the iteration. Calling
/// [`into_iter`](IntoIterator::into_iter) followed by a call to
/// [`skip`](Iterator::skip) would be sufficient, but for many compressed and
/// succinct data structures the setup of an iterator can be expensive, and that
/// setup is usually required again after a skip.
///
/// We provide implementations for (references to) slices, vectors, and boxed
/// slices.
pub trait IntoIteratorFrom: IntoIterator {
    /// Which kind of iterator are we turning this into?
    type IntoIterFrom: Iterator<Item = <Self as IntoIterator>::Item>;

    /// Creates an iterator from a starting position.
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
/// The purpose of this trait is to allow cheap parallel iteration over multiple
/// structures of the same size. The hosting code can take care that the
/// iteration is safe, and can use this trait to iterate very cheaply over each
/// structure. See the implementation of
/// [`EliasFanoIter`](crate::dict::elias_fano::EliasFanoIter) for an example.
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
/// [`into_unchecked_iter_back`](IntoUncheckedBackIterator::into_unchecked_iter_back)
/// cannot be implemented in terms of
/// [`into_unchecked_iter_back_from`](IntoUncheckedBackIterator::into_unchecked_iter_back_from)
/// because we cannot know which is the last position.
pub trait IntoUncheckedBackIterator: Sized {
    type Item;
    type IntoUncheckedIterBack: UncheckedIterator<Item = Self::Item>;

    /// Creates a backward unchecked iterator starting from the end.
    fn into_unchecked_iter_back(self) -> Self::IntoUncheckedIterBack;
    /// Creates a backward unchecked iterator starting from the given position.
    fn into_unchecked_iter_back_from(self, from: usize) -> Self::IntoUncheckedIterBack;
}

/// Conversion into a backward [`Iterator`].
///
/// This is the backward counterpart of [`IntoIterator`]. It provides a way to
/// obtain an iterator that yields elements in backward (decreasing) order.
pub trait IntoBackIterator: Sized {
    type Item;
    type IntoIterBack: Iterator<Item = Self::Item>;

    /// Creates a backward iterator starting from the last element.
    fn into_iter_back(self) -> Self::IntoIterBack;
}

/// Conversion into a backward [`Iterator`] starting from a given position.
///
/// This is the backward counterpart of [`IntoIteratorFrom`]. The iterator
/// returned by [`into_iter_back_from`](IntoBackIteratorFrom::into_iter_back_from)
/// yields element `from` first, then preceding elements in decreasing order
/// (as [`IntoUncheckedBackIterator::into_unchecked_iter_back_from`]).
pub trait IntoBackIteratorFrom: IntoBackIterator {
    type IntoIterBackFrom: Iterator<Item = <Self as IntoBackIterator>::Item>;

    /// Creates a backward iterator starting from the given position.
    fn into_iter_back_from(self, from: usize) -> Self::IntoIterBackFrom;
}

/// A bidirectional iterator that can move both forward and backward.
///
/// This trait extends [`Iterator`] with a `prev()` method and an associated
/// [`SwappedIter`](BidiIterator::SwappedIter) type that swaps the directions. The
/// involution constraint `SwappedIter: BidiIterator<SwappedIter = Self>` guarantees
/// that wrapping and unwrapping are inverses.
///
/// Call [`swap`](BidiIterator::swap) to obtain an iterator whose
/// `next` goes in the opposite direction. The canonical implementation is
/// [`SwappedIter`], a zero-cost `#[repr(transparent)]` wrapper.
pub trait BidiIterator: Iterator + Sized {
    /// The type of the iterator obtained by reversing the direction.
    ///
    /// The constraint `SwappedIter: BidiIterator<Item = Self::Item, SwappedIter =
    /// Self>` enforces that reversing twice yields the original type.
    type SwappedIter: BidiIterator<Item = Self::Item, SwappedIter = Self>;

    /// Converts this iterator into one that iterates in the opposite direction.
    fn swap(self) -> Self::SwappedIter;

    /// Advances the iterator backward and returns the previous item, or
    /// [`None`] if there are no more items.
    fn prev(&mut self) -> Option<Self::Item>;

    /// Returns bounds on the number of remaining items in the backward
    /// direction, analogous to [`Iterator::size_hint`].
    fn prev_size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    /// Advances the iterator backward by `n` items. Returns `Ok(())` if
    /// successful, or `Err(k)` where `k` is the non-zero number of remaining
    /// steps that could not be advanced.
    #[cfg(feature = "iter_advance_by")]
    fn prev_advance_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        for i in 0..n {
            if self.prev().is_none() {
                // SAFETY: n - i > 0 because i < n
                return Err(unsafe { std::num::NonZero::new_unchecked(n - i) });
            }
        }
        Ok(())
    }

    /// Returns the `n`-th item from the back, skipping `n` items.
    ///
    /// Analogous to [`Iterator::nth`] but in the backward direction.
    fn prev_nth(&mut self, n: usize) -> Option<Self::Item> {
        for _ in 0..n {
            self.prev()?;
        }
        self.prev()
    }

    /// Folds every element into an accumulator by applying a closure,
    /// moving backward. Analogous to [`Iterator::fold`].
    fn prev_fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        while let Some(x) = self.prev() {
            accum = f(accum, x);
        }
        accum
    }

    /// Calls a closure on each element, moving backward.
    /// Analogous to [`Iterator::for_each`].
    fn prev_for_each<F>(self, mut f: F)
    where
        F: FnMut(Self::Item),
    {
        self.prev_fold((), |(), x| f(x));
    }

    /// Consumes the iterator backward, counting the number of remaining
    /// elements. Analogous to [`Iterator::count`].
    fn prev_count(self) -> usize {
        self.prev_fold(0, |count, _| count + 1)
    }

    /// Consumes the iterator backward, returning the last element reached.
    /// Analogous to [`Iterator::last`] but in the backward direction.
    fn prev_last(mut self) -> Option<Self::Item> {
        let mut last = None;
        while let Some(x) = self.prev() {
            last = Some(x);
        }
        last
    }
}

/// An exact-size bidirectional iterator.
///
/// This is the backward counterpart of [`ExactSizeIterator`]: implementors
/// must know exactly how many items remain in the backward direction.
pub trait ExactSizeBidiIterator: BidiIterator {
    /// Returns the exact number of remaining items in the backward direction.
    fn prev_len(&self) -> usize;
}

/// A fused bidirectional iterator.
///
/// A `FusedBidiIterator` guarantees that [`prev`](BidiIterator::prev) keeps
/// returning [`None`] once exhausted, mirroring
/// [`FusedIterator`](std::iter::FusedIterator) for the backward direction.
pub trait FusedBidiIterator: BidiIterator {}

/// A zero-cost wrapper that swaps the forward and backward directions of a
/// [`BidiIterator`].
///
/// Wrapping a `BidiIterator` in [`SwappedIter`] causes [`Iterator::next`] to
/// delegate to [`BidiIterator::prev`] and vice versa. Calling
/// [`swap`](BidiIterator::swap) on a [`SwappedIter`] unwraps back to the
/// inner iterator, so the operation is an involution.
#[derive(Clone, Debug, MemDbg, MemSize)]
#[repr(transparent)]
pub struct SwappedIter<I>(pub I);

impl<I: BidiIterator<SwappedIter = SwappedIter<I>>> Iterator for SwappedIter<I> {
    type Item = I::Item;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.prev()
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.prev_size_hint()
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.prev_nth(n)
    }

    #[cfg(feature = "iter_advance_by")]
    #[inline(always)]
    fn advance_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        self.0.prev_advance_by(n)
    }

    #[inline(always)]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.0.prev_fold(init, f)
    }

    #[inline(always)]
    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        self.0.prev_for_each(f);
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.0.prev_count()
    }

    #[inline(always)]
    fn last(self) -> Option<Self::Item> {
        self.0.prev_last()
    }
}

impl<I: BidiIterator<SwappedIter = SwappedIter<I>>> BidiIterator for SwappedIter<I> {
    type SwappedIter = I;

    #[inline(always)]
    fn swap(self) -> I {
        self.0
    }

    #[inline(always)]
    fn prev(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline(always)]
    fn prev_size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[cfg(feature = "iter_advance_by")]
    #[inline(always)]
    fn prev_advance_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        self.0.advance_by(n)
    }

    #[inline(always)]
    fn prev_nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }

    #[inline(always)]
    fn prev_fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.0.fold(init, f)
    }

    #[inline(always)]
    fn prev_for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        self.0.for_each(f);
    }

    #[inline(always)]
    fn prev_count(self) -> usize {
        self.0.count()
    }

    #[inline(always)]
    fn prev_last(self) -> Option<Self::Item> {
        self.0.last()
    }
}

impl<I: ExactSizeBidiIterator<SwappedIter = SwappedIter<I>>> ExactSizeIterator for SwappedIter<I> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.prev_len()
    }
}

impl<I: BidiIterator<SwappedIter = SwappedIter<I>> + ExactSizeIterator> ExactSizeBidiIterator
    for SwappedIter<I>
{
    #[inline(always)]
    fn prev_len(&self) -> usize {
        self.0.len()
    }
}

impl<I: FusedBidiIterator<SwappedIter = SwappedIter<I>>> std::iter::FusedIterator
    for SwappedIter<I>
{
}

impl<I: BidiIterator<SwappedIter = SwappedIter<I>> + std::iter::FusedIterator> FusedBidiIterator
    for SwappedIter<I>
{
}

/// Conversion into a [`BidiIterator`].
///
/// This is the bidirectional counterpart of [`IntoIterator`]. It provides a
/// way to obtain a bidirectional iterator positioned at the first element.
pub trait IntoBidiIterator: Sized {
    type Item;
    type IntoIterBidi: BidiIterator<Item = Self::Item>;

    /// Creates a bidirectional iterator positioned at the first element.
    fn into_iter_bidi(self) -> Self::IntoIterBidi;
}

/// Conversion into a [`BidiIterator`] starting from a given position.
///
/// This is the bidirectional counterpart of [`IntoIteratorFrom`]. It provides
/// a way to obtain a bidirectional iterator starting from a given position.
///
/// Calling [`into_iter_bidi`](IntoBidiIterator::into_iter_bidi) followed by a
/// call to [`skip`](Iterator::skip) would be sufficient, but for many
/// compressed and succinct data structures the setup of an iterator can be
/// expensive, and that setup is usually required again after a skip.
///
/// The cursor position `from` is interpreted as for
/// [`IntoIteratorFrom`]: the first call to `next()` yields element `from`,
/// while the first call to [`prev()`](BidiIterator::prev) yields element
/// `from - 1`.
pub trait IntoBidiIteratorFrom: IntoBidiIterator {
    type IntoIterBidiFrom: BidiIterator<Item = <Self as IntoBidiIterator>::Item>;

    /// Creates a bidirectional iterator positioned at the given index.
    fn into_iter_bidi_from(self, from: usize) -> Self::IntoIterBidiFrom;
}
