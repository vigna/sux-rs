/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Additional iteration-related traits.

use impl_tools::autoimpl;
use mem_dbg::{MemDbg, MemSize};

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
/// [`into_rev_unchecked_iter`](IntoReverseUncheckedIterator::into_rev_unchecked_iter)
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

/// Conversion into a [`BidiIterator`].
///
/// This is the bidirectional counterpart of [`IntoIterator`]. It provides a
/// way to obtain a bidirectional iterator positioned at the first element.
pub trait IntoBidiIterator: Sized {
    type Item;
    type IntoBidiIter: BidiIterator<Item = Self::Item>;

    /// Creates a bidirectional iterator positioned at the first element.
    fn into_bidi_iter(self) -> Self::IntoBidiIter;
}

/// Conversion into a [`BidiIterator`] starting from a given position.
///
/// This is the bidirectional counterpart of [`IntoIteratorFrom`]. It provides
/// a way to obtain a bidirectional iterator starting from a given position.
///
/// The cursor position `from` is interpreted as for
/// [`IntoIteratorFrom`]: the first call to `next()` yields element `from`,
/// while the first call to [`prev()`](BidiIterator::prev) yields element
/// `from - 1`.
pub trait IntoBidiIteratorFrom: IntoBidiIterator {
    type IntoBidiIterFrom: BidiIterator<Item = <Self as IntoBidiIterator>::Item>;

    /// Creates a bidirectional iterator positioned at the given index.
    fn into_bidi_iter_from(self, from: usize) -> Self::IntoBidiIterFrom;
}

/// A bidirectional iterator that can move both forward and backward.
///
/// This trait extends [`Iterator`] with a `prev()` method and an associated
/// [`PrevIter`](BidiIterator::PrevIter) type that swaps the directions. The
/// involution constraint `PrevIter: BidiIterator<PrevIter = Self>` guarantees
/// that wrapping and unwrapping are inverses.
///
/// Call [`prev_iter`](BidiIterator::prev_iter) to obtain an iterator whose
/// `next` goes in the opposite direction. The canonical implementation is
/// [`PrevIter`], a zero-cost `#[repr(transparent)]` wrapper.
pub trait BidiIterator: Iterator + Sized {
    /// The type of the iterator obtained by reversing the direction.
    ///
    /// The constraint `PrevIter: BidiIterator<Item = Self::Item, PrevIter =
    /// Self>` enforces that reversing twice yields the original type.
    type PrevIter: BidiIterator<Item = Self::Item, PrevIter = Self>;

    /// Converts this iterator into one that iterates in the opposite direction.
    fn prev_iter(self) -> Self::PrevIter;

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
        if self.prev_advance_by(n).is_err() {
            return None;
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
/// Wrapping a `BidiIterator` in [`PrevIter`] causes [`Iterator::next`] to
/// delegate to [`BidiIterator::prev`] and vice versa. Calling
/// [`prev_iter`](BidiIterator::prev_iter) on a [`PrevIter`] unwraps back to the
/// inner iterator, so the operation is an involution.
#[derive(Clone, Debug, MemDbg, MemSize)]
#[repr(transparent)]
pub struct PrevIter<I>(pub I);

impl<I: BidiIterator<PrevIter = PrevIter<I>>> Iterator for PrevIter<I> {
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

impl<I: BidiIterator<PrevIter = PrevIter<I>>> BidiIterator for PrevIter<I> {
    type PrevIter = I;

    #[inline(always)]
    fn prev_iter(self) -> I {
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

    #[cfg(not(feature = "iter_advance_by"))]
    #[inline(always)]
    fn prev_advance_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        for i in 0..n {
            if self.0.next().is_none() {
                // SAFETY: n - i > 0 because i < n
                return Err(unsafe { std::num::NonZero::new_unchecked(n - i) });
            }
        }
        Ok(())
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

impl<I: ExactSizeBidiIterator<PrevIter = PrevIter<I>>> ExactSizeIterator for PrevIter<I> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.prev_len()
    }
}

impl<I: BidiIterator<PrevIter = PrevIter<I>> + ExactSizeIterator> ExactSizeBidiIterator
    for PrevIter<I>
{
    #[inline(always)]
    fn prev_len(&self) -> usize {
        self.0.len()
    }
}

impl<I: FusedBidiIterator<PrevIter = PrevIter<I>>> std::iter::FusedIterator for PrevIter<I> {}

impl<I: BidiIterator<PrevIter = PrevIter<I>> + std::iter::FusedIterator> FusedBidiIterator
    for PrevIter<I>
{
}
