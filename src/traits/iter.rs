/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Traits for iterators, possibly [unchecked](crate::traits::iter::IntoUncheckedValueIterator).

*/

/// A trait for iterating on values very quickly and very unsafely.
///
/// The purpose of this trait is to allow cheap parallel iteration over
/// multiple structures of the same size. The hosting code can take care
/// that the iteration is safe, and can use this unsafe
/// trait to iterate very cheaply over each structure. See the implementation
/// of [`EliasFanoIterator`](crate::dict::elias_fano::EliasFanoIterator) for an example.
pub trait UncheckedValueIterator {
    type Item;
    /// Return the next item in the iterator. If there is no next item,
    /// the result is undefined.
    /// # Safety
    /// The caller must ensure that there is a next item.
    unsafe fn next_unchecked(&mut self) -> Self::Item;
}

/// A trait for types that can generate an
/// [unchecked iterator on values](UncheckedValueIterator), rather than on references.
pub trait IntoUncheckedValueIterator {
    type Item;
    type IntoUncheckedValueIter<'a>: UncheckedValueIterator<Item = Self::Item> + 'a
    where
        Self: 'a;

    fn into_val_iter_unchecked(&self) -> Self::IntoUncheckedValueIter<'_> {
        self.into_val_iter_from_unchecked(0)
    }

    fn into_val_iter_from_unchecked(&self, from: usize) -> Self::IntoUncheckedValueIter<'_>;
}
