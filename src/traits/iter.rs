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
pub trait UncheckedIterator {
    type Item;
    /// Return the next item in the iterator. If there is no next item,
    /// the result is undefined.
    /// # Safety
    /// The caller must ensure that there is a next item.
    unsafe fn next_unchecked(&mut self) -> Self::Item;
}

/// A trait for types that can turn into an
/// [unchecked iterator on values](UncheckedValueIterator), rather than on references.
pub trait IntoUncheckedIterator: Sized {
    type Item;
    type IntoUncheckedIter: UncheckedIterator<Item = Self::Item>;

    fn into_unchecked_iter(self) -> Self::IntoUncheckedIter {
        self.into_unchecked_iter_from(0)
    }

    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter;
}
