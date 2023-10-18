/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Traits for iterators based on [values](crate::traits::iter::IntoValueIterator),
possibly [unchecked](crate::traits::iter::IntoUncheckedValueIterator).

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

/// A trait for types that can generate an iterator on values, rather than on references.
///
/// To avoid clashes in types that implement both [`IntoValueIterator`] and
/// [`IntoIterator`], the methods of this trait have the
/// infix `_val`. However, implementors are invited, if it is convenient,
/// to implement [`IntoIterator`] on a reference to the type
/// delegating to the implementation of this trait, and implement also on the type
/// an `into_iter_from` method (see, for examples, the implementation in
/// [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec)).
pub trait IntoValueIterator {
    type Item;
    type IntoValueIter<'a>: Iterator<Item = Self::Item> + 'a
    where
        Self: 'a;
    fn into_val_iter(&self) -> Self::IntoValueIter<'_> {
        self.into_val_iter_from(0)
    }

    fn into_val_iter_from(&self, from: usize) -> Self::IntoValueIter<'_>;
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
