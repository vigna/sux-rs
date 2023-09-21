/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

pub mod convert_to;
pub mod indexed_dict;
pub mod rank_sel;
pub mod vslice;

pub mod prelude {
    pub use super::convert_to::*;
    pub use super::indexed_dict::*;
    pub use super::rank_sel::*;
    pub use super::vslice::*;
}

/// A trait for iterating very quickly and very unsafely.
///
/// The purpose of this trait is to allow cheap parallel iteration over
/// multiple structures of the same size. The hosting code can take care
/// that the iteration is safe, and can use this unsafe
/// trait to iterate very cheaply over each structure. See the implementation
/// of the [`EliasFano`](crate::dict::elias_fano::EliasFano)
/// [iterator](crate::traits::indexed_dict::IndexedDict::iter) for an example.
pub trait UncheckedIterator {
    type Item;
    /// Return the next item in the iterator. If there is no next item,
    /// the result is undefined.
    unsafe fn next_unchecked(&mut self) -> Self::Item;
}
