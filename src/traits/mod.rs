/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Main traits used in the implementation of succinct and compressed data
//! structures.
//!
//! See the discussion in [`bit_field_slice`] about the re-export of its traits.

pub mod bit_field_slice;
use ambassador::delegatable_trait;
pub use bit_field_slice::*;

pub mod indexed_dict;
use impl_tools::autoimpl;
pub use indexed_dict::*;

pub mod iter;
pub use iter::*;

pub mod rank_sel;
pub use rank_sel::*;

pub mod bit_vec_ops;
pub use bit_vec_ops::*;

/// The basic trait defining backends.
///
/// Backends are types that can be seen as slices of words. The type of the word
/// is given by the [`Word`](Self::Word) associated type. They make it possible
/// to write generic code that can work with any backend by providing the word
/// type.
///
/// This trait is delegated by every [rank/select structure](crate::rank_sel) to
/// its backend (an inner field): as a result, every structure can be used as
/// the backend of a further structure.
///
/// Usually, this trait is coupled with [`AsRef<[<Self as
/// Backend>::Word]>`](core::convert::AsRef) or [`AsMut<[<Self as
/// Backend>::Word]>`](core::convert::AsMut) to allow access to the underlying
/// slice of words, but this is not strictly required (see, e.g., the
/// [`BitWidth`] trait).
///
/// We implement this trait for slices, vectors, and arrays; moreover,
/// we delegate it automatically to references and boxed types.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait Backend {
    /// The word type used by this backend.
    ///
    /// Since we have backends based on both atomic and non-atomic primitive
    /// types, we do not require the word type to implement any specific trait.
    type Word;
}

impl<W> Backend for [W] {
    type Word = W;
}

impl<W> Backend for Vec<W> {
    type Word = W;
}

impl<W, const N: usize> Backend for [W; N] {
    type Word = W;
}
