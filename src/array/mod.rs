/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Array data structures.
//!
//! Presently, this module provides [partial arrays], that is, immutable arrays
//! with holes: only some positions contain a value, and accessing a position
//! returns an [`Option`]. Values are stored contiguously, and an index maps
//! each position containing a value to the index of the value. There are two
//! kinds of index:
//!
//! - [dense partial arrays] mark the positions containing a value in a bit
//!   vector with [ranking support], using about 1.25 bits per position, and
//!   provide very fast access;
//!
//! - [sparse partial arrays] store the positions containing a value in an
//!   [Elias–Fano] structure, using about 2 + log₂(*n* / *m*) bits per value,
//!   where *n* is the length of the array and *m* is the number of values;
//!   access is slower, but they use less space when less than about a quarter
//!   of the positions contain a value.
//!
//! You build a partial array with a [`PartialArrayBuilder`], which you obtain
//! from [`new_dense`]/[`PartialArrayBuilder::dense`] or
//! [`new_sparse`]/[`PartialArrayBuilder::sparse`]. The resulting types are
//! [`DensePartialArray`] and [`SparsePartialArray`], respectively.
//!
//! [partial arrays]: PartialArray
//! [dense partial arrays]: DensePartialArray
//! [sparse partial arrays]: SparsePartialArray
//! [ranking support]: crate::rank_sel::Rank9
//! [Elias–Fano]: crate::dict::EliasFano

pub mod partial_array;
#[doc(inline)]
pub use partial_array::{
    DensePartialArray, PartialArray, PartialArrayBuilder, SparsePartialArray, new_dense, new_sparse,
};
