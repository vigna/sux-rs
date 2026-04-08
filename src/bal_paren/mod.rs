/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Balanced parentheses data structures.
//!
//! This module provides succinct data structures for balanced parentheses
//! sequences, where open parentheses are represented as 1-bits and close
//! parentheses as 0-bits (LSB first within each `usize` word). They are
//! frequently used to represent structure such as binary trees, trees, forests,
//! and planar graphs.
//!
//! The [`BalParen`] trait abstracts the query interface; currently, the only
//! implementation is [`JacobsonBalParen`].

use ambassador::delegatable_trait;
use impl_tools::autoimpl;

/// A balanced parentheses structure supporting
/// [`find_close`](BalParen::find_close) queries.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait BalParen {
    /// Returns the position of the matching close parenthesis for the open
    /// parenthesis at bit position `pos`, or `None` if `pos` is out of
    /// bounds or is not an open parenthesis.
    fn find_close(&self, pos: usize) -> Option<usize>;
}

pub mod jacobson;
pub use jacobson::*;
