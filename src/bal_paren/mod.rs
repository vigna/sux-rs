/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Balanced parentheses data structures.
//!
//! This module provides succinct data structures for balanced parentheses
//! sequences, where open parentheses are represented as 1-bits and close
//! parentheses as 0-bits (LSB first within each `usize` word).
//!
//! The [`BalParen`] trait abstracts the query interface; currently, the only
//! implementation is [`JacobsonBalParen`].

use mem_dbg::MemSize;

/// A balanced parentheses structure supporting
/// [`find_close`](BalParen::find_close) queries.
///
/// Open parentheses are 1-bits and close parentheses are 0-bits, packed
/// LSB-first in `usize` words accessible via [`words`](BalParen::words).
pub trait BalParen: MemSize + std::fmt::Debug {
    /// Returns the underlying words of the bit vector.
    fn words(&self) -> &[usize];

    /// Returns the position of the matching close parenthesis for the open
    /// parenthesis at bit position `pos`, or `None` if `pos` is out of
    /// bounds or is not an open parenthesis.
    fn find_close(&self, pos: usize) -> Option<usize>;
}

pub mod jacobson;
pub use jacobson::*;
