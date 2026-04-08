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

pub use crate::traits::bal_paren::{BalParen, ambassador_impl_BalParen};

pub mod jacobson;
pub use jacobson::*;
