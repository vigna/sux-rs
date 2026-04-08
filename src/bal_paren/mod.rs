/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Balanced parentheses data structures.
//!
//! This module provides succinct data structures for balanced parentheses
//! sequences, where open parentheses are represented as 1-bits and close
//! parentheses as 0-bits (LSB first within each 64-bit word).
//!
//! Currently, the only implementation is [`JacobsonBalParen`], which supports
//! [`find_close`](JacobsonBalParen::find_close).

pub mod jacobson;
pub use jacobson::*;
