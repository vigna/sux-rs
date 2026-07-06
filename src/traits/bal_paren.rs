/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Balanced parentheses.

use ambassador::delegatable_trait;
use impl_tools::autoimpl;

/// A balanced parentheses structure supporting
/// [`find_close`] queries.
///
/// [`find_close`]: BalParen::find_close
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait(inline = "always")]
pub trait BalParen {
    /// Returns the position of the matching close parenthesis for the open
    /// parenthesis at bit position `pos`, or `None` if `pos` is not an
    /// open parenthesis.
    ///
    /// # Panics
    ///
    /// Panics if `pos` is out of bounds.
    fn find_close(&self, pos: usize) -> Option<usize>;
}
