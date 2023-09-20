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
pub trait UncheckedIterator: Iterator {
    unsafe fn next_unchecked(&mut self) -> Self::Item {
        self.next().unwrap()
    }
}
