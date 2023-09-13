/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! # Traits
//! This modules contains basic traits related to succinct data structures.
//!
//! Traits are collected into a module so you can do `use sux::traits::*;`
//! for ease of use.

mod vslice;
pub use vslice::*;

mod indexed_dict;
pub use indexed_dict::*;

mod convert_to;
pub use convert_to::*;

mod rank_sel;
pub use rank_sel::*;

pub mod prelude {
    pub use super::convert_to::*;
    pub use super::indexed_dict::*;
    pub use super::rank_sel::*;
    pub use super::vslice::*;
}
