/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!("../README.md")]
#![deny(unconditional_recursion)]

pub mod bits;
pub mod count;
pub mod dict;
pub mod func;
pub mod rank_sel;
pub mod traits;
pub mod utils;

#[cfg(feature = "fuzz")]
pub mod fuzz;

pub mod prelude {
    pub use crate::bits::*;
    pub use crate::count::*;
    pub use crate::dict::*;
    pub use crate::func::*;
    pub use crate::rank_sel::*;
    pub use crate::traits::bit_field_slice;
    pub use crate::traits::*;
    pub use crate::utils::*;
}
