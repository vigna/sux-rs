/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!("../README.md")]
#![deny(unconditional_recursion)]

pub mod bits;
pub mod dict;
pub mod func;
pub mod rank_sel;
pub mod traits;
pub mod utils;

pub mod prelude {
    pub use crate::bits::prelude::*;
    pub use crate::dict::prelude::*;
    pub use crate::func::*;
    pub use crate::rank_sel::prelude::*;
    pub use crate::traits::prelude::*;
    pub use crate::utils::*;
}
