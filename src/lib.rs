/*
 * SPDX-FileCopyrightText: 2023 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!("../README.md")]
#![deny(unconditional_recursion)]

pub mod bits;
pub mod dict;
pub mod hash;
pub mod mph;
pub mod ranksel;
pub mod sf;
pub mod traits;
pub mod utils;

pub mod prelude {
    pub use crate::bits::*;
    pub use crate::dict::prelude::*;
    pub use crate::hash::*;
    pub use crate::mph::*;
    pub use crate::ranksel::prelude::*;
    pub use crate::sf::*;
    pub use crate::traits::*;
}
