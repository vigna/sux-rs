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

#[allow(clippy::len_without_is_empty)]
pub trait BitOps {
    fn ceil_log2(self) -> u32;
    fn len(self) -> u32;
}

macro_rules! impl_bit_ops {
    ($($ty:ty),*) => {$(
        impl BitOps for $ty {
            fn ceil_log2(self) -> u32 {
                if self <= 2 {
                    self as u32
                } else {
                    (self - 1).ilog2() + 1
                }
            }
            fn len(self) -> u32 {
                if self == 0 {
                    1
                } else {
                    self.ilog2() + 1
                }
            }
        }
    )*};
}

impl_bit_ops!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);
