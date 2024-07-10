/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![deny(unconditional_recursion)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::len_without_is_empty)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("`target_pointer_width` must be 64");

pub mod bits;
pub mod dict;
pub mod rank_sel;
pub mod traits;
pub mod utils;

#[cfg(feature = "fuzz")]
pub mod fuzz;

pub mod prelude {
    pub use crate::bit_field_vec;
    pub use crate::bit_vec;
    pub use crate::bits::*;
    pub use crate::dict::*;
    pub use crate::rank_sel::*;
    pub use crate::rank_small;
    pub use crate::traits::bit_field_slice;
    pub use crate::traits::*;
    pub use crate::DivCeilUnchecked;
}

#[ambassador::delegatable_trait_remote]
pub(crate) trait AsRef<T> {
    fn as_ref(&self) -> &T;
}

#[ambassador::delegatable_trait_remote]
pub(crate) trait Index<Idx> {
    type Output;
    fn index(&self, index: Idx) -> &Self::Output;
}

/// A trait for performing division with ceiling rounding without checking for
/// division by zero or overflow.
pub trait DivCeilUnchecked {
    /// Divides `self` by `rhs` and returns the result rounded up to the nearest
    /// integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use sux::prelude::*;
    /// assert_eq!(5_usize.div_ceil_unchecked(2), 3);
    /// assert_eq!(10_u64.div_ceil_unchecked(3), 4);
    /// ```
    fn div_ceil_unchecked(self, rhs: Self) -> Self;
}

impl DivCeilUnchecked for usize {
    #[inline(always)]
    fn div_ceil_unchecked(self, rhs: Self) -> Self {
        (self + rhs - 1) / rhs
    }
}

impl DivCeilUnchecked for u64 {
    #[inline(always)]
    fn div_ceil_unchecked(self, rhs: Self) -> Self {
        (self + rhs - 1) / rhs
    }
}
