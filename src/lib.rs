/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
// TODO: this will disappear as soon as Ambassador emits unsafe blocks
#![allow(unsafe_op_in_unsafe_fn)]
#![deny(unconditional_recursion)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::len_without_is_empty)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("`target_pointer_width` must be 64");

pub mod array;
pub mod bits;
pub mod dict;
pub mod func;
pub mod rank_sel;
pub mod traits;
pub mod utils;

#[cfg(feature = "fuzz")]
pub mod fuzz;

/// Imports the most common items. Note that
/// [`bit_field_slice`](crate::traits::bit_field_slice) and
/// [`indexed_dict`](crate::traits::indexed_dict) are not included in the
/// prelude, as they may cause ambiguities in some contexts.
pub mod prelude {
    pub use crate::array::*;
    pub use crate::bit_field_vec;
    pub use crate::bit_vec;
    pub use crate::bits::*;
    pub use crate::dict::*;
    pub use crate::func::*;
    pub use crate::rank_sel::*;
    pub use crate::rank_small;
    pub use crate::traits::bit_field_slice;
    pub use crate::traits::indexed_dict;
    pub use crate::traits::{iter::*, rank_sel::*};
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

/// Parallel iterators performing very fast operations, such as [zeroing a
/// bit](crate::bits::BitVec::reset) vector, should pass this argument to
/// [IndexedParallelIterator::with_min_len](`rayon::iter::IndexedParallelIterator::with_min_len`).
pub const RAYON_MIN_LEN: usize = 100_000;

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Index out of bounds: {} >= {}", $index, $len)
        }
    };
}
pub(crate) use panic_if_out_of_bounds;

macro_rules! panic_if_value {
    ($value: expr, $mask: expr, $bit_width: expr) => {
        if $value & $mask != $value {
            panic!("Value {} does not fit in {} bits", $value, $bit_width);
        }
    };
}
pub(crate) use panic_if_value;

macro_rules! debug_assert_bounds {
    ($index: expr, $len: expr) => {
        debug_assert!(
            $index < $len || ($index == 0 && $len == 0),
            "Index out of bounds: {} >= {}",
            $index,
            $len
        );
    };
}

pub(crate) use debug_assert_bounds;
