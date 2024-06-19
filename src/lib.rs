/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![deny(unconditional_recursion)]
#![allow(clippy::duplicated_attributes)]

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
    pub use crate::bits::*;
    pub use crate::dict::*;
    pub use crate::rank_sel::*;
    pub use crate::traits::bit_field_slice;
    pub use crate::traits::*;
    pub use crate::utils::*;
}

macro_rules! forward_mult {
    ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident; $macro:path $(, $macros:path)*) => {
		$macro![$name < $( $([$const])? $generic $(:$t)? ),* >; $type; $field];
		crate::forward_mult![$name < $( $([$const])? $generic $(:$t)? ),* >; $type; $field; $($macros),* ];
	};

	($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident; ) => {}
}

pub(crate) use forward_mult;

macro_rules! forward_as_ref_slice_usize {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > std::convert::AsRef<[usize]> for $name < $($generic,)* > where $type: std::convert::AsRef<[usize]> {
            #[inline(always)]
            fn as_ref(&self) -> &[usize] {
                std::convert::AsRef::<[usize]>::as_ref(&self.$field)
            }
        }
    };
}

pub(crate) use forward_as_ref_slice_usize;

macro_rules! forward_index_bool {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > std::ops::Index<usize> for $name < $($generic,)* > where $type: std::ops::Index<usize, Output = bool> {
            type Output = bool;

            #[inline(always)]
            fn index(&self, index: usize) -> &Self::Output {
                std::ops::Index::<usize>::index(&self.$field, index)
            }
        }
    };
}

pub(crate) use forward_index_bool;
