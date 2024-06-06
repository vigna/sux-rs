/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![deny(unconditional_recursion)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("`target_pointer_width` must be 64");

pub mod bits;
pub mod dict;
pub mod func;
pub mod rank_sel;
pub mod traits;
pub mod utils;

#[cfg(feature = "fuzz")]
pub mod fuzz;

pub mod prelude {
    pub use crate::bits::*;
    pub use crate::dict::*;
    pub use crate::func::*;
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
