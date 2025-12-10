/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Main traits used in the implementation of succinct and compressed data
//! structures.
//!
//! See the discussion in [`bit_field_slice`] about the re-export of its traits.

pub mod bit_field_slice;
pub use bit_field_slice::*;

pub mod indexed_dict;
pub use indexed_dict::*;

pub mod iter;
pub use iter::*;

pub mod rank_sel;
pub use rank_sel::*;

pub mod bit_vec_ops;
pub use bit_vec_ops::*;
