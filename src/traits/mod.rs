/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Main traits for succinct data structures.

*/

pub mod bit_field_slice;
pub use bit_field_slice::BitFieldSliceCore;
pub use bit_field_slice::BitFieldSliceIterator;

pub mod convert_to;
pub use convert_to::*;

pub mod indexed_dict;
pub use indexed_dict::*;

pub mod iter;
pub use iter::*;

pub mod rank_sel;
pub use rank_sel::*;
