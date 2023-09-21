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
pub mod convert_to;
pub mod indexed_dict;
pub mod iter;
pub mod rank_sel;

pub mod prelude {
    pub use super::bit_field_slice::*;
    pub use super::convert_to::*;
    pub use super::indexed_dict::*;
    pub use super::iter::*;
    pub use super::rank_sel::*;
}
