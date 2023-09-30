/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Classes for [bit vectors](`bit_vec::BitVec`) and
[arrays of values of bounded bit width](`compact_array::CompactArray`).

*/

pub mod bit_vec;
pub mod compact_array;

pub mod prelude {
    pub use super::bit_vec::*;
    pub use super::compact_array::*;
}
