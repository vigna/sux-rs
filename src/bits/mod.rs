/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Classes for [bit vectors](`bit_vec::BitVec`) and
[arrays of values of bounded bit width](`bit_field_vec::BitFieldVec`).

*/

pub mod bit_field_vec;
pub mod bit_vec;

pub mod prelude {
    pub use super::bit_field_vec::*;
    pub use super::bit_vec::*;
}
