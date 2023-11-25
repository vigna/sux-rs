/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Support for rank and select operations.

Selection is supported by means of structures implementing the
[`Select`](crate::traits::Select)/[`SelectZero`](crate::traits::SelectZero),
usually adding an index to structures, such as [`BitVec`](crate::bits::bit_vec::BitVec),
which implement [`SelectHinted`](crate::traits::Select)/[`SelectZeroHinted`](crate::traits::SelectZero).

The "fixed" selection structures are very simple and fast, but they require that
the bit vector has a reasonably uniform distribution of zeroes and ones. The default parameters
are a good choice for a vector with approximately the same number of zeroes and ones, such as
the high bits of the [Elias–Fano representation of monotone sequences](crate::dict::elias_fano::EliasFano),
but they can be tuned for other densities.

*/
mod select_fixed1;
pub use select_fixed1::*;

mod select_zero_fixed1;
pub use select_zero_fixed1::*;

mod select_fixed2;
pub use select_fixed2::*;

mod select_zero_fixed2;
pub use select_zero_fixed2::*;
