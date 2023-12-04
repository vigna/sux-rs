/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Support for rank and select operations.

## Design

Rank and selection structures can be combined arbitrarily with a mix-and-match approach.
The base class, usually [`BitVec`](crate::bits::bit_vec::BitVec), must implement the hinted
versions of the desired traits (i.e.,  [`RankHinted`](crate::traits::RankHinted),
[`SelectHinted`](crate::traits::SelectHinted), and [`SelectZeroHinted`](crate::traits::SelectZeroHinted)).
At that point, one can add an optional ranking structure, an optional selection structure, and an optional zero-selection
structure, in this order. The structures implement the ranking or selection traits by providing
a hint and then delegating to the base class. Moreover, each class forwards the methods it
does not implement to the next class in the chain.

A few of the structures in module are based on _broadword programming_ and have been described
by Sebastiano Vigna in “<a href="https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12">Broadword
Implementation of Rank/Select Queries</a>”, _Proc. of the 7th International Workshop
on Experimental Algorithms, WEA 2008_, volume 5038 of Lecture Notes in Computer Science, pages
154–168. Springer, 2008.

## Select

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
