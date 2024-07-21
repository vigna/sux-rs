/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Support for rank and select operations.
//!
//! # Design
//!
//! Rank and selection structures can be combined arbitrarily with a
//! mix-and-match approach. There is a base class, usually
//! [`BitVec`](crate::bits::bit_vec::BitVec), over which different structures
//! can be layered. Each structure forwards the methods it does not implement to
//! the next structure in the chain.
//!
//! A few of the structures in module are based on _broadword programming_ and
//! have been described by Sebastiano Vigna in “[Broadword Implementation of
//! Rank/Select
//! Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
//! _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
//! 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
//! Springer, 2008.
//!
//! # Examples
//!
//! Assuming we want to implement selection over a bit vector, we could do as
//! follows:
//!
//! ```rust
//! use sux::bit_vec;
//! use sux::rank_sel::SelectAdapt;
//! use sux::traits::SelectUnchecked;
//!
//! let bv = bit_vec![0, 1, 0, 1, 1, 0, 1, 0];
//! let select = SelectAdapt::new(bv, 3);
//!
//! assert_eq!(unsafe { select.select_unchecked(0) }, 1);
//! ```
//!
//! Note that we invoked
//! [`select_unchecked`](crate::traits::SelectUnchecked::select_unchecked). The
//! [`select`](crate::traits::Select::select) method, indeed, requires the
//! knowledge of the number of ones in the bit vector to perform bound checks,
//! and this number is not available in constant time in a
//! [`BitVec`](crate::bits::BitVec); we need
//! [`AddNumBits`](crate::traits::AddNumBits), a thin immutable wrapper around a
//! bit vector that stores internally the number of ones and thus implements the
//! [`NumBits`](crate::traits::NumBits) trait:
//!
//! ```rust
//! use sux::bit_vec;
//! use sux::rank_sel::SelectAdapt;
//! use sux::traits::{AddNumBits, Select};
//!
//! let bv: AddNumBits<_> = bit_vec![0, 1, 0, 1, 1, 0, 1, 0].into();
//! let select = SelectAdapt::new(bv, 3);
//!
//! assert_eq!(select.select(0), Some(1));
//! ```
//!
//! Suppose instead we want to build our selection structure around a [`Rank9`]
//! structure: in this case, [`Rank9`] implements directly
//! [`NumBits`](crate::traits::NumBits), so we can just use it:
//!
//! ```rust
//! use sux::{bit_vec, rank_small};
//! use sux::rank_sel::{Rank9, SelectAdapt};
//! use sux::traits::{Rank, Select};
//!
//! let bv = bit_vec![0, 1, 0, 1, 1, 0, 1, 0];
//! let sel_rank9 = SelectAdapt::new(Rank9::new(bv), 3);
//!
//! assert_eq!(sel_rank9.select(0), Some(1));
//! assert_eq!(sel_rank9.rank(4), 2);
//! assert!(!sel_rank9[0]);
//! assert!(sel_rank9[1]);
//!
//! let sel_rank_small = unsafe { sel_rank9.map(|x| rank_small![4; x.into_inner()]) };
//! ```
//!
//! Note how [`SelectAdapt`] forwards not only [`Rank`](crate::traits::Rank) but
//! also [`Index`](std::ops::Index), which gives access to the bits of the
//! underlying bit vector. The last line uses the [`map`](SelectAdapt::map)
//! method to replace the underlying [`Rank9`] structure with one that is slower
//! but uses much less space: the method is unsafe because in principle you
//! might replace the structure with something built on a different bit vector,
//! leading to an inconsistent state; note how we use `into_inner()` to get rid
//! of the [`AddNumBits`](crate::traits::AddNumBits) wrapper.
//!
//! Some structures depend on the internals of others, and thus cannot be
//! composed freely: for example, a [`Select9`] must necessarily wrap a
//! [`Rank9`]. In general, in any case, we suggest embedding structure in the
//! order rank, select, and zero select, from inner to outer, because ranking
//! structures usually implement [`NumBits`](crate::traits::NumBits).

mod select_adapt;
pub use select_adapt::*;

mod select_zero_adapt;
pub use select_zero_adapt::*;

mod select_adapt_const;
pub use select_adapt_const::*;

mod select_zero_adapt_const;
pub use select_zero_adapt_const::*;

mod rank_small;
pub use rank_small::*;

mod select_small;
pub use select_small::*;

mod rank9;
pub use rank9::*;

mod select9;
pub use select9::*;

mod simple_select;
pub use simple_select::*;
