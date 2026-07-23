/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Support for rank and select operations.
//!
//! # Design
//!
//! Rank and selection structures can be combined arbitrarily with a
//! mix-and-match approach. There is a base class, usually
//! [`BitVec`], over which different structures can be layered. Each structure
//! forwards traits it does not implement to the next structure in the chain,
//! and also implements [`Deref`] with the next structure as target.
//!
//! A few of the structures in this module have been described by Sebastiano
//! Vigna in “[Broadword Implementation of Rank/Select Queries]”, _Proc. of the
//! 7th International Workshop on Experimental Algorithms, WEA 2008_, volume
//! 5038 of Lecture Notes in Computer Science, pages 154–168, Springer,
//! 2008.
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
//! let select = SelectAdapt::new(bv);
//!
//! assert_eq!(unsafe { select.select_unchecked(0) }, 1);
//! ```
//!
//! Note that we invoked [`select_unchecked`]. The [`select`] method,
//! indeed, requires the knowledge of the number of ones in the bit vector
//! to perform bound checks, and this number is not available in constant
//! time in a [`BitVec`]; we need [`AddNumBits`], a thin immutable wrapper
//! around a bit vector that stores internally the number of ones and thus
//! implements the [`NumBits`] trait:
//!
//! ```rust
//! use sux::bit_vec;
//! use sux::rank_sel::SelectAdapt;
//! use sux::traits::{AddNumBits, Select};
//!
//! let bv: AddNumBits<_> = bit_vec![0, 1, 0, 1, 1, 0, 1, 0].into();
//! let select = SelectAdapt::new(bv);
//!
//! assert_eq!(select.select(0), Some(1));
//! ```
//!
//! Suppose instead we want to build our selection structure around a [`Rank9`]
//! structure: in this case, [`Rank9`] implements directly [`NumBits`], so
//! we can just use it:
//!
//! ```rust
//! # #[cfg(target_pointer_width = "64")]
//! # {
//! use sux::{bit_vec, rank_small};
//! use sux::rank_sel::{Rank9, SelectAdapt};
//! use sux::traits::{Rank, Select};
//!
//! let bv = bit_vec![0, 1, 0, 1, 1, 0, 1, 0];
//! let sel_rank9 = SelectAdapt::new(Rank9::new(bv));
//!
//! assert_eq!(sel_rank9.select(0), Some(1));
//! assert_eq!(sel_rank9.rank(4), 2);
//! assert!(!sel_rank9[0]);
//! assert!(sel_rank9[1]);
//!
//! let sel_rank_small = unsafe { sel_rank9.map(|x| rank_small![4; x.into_inner()]) };
//! # }
//! ```
//!
//! Note how [`SelectAdapt`] forwards not only [`Rank`] but also [`Index`],
//! which gives access to the bits of the underlying bit vector. The last
//! line uses the [`map`] method to replace the underlying [`Rank9`]
//! structure with one that is slower but uses much less space: the method
//! is unsafe because in principle you might replace the structure with
//! something built on a different bit vector, leading to an inconsistent
//! state; note how we use `into_inner()` to get rid of the [`Rank9`]
//! wrapper.
//!
//! Some structures depend on the internals of others, and thus cannot be
//! composed freely: for example, a [`Select9`] must necessarily wrap a
//! [`Rank9`]. In general, in any case, we suggest embedding structures in the
//! order rank, select, and zero select, from inner to outer, because ranking
//! structures usually implement [`NumBits`].
//!
//! [`BitVec`]: crate::bits::bit_vec::BitVec
//! [Broadword Implementation of Rank/Select Queries]: https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12
//! [`select_unchecked`]: crate::traits::SelectUnchecked::select_unchecked
//! [`select`]: crate::traits::Select::select
//! [`AddNumBits`]: crate::traits::AddNumBits
//! [`NumBits`]: crate::traits::NumBits
//! [`Rank`]: crate::traits::Rank
//! [`Deref`]: core::ops::Deref
//! [`Index`]: std::ops::Index
//! [`map`]: SelectAdapt::map

pub mod select_adapt;
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

mod select_zero_small;
pub use select_zero_small::*;

mod rank9;
pub use rank9::*;

mod select9;
pub use select9::*;

use crate::traits::Word;

/// Returns the mask that keeps only the low `residual` logical bits of a word,
/// or `W::MAX` when `residual == 0` (the last word is full, nothing to mask).
///
/// A valid [`BitVec`](crate::bits::BitVec) may legally carry arbitrary bits
/// past its logical length (its own `count_ones` masks them), so rank/select
/// builders that sum raw word populations must mask the final word or they
/// over-count. `residual` is `len % W::BITS`. This is computed once per build
/// and fed to [`mask_tail_word`] inside the counting loop.
#[inline(always)]
pub(crate) fn tail_mask<W: Word>(residual: usize) -> W {
    if residual == 0 {
        W::MAX
    } else {
        W::MAX >> (W::BITS as usize - residual)
    }
}

/// Clears the dirty padding bits of the final word without a data-dependent
/// branch.
///
/// Returns `word & tail_mask` for the last word and `word` unchanged
/// otherwise. Both arms of the selection are trivial, so it lowers to a
/// conditional move rather than a branch: the counting loops call this once
/// per word, and masking a full mid-vector word with `W::MAX` is a no-op.
/// `tail_mask` is [`tail_mask`] precomputed outside the loop.
#[inline(always)]
pub(crate) fn mask_tail_word<W: Word>(word: W, is_last: bool, tail_mask: W) -> W {
    word & if is_last { tail_mask } else { W::MAX }
}
