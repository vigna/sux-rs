/*
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Array data structures.
//!
//! This module provides array-like data structures with specialized
//! characteristics for succinct data structure applications.

use mem_dbg::*;

use crate::bits::BitVec;
use crate::dict::elias_fano::EliasFano;
use crate::rank_sel::{Rank9, SelectZeroAdaptConst};
use crate::traits::SuccUnchecked;

pub mod partial_array;
pub use partial_array::{PartialArray, PartialArrayBuilder, new_dense, new_sparse};
pub mod partial_value_array;
pub use partial_value_array::{PartialValueArray, PartialValueArrayBuilder};

/// An internal index for sparse partial arrays.
///
/// We cannot use directly an [Elias–Fano](crate::dict::EliasFano) structure
/// because we need to keep track of the first invalid position; and we need to
/// keep track of the first invalid position because we want to implement just
/// [`SuccUnchecked`](crate::traits::SuccUnchecked) on the Elias–Fano structure,
/// because it requires just
/// [`SelectZeroUnchecked`](crate::traits::SelectZeroUnchecked), whereas
/// [`Succ`](crate::traits::Succ) would require
/// [`SelectUnchecked`](crate::traits::SelectUnchecked) as well.
#[doc(hidden)]
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseIndex<D> {
    ef: EliasFano<SelectZeroAdaptConst<BitVec<D>, D, 12, 3>>,
    /// self.ef should be not be queried for values >= self.first_invalid_position
    first_invalid_pos: usize,
}

impl<D: AsRef<[usize]>> SparseIndex<D> {
    /// Given a `position`, returns the first `(index, position2)` with `position2 >= position`
    ///
    fn get_next_pos(&self, position: usize) -> Option<(usize, usize)> {
        if position >= self.first_invalid_pos {
            return None;
        }

        // SAFETY: we just checked it
        Some(unsafe { self.ef.succ_unchecked::<false>(position) })
    }
}

type DenseIndex = Rank9<BitVec<Box<[usize]>>>;
