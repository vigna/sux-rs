/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Basic traits for bit vectors, including [`Rank`] and [`Select`].

*/

/// A trait for succinct data structures that expose the
/// length of the underlying bit vector.
#[allow(clippy::len_without_is_empty)]
pub trait BitLength {
    /// Return the length in bits of the underlying bit vector.
    fn len(&self) -> usize;
}

/// A trait for succinct data structures that expose the
/// numer of ones of the underlying bit vector.
pub trait BitCount {
    /// Return the number of ones in the underlying bit vector.
    fn count(&self) -> usize;
}

/// Rank over a bit vector.
pub trait Rank: BitLength {
    /// Return the number of ones preceding the specified position.
    ///
    /// # Arguments
    /// * `pos` : `usize` - The position to query.
    fn rank(&self, pos: usize) -> usize {
        unsafe { self.rank_unchecked(pos.min(self.len())) }
    }

    /// Return the number of ones preceding the specified position.
    ///
    /// # Arguments
    /// * `pos` : `usize` - The position to query; see Safety below for valid values.
    ///
    /// # Safety
    ///
    /// `pos` must be between 0 (included) and the [length of the underlying bit
    /// vector](`BitLength::len`) (included).
    unsafe fn rank_unchecked(&self, pos: usize) -> usize;
}

/// Rank zeros over a bit vector.
pub trait RankZero: Rank {
    /// Return the number of zeros preceding the specified position.
    ///
    /// # Arguments
    /// * `pos` : `usize` - The position to query.
    fn rank_zero(&self, pos: usize) -> usize {
        pos - self.rank(pos)
    }
    /// Return the number of zeros preceding the specified position.
    ///
    /// # Arguments
    /// * `pos` : `usize` - The position to query; see Safety below for valid values.
    ///
    /// # Safety
    ///
    /// `pos` must be between 0 and the [length of the underlying bit
    /// vector](`BitLength::len`) (included).
    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        pos - self.rank_unchecked(pos)
    }
}

/// Select over a bit vector.
pub trait Select: BitCount {
    /// Return the position of the one of given rank.
    ///
    /// # Arguments
    /// * `rank` : `usize` - The rank to query. If there is no
    /// one of given rank, this function return `None`.
    fn select(&self, rank: usize) -> Option<usize> {
        if rank >= self.count() {
            None
        } else {
            Some(unsafe { self.select_unchecked(rank) })
        }
    }

    /// Return the position of the one of given rank.
    ///
    /// # Arguments
    /// * `rank` : `usize` - The rank to query; see Safety below for valid values.
    ///
    /// # Safety
    ///
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded).
    unsafe fn select_unchecked(&self, rank: usize) -> usize;
}

/// Select zeros over a bit vector.
pub trait SelectZero: BitLength + BitCount {
    /// Return the position of the zero of given rank.
    ///
    /// # Arguments
    /// * `rank` : `usize` - The rank to query. If there is no
    /// zero of given rank, this function return `None`.
    fn select_zero(&self, rank: usize) -> Option<usize> {
        if rank >= self.len() - self.count() {
            None
        } else {
            Some(unsafe { self.select_zero_unchecked(rank) })
        }
    }

    /// Return the position of the zero of given rank.
    ///
    /// # Arguments
    /// * `rank` : `usize` - The rank to query; see Safety below for valid values
    ///
    /// # Safety
    ///
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded).
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize;
}

pub trait SelectHinted: Select {
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded). `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a one in the underlying bit vector.
    /// `rank_at_pos` must be the number of ones in the underlying bit vector
    /// before `pos`.
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize;
}

pub trait SelectZeroHinted: SelectZero {
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded). `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a zero in the underlying bit vector.
    /// `rank_at_pos` must be the number of zeros in the underlying bit vector
    /// before `pos`.
    unsafe fn select_zero_hinted_unchecked(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize;
}
