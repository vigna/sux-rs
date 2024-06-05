/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Basic traits for succinct operations on bit vectors, including [`Rank`] and [`Select`].

*/

/// A trait for succinct data structures that expose the
/// length of the underlying bit vector.
#[allow(clippy::len_without_is_empty)]
pub trait BitLength {
    /// Return the length in bits of the underlying bit vector.
    fn len(&self) -> usize;
}

#[macro_export]
macro_rules! forward_bit_length {
        ($name:ident
        <
            $( $([$const:ident])? $generic:ident $(:$t:ty)?  ),*
        >;
        $type:ident;
        $field:ident) => {
        impl <$( $($const)? $generic $(:$t)?,)* > BitLength for $name < $($generic,)* > where $type: BitLength {
            #[inline(always)]
            fn len(&self) -> usize {
                    BitLength::len(&self.$field)
                }
            }
    };
}

/// A trait for succinct data structures that expose the
/// numer of ones and zeros of the underlying bit vector.
pub trait BitCount: BitLength {
    /// Return the number of ones in the underlying bit vector.
    fn count_ones(&self) -> usize;
    /// Return the number of zeros in the underlying bit vector.
    #[inline(always)]
    fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }
}

#[macro_export]
macro_rules! forward_bit_count {
        ($name:ident
        <
            $( $([$const:ident])? $generic:ident $(:$t:ty)?  ),*
        >;
        $type:ident;
        $field:ident) => {
        impl < $( $($const)? $generic $(:$t)?,)* > BitCount for $name < $($generic,)* > where $type: BitCount {
            #[inline(always)]
            fn count_ones(&self) -> usize {
                BitCount::count_ones(&self.$field)
            }
            #[inline(always)]
            fn count_zeros(&self) -> usize {
                BitCount::count_zeros(&self.$field)
            }
        }
    };
}

/// Rank over a bit vector.
pub trait Rank: BitLength {
    /// Return the number of ones preceding the specified position.
    ///
    /// The bit vector is virtually zero-extended. If `pos` is greater than or equal to the
    /// [length of the underlying bit vector](`BitLength::len`), the number of
    /// ones in the underlying bit vector is returned.
    fn rank(&self, pos: usize) -> usize {
        unsafe { self.rank_unchecked(pos.min(self.len())) }
    }

    /// Return the number of ones preceding the specified position.
    ///
    /// # Safety
    /// `pos` must be between 0 (included) and the [length of the underlying bit
    /// vector](`BitLength::len`) (included).
    unsafe fn rank_unchecked(&self, pos: usize) -> usize;
}

/// Rank zeros over a bit vector.
pub trait RankZero: Rank {
    /// Return the number of zeros preceding the specified position.
    fn rank_zero(&self, pos: usize) -> usize {
        pos - self.rank(pos)
    }
    /// Return the number of zeros preceding the specified position.
    ///
    /// # Safety
    /// `pos` must be between 0 and the [length of the underlying bit
    /// vector](`BitLength::len`) (included).
    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        pos - self.rank_unchecked(pos)
    }
}

/// Rank over a bit vector, with a hint.
///
/// This trait is used to implement fast ranking by adding to bit vectors
/// counters of different kind.
pub trait RankHinted<const HINT_BIT_SIZE: usize> {
    /// Return the number of ones preceding the specified position,
    /// provided a preceding position and its associated rank.
    ///
    /// # Safety
    /// `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included).
    /// `hint_pos` * `HINT_BIT_SIZE` must be between 0 (included) and
    /// `pos` (included).
    /// `hint_rank` must be the number of ones in the underlying bit vector
    /// before `hint_pos` * `HINT_BIT_SIZE`.
    unsafe fn rank_hinted_unchecked(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> usize;
    /// Return the number of ones preceding the specified position,
    /// provided a preceding position `hint_pos` * `HINT_BIT_SIZE` and
    /// the associated rank.
    fn rank_hinted(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> Option<usize>;
}

/// Select over a bit vector.
pub trait Select: BitCount {
    /// Return the position of the one of given rank, or `None` if no such
    /// bit exist.
    fn select(&self, rank: usize) -> Option<usize> {
        if rank >= self.count_ones() {
            None
        } else {
            Some(unsafe { self.select_unchecked(rank) })
        }
    }

    /// Return the position of the one of given rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded).
    unsafe fn select_unchecked(&self, rank: usize) -> usize;
}

/// Select zeros over a bit vector.
pub trait SelectZero: BitLength + BitCount {
    /// Return the position of the zero of given rank, or `None` if no such
    /// bit exist.
    fn select_zero(&self, rank: usize) -> Option<usize> {
        if rank >= self.count_zeros() {
            None
        } else {
            Some(unsafe { self.select_zero_unchecked(rank) })
        }
    }

    /// Return the position of the zero of given rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded).
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize;
}

/// Select over a bit vector, with a hint.
///
/// This trait is used to implement fast selection by adding to bit vectors
/// indices of different kind. See, for example,
/// [`SimpleSelect`](crate::rank_sel::SimpleSelect).
pub trait SelectHinted {
    /// Select the one of given rank, provided the position of a preceding one
    /// and its rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded). `hint_pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a one in the underlying bit vector.
    /// `hint_rank` must be the number of ones in the underlying bit vector
    /// before `hint_pos`, and must be less than or equal to `rank`.
    unsafe fn select_hinted_unchecked(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;
    /// Select the one of given rank, provided the position of a preceding one
    /// and its rank.
    fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize>;
}

/// Select zeros over a bit vector, with a hint.
///
/// This trait is used to implement fast selection over zeros by adding to bit
/// vectors indices of different kind.
pub trait SelectZeroHinted {
    /// Select the zero of given rank, provided the position of a preceding zero
    /// and its rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded). `hint_pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a zero in the underlying bit vector.
    /// `hint_rank` must be the number of zeros in the underlying bit vector
    /// before `hint_pos`, and must be less than or equal to `rank`.
    unsafe fn select_zero_hinted_unchecked(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;

    /// Select the zero of given rank, provided the position of a preceding zero
    /// and its rank.
    fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize>;
}
