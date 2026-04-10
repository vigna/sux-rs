/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Basic traits for succinct operations on bit vectors, including [`Rank`] and
//! [`Select`].
//!
//! All traits in this module are automatically implemented for references,
//! mutable references, and boxes. Moreover, usually they are all forwarded to
//! underlying implementations.

use crate::ambassador_impl_Index;
use crate::traits::Backend;
use ambassador::{Delegate, delegatable_trait};
use impl_tools::autoimpl;
use mem_dbg::{MemDbg, MemSize};
use std::ops::Deref;
use std::ops::Index;

use crate::traits::ambassador_impl_Backend;

use crate::traits::bit_vec_ops::{BitLength, ambassador_impl_BitLength};

/// Potentially expensive bit-counting methods.
///
/// The methods in this trait compute the number of ones or zeros
/// in a bit vector (possibly underlying a succinct data structure).
/// The computation can be expensive: if you need a constant-time
/// version, use [`NumBits`]. If you need to cache the result
/// of these methods, you can use [`AddNumBits`].
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait BitCount: BitLength {
    /// Returns the number of ones in the underlying bit vector,
    /// with a possibly expensive computation; see [`NumBits::num_ones`]
    /// for constant-time version.
    fn count_ones(&self) -> usize;
    /// Returns the number of zeros in the underlying bit vector,
    /// with a possibly expensive computation; see [`NumBits::num_zeros`]
    /// for constant-time version.
    #[inline(always)]
    fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }
}

/// Constant-time bit-counting methods.
///
/// The methods in this trait compute the number of ones or zeros
/// in a bit vector (possibly underlying a succinct data structure)
/// in constant time. If you can be content with a potentially
/// expensive computation, use [`BitCount`].
///
/// If you need to implement this trait on a structure that already
/// implements [`BitCount`], you can use [`AddNumBits`].
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait NumBits: BitLength {
    /// Returns the number of ones in the underlying bit vector
    /// in constant time. If you can be content with a potentially
    /// expensive computation, use [`BitCount::count_ones`].
    fn num_ones(&self) -> usize;
    /// Returns the number of zeros in the underlying bit vector
    /// in constant time. If you can be content with a potentially
    /// expensive computation, use [`BitCount::count_zeros`].
    #[inline(always)]
    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }
}

/// Ranking over a bit vector.
///
/// # Examples
///
/// ```rust
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// # use sux::prelude::*;
/// let rank9 = Rank9::new(bit_vec![1, 0, 1, 1, 0, 1, 0, 1]);
/// assert_eq!(rank9.rank(0), 0);
/// assert_eq!(rank9.rank(4), 3);
/// assert_eq!(rank9.rank(8), 5);
/// # }
/// ```
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait Rank: BitLength + NumBits + RankUnchecked {
    /// Returns the number of ones preceding the specified position.
    ///
    /// The bit vector is virtually zero-extended. If `pos` is greater than or equal to the
    /// [length of the underlying bit vector](`BitLength::len`), the number of
    /// ones in the underlying bit vector is returned.
    #[inline(always)]
    fn rank(&self, pos: usize) -> usize {
        if pos >= self.len() {
            self.num_ones()
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }
}

/// Rank over a bit vector without bounds checks.
///
/// This is the unchecked counterpart of [`Rank`], providing the core
/// [`rank_unchecked`](RankUnchecked::rank_unchecked) operation. Use [`Rank`]
/// for a checked version that verifies the position is within bounds.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait RankUnchecked {
    /// Returns the number of ones preceding the specified position.
    ///
    /// # Safety
    /// `pos` must be between 0 (included) and the [length of the underlying bit
    /// vector](`BitLength::len`) (excluded).
    ///
    /// Some implementation might accept the length as a valid argument. If
    /// you need to be sure that the length is a valid argument, just
    /// add a padding zero bit at the end of your vector (at which
    /// point the original length will fall within the valid range).
    unsafe fn rank_unchecked(&self, pos: usize) -> usize;

    /// Prefetches the cache lines needed to compute
    /// [`rank_unchecked(pos)](#tymethod.rank_unchecked).
    ///
    /// This can speed up computing the rank of many positions in parallel.
    ///
    /// # Examples
    ///
    /// For example, take the following for loop:
    /// ```
    /// # use sux::prelude::RankUnchecked;
    /// fn query_all(rank: &impl RankUnchecked, positions: &[usize]) {
    ///    for i in 0..positions.len() {
    ///        let r = unsafe { rank.rank_unchecked(positions[i]) };
    ///        // ...
    ///    }
    /// }
    /// ```
    /// By prefetching cache lines some iterations ahead, we can make sure that
    /// they are already loaded in memory by the time we get to that loop iteration:
    /// ```
    /// # use sux::prelude::RankUnchecked;
    /// fn query_all(rank: &impl RankUnchecked, positions: &[usize]) {
    ///    for i in 0..positions.len() {
    ///        rank.prefetch(positions[(i + 32).min(positions.len() - 1)]);
    ///        let r = unsafe { rank.rank_unchecked(positions[i]) };
    ///        // ...
    ///    }
    /// }
    /// ```
    ///
    /// For [`Rank9`](crate::rank_sel::Rank9) and
    /// [`RankSmall`](crate::rank_sel::RankSmall), this gives around 10% to 30%
    /// speedup when there are 16 billion keys.
    ///
    /// Prefetching out-of-bounds is never unsafe, and neither is this method.
    fn prefetch(&self, _pos: usize) {
        // Default implementation is no-op
    }
}

/// Ranking zeros over a bit vector.
///
/// Note that this is just an extension trait for [`Rank`].
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait RankZero: Rank {
    /// Returns the number of zeros preceding the specified position.
    ///
    /// The bit vector is virtually zero-extended. If `pos` is greater than or
    /// equal to the [length of the underlying bit vector](`BitLength::len`),
    /// the `pos` minus the number of ones in the underlying bit vector is
    /// returned.
    fn rank_zero(&self, pos: usize) -> usize {
        pos - self.rank(pos)
    }

    /// Returns the number of zeros preceding the specified position.
    ///
    /// # Safety
    /// `pos` must be between 0 and the [length of the underlying bit
    /// vector](`BitLength::len`) (excluded).
    ///
    /// Some implementation might accept the length as a valid argument.
    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        pos - unsafe { self.rank_unchecked(pos) }
    }
}

/// Ranking over a bit vector, with a hint.
///
/// This trait is used to implement fast ranking by adding to bit vectors
/// counters of different kind.
///
/// The hint position is expressed as a multiple of the word bit size.
///
/// The const parameter `WORDS_PER_SUBBLOCK` bounds the maximum number of
/// words that will be scanned from `hint_pos` to reach `pos`. Knowing this
/// bound at compile time lets the compiler fully unroll the inner loop.
///
/// If the bound is not known at compile time, pass `usize::MAX` to fall back to
/// an unbounded scan.
#[delegatable_trait]
pub trait RankHinted {
    /// Returns the number of ones preceding the specified position,
    /// provided a preceding position and its associated rank.
    ///
    /// The hinted position, `hint_pos`, is expressed as a word index
    /// (i.e., a multiple of the word bit size). This parameter is necessary
    /// as some rank implementations can accept only hints at specific
    /// positions (usually, multiples of the word size).
    ///
    /// # Safety
    ///
    /// `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (excluded).
    /// `hint_pos` must be between 0 (included) and
    /// `pos` (included), expressed in words.
    /// `hint_rank` must be the number of ones in the underlying bit vector
    /// before the bit at the start of word `hint_pos`.
    ///
    /// Some implementation might accept the length as a valid argument.
    unsafe fn rank_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        pos: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;
}

impl<T: RankHinted + ?Sized> RankHinted for &T {
    #[inline(always)]
    unsafe fn rank_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        pos: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).rank_hinted::<WORDS_PER_SUBBLOCK>(pos, hint_pos, hint_rank) }
    }
}

impl<T: RankHinted + ?Sized> RankHinted for &mut T {
    #[inline(always)]
    unsafe fn rank_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        pos: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).rank_hinted::<WORDS_PER_SUBBLOCK>(pos, hint_pos, hint_rank) }
    }
}

impl<T: RankHinted + ?Sized> RankHinted for Box<T> {
    #[inline(always)]
    unsafe fn rank_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        pos: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).rank_hinted::<WORDS_PER_SUBBLOCK>(pos, hint_pos, hint_rank) }
    }
}

/// Selection over a bit vector without bound checks.
///
/// This is the unchecked counterpart of [`Select`], providing the core
/// [`select_unchecked`](SelectUnchecked::select_unchecked) operation. Use [`Select`]
/// for a checked version that verifies that there is a one of the given rank.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait SelectUnchecked {
    /// Returns the position of the one of given rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded).
    unsafe fn select_unchecked(&self, rank: usize) -> usize;
}

/// Selection over a bit vector.
///
/// # Examples
///
/// ```rust
/// # use sux::prelude::*;
/// // SelectAdapt needs NumBits
/// let bits: AddNumBits<_> = bit_vec![1, 0, 1, 1, 0, 1, 0, 1].into();
/// let sel = SelectAdapt::new(bits);
/// assert_eq!(sel.select(0), Some(0));
/// assert_eq!(sel.select(2), Some(3));
/// assert_eq!(sel.select(4), Some(7));
/// assert_eq!(sel.select(5), None);
/// ```
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait Select: SelectUnchecked + NumBits {
    /// Returns the position of the one of given rank, or `None` if no such
    /// bit exists.
    fn select(&self, rank: usize) -> Option<usize> {
        if rank >= self.num_ones() {
            None
        } else {
            Some(unsafe { self.select_unchecked(rank) })
        }
    }
}

/// Selection of zeros over a bit vector without bound checks.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait SelectZeroUnchecked {
    /// Returns the position of the zero of given rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded).
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize;
}

/// Selection of zeros over a bit vector.
///
/// # Examples
///
/// ```rust
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// # use sux::prelude::*;
/// let sel = SelectZeroAdapt::new(Rank9::new(bit_vec![1, 0, 1, 1, 0, 1, 0, 1]));
/// assert_eq!(sel.select_zero(0), Some(1));
/// assert_eq!(sel.select_zero(1), Some(4));
/// assert_eq!(sel.select_zero(2), Some(6));
/// assert_eq!(sel.select_zero(3), None);
/// # }
/// ```
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait SelectZero: SelectZeroUnchecked + NumBits {
    /// Returns the position of the zero of given rank, or `None` if no such
    /// bit exists.
    fn select_zero(&self, rank: usize) -> Option<usize> {
        if rank >= self.num_zeros() {
            None
        } else {
            Some(unsafe { self.select_zero_unchecked(rank) })
        }
    }
}

/// Selection over a bit vector, with a hint.
///
/// This trait is used to implement fast selection by adding to bit vectors
/// indices of different kind. See, for example,
/// [`SelectAdapt`](crate::rank_sel::SelectAdapt).
///
/// The const parameter `WORDS_PER_SUBBLOCK` bounds the maximum number of words
/// scanned from `hint_pos`. If the bound is not known (as it is typical in
/// selection), pass `usize::MAX` to fall back to an unbounded scan.
#[delegatable_trait]
pub trait SelectHinted {
    /// Selects the one of given rank, provided the position of a preceding one
    /// and its rank.
    ///
    /// # Safety
    ///
    /// `rank` must be between zero (included) and the number of ones
    /// in the underlying bit vector (excluded). `hint_pos` must be between 0
    /// (included) and the [length of the underlying bit
    /// vector](`BitLength::len`) (included), and must be the position of a one
    /// in the underlying bit vector. `hint_rank` must be the number of ones in
    /// the underlying bit vector before `hint_pos`, and must be less than or
    /// equal to `rank`.
    unsafe fn select_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;
}

impl<T: SelectHinted + ?Sized> SelectHinted for &T {
    #[inline(always)]
    unsafe fn select_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).select_hinted::<WORDS_PER_SUBBLOCK>(rank, hint_pos, hint_rank) }
    }
}

impl<T: SelectHinted + ?Sized> SelectHinted for &mut T {
    #[inline(always)]
    unsafe fn select_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).select_hinted::<WORDS_PER_SUBBLOCK>(rank, hint_pos, hint_rank) }
    }
}

impl<T: SelectHinted + ?Sized> SelectHinted for Box<T> {
    #[inline(always)]
    unsafe fn select_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).select_hinted::<WORDS_PER_SUBBLOCK>(rank, hint_pos, hint_rank) }
    }
}

/// Selection of zeros over a bit vector, with a hint.
///
/// This trait is used to implement fast selection over zeros by adding to bit
/// vectors indices of different kind.
///
/// The const parameter `WORDS_PER_SUBBLOCK` bounds the maximum number of words
/// scanned from `hint_pos`. If the bound is not known (as it is typical in
/// selection), pass `usize::MAX` to fall back to an unbounded scan.
#[delegatable_trait]
pub trait SelectZeroHinted {
    /// Selects the zero of given rank, provided the position of a preceding zero
    /// and its rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded). `hint_pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a zero in the underlying bit vector.
    /// `hint_rank` must be the number of zeros in the underlying bit vector
    /// before `hint_pos`, and must be less than or equal to `rank`.
    unsafe fn select_zero_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;
}

impl<T: SelectZeroHinted + ?Sized> SelectZeroHinted for &T {
    #[inline(always)]
    unsafe fn select_zero_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).select_zero_hinted::<WORDS_PER_SUBBLOCK>(rank, hint_pos, hint_rank) }
    }
}

impl<T: SelectZeroHinted + ?Sized> SelectZeroHinted for &mut T {
    #[inline(always)]
    unsafe fn select_zero_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).select_zero_hinted::<WORDS_PER_SUBBLOCK>(rank, hint_pos, hint_rank) }
    }
}

impl<T: SelectZeroHinted + ?Sized> SelectZeroHinted for Box<T> {
    #[inline(always)]
    unsafe fn select_zero_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        unsafe { (**self).select_zero_hinted::<WORDS_PER_SUBBLOCK>(rank, hint_pos, hint_rank) }
    }
}

/// A thin wrapper implementing [`NumBits`] by caching the result of
/// [`BitCount::count_ones`].
///
/// This structure forwards to the wrapped structure all traits defined in [this
/// module](crate::traits::rank_sel) except for [`NumBits`] and [`BitCount`]. It
/// is typically used to provide [`NumBits`] to [`Select`]/[`SelectZero`]
/// implementations; see,
/// for example, [`SelectAdapt`](crate::rank_sel::SelectAdapt).
///
/// # Examples
///
/// ```rust
/// # use sux::prelude::*;
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 0, 1];
/// let bits: AddNumBits<_> = bits.into();
/// assert_eq!(bits.num_ones(), 5);
/// assert_eq!(bits.num_zeros(), 4);
/// ```
#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::Backend, target = "bits")]
#[delegate(crate::traits::bit_vec_ops::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::Rank, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::RankUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::RankZero, target = "bits")]
#[delegate(crate::traits::rank_sel::Select, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "bits")]
pub struct AddNumBits<B> {
    bits: B,
    number_of_ones: usize,
}

impl<B> AddNumBits<B> {
    /// Returns the underlying bit structure.
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Creates a new `AddNumBits` from raw parts.
    ///
    /// # Safety
    ///
    /// `number_of_ones` must be the actual number of ones in `bits`. No
    /// validation is performed to verify this invariant.
    #[inline(always)]
    pub const unsafe fn from_raw_parts(bits: B, number_of_ones: usize) -> Self {
        Self {
            bits,
            number_of_ones,
        }
    }

    /// Decomposes this `AddNumBits` into its raw parts.
    ///
    /// Returns a tuple containing the underlying bit structure and the cached
    /// number of ones.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.bits, self.number_of_ones)
    }
}

impl<B: BitLength> AddNumBits<B> {
    /// Returns the number of bits in the underlying bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<B: BitLength> NumBits for AddNumBits<B> {
    #[inline(always)]
    fn num_ones(&self) -> usize {
        self.number_of_ones
    }
}

impl<B: BitLength> BitCount for AddNumBits<B> {
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.number_of_ones
    }
}

impl<B> Deref for AddNumBits<B> {
    type Target = B;
    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

impl<B: BitCount> From<B> for AddNumBits<B> {
    fn from(bits: B) -> Self {
        let number_of_ones = bits.count_ones();
        AddNumBits {
            bits,
            number_of_ones,
        }
    }
}

// Manual AsRef forwarding (ambassador can't resolve B::Word)
impl<B: Backend + AsRef<[<B as Backend>::Word]>> AsRef<[<B as Backend>::Word]> for AddNumBits<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[<B as Backend>::Word] {
        self.bits.as_ref()
    }
}
