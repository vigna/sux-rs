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

use crate::ambassador_impl_AsRef;
use crate::ambassador_impl_Index;
use ambassador::{delegatable_trait, Delegate};
use epserde::Epserde;
use impl_tools::autoimpl;
use mem_dbg::{MemDbg, MemSize};
use std::ops::Index;

/// A trait expressing a length in bits.
///
/// This trait is typically used in conjunction with `AsRef<[usize]>` to provide
/// word-based access to a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait BitLength {
    /// Returns a length in bits.
    fn len(&self) -> usize;
}

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
/// in constant time. If you can be contented with a potentially
/// expensive computation, use [`BitCount`].
///
/// If you need to implement this trait on a structure that already
/// implements [`BitCount`], you can use [`AddNumBits`].
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait NumBits: BitLength {
    /// Returns the number of ones in the underlying bit vector
    /// in constant time. If you can be contented with a potentially
    /// expensive computation, use [`BitCount::count_ones`].
    fn num_ones(&self) -> usize;
    /// Returns the number of zeros in the underlying bit vector
    /// in constant time. If you can be contented with a potentially
    /// expensive computation, use [`BitCount::count_zeros`].
    #[inline(always)]
    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }
}

/// Ranking over a bit vector.
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

#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait RankUnchecked {
    /// Returns the number of ones preceding the specified position.
    ///
    /// # Safety
    /// `pos` must be between 0 (included) and the [length of the underlying bit
    /// vector](`BitLength::len`) (excluded).
    ///
    /// Some implementation might accept the the length as a valid argument.
    unsafe fn rank_unchecked(&self, pos: usize) -> usize;
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
    /// Some implementation might consider the the length as a valid argument.
    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        pos - self.rank_unchecked(pos)
    }
}

/// Ranking over a bit vector, with a hint.
///
/// This trait is used to implement fast ranking by adding to bit vectors
/// counters of different kind.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait RankHinted<const HINT_BIT_SIZE: usize> {
    /// Returns the number of ones preceding the specified position,
    /// provided a preceding position and its associated rank.
    ///
    /// # Safety
    /// `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (excluded).
    /// `hint_pos` * `HINT_BIT_SIZE` must be between 0 (included) and
    /// `pos` (included).
    /// `hint_rank` must be the number of ones in the underlying bit vector
    /// before `hint_pos` * `HINT_BIT_SIZE`.
    ///
    /// Some implementation might consider the the length as a valid argument.
    unsafe fn rank_hinted(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> usize;
}

/// Selection over a bit vector without bound checks.
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
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait Select: SelectUnchecked + NumBits {
    /// Returns the position of the one of given rank, or `None` if no such
    /// bit exist.
    fn select(&self, rank: usize) -> Option<usize> {
        if rank >= self.num_ones() {
            None
        } else {
            Some(unsafe { self.select_unchecked(rank) })
        }
    }
}

/// Selection zeros over a bit vector without bound checks.
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

/// Selection zeros over a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait SelectZero: SelectZeroUnchecked + NumBits {
    /// Returns the position of the zero of given rank, or `None` if no such
    /// bit exist.
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
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait SelectHinted {
    /// Selection the one of given rank, provided the position of a preceding one
    /// and its rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded). `hint_pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a one in the underlying bit vector.
    /// `hint_rank` must be the number of ones in the underlying bit vector
    /// before `hint_pos`, and must be less than or equal to `rank`.
    unsafe fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize;
}

/// Selection zeros over a bit vector, with a hint.
///
/// This trait is used to implement fast selection over zeros by adding to bit
/// vectors indices of different kind.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait SelectZeroHinted {
    /// Selection the zero of given rank, provided the position of a preceding zero
    /// and its rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded). `hint_pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`BitLength::len`) (included),
    /// and must be the position of a zero in the underlying bit vector.
    /// `hint_rank` must be the number of zeros in the underlying bit vector
    /// before `hint_pos`, and must be less than or equal to `rank`.
    unsafe fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize;
}

/// A thin wrapper implementing [`NumBits`] by caching the result of
/// [`BitCount::count_ones`].
///
/// This structure forwards to the wrapped structure all traits defined in [this
/// module](crate::rank_sel) except for [`NumBits`] and [`BitCount`]. It is typically
/// used to provide [`NumBits`] to [`Select`]/[`SelectZero`] implementations; see,
/// for example, [`SelectAdapt`](crate::rank_sel::SelectAdapt).
#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Delegate)]
#[delegate(AsRef<[usize]>, target = "bits")]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::rank_sel::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::Rank, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted<64>, target = "bits")]
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
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// # Safety
    /// `len` must be between 0 (included) the number of
    /// bits in `data` (included). No test is performed
    /// on the number of ones.
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, number_of_ones: usize) -> Self {
        Self {
            bits,
            number_of_ones,
        }
    }
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

impl<B: BitCount> From<B> for AddNumBits<B> {
    fn from(bits: B) -> Self {
        let number_of_ones = bits.count_ones();
        AddNumBits {
            bits,
            number_of_ones,
        }
    }
}
