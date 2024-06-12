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

use epserde::Epserde;
use impl_tools::autoimpl;

/// A trait expressing a length in bits.
///
/// This trait is typically used in conjunction with `AsRef<[usize]>` to provide
/// word-based access to a bit vector.
#[allow(clippy::len_without_is_empty)]
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait BitLength {
    /// Returns a length in bits.
    fn len(&self) -> usize;
}

macro_rules! forward_bit_length {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::BitLength for $name < $($generic),* > where $type: $crate::traits::rank_sel::BitLength {
            #[inline(always)]
            fn len(&self) -> usize {
                    $crate::traits::rank_sel::BitLength::len(&self.$field)
                }
            }
    };
}

pub(crate) use forward_bit_length;

/// Potentially expensive bit-counting methods.
///
/// The methods in this trait compute the number of ones or zeros
/// in a bit vector (possibly underlying a succinct data structure).
/// The computation can be expensive: if you need a constant-time
/// version, use [`NumBits`]. If you need to cache the result
/// of these methods, you can use [`AddNumBits`].
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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

macro_rules! forward_bit_count {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::BitCount for $name < $($generic,)* > where $type: $crate::traits::rank_sel::BitCount {
            #[inline(always)]
            fn count_ones(&self) -> usize {
                $crate::traits::rank_sel::BitCount::count_ones(&self.$field)
            }
            #[inline(always)]
            fn count_zeros(&self) -> usize {
                $crate::traits::rank_sel::BitCount::count_zeros(&self.$field)
            }
        }
    };
}

pub(crate) use forward_bit_count;

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

macro_rules! forward_num_bits {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::NumBits for $name < $($generic,)* > where $type: $crate::traits::rank_sel::NumBits {
            #[inline(always)]
            fn num_ones(&self) -> usize {
                $crate::traits::rank_sel::NumBits::num_ones(&self.$field)
            }
            #[inline(always)]
            fn num_zeros(&self) -> usize {
                $crate::traits::rank_sel::NumBits::num_zeros(&self.$field)
            }
        }
    };
}

pub(crate) use forward_num_bits;

/// Ranking over a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait Rank: BitLength {
    /// Returns the number of ones preceding the specified position.
    ///
    /// The bit vector is virtually zero-extended. If `pos` is greater than or equal to the
    /// [length of the underlying bit vector](`BitLength::len`), the number of
    /// ones in the underlying bit vector is returned.
    fn rank(&self, pos: usize) -> usize {
        unsafe { self.rank_unchecked(pos.min(self.len())) }
    }

    /// Returns the number of ones preceding the specified position.
    ///
    /// # Safety
    /// `pos` must be between 0 (included) and the [length of the underlying bit
    /// vector](`BitLength::len`) (excluded).
    ///
    /// Some implementation might consider the the length as a valid argument.
    unsafe fn rank_unchecked(&self, pos: usize) -> usize;
}

macro_rules! forward_rank {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::Rank for $name < $($generic,)* > where $type: $crate::traits::rank_sel::Rank {
            #[inline(always)]
            fn rank(&self, pos: usize) -> usize {
                $crate::traits::rank_sel::Rank::rank(&self.$field, pos)
            }
            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> usize {
                $crate::traits::rank_sel::Rank::rank_unchecked(&self.$field, pos)
            }
        }
    };
}

pub(crate) use forward_rank;

/// Ranking zeros over a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
///
/// Note that this is just an extension trait for [`Rank`].
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

macro_rules! forward_rank_zero {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::RankZero for $name < $($generic,)* >
            where Self: $crate::traits::rank_sel::Rank, $type: $crate::traits::rank_sel::RankZero {
            #[inline(always)]
            fn rank_zero(&self, pos: usize) -> usize {
                $crate::traits::rank_sel::RankZero::rank_zero(&self.$field, pos)
            }
            #[inline(always)]
            unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
                $crate::traits::rank_sel::RankZero::rank_zero_unchecked(&self.$field, pos)
            }
        }
    };
}

pub(crate) use forward_rank_zero;

/// Ranking over a bit vector, with a hint.
///
/// This trait is used to implement fast ranking by adding to bit vectors
/// counters of different kind.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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
    unsafe fn rank_hinted_unchecked(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> usize;
    /// Returns the number of ones preceding the specified position,
    /// provided a preceding position `hint_pos` * `HINT_BIT_SIZE` and
    /// the associated rank.
    fn rank_hinted(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> Option<usize>;
}

macro_rules! forward_rank_hinted {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::RankHinted<64> for $name < $($generic,)* > where $type: $crate::traits::rank_sel::RankHinted<64> {
            #[inline(always)]
            unsafe fn rank_hinted_unchecked(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> usize {
                $crate::traits::rank_sel::RankHinted::<64>::rank_hinted_unchecked(&self.$field, pos, hint_pos, hint_rank)
            }
            #[inline(always)]
            fn rank_hinted(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> Option<usize> {
                $crate::traits::rank_sel::RankHinted::<64>::rank_hinted(&self.$field, pos, hint_pos, hint_rank)
            }
        }
    };
}

pub(crate) use forward_rank_hinted;

/// Selection over a bit vector without bound checks.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait SelectUnchecked {
    /// Returns the position of the one of given rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded).
    unsafe fn select_unchecked(&self, rank: usize) -> usize;
}

macro_rules! forward_select_unchecked {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::SelectUnchecked for $name < $($generic,)* >
            where $type: $crate::traits::rank_sel::SelectUnchecked {
            #[inline(always)]
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                $crate::traits::rank_sel::SelectUnchecked::select_unchecked(&self.$field, rank)
            }
        }
    };
}

pub(crate) use forward_select_unchecked;

/// Selection over a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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

macro_rules! forward_select {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::Select for $name < $($generic,)* >
            where
                Self: $crate::traits::rank_sel::NumBits,
                Self: $crate::traits::rank_sel::SelectUnchecked, $type:
                $crate::traits::rank_sel::Select {
            #[inline(always)]
            fn select(&self, rank: usize) -> Option<usize> {
                $crate::traits::rank_sel::Select::select(&self.$field, rank)
            }
        }
    };
}

pub(crate) use forward_select;

/// Selection zeros over a bit vector without bound checks.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait SelectZeroUnchecked {
    /// Returns the position of the zero of given rank.
    ///
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded).
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize;
}

macro_rules! forward_select_zero_unchecked {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::SelectZeroUnchecked for $name < $($generic,)* >
            where $type: $crate::traits::rank_sel::SelectZeroUnchecked {
            #[inline(always)]
            unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
                $crate::traits::rank_sel::SelectZeroUnchecked::select_zero_unchecked(&self.$field, rank)
            }
        }
    };
}

pub(crate) use forward_select_zero_unchecked;
/// Selection zeros over a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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

macro_rules! forward_select_zero {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::SelectZero for $name < $($generic,)* >
            where
                Self: $crate::traits::rank_sel::NumBits,
                Self: $crate::traits::rank_sel::SelectZeroUnchecked,
                $type: $crate::traits::rank_sel::SelectZero {
            #[inline(always)]
            fn select_zero(&self, rank: usize) -> Option<usize> {
                $crate::traits::rank_sel::SelectZero::select_zero(&self.$field, rank)
            }
        }
    };
}

pub(crate) use forward_select_zero;

/// Selection over a bit vector, with a hint.
///
/// This trait is used to implement fast selection by adding to bit vectors
/// indices of different kind. See, for example,
/// [`SimpleSelect`](crate::rank_sel::SimpleSelect).
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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
    unsafe fn select_hinted_unchecked(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;
    /// Selection the one of given rank, provided the position of a preceding one
    /// and its rank.
    fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize>;
}

macro_rules! forward_select_hinted {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::SelectHinted for $name < $($generic,)* > where $type: $crate::traits::rank_sel::SelectHinted {
            #[inline(always)]
            unsafe fn select_hinted_unchecked(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize {
                $crate::traits::rank_sel::SelectHinted::select_hinted_unchecked(&self.$field, rank, hint_pos, hint_rank)
            }
            #[inline(always)]
            fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize> {
                $crate::traits::rank_sel::SelectHinted::select_hinted(&self.$field, rank, hint_pos, hint_rank)
            }
        }
    };
}

pub(crate) use forward_select_hinted;

/// Selection zeros over a bit vector, with a hint.
///
/// This trait is used to implement fast selection over zeros by adding to bit
/// vectors indices of different kind.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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
    unsafe fn select_zero_hinted_unchecked(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize;

    /// Selection the zero of given rank, provided the position of a preceding zero
    /// and its rank.
    fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize>;
}

macro_rules! forward_select_zero_hinted {
        ($name:ident < $( $([$const:ident])? $generic:ident $(:$t:ty)? ),* >; $type:ident; $field:ident) => {
        impl < $( $($const)? $generic $(:$t)? ),* > $crate::traits::rank_sel::SelectZeroHinted for $name < $($generic,)* > where $type: $crate::traits::rank_sel::SelectZeroHinted {
            #[inline(always)]
            unsafe fn select_zero_hinted_unchecked(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize {
                $crate::traits::rank_sel::SelectZeroHinted::select_zero_hinted_unchecked(&self.$field, rank, hint_pos, hint_rank)
            }
            #[inline(always)]
            fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize> {
                $crate::traits::rank_sel::SelectZeroHinted::select_zero_hinted(&self.$field, rank, hint_pos, hint_rank)
            }
        }
    };
}

pub(crate) use forward_select_zero_hinted;
use mem_dbg::{MemDbg, MemSize};

/// A thin wrapper implementing [`NumBits`] by caching the result of [`BitCount::count_ones`].
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct AddNumBits<B> {
    bits: B,
    number_of_ones: usize,
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

impl<B: BitCount> From<B> for AddNumBits<B> {
    fn from(bits: B) -> Self {
        let number_of_ones = bits.count_ones();
        AddNumBits {
            bits,
            number_of_ones,
        }
    }
}

crate::forward_mult![AddNumBits<B>; B; bits;
    crate::forward_as_ref_slice_usize,
    crate::forward_index_bool,
    crate::traits::rank_sel::forward_bit_length,
    crate::traits::rank_sel::forward_rank_hinted,
    crate::traits::rank_sel::forward_rank,
    crate::traits::rank_sel::forward_rank_zero,
    crate::traits::rank_sel::forward_select_hinted,
    crate::traits::rank_sel::forward_select_unchecked,
    crate::traits::rank_sel::forward_select,
    crate::traits::rank_sel::forward_select_zero_hinted,
    crate::traits::rank_sel::forward_select_zero_unchecked,
    crate::traits::rank_sel::forward_select_zero
];
