//! # Traits
//! This modules contains basic traits related to succinct data structures.
//!
//! Traits are collected into a module so you can do `use sux::traits::*;`
//! for ease of use.

use std::sync::atomic::{AtomicU64, AtomicUsize};

mod memory;
pub use memory::*;

mod vslice;
pub use vslice::*;

mod indexed_dict;
pub use indexed_dict::*;

use anyhow::Result;

/// Like [`Into`], but we need to avoid the orphan rule and error
/// [E0210](https://github.com/rust-lang/rust/blob/master/compiler/rustc_error_codes/src/error_codes/E0210.md)
///
/// Reference: https://rust-lang.github.io/chalk/book/clauses/coherence.html
pub trait ConvertTo<B> {
    fn convert_to(self) -> Result<B>;
}

impl ConvertTo<usize> for usize {
    #[inline(always)]
    fn convert_to(self) -> Result<Self> {
        Ok(self)
    }
}
impl ConvertTo<AtomicUsize> for AtomicUsize {
    #[inline(always)]
    fn convert_to(self) -> Result<Self> {
        Ok(self)
    }
}
impl ConvertTo<Vec<u64>> for Vec<u64> {
    #[inline(always)]
    fn convert_to(self) -> Result<Self> {
        Ok(self)
    }
}

impl ConvertTo<Vec<u64>> for Vec<AtomicU64> {
    #[inline(always)]
    fn convert_to(self) -> Result<Vec<u64>> {
        Ok(unsafe { std::mem::transmute::<Vec<AtomicU64>, Vec<u64>>(self) })
    }
}
impl ConvertTo<Vec<AtomicU64>> for Vec<u64> {
    #[inline(always)]
    fn convert_to(self) -> Result<Vec<AtomicU64>> {
        Ok(unsafe { std::mem::transmute::<Vec<u64>, Vec<AtomicU64>>(self) })
    }
}
impl<'a> ConvertTo<&'a [AtomicU64]> for &'a [u64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a [AtomicU64]> {
        Ok(unsafe { std::mem::transmute::<&'a [u64], &'a [AtomicU64]>(self) })
    }
}
impl<'a> ConvertTo<&'a [u64]> for &'a [AtomicU64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a [u64]> {
        Ok(unsafe { std::mem::transmute::<&'a [AtomicU64], &'a [u64]>(self) })
    }
}
impl<'a> ConvertTo<&'a mut [AtomicU64]> for &'a mut [u64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a mut [AtomicU64]> {
        Ok(unsafe { std::mem::transmute::<&'a mut [u64], &'a mut [AtomicU64]>(self) })
    }
}
impl<'a> ConvertTo<&'a mut [u64]> for &'a mut [AtomicU64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a mut [u64]> {
        Ok(unsafe { std::mem::transmute::<&'a mut [AtomicU64], &'a mut [u64]>(self) })
    }
}

/// A trait for succinct data structures that expose the
/// length of the underlying bit vector.
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
    /// vector](`Length::len`) (included).
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
    /// vector](`Length::len`) (included).
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
    /// * `rank` : `usize` - The rank to query; see Seafety below for valid values.
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
    /// `rank` must be between zero (included) and the number of zeroes in the
    /// underlying bit vector (excluded).
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize;
}

pub trait SelectHinted: Select {
    /// # Safety
    /// `rank` must be between zero (included) and the number of ones in the
    /// underlying bit vector (excluded). `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`Length::len`) (included),
    /// and must be the position of a one in the underlying bit vector.
    /// `rank_at_pos` must be the number of ones in the underlying bit vector
    /// before `pos`.
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize;
}

pub trait SelectZeroHinted: SelectZero {
    /// # Safety
    /// `rank` must be between zero (included) and the number of zeros in the
    /// underlying bit vector (excluded). `pos` must be between 0 (included) and
    /// the [length of the underlying bit vector](`Length::len`) (included),
    /// and must be the position of a zero in the underlying bit vector.
    /// `rank_at_pos` must be the number of zeros in the underlying bit vector
    /// before `pos`.
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize;
}
