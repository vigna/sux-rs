//! # Traits
//! This modules contains basic traits related to succinct data structures.
//! The train `Length` provides information about the length of the
//! underlying bit vector, independently of its implementation.
//! 
//! Traits are collected into a module so you can do `use sux::traits::*;`
//! for ease of use. 

/// A trait specifying abstractly the length of the bit vector underlying
/// a succint data structure.
pub trait Length {
	/// Return the length of the underlying bit vector.
	fn len(&self) -> usize;
}

/// Rank over a bit vector.
pub trait Rank: Length {
	/// Return the number of ones preceding the specified position.
	/// 
	/// # Arguments
	/// * `pos` : `usize` - The position to query.
	fn rank(&self, pos: usize) -> usize;
	/// Return the number of ones preceding the specified position.
	/// 
	/// # Arguments
	/// * `pos` : `usize` - The position to query, which must be between 0 and the [length of the underlying bit vector](`Length::len`).
	unsafe fn rank_unchecked(&self, pos: usize) -> usize;
}

/// Rank zeros over a bit vector.
pub trait RankZero: Length {
	/// Return the number of zeros preceding the specified position.
	/// 
	/// # Arguments
	/// * `pos` : `usize` - The position to query.
	fn rank_zero(&self, i: usize) -> usize;
	/// Return the number of zeros preceding the specified position.
	/// 
	/// # Arguments
	/// * `pos` : `usize` - The position to query, which must be between 0 and the [length of the underlying bit vector](`Length::len`).
	unsafe fn rank_zero_unchecked(&self, i: usize) -> usize;
}

/// Select over a bit vector.
pub trait Select: Length {
	/// Return the position of the one of given rank.
	/// 
	/// # Arguments
	/// * `rank` : `usize` - The rank to query. If there is no
	/// one of given rank, this function return `None`.
	fn select(&self, rank: usize) -> Option<usize>;
	/// Return the position of the one of given rank.
	/// 
	/// # Arguments
	/// * `rank` : `usize` - The rank to query, which must be
	/// between zero and the number of ones in the underlying bit vector minus one.
	unsafe fn select_unchecked(&self, rank: usize) -> usize;
}

/// Select zeros over a bit vector.
pub trait SelectZero: Length {
	/// Return the position of the zero of given rank.
	/// 
	/// # Arguments
	/// * `rank` : `usize` - The rank to query. If there is no
	/// zero of given rank, this function return `None`.
	fn select_zero(&self, i: usize) -> Option<usize>;

	/// Return the position of the zero of given rank.
	/// 
	/// # Arguments
	/// * `rank` : `usize` - The rank to query, which must be
	/// between zero and the number of zeroes in the underlying bit vector minus one.
	unsafe fn select_zero_unchecked(&self, i: usize) -> usize;
}

