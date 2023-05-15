//! # VSlice
//!
//! This module defines the `VSlice` and `VSliceMut` traits, which are accessed
//! with a logic similar to slices, but when indexed with `get` return a value.
//! Implementing the slice trait would be more natural, but it would be very complicated
//! because there is no easy way to return a reference to a bit segment
//! (see, e.g., [BitSlice](https://docs.rs/bitvec/latest/bitvec/slice/struct.BitSlice.html)).
//!
//! Each `VSlice` has an associated [`BIT_WIDTH`]. All stored values must fit
//! within this bit width.
//!
//! Implementations must return always zero on a [`VSlice::get`] when the bit
//! width is zero. The behavior of a [`VSlice::set`] in the same context is not defined.

use anyhow::{bail, Result};

pub trait VSlice {
    /// Return the width of the slice. All elements stored in the slice must
    /// fit within this bit width.
    fn bit_width(&self) -> usize;
    /// Return the length of the slice.
    fn len(&self) -> usize;
    /// Return the element of the slice at the given position, without
    /// doing any bounds checking.
    unsafe fn get_unchecked(&self, index: usize) -> u64;

    /// Return the element of the slice at the given position, or `None` if the
    /// position is out of bounds.
    fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len() {
            None
        } else {
            Some(unsafe { self.get_unchecked(index) })
        }
    }
    /// Return if the slice has length zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait VSliceMut: VSlice {
    /// Set the element of the slice at the given position, without
    /// doing any bound or value checking.
    unsafe fn set_unchecked(&mut self, index: usize, value: u64);
    /// Set the element of the slice at the given position, or return `None` if the
    /// position is out of bounds or the value does not fit in [`VSlice::bit_width`] bits.
    fn set(&mut self, index: usize, value: u64) -> Result<u64> {
        if index >= self.len() {
            bail!(
                "Index out of bounds {} on a vector of len {}",
                index,
                self.len()
            )
        }
        let bw = self.bit_width();
        let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
        if value & mask != value {
            bail!("Value {} does not fit in {} bits", value, bw)
        }
        unsafe {
            self.set_unchecked(index, value);
        }
        Ok(value)
    }
}

impl<'a> VSlice for &'a [u64] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[u64]>::get_unchecked(self, index)
    }
}

impl<'a> VSlice for &'a mut [u64] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[u64]>::get_unchecked(self, index)
    }
}

impl<'a> VSliceMut for &'a mut [u64] {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[u64]>::get_unchecked_mut(self, index) = value;
    }
}

impl VSlice for Vec<u64> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <Vec<u64>>::len(self)
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[u64]>::get_unchecked(self, index)
    }
}

impl VSliceMut for Vec<u64> {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[u64]>::get_unchecked_mut(self, index) = value;
    }
}
