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

use anyhow::Result;

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
            Some(unsafe{self.get_unchecked(index)})
        }
    }
}


pub trait VSliceMut: VSlice {
    /// Set the element of the slice at the given position, without
    /// doing any bound or value checking.
    unsafe fn set_unchecked(&self, index: usize, value: u64);
    /// Set the element of the slice at the given position, or return `None` if the
    /// position is out of bounds or the value does not fit in [`VSlice::bit_width`] bits.
    fn set(&self, index: usize, value: u64) -> Result<u64> {
        if index >= self.len() {
            Err(anyhow::anyhow!("Index out of bounds"))
        } else if value & (u64::MAX >> 64 - self.bit_width()) != value {
            Err(anyhow::anyhow!("Value does not fit in {} bits", self.bit_width()))
        } else {
            unsafe {
                self.set_unchecked(index, value);
            }
            Ok(value)
        }
    }
}