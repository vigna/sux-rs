//! # VSlice
//! 
//! This module defines the `VSlice` and `VSliceMut` traits, which are accessed
//! with a logic similar to slices, but when indexed with `get` return a value.
//! Implementing the slice trait would be more natural, but it would be very complicated
//! because there is no easy way to return a reference to a bit segment 
//! (see, e.g., [BitSlice](https://docs.rs/bitvec/latest/bitvec/slice/struct.BitSlice.html)).

pub trait VSlice {
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
    unsafe fn set_unchecked(&self, index: usize, value: u64);
    fn set(&self, index: usize, value: u64) -> Option<()>;
}