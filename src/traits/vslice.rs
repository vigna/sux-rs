//! # VSlice
//!
//! This module defines the `VSlice` and `VSliceMut` traits, which are accessed
//! with a logic similar to slices, but when indexed with `get` return a value.
//! Implementing the slice trait would be more natural, but it would be very complicated
//! because there is no easy way to return a reference to a bit segment
//! (see, e.g., [BitSlice](https://docs.rs/bitvec/latest/bitvec/slice/struct.BitSlice.html)).
//!
//! Each `VSlice` has an associated [`VSlice::bit_width`]. All stored values must fit
//! within this bit width.
//!
//! Implementations must return always zero on a [`VSlice::get`] when the bit
//! width is zero. The behavior of a [`VSliceMut::set`] in the same context is not defined.

pub trait VSlice {
    /// Return the width of the slice. All elements stored in the slice must
    /// fit within this bit width.
    fn bit_width(&self) -> usize;

    /// Return the length of the slice.
    fn len(&self) -> usize;

    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> u64;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    fn get(&self, index: usize) -> u64 {
        if index >= self.len() {
            panic!("Index out of bounds: {} >= {}", index, self.len());
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }
    /// Return if the slice has length zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait VSliceMut: VSlice {
    /// Set the element of the slice at the specified index.
    /// No bounds checking is performed.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
    unsafe fn set_unchecked(&mut self, index: usize, value: u64);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    /// or the value does not fit in [`VSlice::bit_width`] bits.
    fn set(&mut self, index: usize, value: u64) {
        if index >= self.len() {
            panic!(
                "Index out of bounds {} on a vector of len {}",
                index,
                self.len()
            )
        }
        let bw = self.bit_width();
        let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
        if value & mask != value {
            panic!("Value {} does not fit in {} bits", value, bw)
        }
        unsafe {
            self.set_unchecked(index, value);
        }
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

impl VSlice for mmap_rs::Mmap {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.as_ref().len() / 8
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const u64).add(index);
        std::ptr::read(ptr)
    }
}

impl VSlice for mmap_rs::MmapMut {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.as_ref().len() / 8
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const u64).add(index);
        std::ptr::read(ptr)
    }
}

impl VSliceMut for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *mut u64).add(index);
        std::ptr::write(ptr, value);
    }
}
