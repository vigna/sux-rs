/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

 This module defines the `VSlice` and `VSliceMut` traits, which are accessed
 with a logic similar to slices, but when indexed with `get` return a value.
 Implementing the slice trait would be more natural, but it would be very complicated
 because there is no easy way to return a reference to a bit segment
 (see, e.g., [BitSlice](https://docs.rs/bitvec/latest/bitvec/slice/struct.BitSlice.html)).

 Each `VSlice` has an associated [`VSlice::bit_width`]. All stored values must fit
 within this bit width.

 Implementations must return always zero on a [`VSlice::get`] when the bit
 width is zero. The behavior of a [`VSliceMut::set`] in the same context is not defined.
*/
use core::sync::atomic::{AtomicUsize, Ordering};

/// Trait for common bits between [`VSlice`] and [`VSliceAtomic`]
pub trait VSliceCore {
    /// Return the width of the slice. All elements stored in the slice must
    /// fit within this bit width.
    fn bit_width(&self) -> usize;

    /// Return the length of the slice.
    fn len(&self) -> usize;
    /// Return if the slice has length zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

macro_rules! panic_out_of_bounds {
    ($index: expr, $len: expr) => {
        panic!("Index out of bounds: {} >= {}", $index, $len)
    };
}

macro_rules! panic_value {
    ($value: expr, $bit_width: expr) => {
        panic!("Value {} does not fit in {} bits", $value, $bit_width);
    };
}

pub trait VSlice: VSliceCore {
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> usize;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    fn get(&self, index: usize) -> usize {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }
}

pub trait VSliceMut: VSlice {
    /// Set the element of the slice at the specified index.
    /// No bounds checking is performed.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
    unsafe fn set_unchecked(&mut self, index: usize, value: usize);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    /// or the value does not fit in [`VSlice::bit_width`] bits.
    fn set(&mut self, index: usize, value: usize) {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = usize::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as usize;
        if value & mask != value {
            panic_value!(value, bw);
        }
        unsafe {
            self.set_unchecked(index, value);
        }
    }
}

pub trait VSliceAtomic: VSliceCore {
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    fn get_atomic(&self, index: usize, order: Ordering) -> usize {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        } else {
            unsafe { self.get_atomic_unchecked(index, order) }
        }
    }

    /// Return if the slice has length zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Set the element of the slice at the specified index.
    /// No bounds checking is performed.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    /// or the value does not fit in [`VSlice::bit_width`] bits.
    fn set_atomic(&self, index: usize, value: usize, order: Ordering) {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = usize::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as usize;
        if value & mask != value {
            panic_value!(value, bw);
        }
        unsafe {
            self.set_atomic_unchecked(index, value, order);
        }
    }
}

pub trait VSliceMutAtomicCmpExchange: VSliceAtomic {
    #[inline(always)]
    /// Compare and exchange the value at the specified index.
    /// If the current value is equal to `current`, set it to `new` and return
    /// `Ok(current)`. Otherwise, return `Err(current)`.
    fn compare_exchange(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = usize::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as usize;
        if current & mask != current {
            panic!("Value {} does not fit in {} bits", current, bw)
        }
        if new & mask != new {
            panic!("Value {} does not fit in {} bits", new, bw)
        }
        unsafe { self.compare_exchange_unchecked(index, current, new, success, failure) }
    }

    /// Compare and exchange the value at the specified index.
    /// If the current value is equal to `current`, set it to `new` and return
    /// `Ok(current)`. Otherwise, return `Err(current)`.
    ///
    /// # Safety
    /// The caller must ensure that `index` is in [0..[len](`VSlice::len`)) and that
    /// `current` and `new` fit in [`VSlice::bit_width`] bits.
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize>;
}

impl<'a> VSliceCore for &'a [usize] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[usize]>::len(self)
    }
}

impl<'a> VSlice for &'a [usize] {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[usize]>::get_unchecked(self, index)
    }
}

impl<'a> VSliceCore for &'a [AtomicUsize] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[AtomicUsize]>::len(self)
    }
}

impl<'a> VSliceAtomic for &'a [AtomicUsize] {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).store(value, order);
    }
}

impl<'a> VSliceCore for &'a mut [usize] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[usize]>::len(self)
    }
}

impl<'a> VSlice for &'a mut [usize] {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[usize]>::get_unchecked(self, index)
    }
}

impl<'a> VSliceMut for &'a mut [usize] {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[usize]>::get_unchecked_mut(self, index) = value;
    }
}

impl<'a> VSliceCore for &'a mut [AtomicUsize] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[AtomicUsize]>::len(self)
    }
}

impl<'a> VSliceAtomic for &'a mut [AtomicUsize] {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).store(value, order);
    }
}

impl<'a> VSliceMutAtomicCmpExchange for &'a [AtomicUsize] {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).compare_exchange(current, new, success, failure)
    }
}

impl<'a> VSliceMutAtomicCmpExchange for &'a mut [AtomicUsize] {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).compare_exchange(current, new, success, failure)
    }
}

impl VSliceCore for Vec<usize> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[usize]>::len(self)
    }
}

impl VSlice for Vec<usize> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[usize]>::get_unchecked(self, index)
    }
}

impl VSliceMut for Vec<usize> {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[usize]>::get_unchecked_mut(self, index) = value;
    }
}

impl VSliceCore for Vec<AtomicUsize> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[AtomicUsize]>::len(self)
    }
}
impl VSliceAtomic for Vec<AtomicUsize> {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).store(value, order);
    }
}

impl VSliceMutAtomicCmpExchange for Vec<AtomicUsize> {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicUsize]>::get_unchecked(self, index).compare_exchange(current, new, success, failure)
    }
}

impl VSliceCore for mmap_rs::Mmap {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.as_ref().len() / 8
    }
}

impl VSlice for mmap_rs::Mmap {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const usize).add(index);
        std::ptr::read(ptr)
    }
}

impl VSliceCore for mmap_rs::MmapMut {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.as_ref().len() / 8
    }
}

impl VSlice for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const usize).add(index);
        std::ptr::read(ptr)
    }
}

impl VSliceAtomic for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const AtomicUsize).add(index);
        (*ptr).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const AtomicUsize).add(index);
        (*ptr).store(value, order)
    }
}

impl VSliceMut for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *mut usize).add(index);
        std::ptr::write(ptr, value);
    }
}

impl VSliceMutAtomicCmpExchange for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const AtomicUsize).add(index);
        (*ptr).compare_exchange(current, new, success, failure)
    }
}
