/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

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
use core::sync::atomic::{AtomicU64, Ordering};

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
    unsafe fn get_unchecked(&self, index: usize) -> u64;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    fn get(&self, index: usize) -> u64 {
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
    unsafe fn set_unchecked(&mut self, index: usize, value: u64);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    /// or the value does not fit in [`VSlice::bit_width`] bits.
    fn set(&mut self, index: usize, value: u64) {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
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
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    fn get_atomic(&self, index: usize, order: Ordering) -> u64 {
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
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSlice::len`))
    /// or the value does not fit in [`VSlice::bit_width`] bits.
    fn set_atomic(&self, index: usize, value: u64, order: Ordering) {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
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
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        if index >= self.len() {
            panic_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
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
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64>;
}

impl<'a> VSliceCore for &'a [u64] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }
}

impl<'a> VSlice for &'a [u64] {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        *<[u64]>::get_unchecked(self, index)
    }
}

impl<'a> VSliceCore for &'a [AtomicU64] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[AtomicU64]>::len(self)
    }
}

impl<'a> VSliceAtomic for &'a [AtomicU64] {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).store(value, order);
    }
}

impl<'a> VSliceCore for &'a mut [u64] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }
}

impl<'a> VSlice for &'a mut [u64] {
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

impl<'a> VSliceCore for &'a mut [AtomicU64] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[AtomicU64]>::len(self)
    }
}

impl<'a> VSliceAtomic for &'a mut [AtomicU64] {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).store(value, order);
    }
}

impl<'a> VSliceMutAtomicCmpExchange for &'a [AtomicU64] {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).compare_exchange(current, new, success, failure)
    }
}

impl<'a> VSliceMutAtomicCmpExchange for &'a mut [AtomicU64] {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).compare_exchange(current, new, success, failure)
    }
}

impl VSliceCore for Vec<u64> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }
}

impl VSlice for Vec<u64> {
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

impl VSliceCore for Vec<AtomicU64> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        64
    }
    #[inline(always)]
    fn len(&self) -> usize {
        <[AtomicU64]>::len(self)
    }
}
impl VSliceAtomic for Vec<AtomicU64> {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).store(value, order);
    }
}

impl VSliceMutAtomicCmpExchange for Vec<AtomicU64> {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        <[AtomicU64]>::get_unchecked(self, index).compare_exchange(current, new, success, failure)
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
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const u64).add(index);
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
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const u64).add(index);
        std::ptr::read(ptr)
    }
}

impl VSliceAtomic for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const AtomicU64).add(index);
        (*ptr).load(order)
    }
    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const AtomicU64).add(index);
        (*ptr).store(value, order)
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

impl VSliceMutAtomicCmpExchange for mmap_rs::MmapMut {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        debug_assert!(index < self.len(), "{} {}", index, self.len());
        let ptr = (self.as_ptr() as *const AtomicU64).add(index);
        (*ptr).compare_exchange(current, new, success, failure)
    }
}
