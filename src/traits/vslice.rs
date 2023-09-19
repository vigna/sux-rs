/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Traits for value slices, which are accessed
with a logic similar to slices, but when indexed with `get` return an owned value.

Value slices have a limited bit width per element, and are accessed
with a logic similar to slices, but when indexed with `get`
return an owned value---not a reference.

Implementing the [`core::ops::Index`]/[`core::ops::IndexMut`] traits
would be more natural and practical, but in certain cases it is impossible:
in our main use case, [`CompactArray`](crate::bits::compact_array::CompactArray)
we cannot implement [`core::ops::Index`] because there is no way to
return a reference to a bit segment
(see, e.g., [BitSlice](https://docs.rs/bitvec/latest/bitvec/slice/struct.BitSlice.html)).

There are three end-user traits: [`VSlice`], [`VSliceMut`] and [`VSliceAtomic`].
The trait [`VSliceCore`] contains the common methods, and in particular
[`VSliceCore::bit_width`], which returns the bit width of the slice.
 All stored values must fit within this bit width.

Implementations must return always zero on a [`VSlice::get`] when the bit
width is zero. The behavior of a [`VSliceMut::set`] in the same context is not defined.

We provide implementations for `Vec<usize>`, `Vec<AtomicUsize>`, `&[usize]`,
and `&[AtomicUsize]` that view their elements as values with a bit width
equal to that of `usize`. The implementations based on atomic types implements
[`VSliceAtomic`].
*/
use core::sync::atomic::{AtomicUsize, Ordering};

const BITS: usize = core::mem::size_of::<usize>() * 8;

/// Common methods for [`VSlice`], [`VSliceMut`], and [`VSliceAtomic`]
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

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Index out of bounds: {} >= {}", $index, $len)
        }
    };
}
pub(crate) use panic_if_out_of_bounds;

macro_rules! panic_if_value {
    ($value: expr, $mask: expr, $bit_width: expr) => {
        if $value & $mask != $value {
            panic!("Value {} does not fit in {} bits", $value, $bit_width);
        }
    };
}
pub(crate) use panic_if_value;

macro_rules! debug_assert_bounds {
    ($index: expr, $len: expr) => {
        debug_assert!(
            $index < $len || ($index == 0 && $len == 0),
            "Index out of bounds: {} >= {}",
            $index,
            $len
        );
    };
}

/// A value slice.
pub trait VSlice: VSliceCore {
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSliceCore::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> usize;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSliceCore::len`))
    fn get(&self, index: usize) -> usize {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index) }
    }
}

/// A mutable value slice.
pub trait VSliceMut: VSlice {
    /// Set the element of the slice at the specified index.
    /// No bounds checking is performed.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`VSliceCore::len`));
    /// - `value` must fit withing [`VSliceCore::bit_width`] bits.
    /// No bound or bit-width check is performed.
    unsafe fn set_unchecked(&mut self, index: usize, value: usize);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSliceCore::len`))
    /// or the value does not fit in [`VSliceCore::bit_width`] bits.
    fn set(&mut self, index: usize, value: usize) {
        panic_if_out_of_bounds!(index, self.len());
        let bw = self.bit_width();
        let mask = usize::MAX.wrapping_shr(BITS as u32 - bw as u32)
            & !((bw as isize - 1) >> (BITS - 1)) as usize;
        panic_if_value!(value, mask, bw);
        unsafe {
            self.set_unchecked(index, value);
        }
    }
}

/// A thread-safe value slice supporting atomic operations.
pub trait VSliceAtomic: VSliceCore {
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`VSliceCore::len`)).
    /// No bound or bit-width check is performed.
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> usize;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`VSliceCore::len`))
    fn get(&self, index: usize, order: Ordering) -> usize {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index, order) }
    }

    /// Set the element of the slice at the specified index.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`VSliceCore::len`));
    /// - `value` must fit withing [`VSliceCore::bit_width`] bits.
    /// No bound or bit-width check is performed.
    unsafe fn set_unchecked(&self, index: usize, value: usize, order: Ordering);

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSliceCore::len`))
    /// or the value does not fit in [`VSliceCore::bit_width`] bits.
    fn set(&self, index: usize, value: usize, order: Ordering) {
        if index >= self.len() {
            panic_if_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        let mask = usize::MAX.wrapping_shr(BITS as u32 - bw as u32)
            & !((bw as isize - 1) >> (BITS - 1)) as usize;
        panic_if_value!(value, mask, bw);
        unsafe {
            self.set_unchecked(index, value, order);
        }
    }
}

impl<T: AsRef<[usize]>> VSliceCore for T {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        BITS
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<T: AsRef<[usize]>> VSlice for T {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert_bounds!(index, self.len());
        *self.as_ref().get_unchecked(index)
    }
}

impl<T: AsRef<[AtomicUsize]> + AsRef<[usize]>> VSliceAtomic for T {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> usize {
        debug_assert_bounds!(index, self.len());
        <T as AsRef<[AtomicUsize]>>::as_ref(self)
            .get_unchecked(index)
            .load(order)
    }
    #[inline(always)]
    unsafe fn set_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert_bounds!(index, self.len());
        <T as AsRef<[AtomicUsize]>>::as_ref(self)
            .get_unchecked(index)
            .store(value, order);
    }
}

impl<T: AsMut<[usize]> + AsRef<[usize]>> VSliceMut for T {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        debug_assert_bounds!(index, self.len());
        *self.as_mut().get_unchecked_mut(index) = value;
    }
}
