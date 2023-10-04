/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Traits for slices of bit fields of constant width.

Slices of bit fields are accessed with a logic similar to slices, but
when indexed with [`get`](BitFieldSlice::get) return an owned value
of a [fixed bit width](BitFieldSliceCore::bit_width).

Implementing the [`core::ops::Index`]/[`core::ops::IndexMut`] traits
would be more natural and practical, but in certain cases it is impossible:
in our main use case, [`CompactArray`],
we cannot implement [`core::ops::Index`] because there is no way to
return a reference to a bit segment.

There are three end-user traits: [`BitFieldSlice`], [`BitFieldSliceMut`] and [`BitFieldSliceAtomic`].
The trait [`BitFieldSliceCore`] contains the common methods, and in particular
[`BitFieldSliceCore::bit_width`], which returns the bit width the values stored in the slice.
 All stored values must fit within this bit width.

Implementations must return always zero on a [`BitFieldSlice::get`] when the bit
width is zero. The behavior of a [`BitFieldSliceMut::set`] in the same context is not defined.

We provide implementations for `Vec<usize>`, `Vec<AtomicUsize>`, `&[usize]`,
and `&[AtomicUsize]` that view their elements as values with a bit width
equal to that of `usize`; for those, we also implement
[`IntoValueIterator`] using a [helper](BitFieldSliceIterator) structure
that might be useful for other implementations, too.

The implementations based on atomic types implement
[`BitFieldSliceAtomic`].

*/
use crate::prelude::*;
use common_traits::Number;
use common_traits::*;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::marker::PhantomData;

/// Common methods for [`BitFieldSlice`], [`BitFieldSliceMut`], and [`BitFieldSliceAtomic`]
pub trait BitFieldSliceCore<V: Bits> {
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

/// A slice of bit fields of constant bit width.
pub trait BitFieldSlice<V: Bits>: BitFieldSliceCore<V> {
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`BitFieldSliceCore::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> V;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    fn get(&self, index: usize) -> V {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index) }
    }
}

/// A mutable slice of bit fields of constant bit width.
pub trait BitFieldSliceMut<V: Integer>: BitFieldSliceCore<V> {
    /// Set the element of the slice at the specified index.
    /// No bounds checking is performed.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`BitFieldSliceCore::len`));
    /// - `value` must fit withing [`BitFieldSliceCore::bit_width`] bits.
    /// No bound or bit-width check is performed.
    unsafe fn set_unchecked(&mut self, index: usize, value: V);

    /// Set the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    fn set(&mut self, index: usize, value: V) {
        panic_if_out_of_bounds!(index, self.len());
        let bw = self.bit_width();
        // TODO: Maybe testless?
        let mask = if bw == 0 {
            V::ZERO
        } else {
            V::MAX.wrapping_shr(V::BITS as u32 - bw as u32)
        };

        panic_if_value!(value, mask, bw);
        unsafe {
            self.set_unchecked(index, value);
        }
    }
}

/// A thread-safe slice of bit fields of constant bit width supporting atomic operations.
///
/// Different implementations might provide different atomicity guarantees. See
/// [`CompactArray`] for an example.
pub trait BitFieldSliceAtomic<V: Atomic + Integer + Bits>: BitFieldSliceCore<V>
where
    V::NonAtomic: Integer,
{
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`BitFieldSliceCore::len`)).
    /// No bound or bit-width check is performed.
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> V::NonAtomic;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    fn get(&self, index: usize, order: Ordering) -> V::NonAtomic {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index, order) }
    }

    /// Set the element of the slice at the specified index.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`BitFieldSliceCore::len`));
    /// - `value` must fit withing [`BitFieldSliceCore::bit_width`] bits.
    /// No bound or bit-width check is performed.
    unsafe fn set_unchecked(&self, index: usize, value: V::NonAtomic, order: Ordering);

    /// Set the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    fn set(&self, index: usize, value: V::NonAtomic, order: Ordering) {
        if index >= self.len() {
            panic_if_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        // TODO Maybe testless?
        let mask = if bw == 0 {
            V::NonAtomic::ZERO
        } else {
            V::NonAtomic::MAX.wrapping_shr(V::BITS as u32 - bw as u32)
        };
        panic_if_value!(value, mask, bw);
        unsafe {
            self.set_unchecked(index, value, order);
        }
    }
}

/// A ready-made implementation of [`BitFieldSliceIterator`].
///
/// We cannot implement [`IntoValueIterator`] for [`BitFieldSlice`]
/// because it would be impossible to override in implementing classes,
/// but you can implement [`IntoValueIterator`] for your implementation
/// of [`BitFieldSlice`] by using this structure.
pub struct BitFieldSliceIterator<'a, V: Bits, B: BitFieldSlice<V>> {
    slice: &'a B,
    index: usize,
    _marker: PhantomData<V>,
}

impl<'a, V: Bits, B: BitFieldSlice<V>> BitFieldSliceIterator<'a, V, B> {
    pub fn new(slice: &'a B, index: usize) -> Self {
        if index > slice.len() {
            panic!("Start index out of bounds: {} > {}", index, slice.len());
        }
        Self {
            slice,
            index,
            _marker: PhantomData,
        }
    }
}

impl<'a, V: Bits, B: BitFieldSlice<V>> Iterator for BitFieldSliceIterator<'a, V, B> {
    type Item = V;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let res = self.slice.get(self.index);
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }
}

impl<V: Bits, T: AsRef<[V]>> BitFieldSliceCore<V> for T {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        V::BITS
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<V: Bits, T: AsRef<[V]>> BitFieldSlice<V> for T {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> V {
        debug_assert_bounds!(index, self.len());
        *self.as_ref().get_unchecked(index)
    }
}

/*
impl<V, T> IntoValueIterator for T
    where T: AsRef<[V]>
{
    type Item = V;
    type IntoValueIter<'a> = Copied<core::slice::Iter<'a, Self::Item>>
        where
            T: 'a;
    #[inline(always)]
    fn iter_val(&self) -> Self::IntoValueIter<'_> {
        <Self as AsRef<[Self::Item]>>::as_ref(self).iter().copied()
    }

    fn iter_val_from(&self, from: Self::Item) -> Self::IntoValueIter<'_> {
        <Self as AsRef<[Self::Item]>>::as_ref(self)[from..]
            .iter()
            .copied()
    }
} */

impl<V: Bits + Atomic + Integer, T: AsRef<[V]>> BitFieldSliceAtomic<V> for T
where
    V::NonAtomic: Integer,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> V::NonAtomic {
        debug_assert_bounds!(index, self.len());
        <T as AsRef<[V]>>::as_ref(self)
            .get_unchecked(index)
            .load(order)
    }
    #[inline(always)]
    unsafe fn set_unchecked(&self, index: usize, value: V::NonAtomic, order: Ordering) {
        debug_assert_bounds!(index, self.len());
        <T as AsRef<[V]>>::as_ref(self)
            .get_unchecked(index)
            .store(value, order);
    }
}

impl<V: Integer, T: AsMut<[V]> + AsRef<[V]>> BitFieldSliceMut<V> for T {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: V) {
        debug_assert_bounds!(index, self.len());
        *self.as_mut().get_unchecked_mut(index) = value;
    }
}
