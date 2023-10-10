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
of a [fixed bit width](BitFieldSliceCore::bit_width). The associated
implementation is [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec).

Implementing the [`core::ops::Index`]/[`core::ops::IndexMut`] traits
would be more natural and practical, but in certain cases it is impossible:
in our main use case, [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec),
we cannot implement [`core::ops::Index`] because there is no way to
return a reference to a bit segment.

There are three end-user traits: [`BitFieldSlice`], [`BitFieldSliceMut`] and [`BitFieldSliceAtomic`].
The trait [`BitFieldSliceCore`] contains the common methods, and in particular
[`BitFieldSliceCore::bit_width`], which returns the bit width the values stored in the slice.
 All stored values must fit within this bit width.

Note that [`BitFieldSliceAtomic`] has methods with the same names as those in
[`BitFieldSlice`] and [`BitFieldSliceMut`]. Due to the way methods are resolved,
this might cause problems if you have have imported all traits. We suggest that,
unless you are using only of the two variants, you import globally only [`BitFieldSliceCore`],
importing the other traits only in the functions of modules that needs them. For
this reason, only [`BitFieldSliceCore`] is glob-imported in the prelude.

If you need to iterate over a [`BitFieldSlice`], you can use [`BitFieldSliceIterator`].

Implementations must return always zero on a [`BitFieldSlice::get`] when the bit
width is zero. The behavior of a [`BitFieldSliceMut::set`] in the same context is not defined.

We provide implementations for `Vec<usize>`, `Vec<AtomicUsize>`, `&[usize]`,
and `&[AtomicUsize]` that view their elements as values with a bit width
equal to that of `usize`; for those, we also implement
[`IntoValueIterator`](crate::traits::iter::IntoValueIterator) using a [helper](BitFieldSliceIterator) structure
that might be useful for other implementations, too.

The implementations based on atomic types implement
[`BitFieldSliceAtomic`].

*/
use common_traits::Number;
use common_traits::*;
use core::sync::atomic::*;
use std::marker::PhantomData;

/// Common methods for [`BitFieldSlice`], [`BitFieldSliceMut`], and [`BitFieldSliceAtomic`].
///
/// The dependence on `V` is necessary to implement this trait on slices, as
/// we need the bit width of the values stored in the slice.
pub trait BitFieldSliceCore<V> {
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
pub trait BitFieldSlice<V: UnsignedInt>: BitFieldSliceCore<V> {
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
pub trait BitFieldSliceMut<V: UnsignedInt>: BitFieldSliceCore<V> {
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
/// [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec) for an example.
pub trait BitFieldSliceAtomic<V: UnsignedInt + IntoAtomic>:
    BitFieldSliceCore<<V as IntoAtomic>::AtomicType>
{
    /// Return the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`BitFieldSliceCore::len`)).
    /// No bound or bit-width check is performed.
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> V;

    /// Return the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    fn get(&self, index: usize, order: Ordering) -> V {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index, order) }
    }

    /// Set the element of the slice at the specified index.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`BitFieldSliceCore::len`));
    /// - `value` must fit withing [`BitFieldSliceCore::bit_width`] bits.
    /// No bound or bit-width check is performed.
    unsafe fn set_unchecked(&self, index: usize, value: V, order: Ordering);

    /// Set the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    fn set(&self, index: usize, value: V, order: Ordering) {
        if index >= self.len() {
            panic_if_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();
        // TODO Maybe testless?
        let mask = if bw == 0 {
            V::ZERO
        } else {
            V::MAX.wrapping_shr(V::BITS as u32 - bw as u32)
        };
        panic_if_value!(value, mask, bw);
        unsafe {
            self.set_unchecked(index, value, order);
        }
    }
}

/// A ready-made implementation of [`BitFieldSliceIterator`].
///
/// We cannot implement [`IntoValueIterator`](crate::traits::iter::IntoValueIterator) for [`BitFieldSlice`]
/// because it would be impossible to override in implementing classes,
/// but you can implement [`IntoValueIterator`](crate::traits::iter::IntoValueIterator) for your implementation
/// of [`BitFieldSlice`] by using this structure.
pub struct BitFieldSliceIterator<'a, V: UnsignedInt, B: BitFieldSlice<V>> {
    slice: &'a B,
    index: usize,
    _marker: PhantomData<V>,
}

impl<'a, V: UnsignedInt, B: BitFieldSlice<V>> BitFieldSliceIterator<'a, V, B> {
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

impl<'a, V: UnsignedInt, B: BitFieldSlice<V>> Iterator for BitFieldSliceIterator<'a, V, B> {
    type Item = V;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            // SAFETY: self.index is always within bounds
            let res = unsafe { self.slice.get_unchecked(self.index) };
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }
}

macro_rules! impl_core {
    ($($ty:ty),*) => {$(
        impl<T: AsRef<[$ty]>> BitFieldSliceCore<$ty> for T {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.as_ref().len()
            }
        }
    )*};
}

impl_core!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_core_atomic {
    ($($aty:ty),*) => {$(
        impl<T: AsRef<[$aty]>> BitFieldSliceCore<$aty> for T {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.as_ref().len()
            }
        }
    )*};
}

impl_core_atomic!(AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize);

macro_rules! impl_ref {
    ($($ty:ty),*) => {$(
        impl<T: AsRef<[$ty]>> BitFieldSlice<$ty> for T {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> $ty {
                debug_assert_bounds!(index, self.len());
                *self.as_ref().get_unchecked(index)
            }
        }
    )*};
}

impl_ref!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_atomic {
    ($std:ty, $atomic:ty) => {
        impl<T: AsRef<[$atomic]>> BitFieldSliceAtomic<$std> for T {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> $std {
                debug_assert_bounds!(index, self.len());
                self.as_ref().get_unchecked(index).load(order)
            }
            #[inline(always)]
            unsafe fn set_unchecked(&self, index: usize, value: $std, order: Ordering) {
                debug_assert_bounds!(index, self.len());
                self.as_ref().get_unchecked(index).store(value, order);
            }
        }
    };
}

impl_atomic!(u8, AtomicU8);
impl_atomic!(u16, AtomicU16);
impl_atomic!(u32, AtomicU32);
impl_atomic!(u64, AtomicU64);
impl_atomic!(usize, AtomicUsize);

macro_rules! impl_mut {
    ($($ty:ty),*) => {$(
        impl<T: AsMut<[$ty]> + AsRef<[$ty]>> BitFieldSliceMut<$ty> for T {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: $ty) {
                debug_assert_bounds!(index, self.len());
                *self.as_mut().get_unchecked_mut(index) = value;
            }
        }
    )*};
}

impl_mut!(u8, u16, u32, u64, u128, usize);
