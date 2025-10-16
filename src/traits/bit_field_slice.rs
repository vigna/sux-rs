/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Traits for slices of bit fields of fixed width (AKA “compact arrays“,
//! “bit array“, etc.).
//!
//! Slices of bit fields are accessed with a logic similar to slices, but when
//! indexed they return an owned value of a [fixed bit
//! width](BitWidth::bit_width). They are a prototypical example of a [*slice by
//! value*](SliceByValue), and as such they are based on the
//! [`value-traits`](https://crates.io/crates/value-traits) crate.
//! In particular, [`BitFieldSlice`] extends [`SliceByValue`] and
//! [`BitFieldSliceMut`] extends [`SliceByValueMut`]. Both traits also extend
//! [`BitWidth`], which provides the method [`BitWidth::bit_width`] to
//! retrieve the bit width of the values stored in the slice.
//!
//! Finally, the trait [`AtomicBitFieldSlice`] is a specialized trait for
//! slices of bit fields that support atomic operations.
//!
//! All the traits depends on a type parameter `W` that must implement [`Word`],
//! and which default to `usize`, but any type satisfying the [`Word`] trait can
//! be used, with the restriction that the bit width of the slice can be at most
//! the bit width of `W` as defined by [`AsBytes::BITS`]. Additionally, to
//! implement [`AtomicBitFieldSlice`], `W` must implement [`IntoAtomic`]. The
//! methods of all traits accept and return values of type `W`.
//!
//! Implementations must return always zero upon a read operation when the bit
//! width is zero. The behavior of write operations in the same context is not
//! defined.
//!
//! The derive macros from the
//! [`value-traits`](https://crates.io/crates/value-traits) crate can be used to
//! derive implementations of iterator and subslices for types that
//! implement [`BitFieldSlice`] and [`BitFieldSliceMut`].
//!
//! We provide implementations for vectors and slices of all primitive atomic
//! and non-atomic unsigned integer types that view their elements as values
//! with a bit width equal to that of the type.
#![allow(clippy::result_unit_err)]
use common_traits::*;
use core::sync::atomic::*;
#[cfg(feature = "rayon")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use value_traits::slices::{SliceByValue, SliceByValueMut};

use crate::{debug_assert_bounds, panic_if_out_of_bounds, panic_if_value};

/// A derived trait that the types used as a parameter for [`BitFieldSlice`] must satisfy.
/// To be usable in an [`AtomicBitFieldSlice`], the type must also implement [`IntoAtomic`].
pub trait Word: UnsignedInt + FiniteRangeNumber + AsBytes {}
impl<W: UnsignedInt + FiniteRangeNumber + AsBytes> Word for W {}

/// Common method for [`BitFieldSlice`], [`BitFieldSliceMut`], and
/// [`AtomicBitFieldSlice`].
///
/// The dependence on `W` is necessary to implement this trait on vectors and
/// slices, as we need the bit width of the values stored in the slice.
pub trait BitWidth<W> {
    /// Returns the width of the slice.
    ///
    /// All elements stored in the slice must fit within this bit width.
    fn bit_width(&self) -> usize;
}

/// A slice of bit fields of constant bit width.
///
/// This trait combines [`SliceByValue`] and [`BitWidth`]. Additionally,
/// it provides the method [`as_slice`](BitFieldSlice::as_slice) to
/// access the backend of the slice.
pub trait BitFieldSlice<W: Word>: SliceByValue<Value = W> + BitWidth<W> {
    /// Returns the backend of the slice as a slice of `W`.
    fn as_slice(&self) -> &[W];
}

/// A mutable slice of bit fields of constant bit width.
///
/// This trait combines [`BitFieldSlice`] and [`SliceByValueMut`]. Moreover, it
/// provides reset methods and the method
/// [`as_mut_slice`](BitFieldSliceMut::as_mut_slice) to mutate the backend of
/// the slice.
pub trait BitFieldSliceMut<W: Word>: BitFieldSlice<W> + SliceByValueMut<Value = W> {
    /// Returns the mask to apply to values to ensure they fit in
    /// [`bit_width`](BitWidth::bit_width) bits.
    #[inline(always)]
    fn mask(&self) -> W {
        // TODO: Maybe testless?
        if self.bit_width() == 0 {
            W::ZERO
        } else {
            W::MAX >> (W::BITS as u32 - self.bit_width() as u32)
        }
    }

    /// Sets all values to zero.
    fn reset(&mut self);

    /// Sets all values to zero using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_reset(&mut self);

    /// Returns the backend of the slice as a mutable slice of `W`.
    fn as_mut_slice(&mut self) -> &mut [W];
}

/// A (tentatively) thread-safe slice of bit fields of constant bit width
/// supporting atomic operations.
///
/// Different implementations might provide different atomicity guarantees. See
/// [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec) for an example.
pub trait AtomicBitFieldSlice<W: Word + IntoAtomic>: BitWidth<W::AtomicType>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// See [`slice::len`].
    fn len(&self) -> usize;

    /// See [`slice::is_empty`].
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](SliceByValue::len)).
    /// No bound or bit-width check is performed.
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> W;

    /// Returns the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](SliceByValue::len))
    fn get_atomic(&self, index: usize, order: Ordering) -> W {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_atomic_unchecked(index, order) }
    }

    /// Sets the element of the slice at the specified index.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](SliceByValue::len));
    /// - `value` must fit withing [`BitWidth::bit_width`] bits.
    ///
    /// No bound or bit-width check is performed.
    unsafe fn set_atomic_unchecked(&self, index: usize, value: W, order: Ordering);

    /// Sets the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](SliceByValue::len))
    /// or the value does not fit in [`BitWidth::bit_width`] bits.
    fn set_atomic(&self, index: usize, value: W, order: Ordering) {
        if index >= self.len() {
            panic_if_out_of_bounds!(index, self.len());
        }
        let bw = self.bit_width();

        let mask = if bw == 0 {
            W::ZERO
        } else {
            W::MAX >> (W::BITS as u32 - bw as u32)
        };
        panic_if_value!(value, mask, bw);
        unsafe {
            self.set_atomic_unchecked(index, value, order);
        }
    }

    /// Sets all values to zero.
    ///
    /// This method takes an exclusive reference because usually one needs to
    /// reset a vector to reuse it, and the mutable reference makes it
    /// impossible to have any other reference hanging around.
    fn reset_atomic(&mut self, order: Ordering);

    /// Sets all values to zero using a parallel implementation.
    ///
    /// See [`reset_atomic`](AtomicBitFieldSlice::reset_atomic) for more
    /// details.
    #[cfg(feature = "rayon")]
    fn par_reset_atomic(&mut self, order: Ordering);
}

macro_rules! impl_bit_width {
    ($($ty:ty),*) => {$(
        impl BitWidth<$ty> for [$ty] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
        }

        impl BitWidth<$ty> for Vec<$ty> {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
        }

        impl<const N: usize> BitWidth<$ty> for [$ty; N] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
        }
    )*};
}

impl_bit_width!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_bit_width_delegation {
    ($($ty:ty),*) => {$(
        impl<W, T: BitWidth<W> + ?Sized> BitWidth<W> for $ty {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                T::bit_width(self)
            }
        }
)*}}

impl_bit_width_delegation!(&T, &mut T, Box<T>);

macro_rules! impl_slice {
    ($($ty:ty),*) => {$(
        impl BitFieldSlice<$ty> for [$ty] {

            fn as_slice(&self) -> &[$ty] {
                self
            }
        }
        impl BitFieldSlice<$ty> for Vec<$ty> {
            fn as_slice(&self) -> &[$ty] {
                self
            }
        }

        impl<const N: usize> BitFieldSlice<$ty> for [$ty; N] {
            fn as_slice(&self) -> &[$ty] {
                self
            }
        }
    )*};
}

impl_slice!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_slice_delegation {
    ($($ty:ty),*) => {$(
        impl<W: Word, T: BitFieldSlice<W> + ?Sized> BitFieldSlice<W> for $ty {
            #[inline(always)]
            fn as_slice(&self) -> &[W] {
                T::as_slice(self)
            }
        }
)*}}

impl_slice_delegation!(&T, &mut T, Box<T>);

macro_rules! impl_slice_mut {
    ($($ty:ty),*) => {$(
        impl BitFieldSliceMut<$ty> for [$ty] {

            fn reset(&mut self) {
                for idx in 0..<[$ty]>::len(self) {
                    unsafe{ self.set_unchecked(idx, 0) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset(&mut self) {
                self.as_mut()
                    .par_iter_mut()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| { *w = 0 });
            }

            fn as_mut_slice(&mut self) -> &mut [$ty] {
                self
            }
        }
        impl BitFieldSliceMut<$ty> for Vec<$ty> {
            #[inline(always)]
            fn reset(&mut self) {
                for idx in 0..<[$ty]>::len(self) {
                    unsafe{ self.set_unchecked(idx, 0) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset(&mut self) {
                self
                    .par_iter_mut()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| { *w = 0 });
            }

            fn as_mut_slice(&mut self) -> &mut [$ty] {
                self
            }
        }

        impl<const N: usize> BitFieldSliceMut<$ty> for [$ty; N] {
            #[inline(always)]
            fn reset(&mut self) {
                for idx in 0..<[$ty]>::len(self) {
                    unsafe{ self.set_unchecked(idx, 0) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset(&mut self) {
                self
                    .par_iter_mut()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| { *w = 0 });
            }

            fn as_mut_slice(&mut self) -> &mut [$ty] {
                self
            }
        }

    )*};
}

impl_slice_mut!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_slice_mut_delegation {
    ($($ty:ty),*) => {$(
        impl<W: Word, T: BitFieldSliceMut<W> + ?Sized> BitFieldSliceMut<W> for $ty {
            #[inline(always)]
            fn reset(&mut self) {
                T::reset(self)
            }

            #[cfg(feature = "rayon")]
            #[inline(always)]
            fn par_reset(&mut self) {
                T::par_reset(self)
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [W] {
                T::as_mut_slice(self)
            }
        }
)*}}

// Can't delegate on &T because reset requires &mut T
impl_slice_mut_delegation!(&mut T, Box<T>);

// Implementations for slices of atomic types

macro_rules! impl_bit_width_atomic {
    ($($aty:ty),*) => {$(
        impl BitWidth<$aty> for [$aty] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
        }

        impl BitWidth<$aty> for Vec<$aty> {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
        }

        impl<const N: usize> BitWidth<$aty> for [$aty; N] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
        }
    )*};
}

impl_bit_width_atomic!(AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize);

macro_rules! impl_atomic {
    ($std:ty, $atomic:ty) => {
        impl AtomicBitFieldSlice<$std> for [$atomic] {
            #[inline(always)]
            fn len(&self) -> usize {
                self.as_ref().len()
            }

            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> $std {
                debug_assert_bounds!(index, self.len());
                unsafe { self.as_ref().get_unchecked(index).load(order) }
            }

            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: $std, order: Ordering) {
                debug_assert_bounds!(index, self.len());
                unsafe {
                    self.as_ref().get_unchecked(index).store(value, order);
                }
            }

            fn reset_atomic(&mut self, order: Ordering) {
                for idx in 0..self.len() {
                    unsafe { self.set_atomic_unchecked(idx, 0, order) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset_atomic(&mut self, order: Ordering) {
                self.as_ref()
                    .par_iter()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| w.store(0, order));
            }
        }

        impl AtomicBitFieldSlice<$std> for Vec<$atomic> {
            #[inline(always)]
            fn len(&self) -> usize {
                self.len()
            }

            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> $std {
                debug_assert_bounds!(index, self.len());
                unsafe { self.as_slice().get_unchecked(index).load(order) }
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: $std, order: Ordering) {
                debug_assert_bounds!(index, self.len());
                unsafe {
                    self.as_slice().get_unchecked(index).store(value, order);
                }
            }

            fn reset_atomic(&mut self, order: Ordering) {
                for idx in 0..self.len() {
                    unsafe { self.set_atomic_unchecked(idx, 0, order) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset_atomic(&mut self, order: Ordering) {
                self.par_iter()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| w.store(0, order));
            }
        }

        impl<const N: usize> AtomicBitFieldSlice<$std> for [$atomic; N] {
            #[inline(always)]
            fn len(&self) -> usize {
                N
            }

            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> $std {
                debug_assert_bounds!(index, self.len());
                unsafe { self.as_slice().get_unchecked(index).load(order) }
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: $std, order: Ordering) {
                debug_assert_bounds!(index, self.len());
                unsafe {
                    self.as_slice().get_unchecked(index).store(value, order);
                }
            }

            fn reset_atomic(&mut self, order: Ordering) {
                for idx in 0..self.len() {
                    unsafe { self.set_atomic_unchecked(idx, 0, order) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset_atomic(&mut self, order: Ordering) {
                self.par_iter()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| w.store(0, order));
            }
        }
    };
}

impl_atomic!(u8, AtomicU8);
impl_atomic!(u16, AtomicU16);
impl_atomic!(u32, AtomicU32);
impl_atomic!(u64, AtomicU64);
impl_atomic!(usize, AtomicUsize);

macro_rules! impl_atomic_delegation {
    ($($ty:ty),*) => {$(
        impl<W: Word + IntoAtomic, T: AtomicBitFieldSlice<W>> AtomicBitFieldSlice<W> for $ty
        where
            W::AtomicType: AtomicUnsignedInt + AsBytes
        {
            #[inline(always)]
            fn len(&self) -> usize {
                T::len(self)
            }

            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> W {
                unsafe { T::get_atomic_unchecked(self, index, order) }
            }
            #[inline(always)]
            fn get_atomic(&self, index: usize, order: Ordering) -> W {
                T::get_atomic(self, index, order)
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: W, order: Ordering) {
                unsafe { T::set_atomic_unchecked(self, index, value, order) }
            }
            #[inline(always)]
            fn set_atomic(&self, index: usize, value: W, order: Ordering) {
                T::set_atomic(self, index, value, order)
            }
            #[inline(always)]
            fn reset_atomic(&mut self, order: Ordering) {
                T::reset_atomic(self, order)
            }
            #[inline(always)]
            #[cfg(feature = "rayon")]
            fn par_reset_atomic(&mut self, order: Ordering) {
                T::par_reset_atomic(self, order)
            }
        }
)*}}

// Can't delegate on &T because reset requires &mut T
impl_atomic_delegation!(&mut T, Box<T>);
