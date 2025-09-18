/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Traits for slices of bit fields of constant width.
//!
//! Slices of bit fields are accessed with a logic similar to slices, but when
//! indexed with [`get`](BitFieldSlice::get) return an owned value of a [fixed
//! bit width](BitFieldSliceCore::bit_width). The associated implementation is
//! [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec).
//!
//! Implementing the
//! [`Index`](core::ops::Index)/[`IndexMut`](core::ops::IndexMut) traits would
//! be more natural and practical, but in certain cases it is impossible: in our
//! main use case, [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec), we
//! cannot implement [`Index`](core::ops::Index) because there is no way to
//! return a reference to a bit segment.
//!
//! There are three end-user traits: [`BitFieldSlice`], [`BitFieldSliceMut`] and
//! [`AtomicBitFieldSlice`]. The trait [`BitFieldSliceCore`] contains the common
//! methods, and in particular [`BitFieldSliceCore::bit_width`], which returns
//!  the bit width the values stored in the slice. All stored values must fit
//!  within this bit width.
//!
//! All the traits depends on a type parameter `W` that must implement [`Word`],
//! and which default to `usize`, but any type satisfying the [`Word`] trait can
//! be used, with the restriction that the bit width of the slice can be at most
//! the bit width of `W` as defined by [`AsBytes::BITS`]. Additionally, to
//! implement [`AtomicBitFieldSlice`], `W` must implement [`IntoAtomic`]. The
//! methods of all traits accept and return values of type `W`.
//!
//! If you need to iterate over a [`BitFieldSlice`], you can use
//! [`BitFieldSliceIterator`].
//!
//! Implementations must return always zero on a [`BitFieldSlice::get`] when the
//! bit width is zero. The behavior of a [`BitFieldSliceMut::set`] in the same
//! context is not defined.
//!
//! It is suggested that types implementing [`BitFieldSlice`] implement on a
//! reference [`IntoIterator`] with item `W` using [`BitFieldSliceIterator`] as
//! helper.
//!
//! We provide implementations for vectors and slices of all primitive atomic
//! and non-atomic unsigned integer types that view their elements as values
//! with a bit width equal to that of the type.
//!
//! # Simpler methods for atomic slices
//!
//! [`AtomicBitFieldSlice`] has rather cumbersome method names. There is however
//! a trait [`AtomicHelper`] that can be imported that will add to
//! [`AtomicBitFieldSlice`] equivalent methods without the `_atomic` infix. You
//! should be however careful to not mix [`AtomicHelper`] and [`BitFieldSlice`]
//! or a number of ambiguities in trait resolution will arise. In particular, if
//! you plan to use [`AtomicHelper`], we suggest that you do not import the
//! prelude.
//!
//! ```
//! # use sux::traits::bit_field_slice::{AtomicBitFieldSlice,AtomicHelper};
//! # use std::sync::atomic::Ordering;
//! let slice = sux::bits::AtomicBitFieldVec::<usize>::new(3, 3);
//! slice.set(0, 1, Ordering::Relaxed);
//! assert_eq!(slice.get(0, Ordering::Relaxed), 1);
//! ```

#![allow(clippy::result_unit_err)]
use common_traits::*;
use core::sync::atomic::*;
use core::{marker::PhantomData, ops::Deref};
use epserde::deser::{DeserializeInner, MemCase};
use mem_dbg::{MemDbg, MemSize};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{debug_assert_bounds, panic_if_out_of_bounds, panic_if_value};

/// A derived trait that the types used as a parameter for [`BitFieldSlice`] must satisfy.
/// To be usable in an [`AtomicBitFieldSlice`], the type must also implement [`IntoAtomic`].
pub trait Word: UnsignedInt + FiniteRangeNumber + AsBytes {}
impl<W: UnsignedInt + FiniteRangeNumber + AsBytes> Word for W {}

/// Common methods for [`BitFieldSlice`], [`BitFieldSliceMut`], and [`AtomicBitFieldSlice`].
///
/// The dependence on `W` is necessary to implement this trait on vectors and slices, as
/// we need the bit width of the values stored in the slice.
pub trait BitFieldSliceCore<W> {
    /// Returns the width of the slice.
    ///
    /// All elements stored in the slice must fit within this bit width.
    fn bit_width(&self) -> usize;
    /// Returns the length of the slice.
    fn len(&self) -> usize;
    /// Returns true if the slice has length zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A slice of bit fields of constant bit width.
pub trait BitFieldSlice<W: Word>: BitFieldSliceCore<W> {
    /// Returns the value at the specified index.
    ///
    /// # Safety
    ///
    /// `index` must be in [0..[len](`BitFieldSliceCore::len`)). No bounds checking is performed.
    unsafe fn get_unchecked(&self, index: usize) -> W;

    /// Returns the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    fn get(&self, index: usize) -> W {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index) }
    }
}

/// A mutable slice of bit fields of constant bit width.
pub trait BitFieldSliceMut<W: Word>: BitFieldSlice<W> {
    /// Returns the mask to apply to values to ensure they fit in
    /// [`bit_width`](BitFieldSliceCore::bit_width) bits.
    #[inline(always)]
    fn mask(&self) -> W {
        // TODO: Maybe testless?
        if self.bit_width() == 0 {
            W::ZERO
        } else {
            W::MAX >> (W::BITS as u32 - self.bit_width() as u32)
        }
    }

    /// Sets the element of the slice at the specified index.
    /// No bounds checking is performed.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`BitFieldSliceCore::len`));
    /// - `value` must fit withing [`BitFieldSliceCore::bit_width`] bits.
    ///
    /// No bound or bit-width check is performed.
    unsafe fn set_unchecked(&mut self, index: usize, value: W);

    /// Sets the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    fn set(&mut self, index: usize, value: W) {
        panic_if_out_of_bounds!(index, self.len());
        let bit_width = self.bit_width();
        // TODO: Maybe testless?
        let mask = if bit_width == 0 {
            W::ZERO
        } else {
            W::MAX >> (W::BITS as u32 - bit_width as u32)
        };

        panic_if_value!(value, mask, bit_width);
        unsafe {
            self.set_unchecked(index, value);
        }
    }

    /// Sets all values to zero.
    fn reset(&mut self);

    /// Sets all values to zero using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_reset(&mut self);

    /// Copy part of the content of the vector to another vector.
    ///
    /// At most `len` elements are copied, compatibly with the elements
    /// available in both vectors.
    ///
    /// # Arguments
    ///
    /// * `from`: the index of the first element to copy.
    ///
    /// * `dst`: the destination vector.
    ///
    /// * `to`: the index of the first element in the destination vector.
    ///
    /// * `len`: the maximum number of elements to copy.
    ///
    /// # Implementation Notes
    ///
    /// The default implementation is a simple loop that copies the elements one
    /// by one. It is expected to be implemented in a more efficient way.
    fn copy(&self, from: usize, dst: &mut Self, to: usize, len: usize) {
        // Reduce len to the elements available in both vectors
        let len = Ord::min(Ord::min(len, dst.len() - to), self.len() - from);
        for i in 0..len {
            dst.set(to + i, self.get(from + i));
        }
    }

    /// Applies a function to all elements of the slice in place without
    /// checking [bit widths](BitFieldSliceCore::bit_width).
    ///
    /// This method is semantically equivalent to:
    /// ```ignore
    /// for i in 0..self.len() {
    ///     self.set_unchecked(i, f(self.get_unchecked(i)));
    /// }
    /// ```
    /// and this is indeed the default implementation.
    ///
    /// See [`apply_in_place`](BitFieldSliceMut::apply_in_place) for examples.
    ///
    /// # Safety
    /// The function must return a value that fits the the [bit
    ///  width](BitFieldSliceCore::bit_width) of the slice.
    unsafe fn apply_in_place_unchecked<F>(&mut self, mut f: F)
    where
        F: FnMut(W) -> W,
        Self: BitFieldSlice<W>,
    {
        for idx in 0..self.len() {
            let value = unsafe { self.get_unchecked(idx) };
            let new_value = f(value);
            unsafe { self.set_unchecked(idx, new_value) };
        }
    }

    /// Applies a function to all elements of the slice in place.
    ///
    /// This method is semantically equivalent to:
    /// ```ignore
    /// for i in 0..self.len() {
    ///     self.set(i, f(self.get(i)));
    /// }
    /// ```
    /// and this is indeed the default implementation.
    ///
    /// The function is applied from the first element to the last: thus,
    /// it possible to compute cumulative sums as follows:
    ///
    /// ```
    /// # use sux::bits::BitFieldVec;
    /// # use sux::traits::BitFieldSliceMut;
    ///
    /// let mut vec = BitFieldVec::<u16>::new(9, 10);
    ///
    /// for i in 0..10 {
    ///     vec.set(i, i as u16);
    /// }
    ///
    /// let mut total = 0;
    /// vec.apply_in_place(|x| {
    ///     total += x;
    ///     total
    /// });
    /// ```
    fn apply_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(W) -> W,
        Self: BitFieldSlice<W>,
    {
        let bit_width = self.bit_width();
        let mask = self.mask();
        unsafe {
            self.apply_in_place_unchecked(|x| {
                let new_value = f(x);
                panic_if_value!(new_value, mask, bit_width);
                new_value
            });
        }
    }

    type ChunksMut<'a>: Iterator<Item: BitFieldSliceMut<W>>
    where
        Self: 'a;

    /// Tries and returns an iterator over non-overlapping mutable chunks of a
    /// bit-field slice, starting at the beginning of the slice.
    ///
    /// This might not always be possible; implementations must document when
    /// the method will success (see, for example, [the implementation for
    /// `BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec#impl-BitFieldSliceMut<W>-for-BitFieldVec<W,+B>)).
    ///
    /// When the slice len is not evenly divided by the chunk size, the last
    /// chunk of the iteration will be the remainder.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sux::prelude::*;
    /// # use bit_field_slice::*;
    /// # fn main() -> Result<(), ()> {

    /// let mut b = bit_field_vec![32; 4, 500, 2, 3, 1];
    /// for mut c in b.try_chunks_mut(2)? {
    ///     c.set(0, 5);
    /// }
    /// assert_eq!(b, bit_field_vec![32; 5, 500, 5, 3, 5]);
    /// # Ok(())
    /// # }
    /// ```
    fn try_chunks_mut(&mut self, chunk_size: usize) -> Result<Self::ChunksMut<'_>, ()>;

    /// Returns the backend of the slice as a mutable slice of `W`.
    fn as_mut_slice(&mut self) -> &mut [W];
}

/// A (tentatively) thread-safe slice of bit fields of constant bit width supporting atomic operations.
///
/// Different implementations might provide different atomicity guarantees. See
/// [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec) for an example.
pub trait AtomicBitFieldSlice<W: Word + IntoAtomic>: BitFieldSliceCore<W::AtomicType>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Returns the value at the specified index.
    ///
    /// # Safety
    /// `index` must be in [0..[len](`BitFieldSliceCore::len`)).
    /// No bound or bit-width check is performed.
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> W;

    /// Returns the value at the specified index.
    ///
    /// # Panics
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    fn get_atomic(&self, index: usize, order: Ordering) -> W {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_atomic_unchecked(index, order) }
    }

    /// Sets the element of the slice at the specified index.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](`BitFieldSliceCore::len`));
    /// - `value` must fit withing [`BitFieldSliceCore::bit_width`] bits.
    ///
    /// No bound or bit-width check is performed.
    unsafe fn set_atomic_unchecked(&self, index: usize, value: W, order: Ordering);

    /// Sets the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
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

/// An [`Iterator`] implementation returning the elements of a [`BitFieldSlice`].
///
/// You can easily implement [`IntoIterator`] on a reference to your type using this structure.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldSliceIterator<'a, W: Word, B: BitFieldSlice<W>> {
    slice: &'a B,
    index: usize,
    _marker: PhantomData<W>,
}

impl<'a, V: Word, B: BitFieldSlice<V>> BitFieldSliceIterator<'a, V, B> {
    pub fn new(slice: &'a B, from: usize) -> Self {
        if from > slice.len() {
            panic!("Start index out of bounds: {} > {}", from, slice.len());
        }
        Self {
            slice,
            index: from,
            _marker: PhantomData,
        }
    }
}

impl<W: Word, B: BitFieldSlice<W>> Iterator for BitFieldSliceIterator<'_, W, B> {
    type Item = W;
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

macro_rules! impl_core_delegation {
    ($($ty:ty),*) => {$(
        impl<W, T: BitFieldSliceCore<W> + ?Sized> BitFieldSliceCore<W> for $ty {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                T::bit_width(self)
            }
            #[inline(always)]
            fn len(&self) -> usize {
                T::len(self)
            }
            #[inline(always)]
            fn is_empty(&self) -> bool {
                T::is_empty(&self)
            }
        }
)*}}

impl_core_delegation!(&T, &mut T, Box<T>);

macro_rules! impl_core {
    ($($ty:ty),*) => {$(
        impl BitFieldSliceCore<$ty> for [$ty] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.len()
            }
        }

        impl BitFieldSliceCore<$ty> for Vec<$ty> {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.len()
            }
        }

        impl<const N: usize> BitFieldSliceCore<$ty> for [$ty; N] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$ty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.as_slice().len()
            }
        }
    )*};
}

impl_core!(u8, u16, u32, u64, u128, usize);
// This implementation is not necessary, but it avoids ambiguity
// with expressions like [1, 2, 3].len() when using the prelude.
// Without this implementation, the compiler complains that
// BitFieldSliceCore is not implemented for [i32; 3].
impl_core!(i8, i16, i32, i64, i128, isize);

macro_rules! impl_delegation {
    ($($ty:ty),*) => {$(
        impl<W: Word, T: BitFieldSlice<W> + ?Sized> BitFieldSlice<W> for $ty {
            #[inline(always)]
            fn get(&self, index: usize) -> W {
                self.deref().get(index)
            }
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> W {
                debug_assert_bounds!(index, self.len());
                unsafe { T::get_unchecked(self, index) }
            }
        }
)*}}

impl_delegation!(&T, &mut T, Box<T>);

macro_rules! impl_slice {
    ($($ty:ty),*) => {$(
        impl BitFieldSlice<$ty> for [$ty] {
            #[inline(always)]
            fn get(&self, index: usize) -> $ty {
                self[index]
            }

            unsafe fn get_unchecked(&self, index: usize) -> $ty {
                debug_assert_bounds!(index, self.len());
                unsafe { *self.get_unchecked(index) }
            }
        }

        impl BitFieldSlice<$ty> for Vec<$ty> {
            #[inline(always)]
            fn get(&self, index: usize) -> $ty {
                self[index]
            }

            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> $ty {
                debug_assert_bounds!(index, self.len());
                use std::ops::Deref;
                unsafe { *self.deref().get_unchecked(index) }
            }
        }

        impl<const N: usize> BitFieldSlice<$ty> for [$ty; N] {
            #[inline(always)]
            fn get(&self, index: usize) -> $ty {
                self[index]
            }

            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> $ty {
                let as_slice = self.as_slice();
                debug_assert_bounds!(index, as_slice.len());
                unsafe { *as_slice.get_unchecked(index) }
            }
        }
    )*};
}

impl_slice!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_mut_delegation {
    ($($ty:ty),*) => {$(
        impl<W: Word, T: BitFieldSliceMut<W> + ?Sized> BitFieldSliceMut<W> for $ty {
            #[inline(always)]
            fn mask(&self) -> W {
                self.deref().mask()
            }
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: W) {
                debug_assert_bounds!(index, self.len());
                unsafe { T::set_unchecked(self, index, value); }
            }
            #[inline(always)]
            fn set(&mut self, index: usize, value: W) {
                debug_assert_bounds!(index, self.len());
                T::set(self, index, value);
            }
            #[inline(always)]
            fn reset(&mut self) {
                T::reset(self);
            }
            #[cfg(feature = "rayon")]
            #[inline(always)]
            fn par_reset(&mut self) {
                T::par_reset(self);
            }
            #[inline(always)]
            fn copy(&self, from: usize, dst: &mut Self, to: usize, len: usize) {
                T::copy(self, from, dst, to, len)
            }
            #[inline(always)]
            unsafe fn apply_in_place_unchecked<F>(&mut self, f: F)
            where
                F: FnMut(W) -> W,
                Self: BitFieldSlice<W>,
            {
                unsafe { T::apply_in_place_unchecked(self, f); }
            }
            #[inline(always)]
            fn apply_in_place<F>(&mut self, f: F)
            where
                F: FnMut(W) -> W,
                Self: BitFieldSlice<W>,
            {
                T::apply_in_place(self, f);
            }
            type ChunksMut<'a>
                = T::ChunksMut<'a>
            where
                Self: 'a;
            #[inline(always)]
            fn try_chunks_mut(&mut self, chunk_size: usize) -> Result<Self::ChunksMut<'_>, ()> {
                T::try_chunks_mut(self, chunk_size)
            }
            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [W] {
                T::as_mut_slice(self)
            }
        }
)*}}

impl_mut_delegation!(&mut T, Box<T>);

macro_rules! impl_slice_mut {
    ($($ty:ty),*) => {$(
        impl BitFieldSliceMut<$ty> for [$ty] {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: $ty) {
                debug_assert_bounds!(index, <[$ty]>::len(self));
                *unsafe { self.get_unchecked_mut(index) } = value;
            }

            fn reset(&mut self) {
                for idx in 0..<[$ty]>::len(self) {
                    unsafe{BitFieldSliceMut::<$ty>::set_unchecked(self, idx, 0)};
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset(&mut self) {
                self.as_mut()
                    .par_iter_mut()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| { *w = 0 });
            }

            fn copy(&self, from: usize, dst: &mut Self, to: usize, len: usize) {
                let len = Ord::min(Ord::min(len, <[$ty]>::len(dst) - to), self.len() - from);
                dst.as_mut()[to..][..len].copy_from_slice(&self.as_ref()[from..][..len]);
            }

            type ChunksMut<'a> = core::slice::ChunksMut<'a, $ty> where Self: 'a;

            fn try_chunks_mut(&mut self, chunk_size: usize) -> Result<Self::ChunksMut<'_>, ()> {
                Ok(self.chunks_mut(chunk_size))
            }

            fn as_mut_slice(&mut self) -> &mut [$ty] {
                self
            }
        }
        impl BitFieldSliceMut<$ty> for Vec<$ty> {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: $ty) {
                debug_assert_bounds!(index, <Vec<$ty>>::len(self));
                *unsafe { self.get_unchecked_mut(index) } = value;
            }

            fn reset(&mut self) {
                for idx in 0..<[$ty]>::len(self) {
                    unsafe{ BitFieldSliceMut::<$ty>::set_unchecked(self, idx, 0) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset(&mut self) {
                self
                    .par_iter_mut()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| { *w = 0 });
            }

            fn copy(&self, from: usize, dst: &mut Self, to: usize, len: usize) {
                let len = Ord::min(Ord::min(len, <Vec<$ty>>::len(dst) - to), self.len() - from);
                dst[to..][..len].copy_from_slice(&self[from..][..len]);
            }

            type ChunksMut<'a> = core::slice::ChunksMut<'a, $ty> where Self: 'a;

            fn try_chunks_mut(&mut self, chunk_size: usize) -> Result<Self::ChunksMut<'_>, ()> {
                Ok(self.chunks_mut(chunk_size))
            }

            fn as_mut_slice(&mut self) -> &mut [$ty] {
                self
            }
        }

        impl<const N: usize> BitFieldSliceMut<$ty> for [$ty; N] {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: $ty) {
                debug_assert_bounds!(index, self.as_slice().len());
                *unsafe { self.get_unchecked_mut(index) } = value;
            }

            fn reset(&mut self) {
                for idx in 0..<[$ty]>::len(self) {
                    unsafe{ BitFieldSliceMut::<$ty>::set_unchecked(self, idx, 0) };
                }
            }

            #[cfg(feature = "rayon")]
            fn par_reset(&mut self) {
                self
                    .par_iter_mut()
                    .with_min_len(crate::RAYON_MIN_LEN)
                    .for_each(|w| { *w = 0 });
            }

            fn copy(&self, from: usize, dst: &mut Self, to: usize, len: usize) {
                let len = Ord::min(Ord::min(len, dst.as_slice().len() - to), self.as_slice().len() - from);
                dst[to..][..len].copy_from_slice(&self[from..][..len]);
            }

            type ChunksMut<'a> = core::slice::ChunksMut<'a, $ty> where Self: 'a;

            fn try_chunks_mut(&mut self, chunk_size: usize) -> Result<Self::ChunksMut<'_>, ()> {
                Ok(self.chunks_mut(chunk_size))
            }

            fn as_mut_slice(&mut self) -> &mut [$ty] {
                self
            }
        }

    )*};
}

impl_slice_mut!(u8, u16, u32, u64, u128, usize);

// Implementations for slices of atomic types

macro_rules! impl_core_atomic {
    ($($aty:ty),*) => {$(
        impl BitFieldSliceCore<$aty> for [$aty] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.len()
            }
        }

        impl BitFieldSliceCore<$aty> for Vec<$aty> {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.len()
            }
        }

        impl<const N: usize> BitFieldSliceCore<$aty> for [$aty; N] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                <$aty>::BITS as usize
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[$aty; N] as AsRef<[$aty]>>::as_ref(self).len()
            }
        }
    )*};
}

impl_core_atomic!(AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize);

macro_rules! impl_atomic {
    ($std:ty, $atomic:ty) => {
        impl AtomicBitFieldSlice<$std> for [$atomic] {
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

/// Helper trait eliminating `_atomic` from all methods of [`AtomicBitFieldSlice`]
/// using a blanked implementation.
///
/// Note that using this trait and [`BitFieldSlice`] in the same module might cause
/// ambiguity problems.
pub trait AtomicHelper<W: Word + IntoAtomic>: AtomicBitFieldSlice<W>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Delegates to [`AtomicBitFieldSlice::get_atomic_unchecked`]
    /// # Safety
    /// See [`AtomicBitFieldSlice::get_atomic_unchecked`]
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> W {
        unsafe { self.get_atomic_unchecked(index, order) }
    }

    /// Delegates to [`AtomicBitFieldSlice::set_atomic`]
    #[inline(always)]
    fn get(&self, index: usize, order: Ordering) -> W {
        self.get_atomic(index, order)
    }

    /// Delegates to [`AtomicBitFieldSlice::set_atomic_unchecked`]
    /// # Safety
    /// See [`AtomicBitFieldSlice::get_atomic_unchecked`]
    #[inline(always)]
    unsafe fn set_unchecked(&self, index: usize, value: W, order: Ordering) {
        unsafe { self.set_atomic_unchecked(index, value, order) }
    }

    /// Delegates to [`AtomicBitFieldSlice::set_atomic`]
    #[inline(always)]
    fn set(&self, index: usize, value: W, order: Ordering) {
        self.set_atomic(index, value, order)
    }

    /// Delegates to [`AtomicBitFieldSlice::reset_atomic`]
    #[inline(always)]
    fn reset(&mut self, order: Ordering) {
        self.reset_atomic(order);
    }
}

impl<T, W: Word + IntoAtomic> AtomicHelper<W> for T
where
    T: AtomicBitFieldSlice<W>,
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
}

// MemCase delegations

impl<S: DeserializeInner, W> BitFieldSliceCore<W> for MemCase<S>
where
    for<'a> S::DeserType<'a>: BitFieldSliceCore<W>,
{
    fn bit_width(&self) -> usize {
        unsafe { self.uncase_static().bit_width() }
    }

    fn len(&self) -> usize {
        unsafe { self.uncase_static().len() }
    }

    fn is_empty(&self) -> bool {
        unsafe { self.uncase_static().is_empty() }
    }
}

impl<S: DeserializeInner, W: Word> BitFieldSlice<W> for MemCase<S>
where
    for<'a> S::DeserType<'a>: BitFieldSlice<W>,
{
    unsafe fn get_unchecked(&self, index: usize) -> W {
        // SAFETY: We are just using the reference to invoke the method
        unsafe { self.uncase_static().get_unchecked(index) }
    }

    fn get(&self, index: usize) -> W {
        // SAFETY: We are just using the reference to invoke the method
        unsafe { self.uncase_static().get(index) }
    }
}
