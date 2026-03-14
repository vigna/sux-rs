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
//! All the traits depend on a type parameter `W` that must implement [`Word`],
//! and which defaults to `usize`, but any type satisfying the [`Word`] trait
//! can be used, with the restriction that the bit width of the slice can be at
//! most the bit width of `W` as defined by
//! [`PrimitiveInteger::BITS`](num_primitive::PrimitiveInteger::BITS).
//! Additionally, to implement [`AtomicBitFieldSlice`], `W` must implement
//! [`AtomicPrimitive`](atomic_primitive::AtomicPrimitive). The methods of all
//! traits accept and return values of type `W`.
//!
//! Implementations must always return zero upon a read operation when the bit
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
use atomic_primitive::{AtomicPrimitive, PrimitiveAtomicInteger};
use core::sync::atomic::Ordering;
use impl_tools::autoimpl;
use num_primitive::{PrimitiveNumberAs, PrimitiveUnsigned};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use value_traits::slices::{SliceByValue, SliceByValueMut};

use crate::{debug_assert_bounds, panic_if_out_of_bounds, panic_if_value};

/// A derived trait that the types used as a parameter for [`BitFieldSlice`] must satisfy.
/// To be usable in an [`AtomicBitFieldSlice`], the type must also implement
/// [`AtomicPrimitive`].
pub trait Word: PrimitiveUnsigned + PrimitiveNumberAs<u128> {
    const ZERO: Self;
}

macro_rules! impl_word {
    ($($ty:ty),*) => {
        $(impl Word for $ty {
            const ZERO: Self = 0;
        })*
    };
}

impl_word!(u8, u16, u32, u64, u128, usize);

/// The default word type for bit-storage structures on the current platform.
///
/// On 64-bit platforms this is `u64`; on 32-bit platforms (including WASM) it is
/// `u32`. Structures like [`BitVec`](crate::bits::BitVec) default to
/// `Vec<PlatformWord>` backing, keeping storage native to the platform while
/// allowing explicit `u64` backing when 64-bit words are needed
/// (e.g., for [`Rank9`](crate::rank_sel::Rank9)).
#[cfg(target_pointer_width = "64")]
pub type PlatformWord = u64;
#[cfg(not(target_pointer_width = "64"))]
pub type PlatformWord = u32;

/// Common method for [`BitFieldSlice`], [`BitFieldSliceMut`], and
/// [`AtomicBitFieldSlice`].
///
/// The dependence on `W` is necessary to implement this trait on vectors and
/// slices, as we need the bit width of the values stored in the slice.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait BitWidth<W> {
    /// Returns the bit width of the slice.
    ///
    /// All elements stored in the slice must fit within this bit width.
    fn bit_width(&self) -> usize;
}

/// A slice of bit fields of constant bit width.
///
/// This trait combines [`SliceByValue`] and [`BitWidth`]. Additionally,
/// it provides the method [`as_slice`](BitFieldSlice::as_slice) to
/// access the backend of the slice.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
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
#[autoimpl(for<T: trait + ?Sized> &mut T, Box<T>)]
pub trait BitFieldSliceMut<W: Word>: BitFieldSlice<W> + SliceByValueMut<Value = W> {
    /// Returns the mask to apply to values to ensure they fit in
    /// [`bit_width`](BitWidth::bit_width) bits.
    #[inline(always)]
    fn mask(&self) -> W {
        if self.bit_width() == 0 {
            W::ZERO
        } else {
            W::MAX >> (W::BITS - self.bit_width() as u32)
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

/// Bit width for atomic slices.
///
/// This trait is separate from [`BitWidth`] because a blanket impl
/// `impl<A: PrimitiveAtomicInteger> BitWidth<A> for [A]` would conflict
/// with `impl<W: Word> BitWidth<W> for [W]` — the compiler cannot prove
/// that [`Word`] and [`PrimitiveAtomicInteger`] are disjoint. A dedicated
/// trait sidesteps the overlap entirely.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
pub trait AtomicBitWidth {
    /// Returns the bit width of the atomic slice.
    fn atomic_bit_width(&self) -> usize;
}

impl<A: PrimitiveAtomicInteger> AtomicBitWidth for [A] {
    #[inline(always)]
    fn atomic_bit_width(&self) -> usize {
        A::BITS as usize
    }
}

impl<A: PrimitiveAtomicInteger> AtomicBitWidth for Vec<A> {
    #[inline(always)]
    fn atomic_bit_width(&self) -> usize {
        A::BITS as usize
    }
}

impl<A: PrimitiveAtomicInteger, const N: usize> AtomicBitWidth for [A; N] {
    #[inline(always)]
    fn atomic_bit_width(&self) -> usize {
        A::BITS as usize
    }
}

/// A (tentatively) thread-safe slice of bit fields of constant bit width
/// supporting atomic operations.
///
/// Different implementations might provide different atomicity guarantees. See
/// [`BitFieldVec`](crate::bits::bit_field_vec::BitFieldVec) for an example.
#[autoimpl(for<T: trait + ?Sized> &mut T, Box<T>)]
pub trait AtomicBitFieldSlice<W: Word + AtomicPrimitive>: AtomicBitWidth
where
    W::Atomic: PrimitiveAtomicInteger,
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
    /// May panic if the index is not in [0..[len](SliceByValue::len))
    fn get_atomic(&self, index: usize, order: Ordering) -> W {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_atomic_unchecked(index, order) }
    }

    /// Sets the element of the slice at the specified index.
    ///
    /// # Safety
    /// - `index` must be in [0..[len](SliceByValue::len));
    /// - `value` must fit within [`BitWidth::bit_width`] bits.
    ///
    /// No bound or bit-width check is performed.
    unsafe fn set_atomic_unchecked(&self, index: usize, value: W, order: Ordering);

    /// Sets the element of the slice at the specified index.
    ///
    /// May panic if the index is not in [0..[len](SliceByValue::len))
    /// or the value does not fit in [`AtomicBitWidth::atomic_bit_width`] bits.
    fn set_atomic(&self, index: usize, value: W, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len());
        let bw = self.atomic_bit_width();

        let mask = if bw == 0 {
            W::ZERO
        } else {
            W::MAX >> (W::BITS - bw as u32)
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

impl<W: Word> BitWidth<W> for [W] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        W::BITS as usize
    }
}

impl<W: Word> BitWidth<W> for Vec<W> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        W::BITS as usize
    }
}

impl<W: Word, const N: usize> BitWidth<W> for [W; N] {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        W::BITS as usize
    }
}


impl<W: Word> BitFieldSlice<W> for [W] {
    fn as_slice(&self) -> &[W] {
        self
    }
}

impl<W: Word> BitFieldSlice<W> for Vec<W> {
    fn as_slice(&self) -> &[W] {
        self
    }
}

impl<W: Word, const N: usize> BitFieldSlice<W> for [W; N] {
    fn as_slice(&self) -> &[W] {
        self
    }
}


impl<W: Word> BitFieldSliceMut<W> for [W] {
    fn reset(&mut self) {
        self.fill(W::ZERO);
    }

    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        self.as_mut()
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|w| { *w = W::ZERO });
    }

    fn as_mut_slice(&mut self) -> &mut [W] {
        self
    }
}

impl<W: Word> BitFieldSliceMut<W> for Vec<W> {
    #[inline(always)]
    fn reset(&mut self) {
        self.fill(W::ZERO);
    }

    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        self
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|w| { *w = W::ZERO });
    }

    fn as_mut_slice(&mut self) -> &mut [W] {
        self
    }
}

impl<W: Word, const N: usize> BitFieldSliceMut<W> for [W; N] {
    #[inline(always)]
    fn reset(&mut self) {
        self.fill(W::ZERO);
    }

    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        self
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|w| { *w = W::ZERO });
    }

    fn as_mut_slice(&mut self) -> &mut [W] {
        self
    }
}


// Generic implementations for slices/vectors of atomic types.
//
// These impls are parameterized by the atomic type A (e.g., AtomicU64)
// and derive the value type W from A::Value (e.g., u64). This avoids
// the associated-type projection ambiguity that occurs when writing
// `impl<W> ... for [W::Atomic]` — Rust can't resolve methods on
// `[W::Atomic]` because it cannot infer W from the projection.

impl<A: PrimitiveAtomicInteger> AtomicBitFieldSlice<A::Value> for [A]
where
    A::Value: Word + AtomicPrimitive<Atomic = A>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        <[A]>::len(self)
    }

    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> A::Value {
        debug_assert_bounds!(index, self.len());
        unsafe { self.get_unchecked(index).load(order) }
    }

    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: A::Value, order: Ordering) {
        debug_assert_bounds!(index, self.len());
        unsafe {
            self.get_unchecked(index).store(value, order);
        }
    }

    fn reset_atomic(&mut self, order: Ordering) {
        for idx in 0..self.len() {
            unsafe { self.set_atomic_unchecked(idx, A::Value::ZERO, order) };
        }
    }

    #[cfg(feature = "rayon")]
    fn par_reset_atomic(&mut self, order: Ordering) {
        self.par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|w| w.store(A::Value::ZERO, order));
    }
}

impl<A: PrimitiveAtomicInteger> AtomicBitFieldSlice<A::Value> for Vec<A>
where
    A::Value: Word + AtomicPrimitive<Atomic = A>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> A::Value {
        debug_assert_bounds!(index, self.len());
        unsafe { self.as_slice().get_unchecked(index).load(order) }
    }

    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: A::Value, order: Ordering) {
        debug_assert_bounds!(index, self.len());
        unsafe {
            self.as_slice().get_unchecked(index).store(value, order);
        }
    }

    fn reset_atomic(&mut self, order: Ordering) {
        for idx in 0..self.len() {
            unsafe { self.set_atomic_unchecked(idx, A::Value::ZERO, order) };
        }
    }

    #[cfg(feature = "rayon")]
    fn par_reset_atomic(&mut self, order: Ordering) {
        self.par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|w| w.store(A::Value::ZERO, order));
    }
}

impl<A: PrimitiveAtomicInteger, const N: usize> AtomicBitFieldSlice<A::Value> for [A; N]
where
    A::Value: Word + AtomicPrimitive<Atomic = A>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> A::Value {
        debug_assert_bounds!(index, self.len());
        unsafe { self.as_slice().get_unchecked(index).load(order) }
    }

    #[inline(always)]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: A::Value, order: Ordering) {
        debug_assert_bounds!(index, self.len());
        unsafe {
            self.as_slice().get_unchecked(index).store(value, order);
        }
    }

    fn reset_atomic(&mut self, order: Ordering) {
        for idx in 0..self.len() {
            unsafe { self.set_atomic_unchecked(idx, A::Value::ZERO, order) };
        }
    }

    #[cfg(feature = "rayon")]
    fn par_reset_atomic(&mut self, order: Ordering) {
        self.par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|w| w.store(A::Value::ZERO, order));
    }
}

