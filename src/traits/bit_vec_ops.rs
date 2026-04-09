/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Operations on bit vectors.
//!
//! `sux` does not provide a dedicated trait for bit vectors (whereas it
//! provides a trait for [bit-field slices](crate::traits::bit_field_slice)).
//! Rather, it considers anything that is `AsRef<[W]>` (where `W` implements
//! [`Word`]) and implements [`BitLength`] as a bit vector.
//!
//! This approach was chosen for efficiency reasons—all methods are implemented
//! as efficiently as possible for a concrete representation. Other crates opt
//! for a general trait representing a bit vector, but in our experiments the
//! resulting code was not as efficient as the code we obtain.
//!
//! The Rust type system makes the approach quite flexible. We cannot, however,
//! accommodate implicit representation of bit vectors (e.g., compressed or
//! algorithmic). This is in fact in line with Rust's philosophy—the
//! [`Index`](std::ops::Index) trait returns a reference, which forces an
//! explicit representation of sequences (an alternative approach is provided by
//! the [`value-traits`](https://crates.io/crates/value-traits) crate, which is used
//! by bit-field slices).
//!
//! All traits provided in this module are extension traits. They have no
//! unimplemented methods: just pulling them into scope will provide anything
//! that is `AsRef<[W]>` and implements [`BitLength`] with the operations of
//! a bit vector.
//!
//! Iteration on the bits of the vector, or on the positions of the ones or of the
//! zeros, is provided by means of structures that can be reused.
//!
//! The reference implementations using these traits are
//! [`BitVec`](crate::bits::BitVec) and
//! [`AtomicBitVec`](crate::bits::AtomicBitVec).

use ambassador::delegatable_trait;
use impl_tools::autoimpl;

/// A trait expressing a length in bits.
///
/// This trait is typically used in conjunction with [`Backend`](crate::traits::Backend) and
/// [`AsRef<[Backend::Word]>`](std::convert::AsRef) to provide word-based access
/// to a bit vector.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait]
pub trait BitLength {
    /// Returns a length in bits.
    fn len(&self) -> usize;
}

use crate::traits::Word;
use atomic_primitive::PrimitiveAtomicUnsigned;
use mem_dbg::{MemSize, MemDbg};
use num_primitive::PrimitiveInteger;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{iter::FusedIterator, marker::PhantomData, sync::atomic::Ordering};

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Bit index out of bounds: {} >= {}", $index, $len)
        }
    };
}

/// Operations for reading multi-bit values from a bit vector at arbitrary
/// bit positions.
///
/// Unlike [`BitVecOps`] and [`BitVecOpsMut`], this trait does not have a
/// blanket implementation, allowing different types to provide specialized
/// implementations (e.g., using unaligned reads).
pub trait BitVecValueOps<W: Word> {
    /// Reads `width` bits starting at bit position `pos`.
    ///
    /// # Panics
    ///
    /// Panics if `pos + width` exceeds the bit length or if `width` >
    /// `W::BITS`.
    fn get_value(&self, pos: usize, width: usize) -> W;

    /// Reads `width` bits starting at bit position `pos`, without bounds
    /// checks.
    ///
    /// # Safety
    ///
    /// - `pos + width` must not exceed the bit length of the underlying
    ///   storage.
    /// - `width` must be at most `W::BITS`.
    unsafe fn get_value_unchecked(&self, pos: usize, width: usize) -> W;
}

impl<W: Word, T: ?Sized + AsRef<[W]> + BitLength> BitVecOps<W> for T {}

/// Read-only operations on bit vectors.
pub trait BitVecOps<W: Word>: AsRef<[W]> + BitLength {
    /// Returns true if the bit of given index is set.
    #[inline]
    fn get(&self, index: usize) -> bool {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index) }
    }

    /// Returns true if the bit of given index is set, without
    /// bound checks.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded).
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        let bits_per_word = W::BITS as usize;
        let word_index = index / bits_per_word;
        let word = unsafe { *self.as_ref().get_unchecked(word_index) };
        (word >> (index % bits_per_word)) & W::ONE != W::ZERO
    }

    /// Returns an iterator over the bits of this bit vector as booleans.
    #[inline(always)]
    fn iter(&self) -> BitIter<'_, W, [W]> {
        BitIter::new(self.as_ref(), self.len())
    }

    /// Returns an iterator over the positions of the ones in this bit vector.
    fn iter_ones(&self) -> OnesIter<'_, W, [W]> {
        OnesIter::new(self.as_ref(), self.len())
    }

    /// Returns an iterator over the positions of the zeros in this bit vector.
    fn iter_zeros(&self) -> ZerosIter<'_, W, [W]> {
        ZerosIter::new(self.as_ref(), self.len())
    }

    /// A parallel version of
    /// [`BitCount::count_ones`](crate::traits::BitCount::count_ones).
    #[cfg(feature = "rayon")]
    fn par_count_ones(&self) -> usize {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let mut num_ones;
        num_ones = bits[..full_words]
            .par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .map(|x| x.count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones +=
                (self.as_ref()[full_words] << (bits_per_word - residual)).count_ones() as usize
        }

        num_ones
    }
}

impl<W: Word, T: AsRef<[W]> + AsMut<[W]> + BitLength> BitVecOpsMut<W> for T {}

/// Mutation operations on bit vectors.
pub trait BitVecOpsMut<W: Word>: AsRef<[W]> + AsMut<[W]> + BitLength {
    /// Sets the bit of given index to the given value.
    #[inline]
    fn set(&mut self, index: usize, value: bool) {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.set_unchecked(index, value) }
    }

    /// Sets the bit of given index to the given value without bound checks.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded).
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let bits_per_word = W::BITS as usize;
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        let bits = self.as_mut();
        // For constant values, this should be inlined with no test.
        unsafe {
            if value {
                *bits.get_unchecked_mut(word_index) |= W::ONE << bit_index;
            } else {
                *bits.get_unchecked_mut(word_index) &= !(W::ONE << bit_index);
            }
        }
    }

    /// Sets all bits to the given value.
    fn fill(&mut self, value: bool) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        let word_value: W = if value { !W::ZERO } else { W::ZERO };
        bits[..full_words].iter_mut().for_each(|x| *x = word_value);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (word_value & mask);
        }
    }

    /// Sets all bits to the given value using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_fill(&mut self, value: bool) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        let word_value: W = if value { !W::ZERO } else { W::ZERO };
        bits[..full_words]
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| *x = word_value);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (word_value & mask);
        }
    }

    /// Sets all bits to zero.
    fn reset(&mut self) {
        self.fill(false);
    }

    /// Sets all bits to zero using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        self.par_fill(false);
    }

    /// Flip all bits.
    fn flip(&mut self) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        bits[..full_words].iter_mut().for_each(|x| *x = !*x);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
        }
    }

    /// Flips all bits using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_flip(&mut self) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        bits[..full_words]
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| *x = !*x);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
        }
    }
}

/// An iterator over the bits of a bit vector as booleans.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct BitIter<'a, W: Word, B: ?Sized> {
    bits: &'a B,
    len: usize,
    next_bit_pos: usize,
    _phantom: PhantomData<W>,
}

impl<'a, W: Word, B: ?Sized + AsRef<[W]>> BitIter<'a, W, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * W::BITS as usize);
        BitIter {
            bits,
            len,
            next_bit_pos: 0,
            _phantom: PhantomData,
        }
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> Iterator for BitIter<'_, W, B> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let bits_per_word = W::BITS as usize;
        let word_idx = self.next_bit_pos / bits_per_word;
        let bit_idx = self.next_bit_pos % bits_per_word;
        let word = unsafe { *self.bits.as_ref().get_unchecked(word_idx) };
        let bit = (word >> bit_idx) & W::ONE;
        self.next_bit_pos += 1;
        Some(bit != W::ZERO)
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> ExactSizeIterator for BitIter<'_, W, B> {
    fn len(&self) -> usize {
        self.len - self.next_bit_pos
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> FusedIterator for BitIter<'_, W, B> {}

/// An iterator over the positions of the ones in a bit vector.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct OnesIter<'a, W: Word, B: ?Sized> {
    bits: &'a B,
    len: usize,
    word_idx: usize,
    word: W,
}

impl<'a, W: Word, B: ?Sized + AsRef<[W]>> OnesIter<'a, W, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * W::BITS as usize);
        let word = if bits.as_ref().is_empty() {
            W::ZERO
        } else {
            unsafe { *bits.as_ref().get_unchecked(0) }
        };
        Self {
            bits,
            len,
            word_idx: 0,
            word,
        }
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> Iterator for OnesIter<'_, W, B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let bits_per_word = W::BITS as usize;
        // find the next word with ones
        while self.word == W::ZERO {
            self.word_idx += 1;
            if self.word_idx >= self.bits.as_ref().len() {
                return None;
            }
            self.word = unsafe { *self.bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * bits_per_word) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - W::ONE;
            Some(res)
        }
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> FusedIterator for OnesIter<'_, W, B> {}

/// An iterator over the positions of the zeros in a bit vector.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct ZerosIter<'a, W: Word, B: ?Sized> {
    bits: &'a B,
    len: usize,
    word_idx: usize,
    word: W,
}

impl<'a, W: Word, B: ?Sized + AsRef<[W]>> ZerosIter<'a, W, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * W::BITS as usize);
        let word = if bits.as_ref().is_empty() {
            W::ZERO
        } else {
            unsafe { !*bits.as_ref().get_unchecked(0) }
        };
        Self {
            bits,
            len,
            word_idx: 0,
            word,
        }
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> Iterator for ZerosIter<'_, W, B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let bits_per_word = W::BITS as usize;
        // find the next flipped word with zeros
        while self.word == W::ZERO {
            self.word_idx += 1;
            if self.word_idx >= self.bits.as_ref().len() {
                return None;
            }
            self.word = unsafe { !*self.bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest zero bit index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * bits_per_word) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - W::ONE;
            Some(res)
        }
    }
}

impl<W: Word, B: ?Sized + AsRef<[W]>> FusedIterator for ZerosIter<'_, W, B> {}

impl<A: PrimitiveAtomicUnsigned<Value: Word>, T: ?Sized + AsRef<[A]> + BitLength> AtomicBitVecOps<A>
    for T
{
}

/// Operations on atomic bit vectors.
///
/// Parameterized by the atomic type `A` (e.g., `AtomicU64`), not the word type.
/// This avoids method-resolution ambiguity with [`BitVecOpsMut`], because
/// [`PrimitiveAtomicUnsigned`] is only implemented for atomic types, so the compiler can
/// definitively rule out non-atomic backends.
pub trait AtomicBitVecOps<A: PrimitiveAtomicUnsigned<Value: Word>>: AsRef<[A]> + BitLength {
    /// Returns true if the bit of given index is set.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    fn get(&self, index: usize, ordering: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.get_unchecked(index, ordering) }
    }

    /// Sets the bit of given index to the given value.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    fn set(&self, index: usize, value: bool, ordering: Ordering) {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.set_unchecked(index, value, ordering) }
    }

    /// Sets the bit of given index to the given value and returns the previous
    /// value.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    fn swap(&self, index: usize, value: bool, ordering: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len());
        unsafe { self.swap_unchecked(index, value, ordering) }
    }

    /// Returns true if the bit of given index is set.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded).
    #[inline]
    unsafe fn get_unchecked(&self, index: usize, ordering: Ordering) -> bool {
        let bits_per_word = A::Value::BITS as usize;
        let word_index = index / bits_per_word;
        let bits = self.as_ref();
        let word = unsafe { bits.get_unchecked(word_index).load(ordering) };
        (word >> (index % bits_per_word)) & A::Value::ONE != A::Value::ZERO
    }

    /// Sets the bit of given index to the given value.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded).
    #[inline]
    unsafe fn set_unchecked(&self, index: usize, value: bool, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        let bits = self.as_ref();

        // For constant values, this should be inlined with no test.
        unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(A::Value::ONE << bit_index, ordering);
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(A::Value::ONE << bit_index), ordering);
            }
        }
    }

    /// Sets the bit of given index to the given value and returns the previous
    /// value, without bound checks.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded).
    #[inline]
    unsafe fn swap_unchecked(&self, index: usize, value: bool, ordering: Ordering) -> bool {
        let bits_per_word = A::Value::BITS as usize;
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        let bits = self.as_ref();

        let old_word = unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(A::Value::ONE << bit_index, ordering)
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(A::Value::ONE << bit_index), ordering)
            }
        };

        (old_word >> bit_index) & A::Value::ONE != A::Value::ZERO
    }

    /// Sets all bits to the given value.
    fn fill(&mut self, value: bool, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let word_value: A::Value = if value {
            !A::Value::ZERO
        } else {
            A::Value::ZERO
        };
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .iter()
            .for_each(|x| x.store(word_value, ordering));
        if residual != 0 {
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            bits[full_words].store(
                (bits[full_words].load(ordering) & !mask) | (word_value & mask),
                ordering,
            );
        }
    }

    /// Sets all bits to the given value using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_fill(&mut self, value: bool, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let word_value: A::Value = if value {
            !A::Value::ZERO
        } else {
            A::Value::ZERO
        };

        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| x.store(word_value, ordering));
        if residual != 0 {
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            bits[full_words].store(
                (bits[full_words].load(ordering) & !mask) | (word_value & mask),
                ordering,
            );
        }
    }

    /// Sets all bits to zero.
    fn reset(&mut self, ordering: Ordering) {
        self.fill(false, ordering);
    }

    /// Sets all bits to zero using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_reset(&mut self, ordering: Ordering) {
        self.par_fill(false, ordering);
    }

    /// Flips all bits.
    fn flip(&mut self, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .iter()
            .for_each(|x| _ = x.fetch_xor(!A::Value::ZERO, ordering));
        if residual != 0 {
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            let last_word = bits[full_words].load(ordering);
            bits[full_words].store((last_word & !mask) | (!last_word & mask), ordering);
        }
    }

    /// Flips all bits using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_flip(&mut self, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| _ = x.fetch_xor(!A::Value::ZERO, ordering));
        if residual != 0 {
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            let last_word = bits[full_words].load(ordering);
            bits[full_words].store((last_word & !mask) | (!last_word & mask), ordering);
        }
    }

    /// A parallel version of
    /// [`BitCount::count_ones`](`crate::traits::BitCount::count_ones`).
    #[cfg(feature = "rayon")]
    fn par_count_ones(&self) -> usize {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let mut num_ones;
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        num_ones = bits[..full_words]
            .par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words].load(Ordering::Relaxed) << (bits_per_word - residual))
                .count_ones() as usize
        }
        num_ones
    }

    /// Returns an iterator over the bits of this atomic bit vector.
    ///
    /// Note that modifying the bit vector while iterating over it will lead to
    /// behavior depending on processor scheduling and memory model.
    /// Nonetheless, all returned values have been valid at some point during
    /// the iteration.
    #[inline(always)]
    fn iter(&self) -> AtomicBitIter<'_, A, [A]> {
        AtomicBitIter::new(self.as_ref(), self.len())
    }
}

/// An iterator over the bits of an atomic bit vector as booleans.
///
/// Note that modifying the bit vector while iterating over it will lead to
/// behavior depending on processor scheduling and memory model.
#[derive(Debug, MemSize, MemDbg)]
pub struct AtomicBitIter<'a, A, B: ?Sized> {
    bits: &'a B,
    len: usize,
    next_bit_pos: usize,
    _phantom: PhantomData<A>,
}

impl<'a, A: PrimitiveAtomicUnsigned<Value: Word>, B: ?Sized + AsRef<[A]>> AtomicBitIter<'a, A, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * A::Value::BITS as usize);
        AtomicBitIter {
            bits,
            len,
            next_bit_pos: 0,
            _phantom: PhantomData,
        }
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>, B: ?Sized + AsRef<[A]>> Iterator
    for AtomicBitIter<'_, A, B>
{
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let bits_per_word = A::Value::BITS as usize;
        let word_idx = self.next_bit_pos / bits_per_word;
        let bit_idx = self.next_bit_pos % bits_per_word;
        let word = unsafe {
            self.bits
                .as_ref()
                .get_unchecked(word_idx)
                .load(Ordering::Relaxed)
        };
        let bit = (word >> bit_idx) & A::Value::ONE;
        self.next_bit_pos += 1;
        Some(bit != A::Value::ZERO)
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>, B: ?Sized + AsRef<[A]>> ExactSizeIterator
    for AtomicBitIter<'_, A, B>
{
    fn len(&self) -> usize {
        self.len - self.next_bit_pos
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>, B: ?Sized + AsRef<[A]>> FusedIterator
    for AtomicBitIter<'_, A, B>
{
}
