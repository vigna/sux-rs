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
//! Rather, it considers anything that is `AsRef<[usize]>` and implements
//! [`BitLength`] as a bit vector.
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
//! that is `AsRef<[usize]>` and implements [`BitLength`] with the operations of
//! a bit vector.
//!
//! Iteration on the bits of the vector, or on the positions of the ones or of the
//! zeros, is provided by means of structures that can be reused.
//!
//! The reference implementations using these traits are
//! [`BitVec`](crate::bits::BitVec) and
//! [`AtomicBitVec`](crate::bits::AtomicBitVec).

use crate::traits::BitLength;
use mem_dbg::{MemDbg, MemSize};
use std::{
    iter::FusedIterator,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "rayon")]
use crate::RAYON_MIN_LEN;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub const BITS: usize = usize::BITS as usize;

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Bit index out of bounds: {} >= {}", $index, $len)
        }
    };
}

impl<T: ?Sized + AsRef<[usize]> + BitLength> BitVecOps for T {}

/// Read-only operations on bit vectors.
pub trait BitVecOps: AsRef<[usize]> + BitLength {
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
        let word_index = index / BITS;
        let word = unsafe { self.as_ref().get_unchecked(word_index) };
        (word >> (index % BITS)) & 1 != 0
    }

    /// Returns an iterator over the bits of this bit vector as booleans.
    #[inline(always)]
    fn iter(&self) -> BitIter<'_, [usize]> {
        BitIter::new(self.as_ref(), self.len())
    }

    /// Returns an iterator over the positions of the ones in this bit vector.
    fn iter_ones(&self) -> OnesIter<'_, [usize]> {
        OnesIter::new(self.as_ref(), self.len())
    }

    /// Returns an iterator over the positions of the zeros in this bit vector.
    fn iter_zeros(&self) -> ZerosIter<'_, [usize]> {
        ZerosIter::new(self.as_ref(), self.len())
    }

    /// A parallel version of
    /// [`BitCount::count_ones`](crate::traits::BitCount::count_ones).
    #[cfg(feature = "rayon")]
    fn par_count_ones(&self) -> usize {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        let mut num_ones;
        num_ones = bits[..full_words]
            .par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .map(|x| x.count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (self.as_ref()[full_words] << (BITS - residual)).count_ones() as usize
        }

        num_ones
    }
}

impl<T: AsRef<[usize]> + AsMut<[usize]> + BitLength> BitVecOpsMut for T {}

/// Mutation operations on bit vectors.
pub trait BitVecOpsMut: AsRef<[usize]> + AsMut<[usize]> + BitLength {
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
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let bits = self.as_mut();
        // TODO: no test?
        // For constant values, this should be inlined with no test.
        unsafe {
            if value {
                *bits.get_unchecked_mut(word_index) |= 1 << bit_index;
            } else {
                *bits.get_unchecked_mut(word_index) &= !(1 << bit_index);
            }
        }
    }

    /// Sets all bits to the given value.
    fn fill(&mut self, value: bool) {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_mut();
        let word_value = if value { !0 } else { 0 };
        bits[..full_words].iter_mut().for_each(|x| *x = word_value);
        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words] = (bits[full_words] & !mask) | (word_value & mask);
        }
    }

    /// Sets all bits to the given value using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_fill(&mut self, value: bool) {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_mut();
        let word_value = if value { !0 } else { 0 };
        bits[..full_words]
            .par_iter_mut()
            .with_min_len(RAYON_MIN_LEN)
            .for_each(|x| *x = word_value);
        if residual != 0 {
            let mask = (1 << residual) - 1;
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
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_mut();
        bits[..full_words].iter_mut().for_each(|x| *x = !*x);
        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
        }
    }

    /// Flips all bits using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_flip(&mut self) {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_mut();
        bits[..full_words]
            .par_iter_mut()
            .with_min_len(RAYON_MIN_LEN)
            .for_each(|x| *x = !*x);
        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
        }
    }
}

/// An iterator over the bits of a bit vector as booleans.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitIter<'a, B: ?Sized> {
    bits: &'a B,
    len: usize,
    next_bit_pos: usize,
}

impl<'a, B: ?Sized + AsRef<[usize]>> BitIter<'a, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * BITS);
        BitIter {
            bits,
            len,
            next_bit_pos: 0,
        }
    }
}

impl<B: ?Sized + AsRef<[usize]>> Iterator for BitIter<'_, B> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let word_idx = self.next_bit_pos / BITS;
        let bit_idx = self.next_bit_pos % BITS;
        let word = unsafe { *self.bits.as_ref().get_unchecked(word_idx) };
        let bit = (word >> bit_idx) & 1;
        self.next_bit_pos += 1;
        Some(bit != 0)
    }
}

impl<B: ?Sized + AsRef<[usize]>> ExactSizeIterator for BitIter<'_, B> {
    fn len(&self) -> usize {
        self.len - self.next_bit_pos
    }
}

impl<B: ?Sized + AsRef<[usize]>> FusedIterator for BitIter<'_, B> {}

/// An iterator over the positions of the ones in a bit vector.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct OnesIter<'a, B: ?Sized> {
    bits: &'a B,
    len: usize,
    word_idx: usize,
    // BitVec is currently implemented only for AsRef<[usize]>.
    word: usize,
}

impl<'a, B: ?Sized + AsRef<[usize]>> OnesIter<'a, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * BITS);
        let word = if bits.as_ref().is_empty() {
            0
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

impl<B: ?Sized + AsRef<[usize]>> Iterator for OnesIter<'_, B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // find the next word with ones
        while self.word == 0 {
            self.word_idx += 1;
            if self.word_idx >= self.bits.as_ref().len() {
                return None;
            }
            self.word = unsafe { *self.bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * BITS) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - 1;
            Some(res)
        }
    }
}

impl<B: ?Sized + AsRef<[usize]>> FusedIterator for OnesIter<'_, B> {}

/// An iterator over the positions of the zeros in a bit vector.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct ZerosIter<'a, B: ?Sized> {
    bits: &'a B,
    len: usize,
    word_idx: usize,
    // BitVec is currently implemented only for AsRef<[usize]>.
    word: usize,
}

impl<'a, B: ?Sized + AsRef<[usize]>> ZerosIter<'a, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * BITS);
        let word = if bits.as_ref().is_empty() {
            0
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

impl<B: ?Sized + AsRef<[usize]>> Iterator for ZerosIter<'_, B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // find the next flipped word with zeros
        while self.word == 0 {
            self.word_idx += 1;
            if self.word_idx >= self.bits.as_ref().len() {
                return None;
            }
            self.word = unsafe { !*self.bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest zero bit index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * BITS) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - 1;
            Some(res)
        }
    }
}

impl<B: ?Sized + AsRef<[usize]>> FusedIterator for ZerosIter<'_, B> {}

impl<T: ?Sized + AsRef<[AtomicUsize]> + BitLength> AtomicBitVecOps for T {}

/// Operations on atomic bit vectors.
pub trait AtomicBitVecOps: AsRef<[AtomicUsize]> + BitLength {
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
        let word_index = index / BITS;
        let bits = self.as_ref();
        let word = unsafe { bits.get_unchecked(word_index).load(ordering) };
        (word >> (index % BITS)) & 1 != 0
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
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let bits = self.as_ref();

        // For constant values, this should be inlined with no test.
        unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(1 << bit_index, ordering);
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(1 << bit_index), ordering);
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
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let bits = self.as_ref();

        let old_word = unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(1 << bit_index, ordering)
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(1 << bit_index), ordering)
            }
        };

        (old_word >> (bit_index)) & 1 != 0
    }

    /// Sets all bits to the given value.
    fn fill(&mut self, value: bool, ordering: Ordering) {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        let word_value = if value { !0 } else { 0 };
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .iter()
            .for_each(|x| x.store(word_value, ordering));
        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words].store(
                (bits[full_words].load(ordering) & !mask) | (word_value & mask),
                ordering,
            );
        }
    }

    /// Sets all bits to the given value using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_fill(&mut self, value: bool, ordering: Ordering) {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        let word_value = if value { !0 } else { 0 };

        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .for_each(|x| x.store(word_value, ordering));
        if residual != 0 {
            let mask = (1 << residual) - 1;
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
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .iter()
            .for_each(|x| _ = x.fetch_xor(!0, ordering));
        if residual != 0 {
            let mask = (1 << residual) - 1;
            let last_word = bits[full_words].load(ordering);
            bits[full_words].store((last_word & !mask) | (!last_word & mask), ordering);
        }
    }

    /// Flips all bits using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_flip(&mut self, ordering: Ordering) {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .for_each(|x| _ = x.fetch_xor(!0, ordering));
        if residual != 0 {
            let mask = (1 << residual) - 1;
            let last_word = bits[full_words].load(ordering);
            bits[full_words].store((last_word & !mask) | (!last_word & mask), ordering);
        }
    }

    /// A parallel version of
    /// [`BitCount::count_ones`](`crate::traits::BitCount::count_ones`).
    #[cfg(feature = "rayon")]
    fn par_count_ones(&self) -> usize {
        use crate::RAYON_MIN_LEN;

        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        let mut num_ones;
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        num_ones = bits[..full_words]
            .par_iter()
            .with_min_len(RAYON_MIN_LEN)
            .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words].load(Ordering::Relaxed) << (BITS - residual)).count_ones()
                as usize
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
    fn iter(&self) -> AtomicBitIter<'_, [AtomicUsize]> {
        AtomicBitIter::new(self.as_ref(), self.len())
    }
}

/// An iterator over the bits of an atomic bit vector as booleans.
///
/// Note that modifying the bit vector while iterating over it will lead to
/// behavior depending on processor scheduling and memory model.
#[derive(Debug, MemDbg, MemSize)]
pub struct AtomicBitIter<'a, B: ?Sized> {
    bits: &'a B,
    len: usize,
    next_bit_pos: usize,
}

impl<'a, B: ?Sized + AsRef<[AtomicUsize]>> AtomicBitIter<'a, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        debug_assert!(len <= bits.as_ref().len() * BITS);
        AtomicBitIter {
            bits,
            len,
            next_bit_pos: 0,
        }
    }
}

impl<B: ?Sized + AsRef<[AtomicUsize]>> Iterator for AtomicBitIter<'_, B> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let word_idx = self.next_bit_pos / BITS;
        let bit_idx = self.next_bit_pos % BITS;
        let word = unsafe {
            self.bits
                .as_ref()
                .get_unchecked(word_idx)
                .load(Ordering::Relaxed)
        };
        let bit = (word >> bit_idx) & 1;
        self.next_bit_pos += 1;
        Some(bit != 0)
    }
}

impl<B: ?Sized + AsRef<[AtomicUsize]>> ExactSizeIterator for AtomicBitIter<'_, B> {
    fn len(&self) -> usize {
        self.len - self.next_bit_pos
    }
}

impl<B: ?Sized + AsRef<[AtomicUsize]>> FusedIterator for AtomicBitIter<'_, B> {}
