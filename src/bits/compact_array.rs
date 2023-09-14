/*
 * SPDX-FileCopyrightText: 2023 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use anyhow::Result;
use epserde::*;
use std::sync::atomic::{compiler_fence, fence, AtomicUsize, Ordering};

const BITS: usize = core::mem::size_of::<usize>() * 8;

/// A fixed-length array of values of bounded bit width.
///
/// Elements are stored contiguously, with no padding bits (in particular,
/// unless the bit width is a power of two some elements will be stored
/// across word boundaries).
///
/// We provide an implementation
/// based on `Vec<usize>`, and one based on `Vec<AtomicUsize>`. The first
/// implementation implements [`VSlice`] and [`VSliceMut`], the second
/// implementation implements [`VSliceAtomic`].
///
/// In the second case we can provide some concurrency guarantees,
/// albeit not full-fledged thread safety: more precisely, we can
/// guarantee thread-safety if the bit width is a power of two; otherwise,
/// concurrent writes to values that cross word boundaries might end
/// up in different threads succeding in writing only part of a value.
/// If the user can guarantee that no two threads ever write to the same
/// boundary-crossing value, then no race condition can happen.
///

#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompactArray<B = Vec<usize>> {
    /// The underlying storage.
    data: B,
    /// The bit width of the values stored in the array.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: usize,
    /// The length of the array.
    len: usize,
}

fn mask(bit_width: usize) -> usize {
    if bit_width == 0 {
        0
    } else {
        usize::MAX >> (BITS - bit_width)
    }
}

impl CompactArray<Vec<usize>> {
    pub fn new(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = ((len * bit_width + BITS - 1) / BITS).max(1);
        Self {
            data: vec![0; n_of_words],
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }
}

impl CompactArray<Vec<AtomicUsize>> {
    pub fn new_atomic(bit_width: usize, len: usize) -> Self {
        // we need at least two words to avoid branches in the gets
        let n_of_words = ((len * bit_width + BITS - 1) / BITS).max(1);
        Self {
            data: (0..n_of_words).map(|_| AtomicUsize::new(0)).collect(),
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }
}

impl<B> CompactArray<B> {
    /// # Safety
    /// `len` * `bit_width` must be between 0 (included) the number of
    /// bits in `data` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, bit_width: usize, mask: usize, len: usize) -> Self {
        Self {
            data,
            bit_width,
            mask,
            len,
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize, usize) {
        (self.data, self.bit_width, self.mask, self.len)
    }
}

impl<T> VSliceCore for CompactArray<T> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= BITS);
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: AsRef<[usize]>> VSlice for CompactArray<T> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        if bit_index + self.bit_width <= BITS {
            (self.data.as_ref().get_unchecked(word_index) >> bit_index) & self.mask
        } else {
            (self.data.as_ref().get_unchecked(word_index) >> bit_index
                | self.data.as_ref().get_unchecked(word_index + 1) << (BITS - bit_index))
                & self.mask
        }
    }
}

impl VSliceMut for CompactArray<Vec<usize>> {
    // We reimplement set as we have the mask in the structure.

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSliceCore::len`))
    /// or the value does not fit in [`VSliceCore::bit_width`] bits.
    #[inline(always)]
    fn set(&mut self, index: usize, value: usize) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_unchecked(index, value);
        }
    }

    #[inline]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        if bit_index + self.bit_width <= BITS {
            let mut word = self.data.get_unchecked(word_index);
            word &= !(self.mask << bit_index);
            word |= value << bit_index;
            self.data.set_unchecked(word_index, word);
        } else {
            let mut word = self.data.get_unchecked(word_index);
            word &= (1 << bit_index) - 1;
            word |= value << bit_index;
            self.data.set_unchecked(word_index, word);

            let mut word = self.data.get_unchecked(word_index + 1);
            word &= !(self.mask >> (BITS - bit_index));
            word |= value >> (BITS - bit_index);
            self.data.set_unchecked(word_index + 1, word);
        }
    }
}

impl<T: AsRef<[AtomicUsize]>> VSliceAtomic for CompactArray<T> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> usize {
        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        if bit_index + self.bit_width <= BITS {
            (self.data.as_ref().get_unchecked(word_index).load(order) >> bit_index) & self.mask
        } else {
            (self.data.as_ref().get_unchecked(word_index).load(order) >> bit_index
                | self.data.as_ref().get_unchecked(word_index + 1).load(order)
                    << (BITS - bit_index))
                & self.mask
        }
    }

    // We reimplement set as we have the mask in the structure.

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`VSliceCore::len`))
    /// or the value does not fit in [`VSliceCore::bit_width`] bits.
    #[inline(always)]
    fn set(&self, index: usize, value: usize, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_unchecked(index, value, order);
        }
    }

    #[inline]
    unsafe fn set_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert!(self.bit_width != BITS);
        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        if bit_index + self.bit_width <= BITS {
            // this is consistent
            let mut current = self.data.as_ref().get_unchecked(word_index).load(order);
            loop {
                let mut new = current;
                new &= !(self.mask << bit_index);
                new |= value << bit_index;

                match self
                    .data
                    .as_ref()
                    .get_unchecked(word_index)
                    .compare_exchange(current, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => current = e,
                }
            }
        } else {
            let mut word = self.data.as_ref().get_unchecked(word_index).load(order);
            // try to wait for the other thread to finish
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= (1 << bit_index) - 1;
                new |= value << bit_index;

                match self
                    .data
                    .as_ref()
                    .get_unchecked(word_index)
                    .compare_exchange(word, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => word = e,
                }
            }
            fence(Ordering::Release);

            // ensure that the compiler does not reorder the two atomic operations
            // this should increase the probability of having consistency
            // between two concurrent writes as they will both execute the set
            // of the bits in the same order, and the release / acquire fence
            // should try to syncronize the threads as much as possible
            compiler_fence(Ordering::SeqCst);

            let mut word = self.data.as_ref().get_unchecked(word_index + 1).load(order);
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= !(self.mask >> (BITS - bit_index));
                new |= value >> (BITS - bit_index);

                match self
                    .data
                    .as_ref()
                    .get_unchecked(word_index + 1)
                    .compare_exchange(word, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => word = e,
                }
            }
            fence(Ordering::Release);
        }
    }
}

/// Provide conversion betweeen compact arrays whose backends
/// are [convertible](ConvertTo) into one another.
///
/// Many implementations of this trait are then used to
/// implement by delegation a corresponding [`From`].
impl<B, D> ConvertTo<CompactArray<D>> for CompactArray<B>
where
    B: ConvertTo<D>,
{
    #[inline]
    fn convert_to(self) -> Result<CompactArray<D>> {
        Ok(CompactArray {
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            data: self.data.convert_to()?,
        })
    }
}
/// Provide conversion from standard to atomic compact arrays.
impl From<CompactArray<Vec<usize>>> for CompactArray<Vec<AtomicUsize>> {
    #[inline]
    fn from(bm: CompactArray<Vec<usize>>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from atomic to standard compact arrays.
impl From<CompactArray<Vec<AtomicUsize>>> for CompactArray<Vec<usize>> {
    #[inline]
    fn from(bm: CompactArray<Vec<AtomicUsize>>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from references to standard compact arrays to
/// references to atomic compact arrays.
impl<'a> From<CompactArray<&'a [usize]>> for CompactArray<&'a [AtomicUsize]> {
    #[inline]
    fn from(bm: CompactArray<&'a [usize]>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from references to atomic compact arrays to
/// references to standard compact arrays.
impl<'a> From<CompactArray<&'a [AtomicUsize]>> for CompactArray<&'a [usize]> {
    #[inline]
    fn from(bm: CompactArray<&'a [AtomicUsize]>) -> Self {
        bm.convert_to().unwrap()
    }
}
