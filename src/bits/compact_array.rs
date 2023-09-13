/*
 * SPDX-FileCopyrightText: 2023 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::*;
use anyhow::Result;
use epserde::*;
use std::sync::atomic::{compiler_fence, fence, AtomicUsize, Ordering};

const BITS: usize = core::mem::size_of::<usize>() * 8;

/// A fixed-length array of values of bounded bit width. We provide an implementation
/// based on `Vec<usize>`, and one based on `Vec<AtomicUsize>`. In the second case we can
/// provide some concurrency guarantee.
#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompactArray<B> {
    data: B,
    bit_width: usize,
    len: usize,
}

impl CompactArray<Vec<usize>> {
    pub fn new(bit_width: usize, len: usize) -> Self {
        #[cfg(not(any(feature = "testless_read", feature = "testless_write")))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (len * bit_width + BITS - 1) / BITS;
        #[cfg(any(feature = "testless_read", feature = "testless_write"))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (1 + (len * bit_width + BITS - 1) / BITS).max(2);
        Self {
            data: vec![0; n_of_words],
            bit_width,
            len,
        }
    }
}

impl CompactArray<Vec<AtomicUsize>> {
    pub fn new_atomic(bit_width: usize, len: usize) -> Self {
        #[cfg(not(any(feature = "testless_read", feature = "testless_write")))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (len * bit_width + BITS - 1) / BITS;
        #[cfg(any(feature = "testless_read", feature = "testless_write"))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (1 + (len * bit_width + BITS - 1) / BITS).max(2);
        Self {
            data: (0..n_of_words).map(|_| AtomicUsize::new(0)).collect(),
            bit_width,
            len,
        }
    }
}

impl<B> CompactArray<B> {
    /// # Safety
    /// TODO: this function is never used.
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, bit_width: usize, len: usize) -> Self {
        Self {
            data,
            bit_width,
            len,
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.data, self.bit_width, self.len)
    }
}

impl<B: VSliceCore> VSliceCore for CompactArray<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= self.data.bit_width());
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: AsRef<[usize]> + VSliceCore> VSlice for CompactArray<T> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        debug_assert!(self.bit_width != BITS);
        #[cfg(not(feature = "testless_read"))]
        if self.bit_width == 0 {
            return 0;
        }

        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        #[cfg(feature = "testless_read")]
        {
            // ALERT!: this is not the correct mask for width BITS
            let mask = (1_usize << self.bit_width) - 1;

            let lower = self.data.as_ref().get_unchecked(word_index) >> bit_index;
            let higher =
                (self.data.as_ref().get_unchecked(word_index + 1) << (BITS - 1 - bit_index)) << 1;

            (higher | lower) & mask
        }

        #[cfg(not(feature = "testless_read"))]
        {
            let l = BITS - self.bit_width;

            if bit_index <= l {
                self.data.as_ref().get_unchecked(word_index) << (l - bit_index) >> l
            } else {
                self.data.as_ref().get_unchecked(word_index) >> bit_index
                    | self.data.as_ref().get_unchecked(word_index + 1) << (BITS + l - bit_index)
                        >> l
            }
        }
    }
}

impl<T: AsRef<[AtomicUsize]> + VSliceCore> VSliceAtomic for CompactArray<T> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> usize {
        debug_assert!(self.bit_width != BITS);
        if self.bit_width == 0 {
            return 0;
        }

        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        let l = BITS - self.bit_width;
        // we always use the tested to reduce the probability of unconsistent reads
        if bit_index <= l {
            self.data.as_ref().get_unchecked(word_index).load(order) << (l - bit_index) >> l
        } else {
            self.data.as_ref().get_unchecked(word_index).load(order) >> bit_index
                | self.data.as_ref().get_unchecked(word_index + 1).load(order)
                    << (BITS + l - bit_index)
                    >> l
        }
    }
    #[inline]
    unsafe fn set_unchecked(&self, index: usize, value: usize, order: Ordering) {
        debug_assert!(self.bit_width != BITS);
        if self.bit_width == 0 {
            return;
        }

        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        let mask: usize = (1_usize << self.bit_width) - 1;

        let end_word_index = (pos + self.bit_width - 1) / BITS;
        if word_index == end_word_index {
            // this is consistent
            let mut current = self.data.as_ref().get_unchecked(word_index).load(order);
            loop {
                let mut new = current;
                new &= !(mask << bit_index);
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
            // try to wait for the other thread to finish
            let mut word = self.data.as_ref().get_unchecked(word_index).load(order);
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

            let mut word = self.data.as_ref().get_unchecked(end_word_index).load(order);
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= !(mask >> (BITS - bit_index));
                new |= value >> (BITS - bit_index);

                match self
                    .data
                    .as_ref()
                    .get_unchecked(end_word_index)
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

impl VSliceMut for CompactArray<Vec<usize>> {
    #[inline]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        debug_assert!(self.bit_width != BITS);
        #[cfg(not(feature = "testless_write"))]
        if self.bit_width == 0 {
            return;
        }

        let pos = index * self.bit_width;
        let word_index = pos / BITS;
        let bit_index = pos % BITS;

        let mask: usize = (1_usize << self.bit_width) - 1;

        #[cfg(feature = "testless_write")]
        {
            let lower = value << bit_index;
            let higher = (value >> (BITS - 1 - bit_index)) >> 1;

            let lower_word = self.data.get_unchecked(word_index) & !(mask << bit_index);
            self.data.set_unchecked(word_index, lower_word | lower);

            let higher_word =
                self.data.get_unchecked(word_index + 1) & !((mask >> (BITS - 1 - bit_index)) >> 1);
            self.data
                .set_unchecked(word_index + 1, higher_word | higher);
        }

        #[cfg(not(feature = "testless_write"))]
        {
            let end_word_index = (pos + self.bit_width - 1) / BITS;
            if word_index == end_word_index {
                let mut word = self.data.get_unchecked(word_index);
                word &= !(mask << bit_index);
                word |= value << bit_index;
                self.data.set_unchecked(word_index, word);
            } else {
                let mut word = self.data.get_unchecked(word_index);
                word &= (1 << bit_index) - 1;
                word |= value << bit_index;
                self.data.set_unchecked(word_index, word);

                let mut word = self.data.get_unchecked(end_word_index);
                word &= !(mask >> (BITS - bit_index));
                word |= value >> (BITS - bit_index);
                self.data.set_unchecked(end_word_index, word);
            }
        }
    }
}

impl<B, D> ConvertTo<CompactArray<D>> for CompactArray<B>
where
    B: ConvertTo<D> + VSliceCore,
    D: VSliceCore,
{
    #[inline]
    fn convert_to(self) -> Result<CompactArray<D>> {
        Ok(CompactArray {
            len: self.len,
            bit_width: self.bit_width,
            data: self.data.convert_to()?,
        })
    }
}

impl From<CompactArray<Vec<usize>>> for CompactArray<Vec<AtomicUsize>> {
    #[inline]
    fn from(bm: CompactArray<Vec<usize>>) -> Self {
        bm.convert_to().unwrap()
    }
}

impl From<CompactArray<Vec<AtomicUsize>>> for CompactArray<Vec<usize>> {
    #[inline]
    fn from(bm: CompactArray<Vec<AtomicUsize>>) -> Self {
        bm.convert_to().unwrap()
    }
}

impl<'a> From<CompactArray<&'a [AtomicUsize]>> for CompactArray<&'a [usize]> {
    #[inline]
    fn from(bm: CompactArray<&'a [AtomicUsize]>) -> Self {
        bm.convert_to().unwrap()
    }
}

impl<'a> From<CompactArray<&'a [usize]>> for CompactArray<&'a [AtomicUsize]> {
    #[inline]
    fn from(bm: CompactArray<&'a [usize]>) -> Self {
        bm.convert_to().unwrap()
    }
}

impl<'a> From<CompactArray<&'a mut [AtomicUsize]>> for CompactArray<&'a mut [usize]> {
    #[inline]
    fn from(bm: CompactArray<&'a mut [AtomicUsize]>) -> Self {
        bm.convert_to().unwrap()
    }
}

impl<'a> From<CompactArray<&'a mut [usize]>> for CompactArray<&'a mut [AtomicUsize]> {
    #[inline]
    fn from(bm: CompactArray<&'a mut [usize]>) -> Self {
        bm.convert_to().unwrap()
    }
}
