/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use anyhow::Result;
use common_traits::*;
use core::marker::PhantomData;
use epserde::*;
use std::sync::atomic::{compiler_fence, fence, AtomicUsize, Ordering};

/**

A fixed-length array of values of bounded bit width.

Elements are stored contiguously, with no padding bits (in particular,
unless the bit width is a power of two some elements will be stored
across word boundaries).

We provide implementations
based on `AsRef<[usize]>`, `AsMut<[usize]>`, and
`AsRef<[AtomicUsize]>`. They implement
[`BitFieldSlice`], [`BitFieldSliceMut`], and [`BitFieldSliceAtomic`], respectively. Constructors are provided
for storing data in a [`Vec<usize>`](CompactArray::new) (for the first
two implementations) or in a
[`Vec<AtomicUsize>`](CompactArray::new_atomic) (for the third implementation).

In the latter case we can provide some concurrency guarantees,
albeit not full-fledged thread safety: more precisely, we can
guarantee thread-safety if the bit width is a power of two; otherwise,
concurrent writes to values that cross word boundaries might end
up in different threads succeding in writing only part of a value.
If the user can guarantee that no two threads ever write to the same
boundary-crossing value, then no race condition can happen.

*/
#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompactArray<W = usize, M = W, B = Vec<W>> {
    /// The underlying storage.
    data: B,
    /// The bit width of the values stored in the array.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: M,
    /// The length of the array.
    len: usize,

    _marker: PhantomData<W>,
}

fn mask<M: Integer>(bit_width: usize) -> M {
    if bit_width == 0 {
        M::ZERO
    } else {
        M::MAX >> (M::BITS - bit_width)
    }
}

impl<W: Word> CompactArray<W, W, Vec<W>> {
    pub fn new(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        Self {
            data: vec![W::ZERO; n_of_words],
            bit_width,
            mask: mask(bit_width),
            len,
            _marker: PhantomData,
        }
    }
}

impl<W: NonAtomic + Word> CompactArray<W, W, Vec<W>> {
    pub fn new_atomic(bit_width: usize, len: usize) -> CompactArray<W::AtomicType, W> {
        // we need at least two words to avoid branches in the gets
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        CompactArray::<W::AtomicType, W> {
            data: (0..n_of_words)
                .map(|_| W::AtomicType::new(W::ZERO))
                .collect(),
            bit_width,
            mask: mask(bit_width),
            len,
            _marker: PhantomData,
        }
    }
}

impl<W: Bits, M: Word, B> CompactArray<W, M, B> {
    /// # Safety
    /// `len` * `bit_width` must be between 0 (included) the number of
    /// bits in `data` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, bit_width: usize, len: usize) -> Self {
        Self {
            data,
            bit_width,
            mask: mask(bit_width),
            len,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.data, self.bit_width, self.len)
    }
}

impl<W: Bits, M, T> BitFieldSliceCore<W> for CompactArray<W, M, T> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS);
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldSlice<W> for CompactArray<W, W, B> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> W {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;

        if bit_index + self.bit_width <= W::BITS {
            (*self.data.as_ref().get_unchecked(word_index) >> bit_index) & self.mask
        } else {
            (*self.data.as_ref().get_unchecked(word_index) >> bit_index
                | *self.data.as_ref().get_unchecked(word_index + 1) << (W::BITS - bit_index))
                & self.mask
        }
    }
}

pub struct CompactArrayUncheckedIterator<'a, W, M, B> {
    array: &'a CompactArray<W, M, B>,
    word_index: usize,
    window: W,
    fill: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> CompactArrayUncheckedIterator<'a, W, W, B> {
    fn new(array: &'a CompactArray<W, W, B>, index: usize) -> Self {
        if index > array.len() {
            panic!("Start index out of bounds: {} > {}", index, array.len());
        }
        let bit_offset = index * array.bit_width;
        let word_index = bit_offset / usize::BITS as usize;
        let fill;
        let window = if index == array.len() {
            fill = 0;
            W::ZERO
        } else {
            let bit_index = bit_offset % usize::BITS as usize;
            fill = usize::BITS as usize - bit_index;
            unsafe {
                // SAFETY: index has been check at the start and it is within bounds
                *array.data.as_ref().get_unchecked(word_index) >> bit_index
            }
        };
        Self {
            array,
            word_index,
            window,
            fill,
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> UncheckedValueIterator
    for CompactArrayUncheckedIterator<'a, W, W, B>
{
    type Item = W;
    unsafe fn next_unchecked(&mut self) -> W {
        if self.fill >= self.array.bit_width {
            self.fill -= self.array.bit_width;
            let res = self.window & self.array.mask;
            self.window >>= self.array.bit_width;
            return res;
        }

        let res = self.window;
        self.word_index += 1;
        self.window = *self.array.data.as_ref().get_unchecked(self.word_index);
        let res = (res | (self.window << self.fill)) & self.array.mask;
        let used = self.array.bit_width - self.fill;
        self.window >>= used;
        self.fill = usize::BITS as usize - used;
        res
    }
}

impl<W: Word, B: AsRef<[W]>> IntoUncheckedValueIterator for CompactArray<W, W, B> {
    type Item = W;
    type IntoUncheckedValueIter<'a> = CompactArrayUncheckedIterator<'a, W, W, B>
        where B:'a, W:'a ;

    fn iter_val_from_unchecked(&self, from: usize) -> Self::IntoUncheckedValueIter<'_> {
        CompactArrayUncheckedIterator::new(self, from)
    }
}

pub struct CompactArrayIterator<'a, W, M, B> {
    unchecked: CompactArrayUncheckedIterator<'a, W, M, B>,
    index: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> CompactArrayIterator<'a, W, W, B> {
    fn new(array: &'a CompactArray<W, W, B>, index: usize) -> Self {
        if index > array.len() {
            panic!("Start index out of bounds: {} > {}", index, array.len());
        }
        Self {
            unchecked: CompactArrayUncheckedIterator::new(array, index),
            index,
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> Iterator for CompactArrayIterator<'a, W, W, B> {
    type Item = W;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.unchecked.array.len() {
            // SAFETY: index has just been checked.
            let res = unsafe { self.unchecked.next_unchecked() };
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> ExactSizeIterator for CompactArrayIterator<'a, W, W, B> {
    fn len(&self) -> usize {
        self.unchecked.array.len() - self.index
    }
}

impl<W: Word, B: AsRef<[W]>> IntoValueIterator for CompactArray<W, W, B> {
    type Item = W;
    type IntoValueIter<'a> = CompactArrayIterator<'a, W, W, B>
        where B:'a, W: 'a;

    fn iter_val_from(&self, from: usize) -> Self::IntoValueIter<'_> {
        CompactArrayIterator::new(self, from)
    }
}

impl<W: Integer, B: AsRef<[W]> + AsMut<[W]>> BitFieldSliceMut<W> for CompactArray<W, W, B> {
    // We reimplement set as we have the mask in the structure.

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    #[inline(always)]
    fn set(&mut self, index: usize, value: W) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_unchecked(index, value);
        }
    }

    #[inline]
    unsafe fn set_unchecked(&mut self, index: usize, value: W) {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;

        if bit_index + self.bit_width <= W::BITS {
            let mut word = *self.data.as_ref().get_unchecked(word_index);
            word &= !(self.mask << bit_index);
            word |= value << bit_index;
            *self.data.as_mut().get_unchecked_mut(word_index) = word;
        } else {
            let mut word = *self.data.as_ref().get_unchecked(word_index);
            word &= (W::ONE << bit_index) - W::ONE;
            word |= value << bit_index;
            *self.data.as_mut().get_unchecked_mut(word_index) = word;

            let mut word = *self.data.as_ref().get_unchecked(word_index + 1);
            word &= !(self.mask >> (W::BITS - bit_index));
            word |= value >> (W::BITS - bit_index);
            *self.data.as_mut().get_unchecked_mut(word_index + 1) = word;
        }
    }
}

impl<W: Atomic + Bits, T: AsRef<[W]>> BitFieldSliceAtomic<W>
    for CompactArray<W, W::NonAtomicType, T>
where
    W::NonAtomicType: Integer,
{
    #[inline]
    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> W::NonAtomicType {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;

        if bit_index + self.bit_width <= W::BITS {
            (self.data.as_ref().get_unchecked(word_index).load(order) >> bit_index) & self.mask
        } else {
            (self.data.as_ref().get_unchecked(word_index).load(order) >> bit_index
                | self.data.as_ref().get_unchecked(word_index + 1).load(order)
                    << (W::BITS - bit_index))
                & self.mask
        }
    }

    // We reimplement set as we have the mask in the structure.

    /// Set the element of the slice at the specified index.
    ///
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    #[inline(always)]
    fn set(&self, index: usize, value: W::NonAtomicType, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_unchecked(index, value, order);
        }
    }

    #[inline]
    unsafe fn set_unchecked(&self, index: usize, value: W::NonAtomicType, order: Ordering) {
        debug_assert!(self.bit_width != W::BITS);
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;

        if bit_index + self.bit_width <= W::BITS {
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
                new &= (W::NonAtomicType::ONE << bit_index) - W::NonAtomicType::ONE;
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
                new &= !(self.mask >> (W::BITS - bit_index));
                new |= value >> (W::BITS - bit_index);

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
impl<W, V, M, B, C> ConvertTo<CompactArray<V, M, C>> for CompactArray<W, M, B>
where
    B: ConvertTo<C>,
{
    #[inline]
    fn convert_to(self) -> Result<CompactArray<V, M, C>> {
        Ok(CompactArray {
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            data: self.data.convert_to()?,
            _marker: PhantomData,
        })
    }
}

/// Provide conversion from standard to atomic compact arrays.
impl From<CompactArray<usize>> for CompactArray<AtomicUsize, usize> {
    #[inline]
    fn from(bm: CompactArray<usize>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from atomic to standard compact arrays.
impl From<CompactArray<AtomicUsize, usize>> for CompactArray<usize> {
    #[inline]
    fn from(bm: CompactArray<AtomicUsize, usize>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from references to standard compact arrays to
/// references to atomic compact arrays.
impl<'a> From<CompactArray<usize, usize, &'a [usize]>>
    for CompactArray<AtomicUsize, usize, &'a [AtomicUsize]>
{
    #[inline]
    fn from(bm: CompactArray<usize, usize, &'a [usize]>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from references to atomic compact arrays to
/// references to standard compact arrays.
impl<'a> From<CompactArray<AtomicUsize, usize, &'a [AtomicUsize]>>
    for CompactArray<usize, usize, &'a [usize]>
{
    #[inline]
    fn from(bm: CompactArray<AtomicUsize, usize, &'a [AtomicUsize]>) -> Self {
        bm.convert_to().unwrap()
    }
}
