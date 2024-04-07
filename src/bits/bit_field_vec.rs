/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Vectors of values of fixed bit width.

Elements are stored contiguously, with no padding bits (in particular,
unless the bit width is a power of two some elements will be stored
across word boundaries).

We provide implementations
based on `AsRef<[T]>`, `AsMut<[T]>`, and
`AsRef<[A]>`, where `T` is an unsigned type (default: [`usize`]) and `A` is an atomic
unsigned type (default: [`AtomicUsize`]); more generally, the underlying type
must satisfy the trait [`Word`], and additional [`IntoAtomic`] in the second case.
[`BitFieldSlice`], [`BitFieldSliceMut`], and [`AtomicBitFieldSlice`], respectively.
Constructors are provided
for storing data in a [`Vec<T>`](BitFieldVec::new) or
[`Vec<A>`](AtomicBitFieldVec::new).

In the latter case we can provide some concurrency guarantees,
albeit not full-fledged thread safety: more precisely, we can
guarantee thread-safety if the bit width is a power of two; otherwise,
concurrent writes to values that cross word boundaries might end
up in different threads succeding in writing only part of a value.
If the user can guarantee that no two threads ever write to the same
boundary-crossing value, then no race condition can happen.

Note that some care must be exercised when using the methods of
[`BitFieldSlice`], [`BitFieldSliceMut`] and [`AtomicBitFieldSlice`]:
see the discussions in documentation of [`bit_field_slice`].

For high-speed unchecked scanning, we implement
[`IntoUncheckedIterator`] and [`IntoReverseUncheckedIterator`] on a reference
to this type. The are used, for example, to provide
[predecessor](crate::traits::indexed_dict::Pred) and
[successor](crate::traits::indexed_dict::Succ) primitives
for [Elias-Fano](crate::dict::elias_fano::EliasFano).

## Low-level support

The methods [`address_of`](BitFieldVec::address_of)
and [`get_unaligned`](BitFieldVec::get_unaligned) can be used to manually
prefetch parts of the data structure, or read values using unaligned
read, when the bit width makes it possible.
*/

use crate::prelude::*;
use crate::traits::bit_field_slice::*;
use anyhow::{bail, Result};
use common_traits::*;
use epserde::*;
use mem_dbg::*;
use std::sync::atomic::*;

/// A vector of bit fields of fixed width.
#[derive(Epserde, Debug, Clone, Hash, MemDbg, MemSize)]
pub struct BitFieldVec<W: Word = usize, B = Vec<W>> {
    /// The underlying storage.
    data: B,
    /// The bit width of the values stored in the vector.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: W,
    /// The length of the vector.
    len: usize,
}

/// A tentatively thread-safe vector of bit fields of fixed width.
#[derive(Epserde, Debug, Clone, Hash, MemDbg, MemSize)]
pub struct AtomicBitFieldVec<W: Word + IntoAtomic = usize, B = Vec<<W as IntoAtomic>::AtomicType>> {
    /// The underlying storage.
    data: B,
    /// The bit width of the values stored in the vector.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: W,
    /// The length of the vector.
    len: usize,
}

fn mask<W: Word>(bit_width: usize) -> W {
    if bit_width == 0 {
        W::ZERO
    } else {
        W::MAX >> (W::BITS - bit_width)
    }
}

impl<W: Word> BitFieldVec<W, Vec<W>> {
    /// Create a new zero-initialized vector of given bit width and length.
    pub fn new(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        Self {
            data: vec![W::ZERO; n_of_words],
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Create an empty BitFieldVec that doesn't need to reallocate for up to
    /// `capacity` elements.
    pub fn with_capacity(bit_width: usize, capacity: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (capacity * bit_width + W::BITS - 1) / W::BITS);
        Self {
            data: Vec::with_capacity(n_of_words),
            bit_width,
            mask: mask(bit_width),
            len: 0,
        }
    }

    /// Create a new all-ones-initialized vector of given bit width and length.
    pub fn new_ones(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        Self {
            data: vec![!W::ZERO; n_of_words],
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Create a new uninitialized vector of given bit width and length.
    ///
    /// # Safety
    /// this is intherently unsafe as you might read
    /// uninitialized data or write out of bounds.
    pub unsafe fn new_uninit(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        let mut data = Vec::with_capacity(n_of_words);
        #[allow(clippy::uninit_vec)]
        // this is what we want to do, and it's much cleaner than maybeuninit
        data.set_len(n_of_words);
        Self {
            data,
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Set the inner len.
    ///
    /// # Safety
    /// this is intherently unsafe as you might read
    /// uninitialized data or write out of bounds.
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len * self.bit_width <= self.data.len() * W::BITS);
        self.len = len;
    }

    /// Write 0 to all bits in the vector
    pub fn reset(&mut self) {
        self.data.iter_mut().for_each(|x| *x = W::ZERO);
    }

    /// Write 1 to all bits in the vector
    pub fn reset_ones(&mut self) {
        self.data.iter_mut().for_each(|x| *x = !W::ZERO);
    }

    /// Set len to 0
    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    /// Return the bit-width of the values inside this vector.
    pub fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS);
        self.bit_width
    }

    /// Return the mask used to extract values from this vector.
    /// This will keep the lowest `bit_width` bits.
    pub fn mask(&self) -> W {
        self.mask
    }

    /// Create a new vector by copying a slice; the bit width will be the minimum
    /// width sufficient to hold all values in the slice.
    ///
    /// Returns an error if the bit width of the values in `slice` is larger than
    /// `W::BITS`.
    pub fn from_slice<SW>(slice: &impl BitFieldSlice<SW>) -> Result<Self>
    where
        SW: Word + CastableInto<W>,
    {
        let mut max_len: usize = 0;
        for i in 0..slice.len() {
            max_len = Ord::max(max_len, unsafe { slice.get_unchecked(i).len() as usize });
        }

        if max_len > W::BITS {
            bail!(
                "Cannot convert a slice of bit width {} into a slice with W = {}",
                max_len,
                std::any::type_name::<W>()
            );
        }
        let mut result = Self::new(max_len, slice.len());
        for i in 0..slice.len() {
            unsafe { result.set_unchecked(i, slice.get_unchecked(i).cast()) };
        }

        Ok(result)
    }

    /// Add a value at the end of the BitFieldVec
    pub fn push(&mut self, value: W) {
        panic_if_value!(value, self.mask, self.bit_width);
        if (self.len + 1) * self.bit_width > self.data.len() * W::BITS {
            self.data.push(W::ZERO);
        }
        unsafe {
            self.set_unchecked(self.len, value);
        }
        self.len += 1;
    }

    /// Truncate or exted with `value` the BitFieldVec
    pub fn resize(&mut self, new_len: usize, value: W) {
        panic_if_value!(value, self.mask, self.bit_width);
        if new_len > self.len {
            if new_len * self.bit_width > self.data.len() * W::BITS {
                self.data
                    .resize((new_len * self.bit_width + W::BITS - 1) / W::BITS, W::ZERO);
            }
            for i in self.len..new_len {
                unsafe {
                    self.set_unchecked(i, value);
                }
            }
        }
        self.len = new_len;
    }

    /// Remove and return a value from the end of the [`BitFieldVec`].
    /// Return None if the [`BitFieldVec`] is empty.
    pub fn pop(&mut self) -> Option<W> {
        if self.len == 0 {
            return None;
        }
        let value = self.get(self.len - 1);
        self.len -= 1;
        Some(value)
    }
}

impl<W: Word> core::iter::Extend<W> for BitFieldVec<W, Vec<W>> {
    /// Add values from
    fn extend<T: IntoIterator<Item = W>>(&mut self, iter: T) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldVec<W, B> {
    /// Get the address of the item storing (the first part of)
    /// the element of given index.
    ///
    /// This method is mainly useful for manually prefetching
    /// parts of the data structure.
    pub fn address_of(&self, index: usize) -> *const W {
        let pos = index * W::BITS;
        let word_index = pos / W::BITS;
        (&self.data.as_ref()[word_index]) as *const _
    }

    /// Like [`BitFieldSlice::get`], but using unaligned reads.
    ///
    /// # Panic
    /// This methods will panic if the index is out of bounds
    /// or if the bit width is [incompatible with unaligned
    /// reads](BitFieldVec::get_unaligned_unchecked).
    pub fn get_unaligned(&self, index: usize) -> W {
        panic_if_out_of_bounds!(index, self.len);
        assert!(
            self.bit_width % 8 != 3
                && self.bit_width % 8 != 5
                && self.bit_width != 6
                && self.bit_width != 7
        );
        unsafe { self.get_unaligned_unchecked(index) }
    }

    /// Like [`BitFieldSlice::get`], but using unaligned reads.
    ///
    /// # Safety
    /// This methods can be used only if the `bit width % 8` is not
    /// 3, 5, 6, or 7.
    pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> W {
        debug_assert!(
            self.bit_width % 8 != 3
                && self.bit_width % 8 != 5
                && self.bit_width != 6
                && self.bit_width != 7
        );
        let base_ptr = self.data.as_ref().as_ptr() as *const u8;
        let ptr = base_ptr.add(index / W::BYTES) as *const W;
        let word = core::ptr::read_unaligned(ptr);
        (word >> (index % W::BITS)) & self.mask
    }
}

impl<W: Word + IntoAtomic> AtomicBitFieldVec<W> {
    pub fn new(bit_width: usize, len: usize) -> AtomicBitFieldVec<W> {
        // we need at least two words to avoid branches in the gets
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        AtomicBitFieldVec::<W> {
            data: (0..n_of_words)
                .map(|_| W::AtomicType::new(W::ZERO))
                .collect(),
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Create an empty BitFieldVec that doesn't need to reallocate for up to
    /// `capacity` elements.
    pub fn with_capacity(bit_width: usize, capacity: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (capacity * bit_width + W::BITS - 1) / W::BITS);
        Self {
            data: Vec::with_capacity(n_of_words),
            bit_width,
            mask: mask(bit_width),
            len: 0,
        }
    }

    /// Create a new uninitialized vector of given bit width and length.
    ///
    /// # Safety
    /// this is intherently unsafe as you might read
    /// uninitialized data or write out of bounds.
    pub unsafe fn new_uninit(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        let mut data = Vec::with_capacity(n_of_words);
        #[allow(clippy::uninit_vec)]
        // this is what we want to do, and it's much cleaner than maybeuninit
        data.set_len(n_of_words);
        Self {
            data,
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Create a new all-ones-initialized vector of given bit width and length.
    /// This is useful for testing / it's slightly faster than creatin an
    /// uninit vector and then setting all values to ones because we can iterate
    /// over the words and set them all at once.
    pub fn new_ones(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width + W::BITS - 1) / W::BITS);
        Self {
            data: (0..n_of_words)
                .map(|_| W::AtomicType::new(!W::ZERO))
                .collect(),
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Set the inner len.
    /// # Safety
    /// this is intherently unsafe as you might read
    /// uninitialized data or write out of bounds.
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len * self.bit_width <= self.data.len() * W::BITS);
        self.len = len;
    }

    /// Write 0 to all bits in the vector
    pub fn reset(&mut self) {
        self.data
            .iter_mut()
            .for_each(|x| x.store(W::ZERO, Ordering::Relaxed));
    }

    /// Write 1 to all bits in the vector
    pub fn reset_ones(&mut self) {
        self.data
            .iter_mut()
            .for_each(|x| x.store(!W::ZERO, Ordering::Relaxed));
    }

    /// Set len to 0
    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    /// Return the bit-width of the values inside this vector.
    pub fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS);
        self.bit_width
    }

    /// Return the mask used to extract values from this vector.
    /// This will keep the lowest `bit_width` bits.
    pub fn mask(&self) -> W {
        self.mask
    }
}

impl<W: Word, B> BitFieldVec<W, B> {
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
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.data, self.bit_width, self.len)
    }
}

impl<W: Word + IntoAtomic, B> AtomicBitFieldVec<W, B> {
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
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.data, self.bit_width, self.len)
    }
}

impl<W: Word, T> BitFieldSliceCore<W> for BitFieldVec<W, T> {
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

impl<W: Word + IntoAtomic, T> BitFieldSliceCore<W::AtomicType> for AtomicBitFieldVec<W, T> {
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

impl<W: Word, B: AsRef<[W]>> BitFieldSlice<W> for BitFieldVec<W, B> {
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

// Support for unchecked iterators

/// An [`UncheckedIterator`] over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldVectorUncheckedIterator<'a, W, B>
where
    W: Word,
{
    vec: &'a BitFieldVec<W, B>,
    word_index: usize,
    window: W,
    fill: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> BitFieldVectorUncheckedIterator<'a, W, B> {
    fn new(vec: &'a BitFieldVec<W, B>, index: usize) -> Self {
        if index > vec.len() {
            panic!("Start index out of bounds: {} > {}", index, vec.len());
        }
        let (fill, word_index);
        let window = if index == vec.len() {
            word_index = 0;
            fill = 0;
            W::ZERO
        } else {
            let bit_offset = index * vec.bit_width;
            let bit_index = bit_offset % W::BITS;

            word_index = bit_offset / W::BITS;
            fill = W::BITS - bit_index;
            unsafe {
                // SAFETY: index has been check at the start and it is within bounds
                *vec.data.as_ref().get_unchecked(word_index) >> bit_index
            }
        };
        Self {
            vec,
            word_index,
            window,
            fill,
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> crate::traits::UncheckedIterator
    for BitFieldVectorUncheckedIterator<'a, W, B>
{
    type Item = W;
    unsafe fn next_unchecked(&mut self) -> W {
        let bit_width = self.vec.bit_width;

        if self.fill >= bit_width {
            self.fill -= bit_width;
            let res = self.window & self.vec.mask;
            self.window >>= bit_width;
            return res;
        }

        let res = self.window;
        self.word_index += 1;
        self.window = *self.vec.data.as_ref().get_unchecked(self.word_index);
        let res = (res | (self.window << self.fill)) & self.vec.mask;
        let used = bit_width - self.fill;
        self.window >>= used;
        self.fill = W::BITS - used;
        res
    }
}

impl<'a, W: Word, B: AsRef<[W]>> IntoUncheckedIterator for &'a BitFieldVec<W, B> {
    type Item = W;
    type IntoUncheckedIter = BitFieldVectorUncheckedIterator<'a, W, B>;
    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter {
        BitFieldVectorUncheckedIterator::new(self, from)
    }
}

/// An [`UncheckedIterator`] moving backwards over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldVectorReverseUncheckedIterator<'a, W: Word, B> {
    vec: &'a BitFieldVec<W, B>,
    word_index: usize,
    window: W,
    fill: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> BitFieldVectorReverseUncheckedIterator<'a, W, B> {
    fn new(vec: &'a BitFieldVec<W, B>, index: usize) -> Self {
        if index > vec.len() {
            panic!("Start index out of bounds: {} > {}", index, vec.len());
        }
        let (word_index, fill);

        let window = if index == 0 {
            word_index = 0;
            fill = 0;
            W::ZERO
        } else {
            // We have to handle the case of zero bit width
            let bit_offset = (index * vec.bit_width).saturating_sub(1);
            let bit_index = bit_offset % W::BITS;

            word_index = bit_offset / W::BITS;
            fill = bit_index + 1;
            unsafe {
                // SAFETY: index has been check at the start and it is within bounds
                *vec.data.as_ref().get_unchecked(word_index) << (W::BITS - fill)
            }
        };
        Self {
            vec,
            word_index,
            window,
            fill,
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> crate::traits::UncheckedIterator
    for BitFieldVectorReverseUncheckedIterator<'a, W, B>
{
    type Item = W;
    unsafe fn next_unchecked(&mut self) -> W {
        let bit_width = self.vec.bit_width;

        if self.fill >= bit_width {
            self.fill -= bit_width;
            self.window = self.window.rotate_left(bit_width as u32);
            return self.window & self.vec.mask;
        }

        let mut res = self.window.rotate_left(self.fill as u32);
        self.word_index -= 1;
        self.window = *self.vec.data.as_ref().get_unchecked(self.word_index);
        let used = bit_width - self.fill;
        res = ((res << used) | self.window >> (W::BITS - used)) & self.vec.mask;
        self.window <<= used;
        self.fill = W::BITS - used;
        res
    }
}

impl<'a, W: Word, B: AsRef<[W]>> IntoReverseUncheckedIterator for &'a BitFieldVec<W, B> {
    type Item = W;
    type IntoRevUncheckedIter = BitFieldVectorReverseUncheckedIterator<'a, W, B>;

    fn into_rev_unchecked_iter(self) -> Self::IntoRevUncheckedIter {
        BitFieldVectorReverseUncheckedIterator::new(self, self.len())
    }

    fn into_rev_unchecked_iter_from(self, from: usize) -> Self::IntoRevUncheckedIter {
        BitFieldVectorReverseUncheckedIterator::new(self, from)
    }
}

/// An [`Iterator`] over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldVecIterator<'a, W, B>
where
    W: Word,
{
    unchecked: BitFieldVectorUncheckedIterator<'a, W, B>,
    index: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> BitFieldVecIterator<'a, W, B> {
    fn new(vec: &'a BitFieldVec<W, B>, from: usize) -> Self {
        if from > vec.len() {
            panic!("Start index out of bounds: {} > {}", from, vec.len());
        }
        Self {
            unchecked: BitFieldVectorUncheckedIterator::new(vec, from),
            index: from,
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> Iterator for BitFieldVecIterator<'a, W, B> {
    type Item = W;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.unchecked.vec.len() {
            // SAFETY: index has just been checked.
            let res = unsafe { self.unchecked.next_unchecked() };
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }
}

impl<'a, W: Word, B: AsRef<[W]>> ExactSizeIterator for BitFieldVecIterator<'a, W, B> {
    fn len(&self) -> usize {
        self.unchecked.vec.len() - self.index
    }
}

impl<'a, W: Word, B: AsRef<[W]>> IntoIterator for &'a BitFieldVec<W, B> {
    type Item = W;
    type IntoIter = BitFieldVecIterator<'a, W, B>;

    fn into_iter(self) -> Self::IntoIter {
        BitFieldVecIterator::new(self, 0)
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldVec<W, B> {
    pub fn iter_from(&self, from: usize) -> BitFieldVecIterator<W, B> {
        BitFieldVecIterator::new(self, from)
    }

    pub fn iter(&self) -> BitFieldVecIterator<W, B> {
        self.iter_from(0)
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]>> BitFieldSliceMut<W> for BitFieldVec<W, B> {
    // We reimplement set as we have the mask in the structure.

    fn reset(&mut self) {
        for idx in 0..self.len() {
            unsafe { self.set_unchecked(idx, W::ZERO) };
        }
    }

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

impl<W: Word + IntoAtomic, T: AsRef<[W::AtomicType]>> AtomicBitFieldSlice<W>
    for AtomicBitFieldVec<W, T>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    #[inline]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> W {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;
        let data: &[W::AtomicType] = self.data.as_ref();

        if bit_index + self.bit_width <= W::BITS {
            (data.get_unchecked(word_index).load(order) >> bit_index) & self.mask
        } else {
            (data.get_unchecked(word_index).load(order) >> bit_index
                | data.get_unchecked(word_index + 1).load(order) << (W::BITS - bit_index))
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
    fn set_atomic(&self, index: usize, value: W, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_atomic_unchecked(index, value, order);
        }
    }

    #[inline]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: W, order: Ordering) {
        debug_assert!(self.bit_width != W::BITS);
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;
        let data: &[W::AtomicType] = self.data.as_ref();

        if bit_index + self.bit_width <= W::BITS {
            // this is consistent
            let mut current = data.get_unchecked(word_index).load(order);
            loop {
                let mut new = current;
                new &= !(self.mask << bit_index);
                new |= value << bit_index;

                match data
                    .get_unchecked(word_index)
                    .compare_exchange(current, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => current = e,
                }
            }
        } else {
            let mut word = data.get_unchecked(word_index).load(order);
            // try to wait for the other thread to finish
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= (W::ONE << bit_index) - W::ONE;
                new |= value << bit_index;

                match data
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

            let mut word = data.get_unchecked(word_index + 1).load(order);
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= !(self.mask >> (W::BITS - bit_index));
                new |= value >> (W::BITS - bit_index);

                match data
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

    fn reset_atomic(&mut self, order: Ordering) {
        for idx in 0..self.len() {
            unsafe { self.set_atomic_unchecked(idx, W::ZERO, order) };
        }
    }
}

/// Provide conversion from non-atomic to atomic bitfield vectors, provided their
/// backends are [convertible](ConvertTo) into one another.
///
/// Implementations of this trait are then used to
/// implement by delegation a corresponding [`From`].
impl<W: Word + IntoAtomic, B, C> ConvertTo<AtomicBitFieldVec<W, C>> for BitFieldVec<W, B>
where
    B: ConvertTo<C>,
{
    #[inline]
    fn convert_to(self) -> Result<AtomicBitFieldVec<W, C>> {
        Ok(AtomicBitFieldVec {
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            data: self.data.convert_to()?,
        })
    }
}

/// Provide conversion from atomic to non-atomic bitfield vectors, provided their
/// backends are [convertible](ConvertTo) into one another.
///
/// Implementations of this trait are then used to
/// implement by delegation a corresponding [`From`].
impl<W: Word + IntoAtomic, B, C> ConvertTo<BitFieldVec<W, C>> for AtomicBitFieldVec<W, B>
where
    B: ConvertTo<C>,
{
    #[inline]
    fn convert_to(self) -> Result<BitFieldVec<W, C>> {
        Ok(BitFieldVec {
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            data: self.data.convert_to()?,
        })
    }
}

/// Provide conversion betweeen bitfields vectors with different
/// backends, provided that such backends
/// are [convertible](ConvertTo) into one another.
///
/// This is a generalized form of reflexivity of [`ConvertTo`] for bitfield
/// vectors. It is necessary, among other things, for the mechanism with which indexing
/// structures can be added to [`EliasFano`].
impl<W: Word, B, C> ConvertTo<BitFieldVec<W, C>> for BitFieldVec<W, B>
where
    B: ConvertTo<C>,
{
    #[inline]
    fn convert_to(self) -> Result<BitFieldVec<W, C>> {
        Ok(BitFieldVec {
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            data: self.data.convert_to()?,
        })
    }
}

macro_rules! impl_from {
    ($std:ty, $atomic:ty) => {
        impl From<BitFieldVec<$std>> for AtomicBitFieldVec<$std> {
            #[inline]
            fn from(bm: BitFieldVec<$std>) -> Self {
                bm.convert_to().unwrap()
            }
        }

        impl From<AtomicBitFieldVec<$std>> for BitFieldVec<$std> {
            #[inline]
            fn from(bm: AtomicBitFieldVec<$std>) -> Self {
                bm.convert_to().unwrap()
            }
        }

        impl<'a> From<BitFieldVec<$std, &'a [$std]>> for AtomicBitFieldVec<$std, &'a [$atomic]> {
            #[inline]
            fn from(bm: BitFieldVec<$std, &'a [$std]>) -> Self {
                bm.convert_to().unwrap()
            }
        }

        impl<'a> From<AtomicBitFieldVec<$std, &'a [$atomic]>> for BitFieldVec<$std, &'a [$std]> {
            #[inline]
            fn from(bm: AtomicBitFieldVec<$std, &'a [$atomic]>) -> Self {
                bm.convert_to().unwrap()
            }
        }
    };
}

impl_from!(u8, AtomicU8);
impl_from!(u16, AtomicU16);
impl_from!(u32, AtomicU32);
impl_from!(u64, AtomicU64);
impl_from!(usize, AtomicUsize);
