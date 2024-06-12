/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Bit vector implementations.

There are three flavors:
- [`BitVec`], a mutable bit vector;
- [`CountBitVec`], an immutable bit vector that sports an implementation of [`NumBits`];
- [`AtomicBitVec`], a mutable, thread-safe bit vector.

These flavors depends on a backend, and presently we provide:

- `BitVec<Vec<usize>>`: a mutable, growable and resizable bit vector;
- `BitVec<AsRef<[usize]>>`: an immutable bit vector mainly useful for [`epserde`];
- `BitVec<AsRef<[usize]> + AsMut<[usize]>>`: a mutable (but not resizable) bit
   vector;
- `CountBitVec<AsRef<[usize]>>`: an immutable bit vector;
- `AtomicBitVec<AsRef<[AtomicUsize]>>`: a thread-safe, mutable (but not resizable) bit vector.

It is possible to juggle between the three flavors using [`From`].
*/
use common_traits::{IntoAtomic, SelectInWord};
use core::fmt;
use epserde::*;
use mem_dbg::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::traits::rank_sel::*;

const BITS: usize = usize::BITS as usize;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
/// A bit vector.
pub struct BitVec<B = Vec<usize>> {
    data: B,
    len: usize,
}

/// Convenient, [`vec!`](vec!)-like macro to initialize bit vectors.
///
/// # Examples
///
/// ```rust
/// use sux::bit_vec;
///
/// // Empty bit vector
/// let b = bit_vec![];
///
/// // 10 bits set to true
/// let b = bit_vec![true; 10];
/// assert_eq!(b.len(), 10);
/// let b = bit_vec![1; 10];
///
/// // 10 bits set to false
/// let b = bit_vec![false; 10];
/// assert_eq!(b.len(), 10);
/// let b = bit_vec![0; 10];
///
/// // Bit list
/// let b = bit_vec![0, 1, 0, 1, 0, 0];
/// ```

#[macro_export]
macro_rules! bit_vec {
    () => {
        $crate::prelude::BitVec::new(0)
    };
    (false; $n:expr) => {
        $crate::prelude::BitVec::new($n)
    };
    (0; $n:expr) => {
        $crate::prelude::BitVec::new($n)
    };
    (true; $n:expr) => {
        {
            $crate::prelude::BitVec::with_value($n, true)
        }
    };
    (1; $n:expr) => {
        {
            $crate::prelude::BitVec::with_value($n, true)
        }
    };
    ($($x:expr),+ $(,)?) => {
        {
            let mut b = $crate::prelude::BitVec::with_capacity([$($x),+].len());
            $( b.push($x != 0); )*
            b
        }
    };
}

#[derive(Debug, Clone, MemDbg, MemSize)]
/// A thread-safe bit vector.
pub struct AtomicBitVec<B = Vec<AtomicUsize>> {
    data: B,
    len: usize,
}

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Bit index out of bounds: {} >= {}", $index, $len)
        }
    };
}

impl<B> BitLength for BitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

impl<B: AsRef<[usize]>> Index<usize> for BitVec<B> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            false => &false,
            true => &true,
        }
    }
}

impl BitVec<Vec<usize>> {
    /// Creates a new bit vector of length `len` initialized to `false`.
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new bit vector of length `len` initialized to `value`.
    pub fn with_value(len: usize, value: bool) -> Self {
        let n_of_words = (len + BITS - 1) / BITS;
        let extra_bits = (n_of_words * BITS) - len;
        let word_value = if value { !0 } else { 0 };
        let mut data = vec![word_value; n_of_words];
        if extra_bits > 0 {
            let last_word_value = word_value >> extra_bits;
            data[n_of_words - 1] = last_word_value;
        }
        Self { data, len }
    }

    /// Creates a new zero-length bit vector of given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let n_of_words = capacity.div_ceil(BITS);
        Self {
            data: vec![0; n_of_words],
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity() * BITS
    }

    pub fn push(&mut self, b: bool) {
        if self.data.len() * usize::BITS as usize == self.len {
            self.data.push(0);
        }
        let word_index = self.len / BITS;
        let bit_index = self.len % BITS;
        self.data[word_index] |= (b as usize) << bit_index;
        self.len += 1;
    }

    pub fn extend(&mut self, i: impl IntoIterator<Item = bool>) {
        for b in i {
            self.push(b);
        }
    }

    pub fn resize(&mut self, new_len: usize, value: bool) {
        if new_len > self.len {
            if new_len > self.data.len() * usize::BITS as usize {
                self.data.resize(
                    (new_len + usize::BITS as usize - 1) / usize::BITS as usize,
                    0,
                );
            }
            for i in self.len..new_len {
                unsafe {
                    self.set_unchecked(i, value);
                }
            }
        }
        self.len = new_len;
    }
}

impl AtomicBitVec<Vec<AtomicUsize>> {
    /// Creates a new atomic bit vector of length `len` initialized to `false`.
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new atomic bit vector of length `len` initialized to `value`.
    pub fn with_value(len: usize, value: bool) -> Self {
        let n_of_words = (len + BITS - 1) / BITS;
        let extra_bits = (n_of_words * BITS) - len;
        let word_value = if value { !0 } else { 0 };
        let mut data = (0..n_of_words)
            .map(|_| AtomicUsize::new(word_value))
            .collect::<Vec<_>>();
        if extra_bits > 0 {
            let last_word_value = word_value >> extra_bits;
            data[n_of_words - 1] = AtomicUsize::new(last_word_value);
        }
        Self { data, len }
    }
}

impl<B> BitVec<B> {
    #[inline(always)]
    #[allow(clippy::len_without_is_empty)]
    /// Return the number of bits in this bit vector.
    ///
    /// This method is necessary to avoid ambiguity.
    pub fn len(&self) -> usize {
        self.len
    }
    /// # Safety
    /// `len` must be between 0 (included) the number of
    /// bits in `data` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, len: usize) -> Self {
        Self { data, len }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.data, self.len)
    }
}

impl<B> BitLength for AtomicBitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

impl<B> AtomicBitVec<B> {
    #[inline(always)]
    #[allow(clippy::len_without_is_empty)]
    /// Return the number of bits in this bit vector.
    pub fn len(&self) -> usize {
        self.len
    }
    /// # Safety
    /// `len` must be between 0 (included) the number of
    /// bits in `data` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, len: usize) -> Self {
        Self { data, len }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.data, self.len)
    }
}

impl<B: AsRef<[usize]>> BitVec<B> {
    /// Return a [`CountBitVec`] with the same data as this
    /// bit vector and the assuming the given number of ones.
    ///
    /// # Warning
    /// No control is performed on the number of ones, unless
    /// debug assertions are enabled.
    #[inline(always)]
    pub fn with_count(self, number_of_ones: usize) -> AddNumBits<BitVec<B>> {
        debug_assert!(number_of_ones <= self.len);
        debug_assert_eq!(number_of_ones, self.count_ones());
        unsafe { AddNumBits::from_raw_parts(self, number_of_ones) }
    }
}

impl<B: AsRef<[AtomicUsize]>> Index<usize> for AtomicBitVec<B> {
    type Output = bool;

    /// Shorthand for [`Self::get`] using [`Ordering::Relaxed`].
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index, Ordering::Relaxed) {
            false => &false,
            true => &true,
        }
    }
}

impl<B: AsRef<[AtomicUsize]>> BitCount for AtomicBitVec<B> {
    /// Return the number of bits set to 1 in this bit vector.
    ///
    /// If the feature "rayon" is enabled, this function is parallelized.
    fn count_ones(&self) -> usize {
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);

        #[cfg(feature = "rayon")]
        {
            self.data
                .as_ref()
                .par_iter()
                .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
                .sum()
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.data
                .iter()
                .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
                .sum()
        }
    }
}

impl<B: AsRef<[usize]>> BitVec<B> {
    pub fn get(&self, index: usize) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unchecked(index) }
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        let word_index = index / BITS;
        let word = self.data.as_ref().get_unchecked(word_index);
        (word >> (index % BITS)) & 1 != 0
    }
}

impl<B: AsRef<[usize]> + AsMut<[usize]>> BitVec<B> {
    pub fn set(&mut self, index: usize, value: bool) {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.set_unchecked(index, value) }
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let data: &mut [usize] = self.data.as_mut();
        // For constant values, this should be inlined with no test.
        if value {
            *data.get_unchecked_mut(word_index) |= 1 << bit_index;
        } else {
            *data.get_unchecked_mut(word_index) &= !(1 << bit_index);
        }
    }

    pub fn fill(&mut self, value: bool) {
        let data: &mut [usize] = self.data.as_mut();
        if value {
            let end = self.len / BITS;
            let residual = self.len % BITS;
            data[0..end].fill(!0);
            if residual != 0 {
                data[end] = !0 >> (BITS - residual);
            }
        } else {
            data[0..self.len.div_ceil(BITS)].fill(0);
        }
    }

    pub fn flip(&mut self) {
        let data: &mut [usize] = self.data.as_mut();
        let end = self.len / BITS;
        let residual = self.len % BITS;
        data[0..end].iter_mut().for_each(|x| *x ^= !0);
        if residual != 0 {
            data[end] ^= (1 << residual) - 1;
        }
    }
}

impl<B: AsRef<[AtomicUsize]>> AtomicBitVec<B> {
    pub fn get(&self, index: usize, order: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unchecked(index, order) }
    }

    pub fn set(&self, index: usize, value: bool, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.set_unchecked(index, value, order) }
    }

    pub fn swap(&self, index: usize, value: bool, order: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.swap_unchecked(index, value, order) }
    }

    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> bool {
        let word_index = index / BITS;
        let data: &[AtomicUsize] = self.data.as_ref();
        let word = data.get_unchecked(word_index).load(order);
        (word >> (index % BITS)) & 1 != 0
    }
    #[inline(always)]
    unsafe fn set_unchecked(&self, index: usize, value: bool, order: Ordering) {
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let data: &[AtomicUsize] = self.data.as_ref();

        // For constant values, this should be inlined with no test.
        if value {
            data.get_unchecked(word_index)
                .fetch_or(1 << bit_index, order);
        } else {
            data.get_unchecked(word_index)
                .fetch_and(!(1 << bit_index), order);
        }
    }

    #[inline(always)]
    unsafe fn swap_unchecked(&self, index: usize, value: bool, order: Ordering) -> bool {
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let data: &[AtomicUsize] = self.data.as_ref();

        let old_word = if value {
            data.get_unchecked(word_index)
                .fetch_or(1 << bit_index, order)
        } else {
            data.get_unchecked(word_index)
                .fetch_and(!(1 << bit_index), order)
        };

        (old_word >> (bit_index)) & 1 != 0
    }

    pub fn fill(&mut self, value: bool, order: Ordering) {
        let data: &[AtomicUsize] = self.data.as_ref();
        if value {
            let end = self.len / BITS;
            let residual = self.len % BITS;
            data[0..end].iter().for_each(|x| x.store(!0, order));
            if residual != 0 {
                data[end].store(!0 >> (BITS - residual), order);
            }
        } else {
            data[0..self.len.div_ceil(BITS)]
                .iter()
                .for_each(|x| x.store(0, order));
        }
    }

    pub fn flip(&mut self, order: Ordering) {
        let data: &[AtomicUsize] = self.data.as_ref();
        let end = self.len / BITS;
        let residual = self.len % BITS;
        data[0..end].iter().for_each(|x| _ = x.fetch_xor(!0, order));
        if residual != 0 {
            data[end].fetch_xor((1 << residual) - 1, order);
        }
    }
}

impl<B: AsRef<[usize]>> BitCount for BitVec<B> {
    fn count_ones(&self) -> usize {
        #[cfg(feature = "rayon")]
        {
            self.data
                .as_ref()
                .par_iter()
                .map(|x| x.count_ones() as usize)
                .sum()
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.data.iter().map(|x| x.count_ones() as usize).sum()
        }
    }
}

unsafe fn select_hinted_unchecked(
    data: impl AsRef<[usize]>,
    rank: usize,
    pos: usize,
    rank_at_pos: usize,
) -> usize {
    let mut word_index = pos / BITS;
    let bit_index = pos % BITS;
    let mut residual = rank - rank_at_pos;
    let mut word = (data.as_ref().get_unchecked(word_index) >> bit_index) << bit_index;
    loop {
        let bit_count = word.count_ones() as usize;
        if residual < bit_count {
            break;
        }
        word_index += 1;
        word = *data.as_ref().get_unchecked(word_index);
        residual -= bit_count;
    }

    word_index * BITS + word.select_in_word(residual)
}

fn select_hinted(
    data: impl AsRef<[usize]>,
    rank: usize,
    pos: usize,
    rank_at_pos: usize,
) -> Option<usize> {
    let mut word_index = pos / BITS;
    let bit_index = pos % BITS;
    let mut residual = rank - rank_at_pos;
    let mut word = (data.as_ref().get(word_index)? >> bit_index) << bit_index;
    loop {
        let bit_count = word.count_ones() as usize;
        if residual < bit_count {
            break;
        }
        word_index += 1;
        word = *data.as_ref().get(word_index)?;
        residual -= bit_count;
    }

    Some(word_index * BITS + word.select_in_word(residual))
}

unsafe fn select_zero_hinted_unchecked(
    data: impl AsRef<[usize]>,
    rank: usize,
    hint_pos: usize,
    hint_rank: usize,
) -> usize {
    let mut word_index = hint_pos / BITS;
    let bit_index = hint_pos % BITS;
    let mut residual = rank - hint_rank;
    let mut word = (!*data.as_ref().get_unchecked(word_index) >> bit_index) << bit_index;
    loop {
        let bit_count = word.count_ones() as usize;
        if residual < bit_count {
            break;
        }
        word_index += 1;
        word = !data.as_ref().get_unchecked(word_index);
        residual -= bit_count;
    }

    word_index * BITS + word.select_in_word(residual)
}

fn select_zero_hinted(
    data: impl AsRef<[usize]>,
    len: usize,
    rank: usize,
    hint_pos: usize,
    hint_rank: usize,
) -> Option<usize> {
    let mut word_index = hint_pos / BITS;
    let bit_index = hint_pos % BITS;
    let mut residual = rank - hint_rank;
    let mut word = (!data.as_ref().get(word_index)? >> bit_index) << bit_index;
    loop {
        let bit_count = word.count_ones() as usize;
        if residual < bit_count {
            break;
        }
        word_index += 1;
        word = !*data.as_ref().get(word_index)?;
        residual -= bit_count;
    }

    let result = word_index * BITS + word.select_in_word(residual);
    if result >= len {
        None
    } else {
        Some(result)
    }
}

impl<B: AsRef<[usize]>> SelectHinted for BitVec<B> {
    unsafe fn select_hinted_unchecked(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        select_hinted_unchecked(self.data.as_ref(), rank, hint_pos, hint_rank)
    }

    fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize> {
        select_hinted(self.data.as_ref(), rank, hint_pos, hint_rank)
    }
}

impl<B: AsRef<[usize]>> SelectZeroUnchecked for BitVec<B> {
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.select_zero_hinted_unchecked(rank, 0, 0)
    }
}

impl<B: AsRef<[usize]>> SelectZeroHinted for BitVec<B> {
    unsafe fn select_zero_hinted_unchecked(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        select_zero_hinted_unchecked(self.data.as_ref(), rank, hint_pos, hint_rank)
    }

    fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> Option<usize> {
        select_zero_hinted(self.data.as_ref(), self.len(), rank, hint_pos, hint_rank)
    }
}

impl<W: IntoAtomic> From<BitVec<Vec<W>>> for AtomicBitVec<Vec<W::AtomicType>> {
    fn from(value: BitVec<Vec<W>>) -> Self {
        AtomicBitVec {
            data: unsafe { core::mem::transmute::<Vec<W>, Vec<W::AtomicType>>(value.data) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<BitVec<&'a [W]>> for AtomicBitVec<&'a [W::AtomicType]> {
    fn from(value: BitVec<&'a [W]>) -> Self {
        AtomicBitVec {
            data: unsafe { core::mem::transmute::<&'a [W], &'a [W::AtomicType]>(value.data) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<BitVec<&'a mut [W]>> for AtomicBitVec<&'a mut [W::AtomicType]> {
    fn from(value: BitVec<&'a mut [W]>) -> Self {
        AtomicBitVec {
            data: unsafe {
                core::mem::transmute::<&'a mut [W], &'a mut [W::AtomicType]>(value.data)
            },
            len: value.len,
        }
    }
}

impl<W: IntoAtomic> From<AtomicBitVec<Vec<W::AtomicType>>> for BitVec<Vec<W>> {
    fn from(value: AtomicBitVec<Vec<W::AtomicType>>) -> Self {
        BitVec {
            data: unsafe { core::mem::transmute::<Vec<W::AtomicType>, Vec<W>>(value.data) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<AtomicBitVec<&'a [W::AtomicType]>> for BitVec<&'a [W]> {
    fn from(value: AtomicBitVec<&'a [W::AtomicType]>) -> Self {
        BitVec {
            data: unsafe { core::mem::transmute::<&'a [W::AtomicType], &'a [W]>(value.data) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<AtomicBitVec<&'a mut [W::AtomicType]>> for BitVec<&'a mut [W]> {
    fn from(value: AtomicBitVec<&'a mut [W::AtomicType]>) -> Self {
        BitVec {
            data: unsafe {
                core::mem::transmute::<&'a mut [W::AtomicType], &'a mut [W]>(value.data)
            },
            len: value.len,
        }
    }
}

impl<B: AsRef<[usize]>> AsRef<[usize]> for BitVec<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.data.as_ref()
    }
}

impl<B: AsMut<[usize]>> AsMut<[usize]> for BitVec<B> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [usize] {
        self.data.as_mut()
    }
}

impl<B: AsRef<[AtomicUsize]>> AsRef<[AtomicUsize]> for AtomicBitVec<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[AtomicUsize] {
        self.data.as_ref()
    }
}

impl FromIterator<bool> for BitVec<Vec<usize>> {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut res = Self::new(0);
        res.extend(iter);
        res
    }
}

/// An iterator over the ones in an underlying storage.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct OnesIterator<B> {
    mem_words: B,
    word_idx: usize,
    /// This is a usize because BitVec is currently implemented only for `Vec<usize>` and `&[usize]`.
    word: usize,
    len: usize,
}

impl<B: AsRef<[usize]>> OnesIterator<B> {
    pub fn new(mem_words: B, len: usize) -> Self {
        let word = if mem_words.as_ref().is_empty() {
            0
        } else {
            unsafe { *mem_words.as_ref().get_unchecked(0) }
        };
        Self {
            mem_words,
            word_idx: 0,
            word,
            len,
        }
    }
}

impl<B: AsRef<[usize]>> Iterator for OnesIterator<B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        // find the next word with zeros
        while self.word == 0 {
            self.word_idx += 1;
            debug_assert!(self.word_idx < self.mem_words.as_ref().len());
            self.word = unsafe { *self.mem_words.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * BITS) + bit_idx;
        // clear the lowest bit set
        self.word &= self.word - 1;
        self.len -= 1;
        Some(res)
    }
}

// Iterates over the bits as booleans.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitIterator<'a, B> {
    mem_words: &'a B,
    next_bit_pos: usize,
    len: usize,
}

impl<'a> IntoIterator for &'a BitVec<Vec<usize>> {
    type IntoIter = BitIterator<'a, Vec<usize>>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitIterator {
            mem_words: &self.data,
            next_bit_pos: 0,
            len: self.len,
        }
    }
}

impl<'a> Iterator for BitIterator<'a, Vec<usize>> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let word_idx = self.next_bit_pos / BITS;
        let bit_idx = self.next_bit_pos % BITS;
        let word = unsafe { *self.mem_words.get_unchecked(word_idx) };
        let bit = (word >> bit_idx) & 1;
        self.next_bit_pos += 1;
        Some(bit != 0)
    }
}

impl fmt::Display for BitVec<Vec<usize>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for b in self {
            write!(f, "{:b}", b as usize)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[inline(always)]
unsafe fn rank_hinted_unchecked<const HINT_BIT_SIZE: usize>(
    data: impl AsRef<[usize]>,
    pos: usize,
    hint_pos: usize,
    hint_rank: usize,
) -> usize {
    let data = data.as_ref();
    let mut rank = hint_rank;
    let mut hint_pos = hint_pos;

    debug_assert!(
        hint_pos < data.len(),
        "hint_pos: {}, len: {}",
        hint_pos,
        data.len()
    );

    while (hint_pos + 1) * HINT_BIT_SIZE <= pos {
        rank += data.get_unchecked(hint_pos).count_ones() as usize;
        hint_pos += 1;
    }

    rank + (data.get_unchecked(hint_pos) & ((1 << (pos % HINT_BIT_SIZE)) - 1)).count_ones() as usize
}

impl<B: AsRef<[usize]>, const HINT_BIT_SIZE: usize> RankHinted<HINT_BIT_SIZE> for BitVec<B> {
    #[inline(always)]
    unsafe fn rank_hinted_unchecked(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> usize {
        rank_hinted_unchecked::<HINT_BIT_SIZE>(self.data.as_ref(), pos, hint_pos, hint_rank)
    }

    fn rank_hinted(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> Option<usize> {
        if pos > self.len() || hint_pos * HINT_BIT_SIZE > pos {
            return None;
        }
        Some(unsafe {
            rank_hinted_unchecked::<HINT_BIT_SIZE>(self.data.as_ref(), pos, hint_pos, hint_rank)
        })
    }
}
