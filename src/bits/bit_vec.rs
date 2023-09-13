/*
 * SPDX-FileCopyrightText: 2023 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Bit vector implementations. There are three flavors:

- `BitVec<Vec<u64>>`: a mutable bit vector with a `Vec<u64>` as underlying storage;
- `BitVec<Vec<AtomicU64>>`: a thread-safe mutable bit vector
   with a `Vec<AtomicU64>` as underlying storage;
- `CountBitVec<Vec<u64>, usize>`: an immutable bit vector with a `Vec<u64>`
   as underlying storage that implements [`BitCount`].

It is possible to juggle between the three flavors using [`From`].
 */
use crate::traits::*;
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    ops::Index,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Epserde, Debug)]
pub struct BitVec<B> {
    data: B,
    len: usize,
}

impl BitVec<Vec<u64>> {
    pub fn new(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: vec![0; n_of_words],
            len,
        }
    }
}

impl BitVec<Vec<AtomicU64>> {
    pub fn new_atomic(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: (0..n_of_words).map(|_| AtomicU64::new(0)).collect(),
            len,
        }
    }
}

impl<B> BitVec<B> {
    #[inline(always)]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }
    /// # Safety
    /// TODO: this function is never used
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, len: usize) -> Self {
        Self { data, len }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.data, self.len)
    }
}

impl BitVec<Vec<u64>> {
    pub fn count_ones(&self) -> usize {
        #[cfg(feature = "rayon")]
        {
            self.as_ref()
                .par_iter()
                .map(|x| x.count_ones() as usize)
                .sum()
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.as_ref().iter().map(|x| x.count_ones() as usize).sum()
        }
    }

    #[inline(always)]
    pub fn with_count(self, number_of_ones: usize) -> CountBitVec<Vec<u64>> {
        debug_assert!(number_of_ones <= self.len);
        debug_assert_eq!(number_of_ones, self.count_ones());
        CountBitVec {
            data: self.data,
            len: self.len,
            number_of_ones,
        }
    }
}

impl BitVec<Vec<AtomicU64>> {
    pub fn count_ones(&self) -> usize {
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);

        #[cfg(feature = "rayon")]
        {
            self.as_ref()
                .par_iter()
                .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
                .sum()
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.as_ref()
                .iter()
                .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
                .sum()
        }
    }
}

impl<B> BitLength for BitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

impl Index<usize> for BitVec<Vec<u64>> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            false => &false,
            true => &true,
        }
    }
}

impl BitVec<Vec<u64>> {
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        unsafe { self.get_unchecked(index) }
    }

    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < self.len);
        unsafe { self.set_unchecked(index, value) }
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_unchecked(word_index);
        (word >> (index % self.data.bit_width())) & 1 != 0
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        // get the word index, and the bit index in the word
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();
        // get the old word
        let word = self.data.get_unchecked(word_index);
        // clean the old bit in the word
        let mut new_word = word & !(1 << bit_index);
        // and write the new one
        new_word |= if value { 1 } else { 0 } << bit_index;
        // write it back
        self.data.set_unchecked(word_index, new_word);
    }
}

impl BitVec<&[u64]> {
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        unsafe { self.get_unchecked(index) }
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_unchecked(word_index);
        (word >> (index % self.data.bit_width())) & 1 != 0
    }
}

impl BitVec<Vec<AtomicU64>> {
    pub fn get(&self, index: usize, order: Ordering) -> bool {
        assert!(index < self.len);
        unsafe { self.get_unchecked(index, order) }
    }

    pub fn set(&self, index: usize, value: bool, order: Ordering) {
        assert!(index < self.len);
        unsafe { self.set_unchecked(index, value, order) }
    }

    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> bool {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_atomic_unchecked(word_index, order);
        (word >> (index % self.data.bit_width())) & 1 != 0
    }
    unsafe fn set_unchecked(&self, index: usize, value: bool, order: Ordering) {
        // get the word index, and the bit index in the word
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();
        let mut word = self.data.get_atomic_unchecked(word_index, order);
        let mut new_word;
        loop {
            // get the old word
            // clean the old bit in the word
            new_word = word & !(1 << bit_index);
            // and write the new one
            new_word |= if value { 1 } else { 0 } << bit_index;
            // write it back
            // idk if the ordering is reasonable here, the only reasonable is
            // Release
            match self
                .data
                .compare_exchange_unchecked(word_index, word, new_word, order, order)
            {
                Ok(_) => break,
                Err(w) => word = w,
            }
        }
    }
}

/// A bitmap that keeps tracks of the number of ones
#[derive(Epserde, Debug)]
pub struct CountBitVec<B> {
    data: B,
    len: usize,
    number_of_ones: usize,
}

impl<T, B: AsRef<T>> AsRef<T> for CountBitVec<B> {
    fn as_ref(&self) -> &T {
        self.data.as_ref()
    }
}

impl CountBitVec<Vec<u64>> {
    pub fn new(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: vec![0; n_of_words],
            len,
            number_of_ones: 0,
        }
    }
}

impl<B> BitLength for CountBitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B> BitCount for CountBitVec<B> {
    #[inline(always)]
    fn count(&self) -> usize {
        self.number_of_ones
    }
}

impl<B> CountBitVec<B> {
    /// # Safety
    /// TODO: this function is never used
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, len: usize, number_of_ones: usize) -> Self {
        Self {
            data,
            len,
            number_of_ones,
        }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.data, self.len, self.number_of_ones)
    }
}

impl CountBitVec<Vec<u64>> {
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        unsafe { self.get_unchecked(index) }
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_unchecked(word_index);
        (word >> (index % self.data.bit_width())) & 1 != 0
    }
}

impl<B: VSlice> Select for CountBitVec<B> {
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        self.select_unchecked_hinted(rank, 0, 0)
    }
}

impl<B: VSlice> SelectHinted for CountBitVec<B> {
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        let mut word_index = pos / self.data.bit_width();
        let bit_index = pos % self.data.bit_width();
        let mut residual = rank - rank_at_pos;
        let mut word = (self.data.get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                break;
            }
            word_index += 1;
            word = self.data.get_unchecked(word_index);
            residual -= bit_count;
        }

        word_index * self.data.bit_width() + word.select_in_word(residual)
    }
}

impl<B: VSlice> SelectZero for CountBitVec<B> {
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.select_zero_unchecked_hinted(rank, 0, 0)
    }
}

impl<B: VSlice> SelectZeroHinted for CountBitVec<B> {
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        let mut word_index = pos / self.data.bit_width();
        let bit_index = pos % self.data.bit_width();
        let mut residual = rank - rank_at_pos;
        let mut word = (!self.data.get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                break;
            }
            word_index += 1;
            word = !self.data.get_unchecked(word_index);
            residual -= bit_count;
        }

        word_index * self.data.bit_width() + word.select_in_word(residual)
    }
}

impl<B: AsRef<[u64]>, D: AsRef<[u64]>> ConvertTo<BitVec<D>> for BitVec<B>
where
    B: ConvertTo<D>,
{
    fn convert_to(self) -> Result<BitVec<D>> {
        Ok(BitVec {
            len: self.len,
            data: self.data.convert_to()?,
        })
    }
}

impl<B: AsRef<[u64]>> AsRef<[u64]> for BitVec<B> {
    fn as_ref(&self) -> &[u64] {
        self.data.as_ref()
    }
}
impl<B: AsRef<[AtomicU64]>> AsRef<[AtomicU64]> for BitVec<B> {
    fn as_ref(&self) -> &[AtomicU64] {
        self.data.as_ref()
    }
}
impl<B: AsRef<[u64]>> AsRef<[u64]> for CountBitVec<B> {
    fn as_ref(&self) -> &[u64] {
        self.data.as_ref()
    }
}
impl From<BitVec<Vec<u64>>> for BitVec<Vec<AtomicU64>> {
    #[inline]
    fn from(bm: BitVec<Vec<u64>>) -> Self {
        BitVec {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl From<BitVec<Vec<AtomicU64>>> for BitVec<Vec<u64>> {
    #[inline]
    fn from(bm: BitVec<Vec<AtomicU64>>) -> Self {
        BitVec {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitVec<&'a [AtomicU64]>> for BitVec<&'a [u64]> {
    #[inline]
    fn from(bm: BitVec<&'a [AtomicU64]>) -> Self {
        BitVec {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitVec<&'a [u64]>> for BitVec<&'a [AtomicU64]> {
    #[inline]
    fn from(bm: BitVec<&'a [u64]>) -> Self {
        BitVec {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitVec<&'a mut [AtomicU64]>> for BitVec<&'a mut [u64]> {
    #[inline]
    fn from(bm: BitVec<&'a mut [AtomicU64]>) -> Self {
        BitVec {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitVec<&'a mut [u64]>> for BitVec<&'a mut [AtomicU64]> {
    #[inline]
    fn from(bm: BitVec<&'a mut [u64]>) -> Self {
        BitVec {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<B> From<CountBitVec<B>> for BitVec<B> {
    fn from(cb: CountBitVec<B>) -> Self {
        BitVec {
            data: cb.data,
            len: cb.len,
        }
    }
}

impl From<BitVec<Vec<u64>>> for CountBitVec<Vec<u64>> {
    fn from(bitmap: BitVec<Vec<u64>>) -> Self {
        // THIS MIGHT BE SLOW
        let number_of_ones = bitmap.count_ones();
        Self {
            data: bitmap.data,
            len: bitmap.len,
            number_of_ones,
        }
    }
}
