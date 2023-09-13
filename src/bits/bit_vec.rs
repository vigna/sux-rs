/*
 * SPDX-FileCopyrightText: 2023 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Bit vector implementations. There are three flavors:

- `BitVec<Vec<u64>>`: a mutable bit vector with a `Vec<u64>` as underlying storage;
- `BitVec<&[u64]>`: a mutable bit vector with a `&[u64]` as underlying storage,
   mainly useful for [`epserde`];
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

/// A bit vector with selectable backend. We provide implementations
/// for `Vec<u64>` and `Vec<AtomicU64>`.
///
/// In the second case, [`BitVec::get`]
/// and [`BitVec::set`] are both thread-safe, as they both take an immutable reference.
#[derive(Epserde, Debug)]
pub struct BitVec<B> {
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
    /// Create a new bit vector of length `len`.
    pub fn new(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: vec![0; n_of_words],
            len,
        }
    }
}

impl BitVec<Vec<AtomicU64>> {
    /// Create a new atomic bit vector of length `len`.
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

impl BitVec<Vec<u64>> {
    /// Return the number of bits set to 1 in this bit vector.
    ///
    /// If the feature "rayon" is enabled, this function is parallelized.
    pub fn count_ones(&self) -> usize {
        #[cfg(feature = "rayon")]
        {
            self.data.par_iter().map(|x| x.count_ones() as usize).sum()
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.data.iter().map(|x| x.count_ones() as usize).sum()
        }
    }

    /// Return a [`CountBitVec`] with the same data as this
    /// bit vector and the assuming the given number of ones.
    ///
    /// # Warning
    /// No control is performed on the number of ones, unless
    /// debug assertions are enabled.</div>
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
    /// Return the number of bits set to 1 in this bit vector.
    ///
    /// If the feature "rayon" is enabled, this function is parallelized.
    pub fn count_ones(&self) -> usize {
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);

        #[cfg(feature = "rayon")]
        {
            self.data
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

impl BitVec<Vec<u64>> {
    pub fn get(&self, index: usize) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unchecked(index) }
    }

    pub fn set(&mut self, index: usize, value: bool) {
        panic_if_out_of_bounds!(index, self.len);
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
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();

        // For constant values, this should be inlined with no test.
        if value {
            *self.data.get_unchecked_mut(word_index) |= 1 << bit_index;
        } else {
            *self.data.get_unchecked_mut(word_index) &= !(1 << bit_index);
        }
    }
}

impl BitVec<&[u64]> {
    pub fn get(&self, index: usize) -> bool {
        panic_if_out_of_bounds!(index, self.len);
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
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unchecked(index, order) }
    }

    pub fn set(&self, index: usize, value: bool, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.set_unchecked(index, value, order) }
    }

    unsafe fn get_unchecked(&self, index: usize, order: Ordering) -> bool {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_atomic_unchecked(word_index, order);
        (word >> (index % self.data.bit_width())) & 1 != 0
    }
    #[inline(always)]
    unsafe fn set_unchecked(&self, index: usize, value: bool, order: Ordering) {
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();

        // For constant values, this should be inlined with no test.
        if value {
            self.data
                .get_unchecked(word_index)
                .fetch_or(1 << bit_index, order);
        } else {
            self.data
                .get_unchecked(word_index)
                .fetch_and(!(1 << bit_index), order);
        }
    }
}

/// An immutable bit vector that returns the number of ones.
#[derive(Epserde, Debug)]
pub struct CountBitVec<B> {
    data: B,
    len: usize,
    number_of_ones: usize,
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

impl Index<usize> for CountBitVec<Vec<u64>> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            false => &false,
            true => &true,
        }
    }
}

impl<B> CountBitVec<B> {
    /// # Safety
    /// `len` must be between 0 (included) the number of
    /// bits in `data` (included). No test is performed
    /// on the number of ones.
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
        panic_if_out_of_bounds!(index, self.len);
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

/// Provide conversion betweeen bit vectors whose backends
/// are [convertible](ConvertTo) into one another.
///
/// Many implementations of this trait are then used to
/// implement by delegation a corresponding [`From`].
impl<B, D> ConvertTo<BitVec<D>> for BitVec<B>
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

/// Provide conversion from standard to atomic bit vectors.
impl From<BitVec<Vec<u64>>> for BitVec<Vec<AtomicU64>> {
    #[inline]
    fn from(bm: BitVec<Vec<u64>>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from standard to atomic bit vectors.
impl From<BitVec<Vec<AtomicU64>>> for BitVec<Vec<u64>> {
    #[inline]
    fn from(bm: BitVec<Vec<AtomicU64>>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from references to standard bit vectors
/// to references to atomic bit vectors.
impl<'a> From<BitVec<&'a [u64]>> for BitVec<&'a [AtomicU64]> {
    #[inline]
    fn from(bm: BitVec<&'a [u64]>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from references to atomic bit vectors
/// to references to standard bit vectors.
impl<'a> From<BitVec<&'a [AtomicU64]>> for BitVec<&'a [u64]> {
    #[inline]
    fn from(bm: BitVec<&'a [AtomicU64]>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from mutable references to standard bit vectors
/// to mutable references to atomic bit vectors.
impl<'a> From<BitVec<&'a mut [u64]>> for BitVec<&'a mut [AtomicU64]> {
    #[inline]
    fn from(bm: BitVec<&'a mut [u64]>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Provide conversion from mutable references to atomic bit vectors
/// to mutable references to standard bit vectors.
impl<'a> From<BitVec<&'a mut [AtomicU64]>> for BitVec<&'a mut [u64]> {
    #[inline]
    fn from(bm: BitVec<&'a mut [AtomicU64]>) -> Self {
        bm.convert_to().unwrap()
    }
}

/// Forget the number of ones.
impl<B> ConvertTo<BitVec<B>> for CountBitVec<B> {
    fn convert_to(self) -> Result<BitVec<B>> {
        Ok(BitVec {
            data: self.data,
            len: self.len,
        })
    }
}

/// Forget the number of ones.
impl<B> From<CountBitVec<B>> for BitVec<B> {
    fn from(cb: CountBitVec<B>) -> Self {
        cb.convert_to().unwrap()
    }
}

/// Compute the number of ones and return a [`CountBitVec`].
impl ConvertTo<CountBitVec<Vec<u64>>> for BitVec<Vec<u64>> {
    fn convert_to(self) -> Result<CountBitVec<Vec<u64>>> {
        let number_of_ones = self.count_ones();
        Ok(CountBitVec {
            data: self.data,
            len: self.len,
            number_of_ones,
        })
    }
}

/// Compute the number of ones and return a [`CountBitVec`].
impl From<BitVec<Vec<u64>>> for CountBitVec<Vec<u64>> {
    fn from(bitmap: BitVec<Vec<u64>>) -> Self {
        bitmap.convert_to().unwrap()
    }
}
