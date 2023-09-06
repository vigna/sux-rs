use crate::traits::*;
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Wrapper over a bitmap that keeps tracks of the number of ones
#[derive(Debug)]
pub struct CountingBitmap<B, C> {
    bitmap: BitMap<B>,
    number_of_ones: C,
}

impl<C, T, B: AsRef<T>> AsRef<T> for CountingBitmap<B, C>
where
    BitMap<B>: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.bitmap.as_ref()
    }
}

impl CountingBitmap<Vec<u64>, usize> {
    pub fn new(len: usize) -> Self {
        Self {
            bitmap: BitMap::new(len),
            number_of_ones: 0,
        }
    }
}

impl CountingBitmap<Vec<AtomicU64>, AtomicUsize> {
    pub fn new_atomic(len: usize) -> Self {
        Self {
            bitmap: BitMap::new_atomic(len),
            number_of_ones: AtomicUsize::new(0),
        }
    }
}

impl<B, S> BitLength for CountingBitmap<B, S> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.bitmap.len
    }
}

impl<B> BitCount for CountingBitmap<B, usize> {
    #[inline(always)]
    fn count(&self) -> usize {
        self.number_of_ones
    }
}

impl<B> BitCount for CountingBitmap<B, AtomicUsize> {
    #[inline(always)]
    fn count(&self) -> usize {
        self.number_of_ones.load(Ordering::SeqCst)
    }
}

impl<B, S> CountingBitmap<B, S> {
    /// # Safety
    /// TODO: this function is never used
    #[inline(always)]
    pub unsafe fn from_raw_parts(bitmap: BitMap<B>, number_of_ones: S) -> Self {
        Self {
            bitmap,
            number_of_ones: number_of_ones,
        }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (BitMap<B>, S) {
        (self.bitmap, self.number_of_ones)
    }
}

impl<B: VSliceCore, S> VSliceCore for CountingBitmap<B, S> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        self.bitmap.bit_width()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.bitmap.len
    }
}

impl<B: VSlice, S> VSlice for CountingBitmap<B, S> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        self.bitmap.get_unchecked(index)
    }
}

impl<B: VSliceMut> VSliceMut for CountingBitmap<B, usize> {
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        // get the word index, and the bit index in the word
        let word_index = index / self.bitmap.data.bit_width();
        let bit_index = index % self.bitmap.data.bit_width();
        // get the old word
        let word = self.bitmap.data.get_unchecked(word_index);
        // clean the old bit in the word
        let mut new_word = word & !(1 << bit_index);
        // and write the new one
        new_word |= value << bit_index;
        // write it back
        self.bitmap.data.set_unchecked(word_index, new_word);
        // we are safe to use this as we have mut access so we are the only ones
        // and there are no concurrency

        // update the count of ones if we added a one
        self.number_of_ones += (new_word > word) as usize;
        // update the count of ones if we removed a one
        self.number_of_ones -= (new_word < word) as usize;
    }
}

impl<B: VSlice> Select for CountingBitmap<B, usize> {
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        self.select_unchecked_hinted(rank, 0, 0)
    }
}

impl<B: VSlice> SelectHinted for CountingBitmap<B, usize> {
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        let mut word_index = pos / self.bitmap.data.bit_width();
        let bit_index = pos % self.bitmap.data.bit_width();
        let mut residual = rank - rank_at_pos;
        // TODO!: M2L or L2M?
        let mut word = (self.bitmap.data.get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                break;
            }
            word_index += 1;
            word = self.bitmap.data.get_unchecked(word_index);
            residual -= bit_count;
        }

        word_index * self.bitmap.data.bit_width() + word.select_in_word(residual)
    }
}

impl<B: VSlice> SelectZero for CountingBitmap<B, usize> {
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.select_zero_unchecked_hinted(rank, 0, 0)
    }
}

impl<B: VSlice> SelectZeroHinted for CountingBitmap<B, usize> {
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        let mut word_index = pos / self.bitmap.data.bit_width();
        let bit_index = pos % self.bitmap.data.bit_width();
        let mut residual = rank - rank_at_pos;
        // TODO!: M2L or L2M?
        let mut word = (!self.bitmap.data.get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                break;
            }
            word_index += 1;
            word = !self.bitmap.data.get_unchecked(word_index);
            residual -= bit_count;
        }

        word_index * self.bitmap.data.bit_width() + word.select_in_word(residual)
    }
}

impl<B: VSliceMutAtomicCmpExchange> VSliceAtomic for CountingBitmap<B, AtomicUsize> {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        self.bitmap.get_atomic_unchecked(index, order)
    }
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        // get the word index, and the bit index in the word
        let word_index = index / self.bitmap.data.bit_width();
        let bit_index = index % self.bitmap.data.bit_width();
        let mut word = self.bitmap.data.get_atomic_unchecked(word_index, order);
        let mut new_word;
        loop {
            // get the old word
            // clean the old bit in the word
            new_word = word & !(1 << bit_index);
            // and write the new one
            new_word |= value << bit_index;
            // write it back
            // idk if the ordering is reasonable here, the only reasonable is
            // Release
            match self
                .bitmap
                .data
                .compare_exchange_unchecked(word_index, word, new_word, order, order)
            {
                Ok(_) => break,
                Err(w) => word = w,
            }
        }
        // update the count of ones if we added a one
        // update the count of ones if we removed a one
        let inc = (new_word > word) as isize - (new_word < word) as isize;
        // use the isize as usize (which JUST re-interprets the bits)
        // to do a single fetch_add and ensure consistency
        self.number_of_ones
            .fetch_add(inc as usize, Ordering::Relaxed);
    }
}

impl<B: VSliceMutAtomicCmpExchange> VSliceMutAtomicCmpExchange for CountingBitmap<B, AtomicUsize> {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        // get the word index, and the bit index in the word
        let word_index = index / self.bitmap.data.bit_width();
        let bit_index = index % self.bitmap.data.bit_width();
        // get the old word
        let word = self
            .bitmap
            .data
            .get_atomic_unchecked(word_index, Ordering::Acquire);
        // clean the old bit in the word
        let clean_word = word & !(1 << bit_index);
        // and write the new one
        let cur_word = clean_word | (current << bit_index);
        let new_word = clean_word | (new << bit_index);
        // write it back
        let res = self
            .bitmap
            .data
            .compare_exchange_unchecked(word_index, cur_word, new_word, success, failure);
        // if the exchange was successful, update the count of ones
        if res.is_ok() {
            // update the count of ones if we added a one
            // update the count of ones if we removed a one
            let inc = (new > current) as isize - (new < current) as isize;
            // use the isize as usize (which JUST re-interprets the bits)
            // to do a single fetch_add and ensure consistency
            self.number_of_ones
                .fetch_add(inc as usize, Ordering::Relaxed);
        }
        res
    }
}

#[derive(Debug)]
pub struct BitMap<B> {
    data: B,
    len: usize,
}

impl BitMap<Vec<u64>> {
    pub fn new(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: vec![0; n_of_words],
            len,
        }
    }
}

impl BitMap<Vec<AtomicU64>> {
    pub fn new_atomic(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: (0..n_of_words).map(|_| AtomicU64::new(0)).collect(),
            len,
        }
    }
}

impl<B> BitMap<B> {
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

impl BitMap<Vec<u64>> {
    #[inline(always)]
    pub fn with_count(self, number_of_ones: usize) -> CountingBitmap<Vec<u64>, usize> {
        CountingBitmap {
            bitmap: self,
            number_of_ones,
        }
    }
}
impl BitMap<Vec<AtomicU64>> {
    #[inline(always)]
    pub fn with_count(self, number_of_ones: usize) -> CountingBitmap<Vec<AtomicU64>, AtomicUsize> {
        CountingBitmap {
            bitmap: self,
            number_of_ones: AtomicUsize::new(number_of_ones),
        }
    }
}

impl<B> BitLength for BitMap<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: VSliceCore> VSliceCore for BitMap<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(1 <= self.data.bit_width());
        1
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: VSlice> VSlice for BitMap<B> {
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_unchecked(word_index);
        (word >> (index % self.data.bit_width())) & 1
    }
}

impl<B: VSliceMut> VSliceMut for BitMap<B> {
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        // get the word index, and the bit index in the word
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();
        // get the old word
        let word = self.data.get_unchecked(word_index);
        // clean the old bit in the word
        let mut new_word = word & !(1 << bit_index);
        // and write the new one
        new_word |= value << bit_index;
        // write it back
        self.data.set_unchecked(word_index, new_word);
    }
}

impl<B: VSliceMutAtomicCmpExchange> VSliceAtomic for BitMap<B> {
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_atomic_unchecked(word_index, order);
        (word >> (index % self.data.bit_width())) & 1
    }
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
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
            new_word |= value << bit_index;
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

impl<B: VSliceMutAtomicCmpExchange> VSliceMutAtomicCmpExchange for BitMap<B> {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        // get the word index, and the bit index in the word
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();
        // get the old word
        let word = self
            .data
            .get_atomic_unchecked(word_index, Ordering::Acquire);
        // clean the old bit in the word
        let clean_word = word & !(1 << bit_index);
        // and write the new one
        let cur_word = clean_word | (current << bit_index);
        let new_word = clean_word | (new << bit_index);
        // write it back
        self.data
            .compare_exchange_unchecked(word_index, cur_word, new_word, success, failure)
    }
}

impl<B: AsRef<[u64]>, D: AsRef<[u64]>> ConvertTo<BitMap<D>> for BitMap<B>
where
    B: ConvertTo<D>,
{
    fn convert_to(self) -> Result<BitMap<D>> {
        Ok(BitMap {
            len: self.len,
            data: self.data.convert_to()?,
        })
    }
}

impl<B1, C1, B2, C2> ConvertTo<CountingBitmap<B2, C2>> for CountingBitmap<B1, C1>
where
    BitMap<B1>: ConvertTo<BitMap<B2>>,
    C1: ConvertTo<C2>,
{
    #[inline(always)]
    fn convert_to(self) -> Result<CountingBitmap<B2, C2>> {
        Ok(CountingBitmap {
            bitmap: self.bitmap.convert_to()?,
            number_of_ones: self.number_of_ones.convert_to()?,
        })
    }
}

impl<B: AsRef<[u64]>> AsRef<[u64]> for BitMap<B> {
    fn as_ref(&self) -> &[u64] {
        self.data.as_ref()
    }
}
impl<B: AsRef<[AtomicU64]>> AsRef<[AtomicU64]> for BitMap<B> {
    fn as_ref(&self) -> &[AtomicU64] {
        self.data.as_ref()
    }
}

impl From<BitMap<Vec<u64>>> for BitMap<Vec<AtomicU64>> {
    #[inline]
    fn from(bm: BitMap<Vec<u64>>) -> Self {
        BitMap {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl From<BitMap<Vec<AtomicU64>>> for BitMap<Vec<u64>> {
    #[inline]
    fn from(bm: BitMap<Vec<AtomicU64>>) -> Self {
        BitMap {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitMap<&'a [AtomicU64]>> for BitMap<&'a [u64]> {
    #[inline]
    fn from(bm: BitMap<&'a [AtomicU64]>) -> Self {
        BitMap {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitMap<&'a [u64]>> for BitMap<&'a [AtomicU64]> {
    #[inline]
    fn from(bm: BitMap<&'a [u64]>) -> Self {
        BitMap {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitMap<&'a mut [AtomicU64]>> for BitMap<&'a mut [u64]> {
    #[inline]
    fn from(bm: BitMap<&'a mut [AtomicU64]>) -> Self {
        BitMap {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl<'a> From<BitMap<&'a mut [u64]>> for BitMap<&'a mut [AtomicU64]> {
    #[inline]
    fn from(bm: BitMap<&'a mut [u64]>) -> Self {
        BitMap {
            data: bm.data.convert_to().unwrap(),
            len: bm.len,
        }
    }
}

impl From<CountingBitmap<Vec<u64>, usize>> for CountingBitmap<Vec<AtomicU64>, AtomicUsize> {
    #[inline]
    fn from(bm: CountingBitmap<Vec<u64>, usize>) -> Self {
        CountingBitmap {
            bitmap: bm.bitmap.into(),
            number_of_ones: AtomicUsize::new(bm.number_of_ones),
        }
    }
}

impl From<CountingBitmap<Vec<AtomicU64>, AtomicUsize>> for CountingBitmap<Vec<u64>, usize> {
    #[inline]
    fn from(bm: CountingBitmap<Vec<AtomicU64>, AtomicUsize>) -> Self {
        CountingBitmap {
            bitmap: bm.bitmap.into(),
            number_of_ones: bm.number_of_ones.into_inner(),
        }
    }
}

impl<'a> From<CountingBitmap<&'a [AtomicU64], AtomicUsize>> for CountingBitmap<&'a [u64], usize> {
    #[inline]
    fn from(bm: CountingBitmap<&'a [AtomicU64], AtomicUsize>) -> Self {
        CountingBitmap {
            bitmap: bm.bitmap.into(),
            number_of_ones: bm.number_of_ones.into_inner(),
        }
    }
}

impl<'a> From<CountingBitmap<&'a [u64], usize>> for CountingBitmap<&'a [AtomicU64], AtomicUsize> {
    #[inline]
    fn from(bm: CountingBitmap<&'a [u64], usize>) -> Self {
        CountingBitmap {
            bitmap: bm.bitmap.into(),
            number_of_ones: AtomicUsize::new(bm.number_of_ones),
        }
    }
}

impl<'a> From<CountingBitmap<&'a mut [AtomicU64], AtomicUsize>>
    for CountingBitmap<&'a mut [u64], usize>
{
    #[inline]
    fn from(bm: CountingBitmap<&'a mut [AtomicU64], AtomicUsize>) -> Self {
        CountingBitmap {
            bitmap: bm.bitmap.into(),
            number_of_ones: bm.number_of_ones.into_inner(),
        }
    }
}

impl<'a> From<CountingBitmap<&'a mut [u64], usize>>
    for CountingBitmap<&'a mut [AtomicU64], AtomicUsize>
{
    #[inline]
    fn from(bm: CountingBitmap<&'a mut [u64], usize>) -> Self {
        CountingBitmap {
            bitmap: bm.bitmap.into(),
            number_of_ones: AtomicUsize::new(bm.number_of_ones),
        }
    }
}

impl<B, C> From<CountingBitmap<B, C>> for BitMap<B> {
    fn from(cb: CountingBitmap<B, C>) -> Self {
        cb.bitmap
    }
}

impl From<BitMap<Vec<u64>>> for CountingBitmap<Vec<u64>, usize> {
    fn from(bitmap: BitMap<Vec<u64>>) -> Self {
        // THIS MIGHT BE SLOW
        let number_of_ones = bitmap
            .as_ref()
            .iter()
            .map(|x| x.count_ones() as usize)
            .sum();
        Self {
            bitmap,
            number_of_ones,
        }
    }
}

impl From<BitMap<Vec<AtomicU64>>> for CountingBitmap<Vec<AtomicU64>, AtomicUsize> {
    fn from(bitmap: BitMap<Vec<AtomicU64>>) -> Self {
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);

        // THIS MIGHT BE SLOW
        let number_of_ones = bitmap
            .as_ref()
            .iter()
            .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
            .sum();

        Self {
            bitmap,
            number_of_ones: AtomicUsize::new(number_of_ones),
        }
    }
}
