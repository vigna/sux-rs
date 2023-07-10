use crate::traits::*;
use crate::utils::select_in_word;
use anyhow::Result;
use std::io::{Seek, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[derive(Debug)]
pub struct BitMap<B> {
    data: B,
    len: usize,
    number_of_ones: AtomicUsize,
}

impl BitMap<Vec<u64>> {
    pub fn new(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: vec![0; n_of_words],
            len,
            number_of_ones: AtomicUsize::new(0),
        }
    }
}

impl BitMap<Vec<AtomicU64>> {
    pub fn new_atomic(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: (0..n_of_words).map(|_| AtomicU64::new(0)).collect(),
            len,
            number_of_ones: AtomicUsize::new(0),
        }
    }
}

impl<B> BitMap<B> {
    /// # Safety
    /// TODO: this function is never used
    #[inline]
    pub unsafe fn from_raw_parts(data: B, len: usize, number_of_ones: usize) -> Self {
        Self {
            data,
            len,
            number_of_ones: AtomicUsize::new(number_of_ones),
        }
    }
}

impl<B> BitLength for BitMap<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.number_of_ones.load(Ordering::SeqCst)
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

impl<B: VSliceAtomic> VSliceAtomic for BitMap<B> {
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_atomic_unchecked(word_index, order);
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
        // we are safe to use this as we have mut access so we are the only ones
        // and there are no concurrency

        // update the count of ones if we added a one
        self.number_of_ones
            .fetch_add((new_word > word) as usize, Ordering::Relaxed);
        // update the count of ones if we removed a one
        self.number_of_ones
            .fetch_sub((new_word < word) as usize, Ordering::Relaxed);
    }
}

impl<B: VSliceMutAtomic> VSliceMutAtomic for BitMap<B> {
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        // get the word index, and the bit index in the word
        let word_index = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();
        let (word, new_word) = loop {
            // get the old word
            let word = self
                .data
                .get_atomic_unchecked(word_index, Ordering::Acquire);
            // clean the old bit in the word
            let mut new_word = word & !(1 << bit_index);
            // and write the new one
            new_word |= value << bit_index;
            // write it back
            // idk if the ordering is reasonable here, the only reasonable is
            // Release
            if self
                .data
                .compare_exchange_unchecked(word_index, word, new_word, order, order)
                .is_ok()
            {
                break (word, new_word);
            }
        };
        // update the count of ones if we added a one
        // update the count of ones if we removed a one
        let inc = (new_word > word) as isize - (new_word < word) as isize;
        // use the isize as usize (which JUST re-interprets the bits)
        // to do a single fetch_add and ensure consistency
        self.number_of_ones
            .fetch_add(inc as usize, Ordering::Relaxed);
    }

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
        let res = self
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

impl<B: VSlice> Select for BitMap<B> {
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        self.select_unchecked_hinted(rank, 0, 0)
    }
}

impl<B: VSlice> SelectHinted for BitMap<B> {
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        let mut word_index = pos / self.data.bit_width();
        let bit_index = pos % self.data.bit_width();
        let mut residual = rank - rank_at_pos;
        // TODO!: M2L or L2M?
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

        word_index * self.data.bit_width() + select_in_word(word, residual)
    }
}

impl<B: VSlice> SelectZero for BitMap<B> {
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.select_zero_unchecked_hinted(rank, 0, 0)
    }
}

impl<B: VSlice> SelectZeroHinted for BitMap<B> {
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        let mut word_index = pos / self.data.bit_width();
        let bit_index = pos % self.data.bit_width();
        let mut residual = rank - rank_at_pos;
        // TODO!: M2L or L2M?
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

        word_index * self.data.bit_width() + select_in_word(word, residual)
    }
}

impl<B: AsRef<[u64]>, D: AsRef<[u64]>> ConvertTo<BitMap<D>> for BitMap<B>
where
    B: ConvertTo<D>,
{
    fn convert_to(self) -> Result<BitMap<D>> {
        Ok(BitMap {
            len: self.len,
            number_of_ones: self.number_of_ones,
            data: self.data.convert_to()?,
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

impl<B: AsRef<[u64]> + Serialize> Serialize for BitMap<B> {
    fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
        let mut bytes = 0;
        bytes += self.len.serialize(backend)?;
        bytes += self
            .number_of_ones
            .load(Ordering::SeqCst)
            .serialize(backend)?;
        bytes += self.data.serialize(backend)?;
        Ok(bytes)
    }
}

impl<'a, B: AsRef<[u64]> + Deserialize<'a>> Deserialize<'a> for BitMap<B> {
    fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
        let (len, backend) = usize::deserialize(backend)?;
        let (number_of_ones, backend) = usize::deserialize(backend)?;
        let (data, backend) = B::deserialize(backend)?;

        Ok((
            Self {
                len,
                number_of_ones: AtomicUsize::new(number_of_ones),
                data,
            },
            backend,
        ))
    }
}

impl<B: MemSize> MemSize for BitMap<B> {
    fn mem_size(&self) -> usize {
        self.len.mem_size()
            + self.number_of_ones.load(Ordering::Relaxed).mem_size()
            + self.data.mem_size()
    }
    fn mem_used(&self) -> usize {
        self.len.mem_used()
            + self.number_of_ones.load(Ordering::Relaxed).mem_used()
            + self.data.mem_used()
    }
}

impl From<BitMap<Vec<u64>>> for BitMap<Vec<AtomicU64>> {
    #[inline]
    fn from(bm: BitMap<Vec<u64>>) -> Self {
        let data = unsafe { std::mem::transmute::<Vec<u64>, Vec<AtomicU64>>(bm.data) };
        BitMap {
            data: data,
            len: bm.len,
            number_of_ones: bm.number_of_ones,
        }
    }
}

impl From<BitMap<Vec<AtomicU64>>> for BitMap<Vec<u64>> {
    #[inline]
    fn from(bm: BitMap<Vec<AtomicU64>>) -> Self {
        let data = unsafe { std::mem::transmute::<Vec<AtomicU64>, Vec<u64>>(bm.data) };
        BitMap {
            data: data,
            len: bm.len,
            number_of_ones: bm.number_of_ones,
        }
    }
}

impl<'a> From<BitMap<&'a [AtomicU64]>> for BitMap<&'a [u64]> {
    #[inline]
    fn from(bm: BitMap<&'a [AtomicU64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a [AtomicU64], &'a [u64]>(bm.data) };
        BitMap {
            data: data,
            len: bm.len,
            number_of_ones: bm.number_of_ones,
        }
    }
}

impl<'a> From<BitMap<&'a [u64]>> for BitMap<&'a [AtomicU64]> {
    #[inline]
    fn from(bm: BitMap<&'a [u64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a [u64], &'a [AtomicU64]>(bm.data) };
        BitMap {
            data: data,
            len: bm.len,
            number_of_ones: bm.number_of_ones,
        }
    }
}

impl<'a> From<BitMap<&'a mut [AtomicU64]>> for BitMap<&'a mut [u64]> {
    #[inline]
    fn from(bm: BitMap<&'a mut [AtomicU64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a mut [AtomicU64], &'a mut [u64]>(bm.data) };
        BitMap {
            data: data,
            len: bm.len,
            number_of_ones: bm.number_of_ones,
        }
    }
}

impl<'a> From<BitMap<&'a mut [u64]>> for BitMap<&'a mut [AtomicU64]> {
    #[inline]
    fn from(bm: BitMap<&'a mut [u64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a mut [u64], &'a mut [AtomicU64]>(bm.data) };
        BitMap {
            data: data,
            len: bm.len,
            number_of_ones: bm.number_of_ones,
        }
    }
}
