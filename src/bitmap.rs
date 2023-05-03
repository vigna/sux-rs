use anyhow::Result;
use crate::traits::*;
use crate::utils::{select_in_word, select_zero_in_word};

pub struct BitMap<B> {
    data: B,
    len: usize,
    number_of_ones: usize,
}

impl BitMap<Vec<u64>> {
    pub fn new(len: usize) -> Self {
        let n_of_words = (len + 63) / 64;
        Self {
            data: vec![0; n_of_words],
            len,
            number_of_ones: 0,
        }
    }
}

impl<B: VSlice> BitMap<B> {
    pub unsafe fn from_raw_parts(data: B, len: usize, number_of_ones: usize) -> Self {
        Self {
            data,
            len,
            number_of_ones,
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
        self.number_of_ones
    }
}

impl<B: VSlice> VSlice for BitMap<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        1
    }
    
    #[inline(always)]
    fn len(&self) -> usize {
        self.number_of_ones
    }

    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        let word_idx = index / self.data.bit_width();
        let word = self.data.get_unchecked(word_idx);
        (word >> (index % self.data.bit_width())) & 1        
    }
}

impl<B: VSliceMut> VSliceMut for BitMap<B> {
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        let word_idx = index / self.data.bit_width();
        let mut word = self.data.get_unchecked(word_idx);
        word |= value << (index % self.data.bit_width());
        self.data.set_unchecked(word_idx, word);
        // TODO!: increase number of zeros?
    }
}

impl<B: VSlice> SelectHinted for BitMap<B> {
	unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, mut rank_at_pos: usize) -> usize {
        let mut word_idx = pos / self.data.bit_width();
        let bit_idx  = pos % self.data.bit_width();
        // TODO!: M2L or L2M?
        let mut word = self.data.get_unchecked(word_idx) >> bit_idx;
        loop {
            let tmp_rank = rank_at_pos + (word.count_ones() as usize);
            if tmp_rank > rank {
                break
            }
            word_idx += 1;
            word = self.data.get_unchecked(word_idx);
            rank_at_pos = tmp_rank;
        }

        word_idx * self.data.bit_width() + bit_idx 
            + select_in_word(word, rank - rank_at_pos)
    }
}

impl<B: VSlice> SelectZeroHinted for BitMap<B> {
	unsafe fn select_zero_unchecked_hinted(&self, rank: usize, pos: usize, mut rank_at_pos: usize) -> usize {
        let mut word_idx = pos / self.data.bit_width();
        let bit_idx  = pos % self.data.bit_width();
        // TODO!: M2L or L2M?
        let mut word = self.data.get_unchecked(word_idx) >> bit_idx;
        loop {
            let tmp_rank = rank_at_pos + (word.count_zeros() as usize);
            if tmp_rank > rank {
                break
            }
            word_idx += 1;
            word = self.data.get_unchecked(word_idx);
            rank_at_pos = tmp_rank;
        }

        word_idx * self.data.bit_width() + bit_idx 
            + select_zero_in_word(word, rank - rank_at_pos)
    }
}

impl<B, D> ConvertTo<BitMap<D>> for BitMap<B> 
where
    B: ConvertTo<D>
{
    fn convert_to(self) -> Result<BitMap<D>> {
        Ok(BitMap {
            len: self.len,
            number_of_ones: self.number_of_ones,
            data: self.data.convert_to()?,
        })
    }
}