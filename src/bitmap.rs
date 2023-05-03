use crate::traits::*;
use crate::utils::{select_in_word, select_zero_in_word};

pub struct BitMap<B> {
    data: B,
    len: usize,
    number_of_ones: usize,
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