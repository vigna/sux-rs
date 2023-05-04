use anyhow::Result;
use crate::traits::*;
use crate::utils::select_in_word;

pub struct BitMap<B: AsRef<[u64]>> {
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

impl<B: VSlice + AsRef<[u64]>> BitMap<B> {
    pub unsafe fn from_raw_parts(data: B, len: usize, number_of_ones: usize) -> Self {
        Self {
            data,
            len,
            number_of_ones,
        }
    }
}

impl<B: AsRef<[u64]>> BitLength for BitMap<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.number_of_ones
    }
}

impl<B: VSlice + AsRef<[u64]>> VSlice for BitMap<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        1
    }
    
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        let word_index = index / self.data.bit_width();
        let word = self.data.get_unchecked(word_index);
        (word >> (index % self.data.bit_width())) & 1        
    }
}

impl<B: VSliceMut + AsRef<[u64]>> VSliceMut for BitMap<B> {
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        // get the word index, and the bit index in the word
        let word_index  = index / self.data.bit_width();
        let bit_index = index % self.data.bit_width();
        // get the old word
        let word = self.data.get_unchecked(word_index);
        // clean the old bit in the word
        let mut new_word = word & !(1 << bit_index);
        // and write the new one
        new_word |= value << bit_index;
        // write it back
        self.data.set_unchecked(word_index, new_word);
        // update the count of ones if we added a one
        self.number_of_ones += (new_word > word) as usize;
        // update the count of ones if we removed a one
        self.number_of_ones -= (new_word < word) as usize;
    }
}

impl<B: VSlice + AsRef<[u64]>> Select for BitMap<B> {
	#[inline(always)]
	unsafe fn select_unchecked(&self, rank: usize) -> usize {
		self.select_unchecked_hinted(rank, 0, 0)
	}
}

impl<B: VSlice + AsRef<[u64]>> SelectHinted for BitMap<B> {
	unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        let mut word_index = pos / self.data.bit_width();
        let bit_index  = pos % self.data.bit_width();
        let mut residual = rank - rank_at_pos;
        // TODO!: M2L or L2M?
        let mut word = (self.data.get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                break
            }
            word_index += 1;
            word = self.data.get_unchecked(word_index);
            residual -= bit_count;
        }

        word_index * self.data.bit_width() 
            + select_in_word(word, residual)
    }
}

impl<B: VSlice + AsRef<[u64]>> SelectZero for BitMap<B> {
	#[inline(always)]
	unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
		self.select_zero_unchecked_hinted(rank, 0, 0)
	}
}

impl<B: VSlice + AsRef<[u64]>> SelectZeroHinted for BitMap<B> {
	unsafe fn select_zero_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        let mut word_index = pos / self.data.bit_width();
        let bit_index  = pos % self.data.bit_width();
        let mut residual = rank - rank_at_pos;
        // TODO!: M2L or L2M?
        let mut word = (!self.data.get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                break
            }
            word_index += 1;
            word = !self.data.get_unchecked(word_index);
            residual -= bit_count;
        }

        word_index * self.data.bit_width()
            + select_in_word(word, residual)
    }
}

impl<B:AsRef<[u64]>, D:AsRef<[u64]>> ConvertTo<BitMap<D>> for BitMap<B> 
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

impl<B: AsRef<[u64]>> AsRef<[u64]> for BitMap<B> {
    fn as_ref(&self) -> &[u64] {
        self.data.as_ref()
    }
}  