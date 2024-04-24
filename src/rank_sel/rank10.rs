use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Count96 {
    d1: u32,
    d2: u64,
}

impl Count96 {
    pub fn new(d1: u32, d2: u64) -> Self {
        Self { d1, d2 }
    }
    pub fn set_upper(&mut self, upper: u32) {
        self.d1 = upper;
    }
    pub fn set_lower(&mut self, lower: u32, idx: usize) {
        debug_assert!(idx < 7);
        debug_assert!(lower < 1 << 10); // block is 896 bits
        self.d2 |= (lower as u64) << (10 * idx);
    }
    pub fn upper(&self) -> u32 {
        self.d1
    }
    pub fn lower(&self, idx: usize) -> u64 {
        debug_assert!(idx < 7);
        let mut offset = idx.wrapping_sub(1);

        offset = offset.wrapping_add(offset >> 60 & 0x7);

        (self.d2 >> (10 * offset)) & ((1 << 10) - 1)
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank10<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    //pub(super) major_counts: Vec<u64>,
    pub(super) counts: Vec<Count96>,
    pub(super) num_ones: u64,
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > Rank10<B, HINT_BIT_SIZE>
{
    const UPPER_BLOCK_SIZE: usize = 896;
    const WORD_UPPER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / 64;
    const NUM_LOWER_BLOCKS: usize = 7;
    const LOWER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / Self::NUM_LOWER_BLOCKS;
    const WORD_LOWER_BLOCK_SIZE: usize = Self::LOWER_BLOCK_SIZE / 64;
}

impl<const HINT_BIT_SIZE: usize> Rank10<BitVec, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        //let num_major_counts = (num_bits + (1 << 32) - 1) / (1 << 32);
        let num_counts = (num_bits + 896 - 1) / 896;

        //let mut major_counts = Vec::<u64>::with_capacity(num_major_counts);
        let mut counts = Vec::<Count96>::with_capacity(num_counts);

        let mut num_ones = 0u32;

        for i in (0..num_words).step_by(Self::WORD_UPPER_BLOCK_SIZE) {
            let mut count = Count96::new(0, 0);
            count.set_upper(num_ones);
            num_ones += bits.as_ref()[i].count_ones();
            for j in 1..Self::WORD_UPPER_BLOCK_SIZE {
                if j % Self::WORD_LOWER_BLOCK_SIZE == 0 {
                    count.set_lower(
                        num_ones - count.upper(),
                        j / Self::WORD_LOWER_BLOCK_SIZE - 1,
                    );
                }
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones();
                }
            }
            counts.push(count);
        }

        Self {
            bits,
            //major_counts,
            counts,
            num_ones: num_ones as u64,
        }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > BitLength for Rank10<B, HINT_BIT_SIZE>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > BitCount for Rank10<B, HINT_BIT_SIZE>
{
    fn count(&self) -> usize {
        self.rank(self.bits.len())
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > Rank for Rank10<B, HINT_BIT_SIZE>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let block = word / Self::WORD_UPPER_BLOCK_SIZE;

        let counts_ref = <Vec<Count96> as AsRef<[Count96]>>::as_ref(&self.counts);
        let upper = counts_ref.get_unchecked(block).upper();
        let lower = counts_ref
            .get_unchecked(block)
            .lower((word % Self::WORD_UPPER_BLOCK_SIZE) / Self::WORD_LOWER_BLOCK_SIZE);

        let hint_rank = upper + lower as u32;

        let hint_pos = word - ((word % Self::WORD_UPPER_BLOCK_SIZE) % Self::WORD_LOWER_BLOCK_SIZE);

        RankHinted::<HINT_BIT_SIZE>::rank_hinted_unchecked(
            &self.bits,
            pos,
            hint_pos,
            hint_rank as usize,
        )
    }

    fn rank(&self, pos: usize) -> usize {
        if pos >= self.bits.len() {
            self.num_ones as usize
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }
}

#[cfg(test)]
mod test_rank10 {
    use crate::prelude::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_rank10() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank10: Rank10 = Rank10::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(rank10.rank(i), ranks[i],);
            }
            assert_eq!(rank10.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 16], (1 << 16) * 64) };

        let rank10: Rank10 = Rank10::new(bits);

        assert_eq!(rank10.rank(rank10.len()), rank10.bits.count());
    }
}
