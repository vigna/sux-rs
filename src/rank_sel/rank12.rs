use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Count256 {
    d1: u128,
    d2: u128,
}

impl Count256 {
    pub fn new(d1: u128, d2: u128) -> Self {
        Self { d1, d2 }
    }
    pub fn set_upper(&mut self, upper: u64) {
        debug_assert!(upper < 1 << 56);
        self.d1 = (self.d1 & ((1u128 << 72) - 1)) | ((upper as u128) << 72);
    }
    pub fn set_lower(&mut self, lower: u128, idx: usize) {
        // there are 16 lower counts of 12 bits in a Count256, the first 6 are in d1, the rest are in d2
        debug_assert!(idx < 16);
        debug_assert!(lower < 1 << 12);
        if idx < 6 {
            self.d1 |= lower << (12 * idx);
        } else {
            self.d2 |= lower << (12 * (idx - 6));
        }
    }
    pub fn upper(&self) -> u64 {
        (self.d1 >> 72) as u64
    }
    pub fn lower(&self, idx: usize) -> u64 {
        debug_assert!(idx < 16);
        if idx < 6 {
            ((self.d1 >> (12 * idx)) & ((1 << 12) - 1)) as u64
        } else {
            ((self.d2 >> (12 * (idx - 6))) & ((1 << 12) - 1)) as u64
        }
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank12<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    pub(super) counts: Vec<Count256>,
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > Rank12<B, HINT_BIT_SIZE>
{
    const LOG2_UPPER_BLOCK_SIZE: usize = 12;
    const UPPER_BLOCK_SIZE: usize = 1 << Self::LOG2_UPPER_BLOCK_SIZE;
    const WORD_UPPER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / 64;
    const NUM_LOWER_BLOCKS: usize = 16;
    const LOWER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / Self::NUM_LOWER_BLOCKS;
    const WORD_LOWER_BLOCK_SIZE: usize = Self::LOWER_BLOCK_SIZE / 64;
}

impl<const HINT_BIT_SIZE: usize> Rank12<BitVec, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_counts = (num_bits + Self::UPPER_BLOCK_SIZE - 1) / Self::UPPER_BLOCK_SIZE;

        let mut counts = Vec::<Count256>::with_capacity(num_counts + 1);
        let mut num_ones = 0u64;

        for i in (0..num_words).step_by(Self::WORD_UPPER_BLOCK_SIZE) {
            let mut count = Count256::new(0, 0);
            count.set_upper(num_ones);
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..Self::WORD_UPPER_BLOCK_SIZE {
                if j % Self::WORD_LOWER_BLOCK_SIZE == 0 {
                    count.set_lower(
                        (num_ones - count.upper()) as u128,
                        j / Self::WORD_LOWER_BLOCK_SIZE,
                    );
                }
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as u64;
                }
            }
            counts.push(count);
        }
        let mut last = Count256::new(0, 0);
        last.set_upper(num_ones);
        counts.push(last);
        Self { bits, counts }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > BitLength for Rank12<B, HINT_BIT_SIZE>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > BitCount for Rank12<B, HINT_BIT_SIZE>
{
    fn count(&self) -> usize {
        self.rank(self.bits.len())
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > Rank for Rank12<B, HINT_BIT_SIZE>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let block = word / Self::WORD_UPPER_BLOCK_SIZE;

        let counts_ref = <Vec<Count256> as AsRef<[Count256]>>::as_ref(&self.counts);
        let upper = counts_ref.get_unchecked(block).upper();
        let lower = counts_ref
            .get_unchecked(block)
            .lower((word % Self::WORD_UPPER_BLOCK_SIZE) / Self::WORD_LOWER_BLOCK_SIZE);

        let hint_rank = upper + lower;

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
            self.counts[self.counts.len() - 1].upper() as usize
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }
}

#[cfg(test)]
mod test_rank12 {
    use crate::prelude::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_rank12() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank12: Rank12 = Rank12::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(rank12.rank(i), ranks[i],);
            }
            assert_eq!(rank12.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 16], (1 << 16) * 64) };

        let rank12: Rank12 = Rank12::new(bits);

        assert_eq!(rank12.rank(rank12.len()), rank12.bits.count());
    }
}
