use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Block196 {
    pub upper: u64,
    pub lower: u128,
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank16<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    pub(super) counts: Vec<Block196>,
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > Rank16<B, HINT_BIT_SIZE>
{
    const LOG2_UPPER_BLOCK_SIZE: usize = 16;
    const UPPER_BLOCK_SIZE: usize = 1 << Self::LOG2_UPPER_BLOCK_SIZE;
    const WORD_UPPER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / 64;
    const NUM_LOWER_BLOCKS: usize = 128 / Self::LOG2_UPPER_BLOCK_SIZE;
    const LOWER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / Self::NUM_LOWER_BLOCKS;
    const WORD_LOWER_BLOCK_SIZE: usize = Self::LOWER_BLOCK_SIZE / 64;
    const MASK_LOG2_UPPER_BLOCK_SIZE: u128 = (1 << Self::LOG2_UPPER_BLOCK_SIZE) - 1;
}

impl<const HINT_BIT_SIZE: usize> Rank16<BitVec, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_counts = (num_bits + Self::UPPER_BLOCK_SIZE - 1) / Self::UPPER_BLOCK_SIZE;

        let mut counts = Vec::<Block196>::with_capacity(num_counts + 1);
        let mut num_ones = 0u64;

        for i in (0..num_words).step_by(Self::WORD_UPPER_BLOCK_SIZE) {
            let mut block = Block196 {
                upper: num_ones,
                lower: 0,
            };
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..Self::WORD_UPPER_BLOCK_SIZE {
                if j % Self::WORD_LOWER_BLOCK_SIZE == 0 {
                    block.lower |= ((num_ones - block.upper) as u128)
                        << (Self::LOG2_UPPER_BLOCK_SIZE * (j / Self::WORD_LOWER_BLOCK_SIZE));
                }
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as u64;
                }
            }
            counts.push(block);
        }
        counts.push(Block196 {
            upper: num_ones,
            lower: 0,
        });
        assert_eq!(counts.len(), num_counts + 1);

        Self { bits, counts }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > BitLength for Rank16<B, HINT_BIT_SIZE>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > BitCount for Rank16<B, HINT_BIT_SIZE>
{
    fn count(&self) -> usize {
        self.rank(self.bits.len())
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        const HINT_BIT_SIZE: usize,
    > Rank for Rank16<B, HINT_BIT_SIZE>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let block = word / Self::WORD_UPPER_BLOCK_SIZE;

        let counts_ref = <Vec<Block196> as AsRef<[Block196]>>::as_ref(&self.counts);
        let upper = counts_ref.get_unchecked(block).upper;
        let lower = counts_ref.get_unchecked(block).lower;

        let offset = Self::LOG2_UPPER_BLOCK_SIZE
            * ((word % Self::WORD_UPPER_BLOCK_SIZE) / Self::WORD_LOWER_BLOCK_SIZE);

        let hint_rank = upper + (((lower >> offset) & Self::MASK_LOG2_UPPER_BLOCK_SIZE) as u64);

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
            self.counts[self.counts.len() - 1].upper as usize
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }
}

#[cfg(test)]
mod test_rank16 {
    use crate::prelude::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_rank16() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank16: Rank16 = Rank16::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(rank16.rank(i), ranks[i],);
            }
            assert_eq!(rank16.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 16], (1 << 16) * 64) };

        let rank16: Rank16 = Rank16::new(bits);

        assert_eq!(rank16.rank(rank16.len()), rank16.bits.count());
    }
}
