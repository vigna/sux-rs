use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Counts256 {
    inner: Vec<u128>,
}

impl Counts256 {
    pub fn new(num_counts: usize) -> Self {
        Self {
            inner: vec![0; num_counts * 2],
        }
    }
    pub fn set_upper(&mut self, upper: u64, block: usize) {
        debug_assert!(block < self.inner.len() / 2);
        debug_assert!(upper < 1 << 56);
        self.inner[block * 2] =
            (self.inner[block * 2] & ((1u128 << 72) - 1)) | ((upper as u128) << 72);
    }
    pub fn set_lower(&mut self, lower: u128, block: usize, idx: usize) {
        debug_assert!(block < self.inner.len() / 2);
        debug_assert!(idx < 16);
        debug_assert!(lower < 1 << 12);
        if idx < 6 {
            self.inner[block * 2] |= lower << (12 * idx);
        } else {
            self.inner[block * 2 + 1] |= lower << (12 * (idx - 6));
        }
    }
    #[inline(always)]
    pub unsafe fn upper(&self, block: usize) -> u64 {
        (*<Vec<u128> as AsRef<[u128]>>::as_ref(&self.inner).get_unchecked(block * 2) >> 72) as u64
    }
    #[inline(always)]
    pub unsafe fn lower(&self, block: usize, idx: usize) -> u64 {
        debug_assert!(block < self.inner.len() / 2);
        debug_assert!(idx < 16);
        let counts_ref = <Vec<u128> as AsRef<[u128]>>::as_ref(&self.inner);
        let idx_div = (idx >= 6) as usize;
        ((*counts_ref.get_unchecked(block * 2 + idx_div) >> (12 * (idx - 6 * idx_div)))
            & ((1 << 12) - 1)) as u64
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank12<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    pub(super) counts: Counts256,
    pub(super) num_ones: u64,
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

        let mut counts = Counts256::new(num_counts);
        let mut num_ones = 0u64;

        for (i, block) in (0..num_words).step_by(Self::WORD_UPPER_BLOCK_SIZE).zip(0..) {
            let upper = num_ones;
            counts.set_upper(upper, block);
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..Self::WORD_UPPER_BLOCK_SIZE {
                if j % Self::WORD_LOWER_BLOCK_SIZE == 0 {
                    counts.set_lower(
                        (num_ones - upper) as u128,
                        block,
                        j / Self::WORD_LOWER_BLOCK_SIZE,
                    );
                }
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as u64;
                }
            }
        }
        Self {
            bits,
            counts,
            num_ones,
        }
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

        let upper = self.counts.upper(block);
        let lower = self.counts.lower(
            block,
            (word % Self::WORD_UPPER_BLOCK_SIZE) / Self::WORD_LOWER_BLOCK_SIZE,
        );

        let hint_rank = upper + lower;

        let hint_pos = word - (word % Self::WORD_LOWER_BLOCK_SIZE);

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
mod test_rank12 {
    use crate::prelude::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_rank12() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000))
            .chain([1 << 20, 1 << 22]);
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
    fn test_empty() {
        let len = (1 << 16) * 64;
        let bits = unsafe { BitVec::from_raw_parts(vec![0usize; len / 64], len) };
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

    #[test]
    fn test_full() {
        let len = (1 << 16) * 64;
        let bits = unsafe { BitVec::from_raw_parts(vec![usize::MAX; len / 64], len) };
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

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 16], (1 << 16) * 64) };

        let rank12: Rank12 = Rank12::new(bits);

        assert_eq!(rank12.rank(rank12.len()), rank12.bits.count());
    }
}
