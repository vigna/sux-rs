use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

const UPPER_BLOCK_SIZE: usize = 896;
const WORD_UPPER_BLOCK_SIZE: usize = UPPER_BLOCK_SIZE / 64;
const LOWER_BLOCK_SIZE: usize = UPPER_BLOCK_SIZE / 7;
const WORD_LOWER_BLOCK_SIZE: usize = LOWER_BLOCK_SIZE / 64;
// used to compute word % WORD_UPPER_BLOCK_SIZE
// const REC_WORD_UPPER_BLOCK_SIZE: u64 = u64::MAX / (WORD_UPPER_BLOCK_SIZE as u64) + 1;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Counts {
    major: Vec<u64>,
    minor: Vec<u32>,
}

impl Counts {
    pub fn new(num_major_counts: usize, num_counts: usize) -> Self {
        Self {
            major: vec![0; num_major_counts],
            minor: vec![0; num_counts * 3],
        }
    }
    pub fn set_major(&mut self, major: u64, idx: usize) {
        self.major[idx] = major;
    }
    pub fn set_upper(&mut self, upper: u32, block: usize) {
        debug_assert!(block < self.minor.len() / 3);
        self.minor[block * 3] = upper;
    }
    pub fn set_lower(&mut self, lower: u32, block: usize, idx: usize) {
        debug_assert!(idx < 6);
        debug_assert!(lower < (1 << 10)); // block is 896 bits
        if idx < 3 {
            self.minor[block * 3 + 1] |= lower << (10 * idx);
        } else {
            self.minor[block * 3 + 2] |= lower << (10 * (idx - 3));
        }
    }
    #[inline(always)]
    pub unsafe fn major(&self, idx: usize) -> u64 {
        *<Vec<u64> as AsRef<[u64]>>::as_ref(&self.major).get_unchecked(idx)
    }
    #[inline(always)]
    pub unsafe fn upper(&self, block: usize) -> u32 {
        debug_assert!(block < self.minor.len() / 3);
        *<Vec<u32> as AsRef<[u32]>>::as_ref(&self.minor).get_unchecked(block * 3)
    }
    #[inline(always)]
    pub unsafe fn lower(&self, block: usize, idx: usize) -> u32 {
        debug_assert!(block < self.minor.len() / 3);
        debug_assert!(idx < 7);
        let mut offset = idx.wrapping_sub(1);
        offset = offset.wrapping_add(offset >> 29 & 0x4);
        let idx_div = idx >> 2;

        let counts_ref = <Vec<u32> as AsRef<[u32]>>::as_ref(&self.minor);

        (*counts_ref.get_unchecked(block * 3 + 1 + idx_div) >> (10 * (offset - 3 * idx_div)))
            & ((1 << 10) - 1)
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank10<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    pub(super) counts: Counts,
    pub(super) num_ones: u64,
}

impl<const HINT_BIT_SIZE: usize> Rank10<BitVec, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_major_counts = (num_bits + (1 << 32) - 1) / (1 << 32);
        let num_counts = (num_bits + UPPER_BLOCK_SIZE - 1) / UPPER_BLOCK_SIZE;

        let mut counts = Counts::new(num_major_counts, num_counts);

        let mut num_ones = 0u64;
        let mut major_block = 0u64;

        for (i, block) in (0..num_words).step_by(WORD_UPPER_BLOCK_SIZE).zip(0..) {
            if i % (1 << 26) == 0 {
                major_block = num_ones;
                counts.set_major(major_block, i / (1 << 26));
            }
            let upper = (num_ones - major_block) as u32;
            counts.set_upper(upper, block);
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..WORD_UPPER_BLOCK_SIZE {
                if j % WORD_LOWER_BLOCK_SIZE == 0 {
                    counts.set_lower(
                        (num_ones - major_block) as u32 - upper,
                        block,
                        j / WORD_LOWER_BLOCK_SIZE - 1,
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
        let block = word / WORD_UPPER_BLOCK_SIZE;
        let major_block = word >> 26;

        // computes word % WORD_UPPER_BLOCK_SIZE
        // let fastmod = ((((REC_WORD_UPPER_BLOCK_SIZE).wrapping_mul(word as u64) as u128)
        //     .wrapping_mul(WORD_UPPER_BLOCK_SIZE as u128))
        //     >> 64) as usize;
        let upper_block_remainder = word % WORD_UPPER_BLOCK_SIZE;

        let idx = upper_block_remainder / WORD_LOWER_BLOCK_SIZE;

        let major = self.counts.major(major_block);
        let upper = self.counts.upper(block);
        let lower = self.counts.lower(block, idx);

        let hint_rank = major + (upper + lower) as u64;

        let hint_pos = word - (upper_block_remainder % WORD_LOWER_BLOCK_SIZE);

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
            .chain((100_000..1_000_000).step_by(100_000))
            .chain([1 << 20, 1 << 22]);
        // let lens = [1 << 33];
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
