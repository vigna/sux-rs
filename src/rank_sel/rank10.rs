use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Counts64 {
    pub(super) major: Vec<u64>,
    pub(super) minor: Vec<u64>,
}

impl Counts64 {
    pub fn new(num_major_counts: usize, num_counts: usize) -> Self {
        Self {
            major: vec![0; num_major_counts],
            minor: vec![0; num_counts],
        }
    }
    pub fn set_major(&mut self, major: u64, idx: usize) {
        self.major[idx] = major;
    }
    pub fn set_upper(&mut self, upper: u64, block: usize) {
        debug_assert!(upper < (1 << 32));
        self.minor[block] = upper << 32;
    }
    pub fn set_lower(&mut self, lower: u64, block: usize, idx: usize) {
        debug_assert!(idx < 3);
        debug_assert!(lower <= (1 << 10));

        self.minor[block] |= lower << (10 * idx);
    }
    #[inline(always)]
    pub unsafe fn major(&self, idx: usize) -> u64 {
        *<Vec<u64> as AsRef<[u64]>>::as_ref(&self.major).get_unchecked(idx)
    }
    #[inline(always)]
    pub unsafe fn upper(&self, block: usize) -> u64 {
        debug_assert!(block < self.minor.len());
        *<Vec<u64> as AsRef<[u64]>>::as_ref(&self.minor).get_unchecked(block) >> 32
    }
    #[inline(always)]
    pub unsafe fn lower(&self, block: usize, idx: usize) -> u64 {
        debug_assert!(block < self.minor.len());
        debug_assert!(idx < 4);

        let counts_ref = <Vec<u64> as AsRef<[u64]>>::as_ref(&self.minor);
        let lower = *counts_ref.get_unchecked(block) & 0x00000000FFFFFFFF;

        let mut offset = idx.wrapping_sub(1) as u64;
        offset = offset.wrapping_add(offset >> 61 & 0x4);

        (lower >> (10 * offset)) & ((1 << 10) - 1)
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank10<
    const UPPER_BLOCK_SIZE: usize,
    const HINT_BIT_SIZE: usize = 64,
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
> {
    pub(super) bits: B,
    pub(super) counts: Counts64,
    pub(super) num_ones: u64,
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > Rank10<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    pub(super) const MAJOR_BLOCK_SIZE: usize = 1 << 32;
    pub(super) const WORD_MAJOR_BLOCK_SIZE: usize = Self::MAJOR_BLOCK_SIZE / 64;
    pub(super) const WORD_UPPER_BLOCK_SIZE: usize = UPPER_BLOCK_SIZE / 64;
    pub(super) const LOWER_BLOCK_SIZE: usize = UPPER_BLOCK_SIZE / 4;
    pub(super) const WORD_LOWER_BLOCK_SIZE: usize = Self::LOWER_BLOCK_SIZE / 64;
}

impl<const UPPER_BLOCK_SIZE: usize, const HINT_BIT_SIZE: usize>
    Rank10<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, BitVec>
{
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_major_counts = (num_bits + Self::MAJOR_BLOCK_SIZE - 1) / Self::MAJOR_BLOCK_SIZE;
        let num_counts = (num_bits + UPPER_BLOCK_SIZE - 1) / UPPER_BLOCK_SIZE;

        let mut counts = Counts64::new(num_major_counts, num_counts);

        let mut num_ones = 0u64;
        let mut major_block = 0u64;

        for (i, block) in (0..num_words).step_by(Self::WORD_UPPER_BLOCK_SIZE).zip(0..) {
            if i % Self::WORD_MAJOR_BLOCK_SIZE == 0 {
                major_block = num_ones;
                counts.set_major(major_block, i / Self::WORD_MAJOR_BLOCK_SIZE);
            }
            let upper = num_ones - major_block;
            counts.set_upper(upper, block);
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..Self::WORD_UPPER_BLOCK_SIZE {
                if j % Self::WORD_LOWER_BLOCK_SIZE == 0 {
                    let lower = num_ones - major_block - upper;
                    counts.set_lower(lower, block, j / Self::WORD_LOWER_BLOCK_SIZE - 1);
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
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > BitLength for Rank10<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > BitCount for Rank10<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    fn count(&self) -> usize {
        self.rank(self.bits.len())
    }
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > Rank for Rank10<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let block = word / Self::WORD_UPPER_BLOCK_SIZE;
        let major_block = word / Self::WORD_MAJOR_BLOCK_SIZE;

        let upper_block_remainder = word % Self::WORD_UPPER_BLOCK_SIZE;

        let idx = upper_block_remainder / Self::WORD_LOWER_BLOCK_SIZE;

        let major = self.counts.major(major_block);
        let upper = self.counts.upper(block);
        let lower = self.counts.lower(block, idx);

        let hint_rank = major + (upper + lower) as u64;

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
mod test_rank10 {
    use crate::prelude::*;
    use rand::{Rng, SeedableRng};

    const TEST_UPPER_BLOCK_SIZE: usize = 256;

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
            let rank10: Rank10<TEST_UPPER_BLOCK_SIZE> = Rank10::new(bits.clone());

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
    fn test_empty() {
        let len = (1 << 16) * 64;
        let bits = unsafe { BitVec::from_raw_parts(vec![0usize; len / 64], len) };
        let rank10: Rank10<TEST_UPPER_BLOCK_SIZE> = Rank10::new(bits.clone());

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

    #[test]
    fn test_full() {
        let len = (1 << 16) * 64;
        let bits = unsafe { BitVec::from_raw_parts(vec![usize::MAX; len / 64], len) };
        let rank10: Rank10<TEST_UPPER_BLOCK_SIZE> = Rank10::new(bits.clone());

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

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 16], (1 << 16) * 64) };

        let rank10: Rank10<TEST_UPPER_BLOCK_SIZE> = Rank10::new(bits);

        assert_eq!(rank10.rank(rank10.len()), rank10.bits.count());
    }
}
