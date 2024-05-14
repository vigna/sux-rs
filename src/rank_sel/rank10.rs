use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Counts64 {
    pub(super) l0: Vec<u64>,
    pub(super) l1_l2: Vec<u64>,
}

impl Counts64 {
    pub fn new(num_upper_counts: usize, num_counts: usize) -> Self {
        Self {
            l0: vec![0; num_upper_counts],
            l1_l2: vec![0; num_counts],
        }
    }
    pub fn set_upper(&mut self, upper: u64, idx: usize) {
        self.l0[idx] = upper;
    }
    pub fn set_lower(&mut self, count: u64, lower_block_idx: usize) {
        debug_assert!(count < (1 << 32));
        self.l1_l2[lower_block_idx] = count << 32;
    }
    pub fn set_basic(&mut self, count: u64, lower_block_idx: usize, basic_block_idx: usize) {
        debug_assert!(basic_block_idx < 3);
        debug_assert!(count <= (1 << 10));

        self.l1_l2[lower_block_idx] |= count << (10 * basic_block_idx);
    }
    #[inline(always)]
    pub unsafe fn upper(&self, upper_block_idx: usize) -> u64 {
        *<Vec<u64> as AsRef<[u64]>>::as_ref(&self.l0).get_unchecked(upper_block_idx)
    }
    #[inline(always)]
    pub unsafe fn lower(&self, lower_block_idx: usize) -> u64 {
        debug_assert!(lower_block_idx < self.l1_l2.len());
        *<Vec<u64> as AsRef<[u64]>>::as_ref(&self.l1_l2).get_unchecked(lower_block_idx) >> 32
    }
    #[inline(always)]
    pub unsafe fn basic(&self, lower_block_idx: usize, basic_block_idx: usize) -> u64 {
        debug_assert!(lower_block_idx < self.l1_l2.len());
        debug_assert!(basic_block_idx < 4);

        let counts_ref = <Vec<u64> as AsRef<[u64]>>::as_ref(&self.l1_l2);
        let basic = *counts_ref.get_unchecked(lower_block_idx) & 0x00000000FFFFFFFF;

        let mut offset = basic_block_idx.wrapping_sub(1) as u64;
        offset = offset.wrapping_add(offset >> 61 & 0x4);

        (basic >> (10 * offset)) & ((1 << 10) - 1)
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank10<
    const LOG2_LOWER_BLOCK_SIZE: usize,
    const HINT_BIT_SIZE: usize = 64,
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
> {
    pub(super) bits: B,
    pub(super) counts: Counts64,
    pub(super) num_ones: u64,
}

impl<
        const LOG2_LOWER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > Rank10<LOG2_LOWER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    pub(super) const LOWER_BLOCK_SIZE: usize = 1 << LOG2_LOWER_BLOCK_SIZE;
    pub(super) const UPPER_BLOCK_SIZE: usize = 1 << 32;
    pub(super) const WORD_UPPER_BLOCK_SIZE: usize = Self::UPPER_BLOCK_SIZE / 64;
    pub(super) const WORD_LOWER_BLOCK_SIZE: usize = Self::LOWER_BLOCK_SIZE / 64;
    pub(super) const BASIC_BLOCK_SIZE: usize = Self::LOWER_BLOCK_SIZE / 4;
    pub(super) const WORD_BASIC_BLOCK_SIZE: usize = Self::BASIC_BLOCK_SIZE / 64;
}

impl<const LOG2_LOWER_BLOCK_SIZE: usize, const HINT_BIT_SIZE: usize>
    Rank10<LOG2_LOWER_BLOCK_SIZE, HINT_BIT_SIZE, BitVec>
{
    pub fn new(bits: BitVec) -> Self {
        assert!(
            LOG2_LOWER_BLOCK_SIZE >= 8 && LOG2_LOWER_BLOCK_SIZE <= 10,
            "LOG2_LOWER_BLOCK_SIZE must be between 8 and 10 (inclusive)"
        );
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_upper_counts = (num_bits + Self::UPPER_BLOCK_SIZE - 1) / Self::UPPER_BLOCK_SIZE;
        let num_counts = (num_bits + Self::LOWER_BLOCK_SIZE - 1) / Self::LOWER_BLOCK_SIZE;

        let mut counts = Counts64::new(num_upper_counts, num_counts);

        let mut num_ones = 0u64;
        let mut upper_block = 0u64;

        for (i, lower_block_idx) in (0..num_words).step_by(Self::WORD_LOWER_BLOCK_SIZE).zip(0..) {
            if i % Self::WORD_UPPER_BLOCK_SIZE == 0 {
                upper_block = num_ones;
                counts.set_upper(upper_block, i / Self::WORD_UPPER_BLOCK_SIZE);
            }
            let lower = num_ones - upper_block;
            counts.set_lower(lower, lower_block_idx);
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..Self::WORD_LOWER_BLOCK_SIZE {
                if j % Self::WORD_BASIC_BLOCK_SIZE == 0 {
                    let basic = num_ones - upper_block - lower;
                    counts.set_basic(basic, lower_block_idx, j / Self::WORD_BASIC_BLOCK_SIZE - 1);
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
        const LOG2_LOWER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > BitLength for Rank10<LOG2_LOWER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<
        const LOG2_LOWER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > BitCount for Rank10<LOG2_LOWER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    fn count(&self) -> usize {
        self.rank(self.bits.len())
    }
}

impl<
        const LOG2_LOWER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
    > Rank for Rank10<LOG2_LOWER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let upper_block_idx = word / Self::WORD_UPPER_BLOCK_SIZE;
        let lower_block_idx = word / Self::WORD_LOWER_BLOCK_SIZE;
        let basic_block_idx = (word % Self::WORD_LOWER_BLOCK_SIZE) / Self::WORD_BASIC_BLOCK_SIZE;

        let upper = self.counts.upper(upper_block_idx);
        let lower = self.counts.lower(lower_block_idx);
        let basic = self.counts.basic(lower_block_idx, basic_block_idx);

        let hint_rank = upper + lower + basic;

        let hint_pos = word - (word % Self::WORD_BASIC_BLOCK_SIZE);

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

    const TEST_LOG2_LOWER_BLOCK_SIZE: usize = 10;

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
            let rank10: Rank10<TEST_LOG2_LOWER_BLOCK_SIZE> = Rank10::new(bits.clone());

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
        let rank10: Rank10<TEST_LOG2_LOWER_BLOCK_SIZE> = Rank10::new(bits.clone());

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
        let rank10: Rank10<TEST_LOG2_LOWER_BLOCK_SIZE> = Rank10::new(bits.clone());

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

        let rank10: Rank10<TEST_LOG2_LOWER_BLOCK_SIZE> = Rank10::new(bits);

        assert_eq!(rank10.rank(rank10.len()), rank10.bits.count());
    }
}
