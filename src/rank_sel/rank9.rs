use crate::prelude::{BitCount, BitLength, BitVec, Rank, RankHinted};

pub struct Rank9<
    B: RankHinted<HINT_BIT_SIZE> + BitLength + BitCount + AsRef<[usize]> = BitVec,
    const HINT_BIT_SIZE: usize = 64,
> {
    bits: B,
    counts: Vec<u64>,
}

impl<const HINT_BIT_SIZE: usize> Rank9<BitVec, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec, num_bits: u64) -> Self {
        let num_words = (num_bits + 63) / 64;
        let num_counts = ((num_bits + 64 * 8 - 1) / (64 * 8)) * 2;

        let mut counts = Vec::new();
        counts.resize((num_counts + 2) as usize, 0);

        let mut num_ones = 0u64;

        for (i, pos) in (0..num_words as usize).step_by(8).zip((0..).step_by(2)) {
            counts[pos] = num_ones;
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..8 {
                counts[pos + 1] |= (num_ones - counts[pos]) << 9 * (j - 1);
                if i + j < num_words as usize {
                    num_ones += bits.as_ref()[i + j].count_ones() as u64;
                }
            }
        }

        counts[num_counts as usize] = num_ones;

        Self { bits, counts }
    }
}

impl<const HINT_BIT_SIZE: usize> BitLength for Rank9<BitVec, HINT_BIT_SIZE> {
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<const HINT_BIT_SIZE: usize> Rank for Rank9<BitVec, HINT_BIT_SIZE> {
    fn rank(&self, index: usize) -> usize {
        unsafe { self.rank_unchecked(index.min(self.bits.len())) }
    }

    unsafe fn rank_unchecked(&self, index: usize) -> usize {
        let word = index / 64;
        let block = (word / 4 & !1) as usize;
        let offset = (word % 8).wrapping_sub(1);

        let hint_rank = self.counts[block]
            + (self.counts[block + 1] >> (offset.wrapping_add(offset >> 60 & 0x8)) * 9 & 0x1FF);

        <BitVec as RankHinted<HINT_BIT_SIZE>>::rank_hinted_unchecked(
            &self.bits,
            index,
            word,
            hint_rank as usize,
        )
    }
}

#[cfg(test)]
mod test_rank9 {
    use crate::prelude::*;

    #[test]
    fn test_rank9_1() {
        let bitvec = unsafe { BitVec::from_raw_parts(vec![usize::MAX; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(0), 0);
    }

    #[test]
    fn test_rank9_2() {
        let bitvec = unsafe { BitVec::from_raw_parts(vec![usize::MAX; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(64 * 7 + 63), 64 * 7 + 63);
    }

    #[test]
    fn test_rank9_3() {
        let bitvec = unsafe { BitVec::from_raw_parts(vec![usize::MAX; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(64 * 8), 64 * 8);
    }

    #[test]
    fn test_rank9_4() {
        let bitvec = unsafe { BitVec::from_raw_parts(vec![usize::MAX; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(64 * 99 + 63), 64 * 99 + 63);
    }

    #[test]
    fn test_rank9_5() {
        let bitvec = unsafe { BitVec::from_raw_parts(vec![usize::MAX; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(64 * 100), 64 * 100);
    }

    #[test]
    fn test_rank9_6() {
        let bitvec = unsafe { BitVec::from_raw_parts(vec![usize::MAX; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(64 * 101), 64 * 100);
    }

    #[test]
    fn test_rank9_7() {
        let bitvec =
            unsafe { BitVec::from_raw_parts(vec![0x0101010101010101usize; 100], 64 * 100) };
        let rank9: Rank9<BitVec, 64> = Rank9::new(bitvec, 64 * 100);
        assert_eq!(rank9.rank(64 * 15 + 1), 15 * 8 + 1);
    }
}
