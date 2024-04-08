use epserde::*;
use mem_dbg::*;

use crate::prelude::{BitCount, BitLength, BitVec, Rank, RankHinted};

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank9<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    C: AsRef<[u64]> = Vec<u64>,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    pub(super) counts: C,
}

impl<const HINT_BIT_SIZE: usize> Rank9<BitVec, Vec<u64>, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_counts = ((num_bits + 64 * 8 - 1) / (64 * 8)) * 2;

        let mut counts = vec![0u64; num_counts + 2];

        let mut num_ones = 0u64;

        for (i, pos) in (0..num_words).step_by(8).zip((0..).step_by(2)) {
            counts[pos] = num_ones;
            num_ones += bits.as_ref()[i].count_ones() as u64;
            for j in 1..8 {
                counts[pos + 1] |= (num_ones - counts[pos]) << (9 * (j - 1));
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as u64;
                }
            }
        }

        counts[num_counts] = num_ones;

        Self { bits, counts }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > BitLength for Rank9<B, C, HINT_BIT_SIZE>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > Rank for Rank9<B, C, HINT_BIT_SIZE>
{
    fn rank(&self, pos: usize) -> usize {
        if pos >= self.bits.len() {
            self.counts.as_ref()[self.counts.as_ref().len() - 2] as usize
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let block = (word / 4) & !1;
        let offset = (word % 8).wrapping_sub(1);

        let hint_rank = *self.counts.as_ref().get_unchecked(block)
            + (*self.counts.as_ref().get_unchecked(block + 1)
                >> ((offset.wrapping_add(offset >> 60 & 0x8)) * 9)
                & 0x1FF);

        RankHinted::<HINT_BIT_SIZE>::rank_hinted_unchecked(
            &self.bits,
            pos,
            word,
            hint_rank as usize,
        )
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > BitCount for Rank9<B, C, HINT_BIT_SIZE>
{
    fn count(&self) -> usize {
        self.rank(self.bits.len())
    }
}

#[cfg(test)]
mod test_rank9 {
    use crate::prelude::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn test_rank9() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank9: Rank9 = Rank9::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(rank9.rank(i), ranks[i]);
            }
            assert_eq!(rank9.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 10], (1 << 10) * 64) };

        let rank9: Rank9 = Rank9::new(bits);

        assert_eq!(rank9.rank(rank9.len()), rank9.bits.count());
    }
}
