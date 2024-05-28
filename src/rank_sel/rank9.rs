use std::ops::Index;

use epserde::*;
use mem_dbg::*;

use crate::prelude::{BitCount, BitLength, BitVec, Rank};

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank9<B: AsRef<[usize]> = BitVec, C: AsRef<[usize]> = Vec<usize>> {
    pub(super) bits: B,
    pub(super) counts: C,
}

impl Rank9<BitVec, Vec<usize>> {
    const NUM_BLOCKS: usize = 8;

    /// Creates a new Rank9 structure from a given bit vector.
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = num_bits.div_ceil(usize::BITS as usize);
        let num_counts = ((num_bits + usize::BITS as usize * Self::NUM_BLOCKS - 1)
            / (usize::BITS as usize * Self::NUM_BLOCKS))
            * 2;

        let mut counts = vec![0_usize; num_counts + 2];

        let mut num_ones = 0;

        for (i, pos) in (0..num_words).step_by(8).zip((0..).step_by(2)) {
            counts[pos] = num_ones;
            num_ones += bits.as_ref()[i].count_ones() as usize;
            for j in 1..8 {
                counts[pos + 1] |= (num_ones - counts[pos]) << (9 * (j - 1));
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as usize;
                }
            }
        }

        counts[num_counts] = num_ones;

        Self { bits, counts }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<B: BitLength + AsRef<[usize]>, C: AsRef<[usize]>> Rank for Rank9<B, C> {
    #[inline(always)]
    fn rank(&self, pos: usize) -> usize {
        if pos >= self.bits.len() {
            self.count()
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word = pos / 64;
        let block = (word / 4) & !1;
        let offset = (word % 8).wrapping_sub(1);

        let base_rank = *self.counts.as_ref().get_unchecked(block)
            + (*self.counts.as_ref().get_unchecked(block + 1)
                >> ((offset.wrapping_add(offset >> 60 & 0x8)) * 9)
                & 0x1FF);

        base_rank
            + (self.bits.as_ref().get_unchecked(word) & ((1 << (pos % usize::BITS as usize)) - 1))
                .count_ones() as usize
    }
}

impl<B: BitLength + AsRef<[usize]>, C: AsRef<[usize]>> BitCount for Rank9<B, C> {
    #[inline(always)]
    fn count(&self) -> usize {
        self.counts.as_ref()[self.counts.as_ref().len() - 2] as usize
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<B: AsRef<[usize]> + BitLength, C: AsRef<[usize]>> BitLength for Rank9<B, C> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<B: AsRef<[usize]>, C: AsRef<[usize]>> AsRef<[usize]> for Rank9<B, C> {
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<B: AsRef<[usize]> + Index<usize, Output = bool>, C: AsRef<[usize]>> Index<usize>
    for Rank9<B, C>
{
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.bits[index]
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
