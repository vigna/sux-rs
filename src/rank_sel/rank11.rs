/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::ops::Index;

use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

/// A ranking structure using 6.25% additional space and providing fast ranking.
///
/// `Rank11` stores 64-bit absolute cumulative counters for 2048-bit blocks and 5
/// relative cumulative 11-bit counters (inside another 64 bit word) every 384 bits in a block.
/// The implementation follows the same idea as `Rank9`, but with a linear search inside
/// of the 384-bit blocks. As a consequence, the implementation of this method is significantly
/// slower than that of `Rank9`, which doesn't perform any linear search.
///
/// It was proposed by Simon Gog and Matthias Petri in "Optimized succinct data structures
/// for massive data", `Softw. Pract. Exper.`, 2014.
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank11<
    B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]> = BitVec,
    C: AsRef<[BlockCounters11]> = Vec<BlockCounters11>,
    const HINT_BIT_SIZE: usize = 64,
> {
    pub(super) bits: B,
    pub(super) counts: C,
}

#[derive(Epserde, Copy, Debug, Clone, MemDbg, MemSize, Default)]
#[repr(C)]
#[zero_copy]
pub struct BlockCounters11 {
    pub(super) absolute: usize,
    pub(super) relative: usize,
}

impl BlockCounters11 {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        self.relative >> (60 - 12 * word) & 0x7FF
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative |= counter << (60 - 12 * word);
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > Rank11<B, C, HINT_BIT_SIZE>
{
    const WORDS_PER_BLOCK: usize = 32;
}

impl<const HINT_BIT_SIZE: usize> Rank11<BitVec, Vec<BlockCounters11>, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = num_bits.div_ceil(usize::BITS as usize);
        let num_counts = num_words.div_ceil(Self::WORDS_PER_BLOCK);

        // We use the last counter to store the total number of ones
        let mut counts = vec![BlockCounters11::default(); num_counts + 1];

        let mut num_ones = 0;

        for (i, pos) in (0..num_words).step_by(Self::WORDS_PER_BLOCK).zip(0..) {
            counts[pos].absolute = num_ones;
            num_ones += bits.as_ref()[i].count_ones() as usize;

            for j in 1..Self::WORDS_PER_BLOCK {
                if j % 6 == 0 {
                    let rel_count = num_ones - counts[pos].absolute;
                    counts[pos].set_rel(j / 6, rel_count);
                }
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as usize;
                }
            }
        }

        counts[num_counts].absolute = num_ones;

        Self { bits, counts }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > Rank11<B, C, HINT_BIT_SIZE>
{
    pub fn into_inner(self) -> B {
        self.bits
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + BitLength + AsRef<[usize]>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > Rank for Rank11<B, C, HINT_BIT_SIZE>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word_pos = pos / usize::BITS as usize;
        let block = word_pos / Self::WORDS_PER_BLOCK;
        let offset = (word_pos % Self::WORDS_PER_BLOCK) / 6;
        let counts = self.counts.as_ref().get_unchecked(block);

        let hint_rank = counts.absolute + counts.rel(offset);

        let hint_pos = word_pos - ((word_pos % 32) % 6);

        RankHinted::<HINT_BIT_SIZE>::rank_hinted_unchecked(&self.bits, pos, hint_pos, hint_rank)
    }

    fn rank(&self, pos: usize) -> usize {
        if pos >= self.bits.len() {
            self.count_ones()
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + BitLength + AsRef<[usize]>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > BitCount for Rank11<B, C, HINT_BIT_SIZE>
{
    fn count_ones(&self) -> usize {
        self.counts.as_ref().last().unwrap().absolute
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        B: RankHinted<HINT_BIT_SIZE> + BitLength + AsRef<[usize]>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > BitLength for Rank11<B, C, HINT_BIT_SIZE>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > AsRef<[usize]> for Rank11<B, C, HINT_BIT_SIZE>
{
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]> + Index<usize, Output = bool>,
        C: AsRef<[BlockCounters11]>,
        const HINT_BIT_SIZE: usize,
    > Index<usize> for Rank11<B, C, HINT_BIT_SIZE>
{
    type Output = bool;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.bits[index]
    }
}

#[cfg(test)]
mod test_rank11 {
    use crate::prelude::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_rank11() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank11: Rank11 = Rank11::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(rank11.rank(i), ranks[i]);
            }
            assert_eq!(rank11.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 16], (1 << 16) * 64) };

        let rank11: Rank11 = Rank11::new(bits);

        assert_eq!(rank11.rank(rank11.len()), rank11.bits.count_ones());
    }
}
