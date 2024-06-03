/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::ops::Index;

use epserde::*;
use mem_dbg::*;
use static_assertions::{assert_impl_all, const_assert_eq};

use crate::prelude::{BitCount, BitLength, BitVec, Rank, RankHinted};

macro_rules! rank_small {
    ($n: expr; $bv: expr) => {
        match $n {
            9 => RankSmall::<2, 9>::new($bv),
            10 => RankSmall::<1, 10>::new($bv),
            11 => RankSmall::<1, 11>::new($bv),
            13 => RankSmall::<3, 13>::new($bv),
            _ => panic!("Unsupported pointer size");
        }
    };
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct RankSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    const HINT_BIT_SIZE: usize = 64,
    B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]> = BitVec,
    C1: AsRef<[usize]> = Vec<usize>,
    C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]> = Vec<
        Block32Counters<NUM_U32S, COUNTER_WIDTH>,
    >,
> {
    pub(super) bits: B,
    pub(super) upper_counts: C1,
    pub(super) counts: C2,
    pub(super) num_ones: usize,
}

#[derive(Epserde, Copy, Debug, Clone, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
pub struct Block32Counters<const NUM_U32S: usize, const COUNTER_WIDTH: usize> {
    pub(super) absolute: u32,
    pub(super) relative: [u32; NUM_U32S],
}

impl Block32Counters<1, 11> {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        self.relative[0] as usize >> (11 * (word ^ 3)) & ((1 << 11) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (11 * (word ^ 3));
    }
}

impl<const NUM_U32S: usize> Block32Counters<NUM_U32S> {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        match NUM_U32S {
            1 => (self.relative[0] as usize) >> (11 * (word ^ 3)) & ((1 << 11) - 1),
            2 => {
                let packed = unsafe {
                    std::mem::transmute::<[u32; 2], usize>([self.relative[0], self.relative[1]])
                };
                packed >> (9 * (word ^ 7)) & ((1 << 9) - 1)
            }
            3 => {
                let packed = unsafe {
                    std::mem::transmute::<[u32; 4], u128>([
                        self.relative[0],
                        self.relative[1],
                        self.relative[2],
                        0u32,
                    ])
                };
                (packed >> (13 * (word ^ 7)) & ((1 << 13) - 1)) as usize
            }
            _ => panic!("Unsupported number of u32s"),
        }
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        match NUM_U32S {
            1 => {
                self.relative[0] |= (counter as u32) << (11 * (word ^ 3));
            }
            2 => {
                let mut packed = unsafe {
                    std::mem::transmute::<[u32; 2], usize>([self.relative[0], self.relative[1]])
                };
                packed |= counter << (9 * (word ^ 7));
                let slice = unsafe { std::mem::transmute::<usize, [u32; 2]>(packed) };
                self.relative[0] = slice[0];
                self.relative[1] = slice[1];
            }
            3 => {
                let mut packed = unsafe {
                    std::mem::transmute::<[u32; 4], u128>([
                        self.relative[0],
                        self.relative[1],
                        self.relative[2],
                        0u32,
                    ])
                };
                packed |= (counter as u128) << (13 * (word ^ 7));
                let slice = unsafe { std::mem::transmute::<u128, [u32; 4]>(packed) };
                self.relative[0] = slice[0];
                self.relative[1] = slice[1];
                self.relative[2] = slice[2];
            }
            _ => panic!("Unsupported number of u32s"),
        }
    }
}

impl<const NUM_U32S: usize> Default for Block32Counters<NUM_U32S> {
    fn default() -> Self {
        Self {
            absolute: 0,
            relative: [0; NUM_U32S],
        }
    }
}

impl<
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    pub(super) const WORDS_PER_BLOCK: usize = match NUM_U32S {
        1 => 32,  // poppy: 32 * 64 = 2048 block size
        2 => 8,   // small rank9: 8 * 64 = 512 block size
        3 => 128, // rank13: 128 * 64 = 8192 block size
        _ => panic!("Unsupported number of u32s"),
    };
    pub(super) const WORDS_PER_SUBBLOCK: usize = match NUM_U32S {
        1 => Self::WORDS_PER_BLOCK / 4, // poppy has 4 subblocks
        2 => Self::WORDS_PER_BLOCK / 8, // small rank9 has 8 subblocks
        3 => Self::WORDS_PER_BLOCK / 8, // rank13 has 8 subblocks
        _ => panic!("Unsupported number of u32s"),
    };
}

impl<const NUM_U32S: usize>
    RankSmall<NUM_U32S, 64, BitVec, Vec<usize>, Vec<Block32Counters<NUM_U32S>>>
{
    /// Creates a new RankSmall structure from a given bit vector.
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = num_bits.div_ceil(usize::BITS as usize);
        let num_upper_counts = num_bits.div_ceil(1usize << 32);
        let num_counts = num_bits.div_ceil(usize::BITS as usize * Self::WORDS_PER_BLOCK);

        let mut upper_counts = vec![0; num_upper_counts];
        let mut counts = vec![Block32Counters::default(); num_counts];

        let mut num_ones: usize = 0;
        let mut upper_count = 0;

        for (i, pos) in (0..num_words).step_by(Self::WORDS_PER_BLOCK).zip(0..) {
            if i % (1usize << 26) == 0 {
                upper_count = num_ones;
                upper_counts[i / (1usize << 26)] = upper_count;
            }
            counts[pos].absolute = (num_ones - upper_count) as u32;
            num_ones += bits.as_ref()[i].count_ones() as usize;

            for j in 1..Self::WORDS_PER_BLOCK {
                if j % Self::WORDS_PER_SUBBLOCK == 0 {
                    let rel_count = num_ones - upper_count - counts[pos].absolute as usize;
                    counts[pos].set_rel(j / Self::WORDS_PER_SUBBLOCK, rel_count);
                }
                if i + j < num_words {
                    num_ones += bits.as_ref()[i + j].count_ones() as usize;
                }
            }
        }

        Self {
            bits,
            upper_counts,
            counts,
            num_ones,
        }
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
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    pub fn into_inner(self) -> B {
        self.bits
    }
}

impl<
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + BitLength + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > Rank for RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    #[inline(always)]
    fn rank(&self, pos: usize) -> usize {
        if pos >= self.bits.len() {
            self.count_ones()
        } else {
            unsafe { self.rank_unchecked(pos) }
        }
    }

    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        let word_pos = pos / usize::BITS as usize;
        let block = word_pos / Self::WORDS_PER_BLOCK;
        let offset = (word_pos % Self::WORDS_PER_BLOCK) / Self::WORDS_PER_SUBBLOCK;
        let counts = self.counts.as_ref().get_unchecked(block);
        let upper_count = self
            .upper_counts
            .as_ref()
            .get_unchecked(word_pos / (1usize << 26));

        if Self::WORDS_PER_SUBBLOCK == 1 {
            let word = self.bits.as_ref().get_unchecked(word_pos);
            upper_count
                + counts.absolute as usize
                + counts.rel(offset)
                + (word & ((1 << (pos % usize::BITS as usize)) - 1)).count_ones() as usize
        } else {
            let hint_rank = upper_count + counts.absolute as usize + counts.rel(offset);
            let hint_pos =
                word_pos - ((word_pos % Self::WORDS_PER_BLOCK) % Self::WORDS_PER_SUBBLOCK);

            RankHinted::<HINT_BIT_SIZE>::rank_hinted_unchecked(&self.bits, pos, hint_pos, hint_rank)
        }
    }
}

impl<
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + BitLength + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > BitCount for RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]> + BitLength,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > BitLength for RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > AsRef<[usize]> for RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + AsRef<[usize]> + Index<usize, Output = bool>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S>]>,
    > Index<usize> for RankSmall<NUM_U32S, HINT_BIT_SIZE, B, C1, C2>
{
    type Output = bool;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.bits[index]
    }
}

#[cfg(test)]
mod test_rank_small {
    use crate::prelude::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn test_rank_small1() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small = RankSmall::<1>::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(
                    rank_small.rank(i),
                    ranks[i],
                    "i = {}, len = {}, left = {}, right = {}",
                    i,
                    len,
                    rank_small.rank(i),
                    ranks[i]
                );
            }
            assert_eq!(rank_small.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_rank_small2() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small = RankSmall::<2>::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(
                    rank_small.rank(i),
                    ranks[i],
                    "i = {}, len = {}, left = {}, right = {}",
                    i,
                    len,
                    rank_small.rank(i),
                    ranks[i]
                );
            }
            assert_eq!(rank_small.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_rank_small3() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small = RankSmall::<3>::new(bits.clone());

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(
                    rank_small.rank(i),
                    ranks[i],
                    "i = {}, len = {}, left = {}, right = {}",
                    i,
                    len,
                    rank_small.rank(i),
                    ranks[i]
                );
            }
            assert_eq!(rank_small.rank(bits.len() + 1), bits.count_ones());
        }
    }

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 10], (1 << 10) * 64) };

        let rank_small = RankSmall::<1>::new(bits.clone());
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );

        let rank_small = RankSmall::<2>::new(bits.clone());
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );

        let rank_small = RankSmall::<3>::new(bits.clone());
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );
    }
}
