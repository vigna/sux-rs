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

use crate::prelude::{BitCount, BitLength, BitVec, Rank};

/// A ranking structure using 25% of additional space and providing the fastest
/// available rank operations.
///
/// `Rank9` stores 64-bit absolute cumulative counters for 512-bit blocks, and
/// relative cumulative 9-bit counters for each 64-bit word in a block. The
/// first relative counter is stored implicitly using zero extension, so eight
/// 9-bit counters can be stored in just 64 bits. Moreover, absolute and
/// relative counters are interleaved. These two ideas make it possible to rank
/// using a most two cache misses and no tests or loops.
///
/// This structure has been described by Sebastiano Vigna in “[Broadword
/// Implementation of Rank/Select
/// Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
/// _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
/// 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
/// Springer, 2008.
///
/// # Examples
///
/// ```rust
/// use sux::prelude::{Rank9, bit_vec};
///
/// let rank9 = Rank9::new(bit_vec![1, 0, 1, 1, 0, 1, 0, 1]);
/// assert_eq!(rank9.rank(0), 0);
/// assert_eq!(rank9.rank(1), 1);
/// assert_eq!(rank9.rank(2), 1);
/// assert_eq!(rank9.rank(3), 2);
/// assert_eq!(rank9.rank(4), 2);
/// assert_eq!(rank9.rank(5), 3);
/// assert_eq!(rank9.rank(6), 3);
/// assert_eq!(rank9.rank(7), 4);
/// assert_eq!(rank9.rank(8), 4);
/// ```
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank9<B: AsRef<[usize]> = BitVec, C: AsRef<[BlockCounters]> = Vec<BlockCounters>> {
    pub(super) bits: B,
    pub(super) counts: C,
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Default)]
pub struct BlockCounters {
    pub(super) absolute: usize,
    pub(super) relative: usize,
}

impl BlockCounters {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        self.relative >> (9 * (word ^ 7)) & 0x1FF
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative |= counter << (9 * (word ^ 7));
    }
}

impl Rank9<BitVec, Vec<BlockCounters>> {
    pub(super) const WORDS_PER_BLOCK: usize = 8;

    /// Creates a new Rank9 structure from a given bit vector.
    pub fn new(bits: BitVec) -> Self {
        let num_bits = bits.len();
        let num_words = num_bits.div_ceil(usize::BITS as usize);
        let num_counts = num_bits.div_ceil(usize::BITS as usize * Self::WORDS_PER_BLOCK);

        // We use the last counter to store the total number of ones
        let mut counts = vec![BlockCounters::default(); num_counts + 1];

        let mut num_ones = 0;

        for (i, pos) in (0..num_words).step_by(Self::WORDS_PER_BLOCK).zip(0..) {
            counts[pos].absolute = num_ones;
            num_ones += bits.as_ref()[i].count_ones() as usize;

            for j in 1..8 {
                let rel_count = num_ones - counts[pos].absolute;
                counts[pos].set_rel(j, rel_count);
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

impl<B: BitLength + AsRef<[usize]>, C: AsRef<[BlockCounters]>> Rank9<B, C> {
    pub fn into_inner(self) -> B {
        self.bits
    }
}

impl<B: BitLength + AsRef<[usize]>, C: AsRef<[BlockCounters]>> Rank for Rank9<B, C> {
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
        let block = word_pos / Rank9::WORDS_PER_BLOCK;
        let offset = word_pos % Rank9::WORDS_PER_BLOCK;
        let word = self.bits.as_ref().get_unchecked(word_pos);
        let counts = self.counts.as_ref().get_unchecked(block);

        counts.absolute
            + counts.rel(offset)
            + (word & ((1 << (pos % usize::BITS as usize)) - 1)).count_ones() as usize
    }
}

impl<B: BitLength + BitLength + AsRef<[usize]>, C: AsRef<[BlockCounters]>> BitCount
    for Rank9<B, C>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.counts.as_ref().last().unwrap().absolute
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>> BitLength for Rank9<B, C> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<B: AsRef<[usize]>, C: AsRef<[BlockCounters]>> AsRef<[usize]> for Rank9<B, C> {
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<B: AsRef<[usize]> + Index<usize, Output = bool>, C: AsRef<[BlockCounters]>> Index<usize>
    for Rank9<B, C>
{
    type Output = bool;

    #[inline(always)]
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

        assert_eq!(rank9.rank(rank9.len()), rank9.bits.count_ones());
    }
}
