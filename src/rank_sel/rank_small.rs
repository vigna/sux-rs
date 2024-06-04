/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use epserde::*;
use mem_dbg::*;
use std::{
    ops::Index,
    ptr::{self, read, read_unaligned, write_unaligned},
};

use crate::prelude::{BitCount, BitLength, BitVec, Rank, RankHinted};

/// A family of ranking structures using very little additional space but with
/// slower operations than [`Rank9`](super::Rank9).
///
/// [`RankSmall`] structures combine two ideas from [`Rank9`](super::Rank9),
/// that is, the interleaving of absolute and relative counters and the storage
/// of implicit counters using zero extension, and a design trick from from
/// [poppy](https://link.springer.com/chapter/10.1007/978-3-642-38527-8_15),
/// that is, that the structures are actually designed around bit vectors of at
/// most 2³² bits. This allows the use of 32-bit counters, which use less space,
/// at the expense of a high-level additional list of 64-bit counters that
/// contain the actual absolute cumulative counts for each block of 2³² bits.
/// Since in most applications these counters will be very few, their additional
/// space in neglibigle, and they will usually accessed without cache misses.
///
/// The [`RankSmall`] variants are parameterized by the number of 32-bit word
/// per block and by the size of the relative counters. Only certain
/// combinations are possible, and to simplify construction we provide a
/// [`rank_small`] macro that selects the correct combination given the size of
/// the relative counters.
///
/// Presently we support the following combinations:
///
/// - `rank_small![0; -]` (builds `RankSmall<2, 9>`): 18.75% additional space,
///   speed slightly slower than [`Rank9`](super::Rank9).
/// - `rank_small![1; -]` (builds `RankSmall<1, 9>`): 12.5% additional space.
/// - `rank_small![2; -]` (builds `RankSmall<1, 10>`): 6.25% additional space.
/// - `rank_small![3; -]` (builds `RankSmall<1, 11>`): 3.125% additional
///   space.
/// - `rank_small![4; -]` (builds `RankSmall<3, 13>`): 1.56% additional space.
///
/// The first structure is a space-savvy version of [`Rank9`](super::Rank9),
/// while the other ones provide increasing less space usage at the expense of
/// slower operations.
///
/// `RankSmall<1, 11>` is similar to `poppy`, but instead of storing counters
/// and rebuilding cumulative counters on the fly it stores the cumulative
/// counters directly using implicit zero extension, as in
/// [`Rank9`](super::Rank9).
///

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct RankSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B = BitVec,
    C1 = Vec<usize>,
    C2 = Vec<Block32Counters<NUM_U32S, COUNTER_WIDTH>>,
> {
    pub(super) bits: B,
    pub(super) upper_counts: C1,
    pub(super) counts: C2,
    pub(super) num_ones: usize,
}

/// A convenient macro to build a [`RankSmall`] structure with the correct
/// parameters.
///
/// - `rank_small![0; -]` (builds `RankSmall<2, 9>`): 18.75% additional space,
///   speed slightly slower than [`Rank9`](super::Rank9).
/// - `rank_small![1; -]` (builds `RankSmall<1, 9>`): 12.5% additional space.
/// - `rank_small![2; -]` (builds `RankSmall<1, 10>`): 6.25% additional space.
/// - `rank_small![3; -]` (builds `RankSmall<1, 11>`): 3.125% additional
///   space.
/// - `rank_small![4; -]` (builds `RankSmall<3, 13>`): 1.56% additional space.
///
/// # Examples
///
/// ```rust
/// use sux::{prelude::Rank,bit_vec,rank_small};
/// let bv = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![0; bv];
///
/// assert_eq!(rank_small.rank(0), 0);
/// assert_eq!(rank_small.rank(1), 1);
/// assert_eq!(rank_small.rank(2), 1);
/// assert_eq!(rank_small.rank(3), 2);
/// assert_eq!(rank_small.rank(4), 3);
/// assert_eq!(rank_small.rank(5), 3);
/// assert_eq!(rank_small.rank(6), 4);
/// assert_eq!(rank_small.rank(7), 4);
/// assert_eq!(rank_small.rank(8), 5);
/// ```

#[macro_export]
macro_rules! rank_small {
    (0 ; $bits: expr) => {
        $crate::prelude::RankSmall::<2, 9>::new($bits)
    };
    (1 ; $bits: expr) => {
        $crate::prelude::RankSmall::<1, 9>::new($bits)
    };
    (2 ; $bits: expr) => {
        $crate::prelude::RankSmall::<1, 10>::new($bits)
    };
    (3 ; $bits: expr) => {
        $crate::prelude::RankSmall::<1, 11>::new($bits)
    };
    (4 ; $bits: expr) => {
        $crate::prelude::RankSmall::<3, 13>::new($bits)
    };
}

#[derive(Epserde, Copy, Debug, Clone, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
pub struct Block32Counters<const NUM_U32S: usize, const COUNTER_WIDTH: usize> {
    pub(super) absolute: u32,
    pub(super) relative: [u32; NUM_U32S],
}

impl Block32Counters<2, 9> {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        let packed = unsafe { read(&self.relative as *const [u32; 2] as *const usize) };
        packed >> (9 * (word ^ 7)) & ((1 << 9) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        let mut packed = unsafe { read(&self.relative as *const [u32; 2] as *const usize) };
        packed |= counter << (9 * (word ^ 7));
        self.relative = unsafe { read(&packed as *const usize as *const [u32; 2]) };
    }
}

impl Block32Counters<1, 9> {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        self.relative[0] as usize >> (9 * (word ^ 3)) & ((1 << 9) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (9 * (word ^ 3));
    }
}

impl Block32Counters<1, 10> {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        self.relative[0] as usize >> (10 * (word ^ 3)) & ((1 << 10) - 1)
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        self.relative[0] |= (counter as u32) << (10 * (word ^ 3));
    }
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

impl Block32Counters<3, 13> {
    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        #[cfg(target_endian = "little")]
        let packed = unsafe { read_unaligned(ptr::addr_of!(*self) as *const u128) >> 32 };
        #[cfg(target_endian = "big")]
        let packed = unsafe { read_unaligned(ptr::addr_of!(*self) as *const u128) & (1 << 96) - 1 };

        (packed >> (13 * (word ^ 7)) & ((1 << 13) - 1)) as usize
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        #[cfg(target_endian = "little")]
        let mut packed = unsafe { read_unaligned(ptr::addr_of!(*self) as *const u128) >> 32 };
        #[cfg(target_endian = "big")]
        let mut packed =
            unsafe { read_unaligned(ptr::addr_of!(*self) as *const u128) & (1 << 96) - 1 };

        packed |= (counter as u128) << (13 * (word ^ 7));

        #[cfg(target_endian = "little")]
        unsafe {
            write_unaligned(
                ptr::addr_of!(*self) as *mut u128,
                packed << 32 | self.absolute as u128,
            )
        };
        #[cfg(target_endian = "big")]
        unsafe {
            write_unaligned(
                ptr::addr_of!(*self) as *mut u128,
                packed | (self.absolute as u128) << 96,
            );
        };
    }
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize> Default
    for Block32Counters<NUM_U32S, COUNTER_WIDTH>
{
    fn default() -> Self {
        Self {
            absolute: 0,
            relative: [0; NUM_U32S],
        }
    }
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    pub(super) const WORDS_PER_BLOCK: usize = 1 << (COUNTER_WIDTH - usize::BITS.ilog2() as usize);
    pub(super) const WORDS_PER_SUBBLOCK: usize = match NUM_U32S {
        1 => Self::WORDS_PER_BLOCK / 4, // poppy has 4 subblocks
        2 => Self::WORDS_PER_BLOCK / 8, // small rank9 has 8 subblocks
        3 => Self::WORDS_PER_BLOCK / 8, // rank13 has 8 subblocks
        _ => panic!("Unsupported number of u32s"),
    };
}

macro_rules! impl_rank_small {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        impl
            RankSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                BitVec,
                Vec<usize>,
                Vec<Block32Counters<$NUM_U32S, $COUNTER_WIDTH>>,
            >
        {
            /// Creates a new RankSmall structure from a given bit vector.
            pub fn new(bits: BitVec) -> Self {
                let num_bits = bits.len();
                let num_words = num_bits.div_ceil(64 as usize);
                let num_upper_counts = num_bits.div_ceil(1usize << 32);
                let num_counts = num_bits.div_ceil(64 as usize * Self::WORDS_PER_BLOCK);

                let mut upper_counts = vec![0; num_upper_counts];
                let mut counts =
                    vec![Block32Counters::<$NUM_U32S, $COUNTER_WIDTH>::default(); num_counts];

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
        }
        impl<
                B: RankHinted<64> + BitLength + AsRef<[usize]>,
                C1: AsRef<[usize]>,
                C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            > Rank for RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>
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
                let word_pos = pos / 64 as usize;
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
                        + (word & ((1 << (pos % 64 as usize)) - 1)).count_ones() as usize
                } else {
                    let hint_rank = upper_count + counts.absolute as usize + counts.rel(offset);
                    let hint_pos =
                        word_pos - ((word_pos % Self::WORDS_PER_BLOCK) % Self::WORDS_PER_SUBBLOCK);

                    RankHinted::<64>::rank_hinted_unchecked(&self.bits, pos, hint_pos, hint_rank)
                }
            }
        }
    };
}

impl_rank_small!(2; 9);
impl_rank_small!(1; 9);
impl_rank_small!(1; 10);
impl_rank_small!(1; 11);
impl_rank_small!(3; 13);

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + BitLength + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
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
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    pub fn into_inner(self) -> B {
        self.bits
    }
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + BitLength + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > BitCount for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + AsRef<[usize]> + BitLength,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > BitLength for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + AsRef<[usize]>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > AsRef<[usize]> for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        B: RankHinted<64> + AsRef<[usize]> + Index<usize, Output = bool>,
        C1: AsRef<[usize]>,
        C2: AsRef<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
    > Index<usize> for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    type Output = bool;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.bits[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last() {
        let bits = unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 10], (1 << 10) * 64) };

        let rank_small = rank_small![0; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );

        let rank_small = rank_small![1; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );

        let rank_small = rank_small![2; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );

        let rank_small = rank_small![3; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );

        let rank_small = rank_small![4; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.count_ones()
        );
    }
}
