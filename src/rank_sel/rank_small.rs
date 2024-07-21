/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use ambassador::Delegate;
use epserde::*;
use mem_dbg::*;
use std::ptr::{addr_of, read_unaligned, write_unaligned};

use crate::{
    prelude::{BitLength, BitVec, Rank, RankHinted, RankUnchecked, RankZero},
    traits::{BitCount, NumBits},
};

use crate::ambassador_impl_AsRef;
use crate::ambassador_impl_Index;
use crate::traits::rank_sel::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_RankHinted;
use crate::traits::rank_sel::ambassador_impl_Select;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectUnchecked;
use crate::traits::rank_sel::ambassador_impl_SelectZero;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZeroUnchecked;
use std::ops::Index;

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
/// [`rank_small`](crate::rank_small) macro that selects the correct
/// combination.
///
/// Presently we support the following combinations:
///
/// - `rank_small![0; -]` (builds `RankSmall<2, 9>`): 18.75% additional space,
///   speed slightly slower than [`Rank9`](super::Rank9).
/// - `rank_small![1; -]` (builds `RankSmall<1, 9>`): 12.5% additional space.
/// - `rank_small![2; -]` (builds `RankSmall<1, 10>`): 6.25% additional space.
/// - `rank_small![3; -]` (builds `RankSmall<1, 11>`): 3.125% additional space.
/// - `rank_small![4; -]` (builds `RankSmall<3, 13>`): 1.56% additional space.
///
/// The first structure is a space-savvy version of [`Rank9`](super::Rank9),
/// while the other ones provide increasing less space usage at the expense of
/// slower operations.
///
/// `RankSmall<1, 11>` is similar to
/// [`poppy`](https://link.springer.com/chapter/10.1007/978-3-642-38527-8_15),
/// but instead of storing counters and rebuilding cumulative counters on the
/// fly it stores the cumulative counters directly using implicit zero
/// extension, as in [`Rank9`](super::Rank9).
///
/// # Examples
///
/// ```rust
///
/// use sux::{bit_vec,rank_small};
/// use sux::traits::{Rank, Select};
/// use sux::rank_sel::SelectAdapt;
///
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![0; bits];
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
///
/// // Access to the underlying bit vector is forwarded
/// assert_eq!(rank_small[0], true);
/// assert_eq!(rank_small[1], false);
/// assert_eq!(rank_small[2], true);
/// assert_eq!(rank_small[3], true);
/// assert_eq!(rank_small[4], false);
/// assert_eq!(rank_small[5], true);
/// assert_eq!(rank_small[6], false);
/// assert_eq!(rank_small[7], true);

#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Delegate)]
#[delegate(AsRef<[usize]>, target = "bits")]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::rank_sel::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted<64>, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::Select, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
pub struct RankSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    B = BitVec,
    C1 = Box<[usize]>,
    C2 = Box<[Block32Counters<NUM_U32S, COUNTER_WIDTH>]>,
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
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![0; bits];
///
/// assert_eq!(rank_small.rank(0), 0);
/// assert_eq!(rank_small.rank(1), 1);
/// ```

#[macro_export]
macro_rules! rank_small {
    (0 ; $bits: expr) => {
        $crate::prelude::RankSmall::<2, 9, _, _, _>::new($bits)
    };
    (1 ; $bits: expr) => {
        $crate::prelude::RankSmall::<1, 9, _, _, _>::new($bits)
    };
    (2 ; $bits: expr) => {
        $crate::prelude::RankSmall::<1, 10, _, _, _>::new($bits)
    };
    (3 ; $bits: expr) => {
        $crate::prelude::RankSmall::<1, 11, _, _, _>::new($bits)
    };
    (4 ; $bits: expr) => {
        $crate::prelude::RankSmall::<3, 13, _, _, _>::new($bits)
    };
}

#[doc(hidden)]
#[derive(Epserde, Copy, Debug, Clone, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
pub struct Block32Counters<const NUM_U32S: usize, const COUNTER_WIDTH: usize> {
    pub(super) absolute: u32,
    pub(super) relative: [u32; NUM_U32S],
}

impl Block32Counters<2, 9> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        unsafe { read_unaligned(addr_of!(self.relative) as *const u64) }
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.all_rel() >> (9 * (word ^ 7)) & ((1 << 9) - 1)) as usize
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        let mut packed = unsafe { read_unaligned(addr_of!(self.relative) as *const u64) };
        packed |= (counter as u64) << (9 * (word ^ 7));
        unsafe { write_unaligned(addr_of!(self.relative) as *mut u64, packed) };
    }
}

impl Block32Counters<1, 9> {
    #[inline(always)]
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

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
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

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
    pub fn all_rel(&self) -> u64 {
        self.relative[0] as u64
    }

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
    pub fn all_rel(&self) -> u128 {
        #[cfg(target_endian = "little")]
        unsafe {
            read_unaligned(addr_of!(*self) as *const u128) >> 32
        }
        #[cfg(target_endian = "big")]
        unsafe {
            read_unaligned(addr_of!(*self) as *const u128) & (1 << 96) - 1
        }
    }

    #[inline(always)]
    pub fn rel(&self, word: usize) -> usize {
        (self.all_rel() >> (13 * (word ^ 7)) & ((1 << 13) - 1)) as usize
    }

    #[inline(always)]
    pub fn set_rel(&mut self, word: usize, counter: usize) {
        let mut packed = self.all_rel();
        packed |= (counter as u128) << (13 * (word ^ 7));

        #[cfg(target_endian = "little")]
        unsafe {
            write_unaligned(
                addr_of!(*self) as *mut u128,
                packed << 32 | self.absolute as u128,
            )
        };
        #[cfg(target_endian = "big")]
        unsafe {
            write_unaligned(
                addr_of!(*self) as *mut u128,
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

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2>
    RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
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
        impl<B: AsRef<[usize]> + BitLength + RankHinted<64>>
            RankSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                B,
                Box<[usize]>,
                Box<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            >
        {
            /// Creates a new RankSmall structure from a given bit vector.
            pub fn new(bits: B) -> Self {
                let num_bits = bits.len();
                let num_words = num_bits.div_ceil(64 as usize);
                let num_upper_counts = num_bits.div_ceil(1usize << 32);
                let num_counts = num_bits.div_ceil(64 as usize * Self::WORDS_PER_BLOCK);

                let mut upper_counts = Vec::with_capacity(num_upper_counts);
                let mut counts = Vec::with_capacity(num_counts);

                let mut past_ones = 0;
                let mut upper_count = 0;

                for i in (0..num_words).step_by(Self::WORDS_PER_BLOCK) {
                    if i % (1usize << 26) == 0 {
                        upper_count = past_ones;
                        upper_counts.push(upper_count);
                    }
                    let mut count = Block32Counters::<$NUM_U32S, $COUNTER_WIDTH>::default();
                    count.absolute = (past_ones - upper_count) as u32;
                    past_ones += bits.as_ref()[i].count_ones() as usize;

                    for j in 1..Self::WORDS_PER_BLOCK {
                        #[allow(clippy::modulo_one)]
                        if j % Self::WORDS_PER_SUBBLOCK == 0 {
                            let rel_count = past_ones - upper_count - count.absolute as usize;
                            count.set_rel(j / Self::WORDS_PER_SUBBLOCK, rel_count);
                        }
                        if i + j < num_words {
                            past_ones += bits.as_ref()[i + j].count_ones() as usize;
                        }
                    }

                    counts.push(count);
                }

                assert_eq!(upper_counts.len(), num_upper_counts);
                assert_eq!(counts.len(), num_counts);

                let upper_counts = upper_counts.into_boxed_slice();
                let counts = counts.into_boxed_slice();

                Self {
                    bits,
                    upper_counts,
                    counts,
                    num_ones: past_ones,
                }
            }
        }
        impl<
                B: AsRef<[usize]> + BitLength + RankHinted<64>,
                C1: AsRef<[usize]>,
                C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            > RankUnchecked for RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>
        {
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

                let hint_rank = upper_count + counts.absolute as usize + counts.rel(offset);
                if Self::WORDS_PER_SUBBLOCK == 1 {
                    // Rank<2, 9> works like Rank9.
                    let word = self.bits.as_ref().get_unchecked(word_pos);
                    hint_rank + (word & ((1 << (pos % 64 as usize)) - 1)).count_ones() as usize
                } else {
                    // For the other cases we need a bit more work.
                    #[allow(clippy::modulo_one)]
                    let hint_pos =
                        word_pos - ((word_pos % Self::WORDS_PER_BLOCK) % Self::WORDS_PER_SUBBLOCK);

                    RankHinted::<64>::rank_hinted(&self.bits, pos, hint_pos, hint_rank)
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

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2> Rank
    for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
where
    RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>: BitLength + NumBits + RankUnchecked,
{
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2> RankZero
    for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
where
    RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>: Rank,
{
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B: BitLength, C1, C2>
    RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    /// Returns the number of bits in the bit vector.
    ///
    /// This method is equivalent to
    /// [`BitLength::len`](crate::traits::BitLength::len), but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B, C1, C2>
    RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Replaces the backend with a new one.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new backend is identical to the old one as a bit vector.
    pub unsafe fn map<B1>(
        self,
        f: impl FnOnce(B) -> B1,
    ) -> RankSmall<NUM_U32S, COUNTER_WIDTH, B1, C1, C2>
    where
        B1: AsRef<[usize]> + BitLength,
    {
        RankSmall {
            bits: f(self.bits),
            upper_counts: self.upper_counts,
            counts: self.counts,
            num_ones: self.num_ones,
        }
    }
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B: BitLength, C1, C2> NumBits
    for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn num_ones(&self) -> usize {
        self.num_ones
    }
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, B: BitLength, C1, C2> BitCount
    for RankSmall<NUM_U32S, COUNTER_WIDTH, B, C1, C2>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::traits::AddNumBits;
    use crate::traits::NumBits;

    #[test]
    fn test_last() {
        let bits: AddNumBits<_> =
            unsafe { BitVec::from_raw_parts(vec![!1usize; 1 << 10], (1 << 10) * 64) }.into();

        let rank_small = rank_small![1; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.num_ones()
        );

        let rank_small = rank_small![1; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.num_ones()
        );

        let rank_small = rank_small![2; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.num_ones()
        );

        let rank_small = rank_small![3; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.num_ones()
        );

        let rank_small = rank_small![4; bits.clone()];
        assert_eq!(
            rank_small.rank(rank_small.len()),
            rank_small.bits.num_ones()
        );
    }
}
