/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use super::SmallCounters;
use crate::prelude::*;
use ambassador::Delegate;
use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::ambassador_impl_AsRef;
use crate::ambassador_impl_Index;
use crate::rank_sel::ambassador_impl_SmallCounters;
use crate::traits::rank_sel::ambassador_impl_BitCount;
use crate::traits::rank_sel::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_NumBits;
use crate::traits::rank_sel::ambassador_impl_Rank;
use crate::traits::rank_sel::ambassador_impl_RankHinted;
use crate::traits::rank_sel::ambassador_impl_RankUnchecked;
use crate::traits::rank_sel::ambassador_impl_RankZero;
use crate::traits::rank_sel::ambassador_impl_Select;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectUnchecked;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use std::ops::Index;

// NOTE: to make parallel modifications with SelectSmall as easy as possible,
// "ones" are considered to be zeros in the following code.

/// A version of [`SelectSmall`](super::SelectSmall) implementing [selection on
/// zeros](crate::traits::SelectZero).
///
///
/// # Examples
///
/// ```rust
/// use sux::{bit_vec, rank_small};
/// use sux::rank_sel::{SelectSmall, SelectZeroSmall};
/// use sux::traits::{Select, SelectZero};
///
/// let bits = bit_vec![0, 1, 0, 0, 1, 0, 1, 0];
/// let rank_small = rank_small![1; bits];
/// let sel = SelectZeroSmall::<1, 9, _>::new(rank_small);
///
/// assert_eq!(sel.select_zero(0), Some(0));
/// assert_eq!(sel.select_zero(1), Some(2));
/// assert_eq!(sel.select_zero(2), Some(3));
/// assert_eq!(sel.select_zero(3), Some(5));
/// assert_eq!(sel.select_zero(4), Some(7));
/// assert_eq!(sel.select_zero(5), None);
///
/// // One can also stack a SelectZeroSmall on top of a SelectSmall:
/// let rank_small = sel.into_inner();
/// let sel = SelectZeroSmall::<1, 9, _>::new(SelectSmall::<1, 9, _>::new(rank_small));
///
/// assert_eq!(sel.select_zero(0), Some(0));
/// assert_eq!(sel.select_zero(1), Some(2));
/// assert_eq!(sel.select_zero(2), Some(3));
/// assert_eq!(sel.select_zero(3), Some(5));
/// assert_eq!(sel.select_zero(4), Some(7));
/// assert_eq!(sel.select_zero(5), None);
///
/// assert_eq!(sel.select(0), Some(1));
/// assert_eq!(sel.select(1), Some(4));
/// assert_eq!(sel.select(2), Some(6));
/// assert_eq!(sel.select(3), None);
/// ```
#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Delegate)]
#[delegate(AsRef<[usize]>, target = "small_counters")]
#[delegate(Index<usize>, target = "small_counters")]
#[delegate(crate::traits::rank_sel::BitCount, target = "small_counters")]
#[delegate(crate::traits::rank_sel::BitLength, target = "small_counters")]
#[delegate(crate::traits::rank_sel::NumBits, target = "small_counters")]
#[delegate(crate::traits::rank_sel::Rank, target = "small_counters")]
#[delegate(crate::traits::rank_sel::RankHinted<64>, target = "small_counters")]
#[delegate(crate::traits::rank_sel::RankUnchecked, target = "small_counters")]
#[delegate(crate::traits::rank_sel::RankZero, target = "small_counters")]
#[delegate(crate::traits::rank_sel::Select, target = "small_counters")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "small_counters")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "small_counters")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "small_counters")]
#[delegate(crate::rank_sel::SmallCounters<NUM_U32S, COUNTER_WIDTH>, target = "small_counters")]
pub struct SelectZeroSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    C,
    I = Box<[u32]>,
    O = Box<[usize]>,
> {
    small_counters: C,
    inventory: I,
    inventory_begin: O,
    log2_ones_per_inventory: usize,
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, C, I, O>
    SelectZeroSmall<NUM_U32S, COUNTER_WIDTH, C, I, O>
{
    const SUPERBLOCK_BIT_SIZE: usize = 1 << 32;
    const WORDS_PER_BLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_BLOCK;
    const BLOCK_BIT_SIZE: usize = (Self::WORDS_PER_BLOCK * usize::BITS as usize);

    pub fn into_inner(self) -> C {
        self.small_counters
    }
}

impl<const NUM_U32S: usize, const COUNTER_WIDTH: usize, C: BitLength, I, O>
    SelectZeroSmall<NUM_U32S, COUNTER_WIDTH, C, I, O>
{
    /// Returns the number of bits in the bit vector.
    ///
    /// This method is equivalent to
    /// [`BitLength::len`](crate::traits::BitLength::len), but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.small_counters.len()
    }
}

macro_rules! impl_select_zero_small {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        impl<
                C: SmallCounters<$NUM_U32S, $COUNTER_WIDTH>
                    + AsRef<[usize]>
                    + BitLength
                    + NumBits
                    + SelectZeroHinted,
            > SelectZeroSmall<$NUM_U32S, $COUNTER_WIDTH, C>
        {
            /// Creates a new selection structure with eight [`RankSmall`]
            /// blocks per inventory an average.
            pub fn new(small_counters: C) -> Self {
                Self::with_inv(small_counters, 8)
            }

            /// Creates a new selection structure with a given number of
            /// [`RankSmall`] blocks per inventory on average.
            pub fn with_inv(small_counters: C, blocks_per_inv: usize) -> Self {
                let num_bits = small_counters.len();
                let num_ones = small_counters.len() - small_counters.num_ones();

                let target_inventory_span = blocks_per_inv * Self::BLOCK_BIT_SIZE;
                let log2_ones_per_inventory = (num_ones * target_inventory_span)
                    .div_ceil(num_bits.max(1))
                    .max(1)
                    .ilog2() as usize;

                Self::_new(small_counters, num_ones, log2_ones_per_inventory)
            }

            fn _new(small_counters: C, num_ones: usize, log2_ones_per_inventory: usize) -> Self {
                let ones_per_inventory = 1 << log2_ones_per_inventory;

                let inventory_size = num_ones.div_ceil(ones_per_inventory);
                let mut inventory = Vec::<u32>::with_capacity(inventory_size + 1);

                // The inventory_begin vector stores the index of the first element of
                // each superblock in the inventory.
                let mut inventory_begin =
                    Vec::<usize>::with_capacity(small_counters.upper_counts().len() + 1);

                let mut past_ones: usize = 0;
                let mut next_quantum: usize = 0;

                for superblock in small_counters
                    .as_ref()
                    .chunks(Self::SUPERBLOCK_BIT_SIZE / usize::BITS as usize)
                {
                    let mut first = true;
                    for (i, word) in superblock.iter().copied().map(|b| !b).enumerate() {
                        let ones_in_word = (word.count_ones() as usize).min(num_ones - past_ones);

                        while past_ones + ones_in_word > next_quantum {
                            let in_word_index = word.select_in_word(next_quantum - past_ones);
                            let in_superblock_index = i * usize::BITS as usize + in_word_index;
                            if first {
                                inventory_begin.push(inventory.len());
                                first = false;
                            }
                            inventory.push(in_superblock_index as u32);
                            next_quantum += ones_per_inventory;
                        }

                        past_ones += ones_in_word;
                    }
                }
                assert_eq!(num_ones, past_ones);

                if inventory.is_empty() {
                    inventory.push(0);
                    inventory_begin.push(0);
                } else {
                    inventory_begin.push(small_counters.as_ref().len());
                }

                // assert_eq!(inventory.len(), inventory_size + 1);

                let inventory = inventory.into_boxed_slice();
                let inventory_begin = inventory_begin.into_boxed_slice();

                Self {
                    small_counters,
                    inventory,
                    inventory_begin,
                    log2_ones_per_inventory,
                }
            }
        }

        impl<
                C: SmallCounters<$NUM_U32S, $COUNTER_WIDTH>
                    + AsRef<[usize]>
                    + BitLength
                    + NumBits
                    + SelectZeroHinted,
            > SelectZeroUnchecked for SelectZeroSmall<$NUM_U32S, $COUNTER_WIDTH, C>
        {
            unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
                let upper_counts = self.small_counters.upper_counts();
                let counts = self.small_counters.counts();

                let upper_block_idx =
                    upper_counts.linear_partition_point(|i, &x| (i << 32) - x <= rank) - 1;
                let upper_rank_ones = *upper_counts.get_unchecked(upper_block_idx) as usize;
                let upper_rank = (upper_block_idx << 32) - upper_rank_ones;
                let local_rank = rank - upper_rank;

                let inventory = self.inventory.as_ref();
                let inventory_begin = self.inventory_begin.as_ref();

                // we find the two inventory entries containing the rank and from that
                // the blocks containing the rank.
                // if the span of the two entries is not contained in the same upper block
                // we clip the span to the upper block boundaries since we know that
                // the rank is in the upper block with index upper_block_idx.

                let inv_idx = rank >> self.log2_ones_per_inventory;
                let inv_upper_block_idx =
                    inventory_begin.linear_partition_point(|_, &x| x <= inv_idx) - 1;
                let opt;
                let inv_pos = if inv_upper_block_idx == upper_block_idx {
                    opt = (inv_idx << self.log2_ones_per_inventory) - upper_rank;
                    *inventory.get_unchecked(inv_idx) as usize
                        + upper_block_idx * Self::SUPERBLOCK_BIT_SIZE
                } else {
                    // For extremely sparse and large bit vectors, the inventory entry containing
                    // the rank could fall in previous upper blocks.
                    // For this reason, inv_upper_block_idx is not necessarily equal to upper_block_idx.
                    // Since we know for sure that the rank is in the upper block with index upper_block_idx,
                    // we can safely use that value to compute inv_pos.
                    opt = 0;
                    upper_block_idx * Self::SUPERBLOCK_BIT_SIZE
                };
                let mut block_idx = inv_pos / Self::BLOCK_BIT_SIZE;
                // cs-poppy micro-optimization: each block can contain at most
                // Self::BLOCK_BIT_SIZE ones, so we can skip blocks to which the bit
                // we are looking for cannot possibly belong.
                //
                // It would be more precise by using the absolute counter at
                // block_idx, but in benchmarks the additional memory accesses
                // slow down the search, except in the very dense case. We thus
                // approximate the value with opt: this works because
                //
                // inv_idx * ones_per_inventory - upper_rank =
                // local_rank - local_rank % ones_per_inventory
                // >= counts.get(block_idx).absolute.
                block_idx += (local_rank - opt) / Self::BLOCK_BIT_SIZE;

                let last_block_idx;
                if inv_idx + 1 < inventory.len() {
                    let next_inv_upper_block_idx =
                        inventory_begin.linear_partition_point(|_, &x| x <= inv_idx + 1) - 1; // TODO: +1?
                    last_block_idx = if next_inv_upper_block_idx == upper_block_idx {
                        let next_inv_pos = *inventory.get_unchecked(inv_idx + 1) as usize
                            + upper_block_idx * Self::SUPERBLOCK_BIT_SIZE;
                        next_inv_pos.div_ceil(Self::BLOCK_BIT_SIZE)
                    } else {
                        (upper_block_idx + 1) * (Self::SUPERBLOCK_BIT_SIZE / Self::BLOCK_BIT_SIZE)
                    };
                } else {
                    // TODO
                    // Since we use 32-bit entries, we cannot add a sentinel
                    // with value given by the number of bits. Thus, we must
                    // handle the case in which inv_idx is the the last
                    // inventory entry as a special case.
                    last_block_idx = self.len().div_ceil(Self::BLOCK_BIT_SIZE);
                }

                debug_assert!(block_idx < counts.len());

                debug_assert!(
                    block_idx <= last_block_idx,
                    "{}, {}",
                    block_idx,
                    last_block_idx
                );

                debug_assert!(block_idx < last_block_idx);

                block_idx += counts[block_idx..last_block_idx].linear_partition_point(|i, x| {
                    ((block_idx + i) * Self::BLOCK_BIT_SIZE)
                        - (upper_rank_ones + x.absolute as usize)
                        <= rank
                }) - 1;

                let block_count = counts.get_unchecked(block_idx);
                let hint_pos = block_idx * Self::BLOCK_BIT_SIZE;
                let hint_rank = hint_pos - (upper_rank_ones + block_count.absolute as usize);

                self.complete_select(block_count, hint_pos, rank, hint_rank)
            }
        }

        impl<
                C: SmallCounters<$NUM_U32S, $COUNTER_WIDTH>
                    + AsRef<[usize]>
                    + BitLength
                    + NumBits
                    + SelectZeroHinted,
            > SelectZero for SelectZeroSmall<$NUM_U32S, $COUNTER_WIDTH, C>
        {
        }
    };
}

impl<C: SmallCounters<2, 9> + AsRef<[usize]> + BitLength + NumBits> SelectZeroSmall<2, 9, C> {
    #[inline(always)]
    unsafe fn complete_select(
        &self,
        block_count: &Block32Counters<2, 9>,
        mut hint_pos: usize,
        rank: usize,
        hint_rank: usize,
    ) -> usize {
        const ONES_STEP_9: u64 = (1_u64 << 0)
            | (1_u64 << 9)
            | (1_u64 << 18)
            | (1_u64 << 27)
            | (1_u64 << 36)
            | (1_u64 << 45)
            | (1_u64 << 54);
        const MSBS_STEP_9: u64 = 0x100_u64 * ONES_STEP_9;
        // We cannot put this const together with the rest because we need
        // to use it for definining POS_STEP_9.
        const SUBBLOCK_BIT_SIZE: u64 =
            (usize::BITS as u64) * RankSmall::<2, 9>::WORDS_PER_SUBBLOCK as u64;
        const POS_STEP_9: u64 = (SUBBLOCK_BIT_SIZE << (6 * 9))
            | ((2 * SUBBLOCK_BIT_SIZE) << (5 * 9))
            | ((3 * SUBBLOCK_BIT_SIZE) << (4 * 9))
            | ((4 * SUBBLOCK_BIT_SIZE) << (3 * 9))
            | ((5 * SUBBLOCK_BIT_SIZE) << (2 * 9))
            | ((6 * SUBBLOCK_BIT_SIZE) << 9)
            | (7 * SUBBLOCK_BIT_SIZE);

        macro_rules! ULEQ_STEP_9 {
            ($x:ident, $y:ident) => {
                (((((($y) | MSBS_STEP_9) - (($x) & !MSBS_STEP_9)) | ($x ^ $y)) ^ ($x & !$y))
                    & MSBS_STEP_9)
            };
        }

        let rank_in_block = rank - hint_rank;
        let rank_in_block_step_9 = rank_in_block as u64 * ONES_STEP_9;
        let relative = POS_STEP_9 - block_count.all_rel();
        let offset_in_block = ULEQ_STEP_9!(relative, rank_in_block_step_9).count_ones() as usize;

        let rank_in_word = rank_in_block
            - (offset_in_block * (SUBBLOCK_BIT_SIZE as usize) - block_count.rel(offset_in_block));
        hint_pos += offset_in_block * (SUBBLOCK_BIT_SIZE as usize);

        hint_pos
            + (!self
                .small_counters
                .as_ref()
                .get_unchecked(hint_pos / usize::BITS as usize))
            .select_in_word(rank_in_word)
    }
}

impl<C: SmallCounters<1, 9> + AsRef<[usize]> + BitLength + NumBits + SelectZeroHinted>
    SelectZeroSmall<1, 9, C>
{
    #[inline(always)]
    unsafe fn complete_select(
        &self,
        block_count: &Block32Counters<1, 9>,
        mut hint_pos: usize,
        rank: usize,
        mut hint_rank: usize,
    ) -> usize {
        const ONES_STEP_9: u64 = (1_u64 << 0) | (1_u64 << 9) | (1_u64 << 18);
        const MSBS_STEP_9: u64 = 0x100_u64 * ONES_STEP_9;
        const SUBBLOCK_BIT_SIZE: u64 =
            (usize::BITS as u64) * RankSmall::<1, 9>::WORDS_PER_SUBBLOCK as u64;
        // We cannot put this const together with the rest because we need
        // to use it for definining POS_STEP_9.
        const POS_STEP_9: u64 =
            (SUBBLOCK_BIT_SIZE << 18) | ((2 * SUBBLOCK_BIT_SIZE) << 9) | (3 * SUBBLOCK_BIT_SIZE);

        macro_rules! ULEQ_STEP_9 {
            ($x:ident, $y:ident) => {
                (((((($y) | MSBS_STEP_9) - (($x) & !MSBS_STEP_9)) | ($x ^ $y)) ^ ($x & !$y))
                    & MSBS_STEP_9)
            };
        }

        let rank_in_block = rank - hint_rank;
        let rank_in_block_step_9 = rank_in_block as u64 * ONES_STEP_9;
        let relative = POS_STEP_9 - block_count.all_rel();

        let offset_in_block = ULEQ_STEP_9!(relative, rank_in_block_step_9).count_ones() as usize;

        hint_pos += offset_in_block * (SUBBLOCK_BIT_SIZE as usize);
        hint_rank +=
            offset_in_block * (SUBBLOCK_BIT_SIZE as usize) - block_count.rel(offset_in_block);

        self.select_zero_hinted(rank, hint_pos, hint_rank)
    }
}

impl<C: SmallCounters<1, 10> + AsRef<[usize]> + BitLength + NumBits + SelectZeroHinted>
    SelectZeroSmall<1, 10, C>
{
    #[inline(always)]
    unsafe fn complete_select(
        &self,
        block_count: &Block32Counters<1, 10>,
        mut hint_pos: usize,
        rank: usize,
        mut hint_rank: usize,
    ) -> usize {
        const ONES_STEP_10: u64 = (1_u64 << 0) | (1_u64 << 10) | (1_u64 << 20);
        const MSBS_STEP_10: u64 = 0x200_u64 * ONES_STEP_10;
        // We cannot put this const together with the rest because we need
        // to use it for definining POS_STEP_10.
        const SUBBLOCK_BIT_SIZE: u64 =
            (usize::BITS as u64) * RankSmall::<1, 10>::WORDS_PER_SUBBLOCK as u64;
        const POS_STEP_10: u64 =
            (SUBBLOCK_BIT_SIZE << 20) | ((2 * SUBBLOCK_BIT_SIZE) << 10) | (3 * SUBBLOCK_BIT_SIZE);

        macro_rules! ULEQ_STEP_10 {
            ($x:ident, $y:ident) => {
                (((((($y) | MSBS_STEP_10) - (($x) & !MSBS_STEP_10)) | ($x ^ $y)) ^ ($x & !$y))
                    & MSBS_STEP_10)
            };
        }

        let rank_in_block = rank - hint_rank;
        let rank_in_block_step_10 = rank_in_block as u64 * ONES_STEP_10;
        let relative = POS_STEP_10 - block_count.all_rel();

        let offset_in_block = ULEQ_STEP_10!(relative, rank_in_block_step_10).count_ones() as usize;

        hint_pos += offset_in_block * (SUBBLOCK_BIT_SIZE as usize);
        hint_rank +=
            offset_in_block * (SUBBLOCK_BIT_SIZE as usize) - block_count.rel(offset_in_block);

        self.select_zero_hinted(rank, hint_pos, hint_rank)
    }
}

impl<C: SmallCounters<1, 11> + AsRef<[usize]> + BitLength + NumBits + SelectZeroHinted>
    SelectZeroSmall<1, 11, C>
{
    #[inline(always)]
    unsafe fn complete_select(
        &self,
        block_count: &Block32Counters<1, 11>,
        mut hint_pos: usize,
        rank: usize,
        mut hint_rank: usize,
    ) -> usize {
        const ONES_STEP_11: u64 = (1_u64 << 0) | (1_u64 << 11) | (1_u64 << 22);
        const MSBS_STEP_11: u64 = 0x400_u64 * ONES_STEP_11;
        // We cannot put this const together with the rest because we need
        // to use it for definining POS_STEP_11.
        const SUBBLOCK_BIT_SIZE: u64 =
            (usize::BITS as u64) * RankSmall::<1, 11>::WORDS_PER_SUBBLOCK as u64;
        const POS_STEP_11: u64 =
            (SUBBLOCK_BIT_SIZE << 22) | ((2 * SUBBLOCK_BIT_SIZE) << 11) | (3 * SUBBLOCK_BIT_SIZE);

        macro_rules! ULEQ_STEP_11 {
            ($x:ident, $y:ident) => {
                (((((($y) | MSBS_STEP_11) - (($x) & !MSBS_STEP_11)) | ($x ^ $y)) ^ ($x & !$y))
                    & MSBS_STEP_11)
            };
        }

        let rank_in_block = rank - hint_rank;
        let rank_in_block_step_11 = rank_in_block as u64 * ONES_STEP_11;
        let relative = POS_STEP_11 - block_count.all_rel();

        let offset_in_block = ULEQ_STEP_11!(relative, rank_in_block_step_11).count_ones() as usize;

        hint_pos += offset_in_block * (SUBBLOCK_BIT_SIZE as usize);
        hint_rank +=
            offset_in_block * (SUBBLOCK_BIT_SIZE as usize) - block_count.rel(offset_in_block);

        self.select_zero_hinted(rank, hint_pos, hint_rank)
    }
}

impl<C: SmallCounters<3, 13> + AsRef<[usize]> + BitLength + NumBits + SelectZeroHinted>
    SelectZeroSmall<3, 13, C>
{
    unsafe fn complete_select(
        &self,
        block_count: &Block32Counters<3, 13>,
        mut hint_pos: usize,
        rank: usize,
        mut hint_rank: usize,
    ) -> usize {
        const ONES_STEP_13: u128 = (1_u128 << 0)
            | (1_u128 << 13)
            | (1_u128 << 26)
            | (1_u128 << 39)
            | (1_u128 << 52)
            | (1_u128 << 65)
            | (1_u128 << 78);
        const MSBS_STEP_13: u128 = 0x1000_u128 * ONES_STEP_13;
        // We cannot put this const together with the rest because we need
        // to use it for definining POS_STEP_13.
        const SUBBLOCK_BIT_SIZE: u64 =
            (usize::BITS as u64) * RankSmall::<3, 13>::WORDS_PER_SUBBLOCK as u64;
        const POS_STEP_13: u128 = ((SUBBLOCK_BIT_SIZE as u128) << 78)
            | ((2 * (SUBBLOCK_BIT_SIZE as u128)) << 65)
            | ((3 * (SUBBLOCK_BIT_SIZE as u128)) << 52)
            | ((4 * (SUBBLOCK_BIT_SIZE as u128)) << 39)
            | ((5 * (SUBBLOCK_BIT_SIZE as u128)) << 26)
            | ((6 * (SUBBLOCK_BIT_SIZE as u128)) << 13)
            | (7 * (SUBBLOCK_BIT_SIZE as u128));

        macro_rules! ULEQ_STEP_13 {
            ($x:ident, $y:ident) => {
                (((((($y) | MSBS_STEP_13) - (($x) & !MSBS_STEP_13)) | ($x ^ $y)) ^ ($x & !$y))
                    & MSBS_STEP_13)
            };
        }

        let rank_in_block = rank - hint_rank;
        let rank_in_block_step_13 = rank_in_block as u128 * ONES_STEP_13;
        let relative = POS_STEP_13 - block_count.all_rel();

        let offset_in_block = ULEQ_STEP_13!(relative, rank_in_block_step_13).count_ones() as usize;

        hint_pos += offset_in_block * (SUBBLOCK_BIT_SIZE as usize);
        hint_rank +=
            offset_in_block * (SUBBLOCK_BIT_SIZE as usize) - block_count.rel(offset_in_block);

        self.select_zero_hinted(rank, hint_pos, hint_rank)
    }
}

impl_select_zero_small!(2; 9);
impl_select_zero_small!(1; 9);
impl_select_zero_small!(1; 10);
impl_select_zero_small!(1; 11);
impl_select_zero_small!(3; 13);

/// A trait providing the semantics of
/// [`partition_point`](std::slice::partition_point), but using a linear search.
trait LinearPartitionPointExt<T>: AsRef<[T]> {
    fn linear_partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(usize, &T) -> bool,
    {
        let as_ref = self.as_ref();
        as_ref
            .iter()
            .enumerate()
            .position(|(i, x)| !pred(i, x))
            .unwrap_or(as_ref.len())
    }
}

impl<T> LinearPartitionPointExt<T> for [T] {}
