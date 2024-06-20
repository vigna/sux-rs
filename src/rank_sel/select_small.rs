/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use ambassador::Delegate;
use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

crate::forward_mult![
    SelectSmall<[const] NUM_U32S: usize, [const] COUNTER_WIDTH: usize, [const] LOG2_ONES_PER_INVENTORY: usize, R, I>; R; rank_small;
    crate::forward_as_ref_slice_usize,
    crate::forward_index_bool,
    crate::traits::forward_rank_hinted
];

use crate::traits::rank_sel::ambassador_impl_BitCount;
use crate::traits::rank_sel::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_NumBits;
use crate::traits::rank_sel::ambassador_impl_Rank;
use crate::traits::rank_sel::ambassador_impl_RankZero;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZero;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZeroUnchecked;

/// A selection structure over [`RankSmall`] using negligible additional space
/// and providing constant-time selection.
///
/// [`SelectSmall`] adds a very sparse first-level inventory to a [`RankSmall`]
/// structure locates approximately the position of the desired one; the bit is
/// then located using binary searches over [`RankSmall`]'s counters; this
/// technique is called _hinted bsearch_ and is described in Sebastiano Vigna in
/// “[Broadword Implementation of Rank/Select
/// Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
/// _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
/// 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
/// Springer, 2008.
///
///
/// The resulting selection methods are very slow, and in general it is
/// convenient and faster to use [`SelectAdapt`], even with `M` set to 1 (in
/// which case the additional space is 1.5-3% of the original bit vector).
///
/// # Examples
///
/// ```rust
/// use sux::{rank_small, bit_vec};
/// use sux::rank_sel::SelectSmall;
/// use sux::traits::{Rank, Select};
///
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![1; bits];
/// let rank_sel_small = SelectSmall::<1, 9>::new(rank_small);
///
/// assert_eq!(rank_sel_small.select(0), Some(0));
/// assert_eq!(rank_sel_small.select(1), Some(2));
/// assert_eq!(rank_sel_small.select(2), Some(3));
/// assert_eq!(rank_sel_small.select(3), Some(5));
/// assert_eq!(rank_sel_small.select(4), Some(7));
/// assert_eq!(rank_sel_small.select(5), None);
///
/// // Rank methods are forwarded
/// assert_eq!(rank_sel_small.rank(0), 0);
/// assert_eq!(rank_sel_small.rank(1), 1);
/// assert_eq!(rank_sel_small.rank(2), 1);
/// assert_eq!(rank_sel_small.rank(3), 2);
/// assert_eq!(rank_sel_small.rank(4), 3);
/// assert_eq!(rank_sel_small.rank(5), 3);
/// assert_eq!(rank_sel_small.rank(6), 4);
/// assert_eq!(rank_sel_small.rank(7), 4);
/// assert_eq!(rank_sel_small.rank(8), 5);
///
/// // Access to the underlying bit vector is forwarded, too
/// assert_eq!(rank_sel_small[0], true);
/// assert_eq!(rank_sel_small[1], false);
/// assert_eq!(rank_sel_small[2], true);
/// assert_eq!(rank_sel_small[3], true);
/// assert_eq!(rank_sel_small[4], false);
/// assert_eq!(rank_sel_small[5], true);
/// assert_eq!(rank_sel_small[6], false);
/// assert_eq!(rank_sel_small[7], true);
/// ```

#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Delegate)]
#[delegate(crate::traits::rank_sel::BitCount, target = "rank_small")]
#[delegate(crate::traits::rank_sel::BitLength, target = "rank_small")]
#[delegate(crate::traits::rank_sel::NumBits, target = "rank_small")]
#[delegate(crate::traits::rank_sel::Rank, target = "rank_small")]
#[delegate(crate::traits::rank_sel::RankZero, target = "rank_small")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "rank_small")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "rank_small")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "rank_small")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "rank_small")]
pub struct SelectSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    const LOG2_ONES_PER_INVENTORY: usize = 12,
    R = RankSmall<NUM_U32S, COUNTER_WIDTH>,
    I = Box<[u32]>,
> {
    rank_small: R,
    inventory: I,
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        R,
        I,
    > SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, R, I>
{
    const SUPERBLOCK_SIZE: usize = 1 << 32;
    const WORDS_PER_BLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_BLOCK;
    const WORDS_PER_SUBBLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_SUBBLOCK;
    const BLOCK_SIZE: usize = (Self::WORDS_PER_BLOCK * usize::BITS as usize);
    const SUBBLOCK_SIZE: usize = (Self::WORDS_PER_SUBBLOCK * usize::BITS as usize);
    const ONES_PER_INVENTORY: usize = 1 << LOG2_ONES_PER_INVENTORY;

    pub fn into_inner(self) -> R {
        self.rank_small
    }
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        R: BitLength,
        I,
    > SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, R, I>
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

macro_rules! impl_rank_small_sel {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        impl<
                const LOG2_ONES_PER_INVENTORY: usize,
                B: AsRef<[usize]> + BitLength,
                C1: AsRef<[usize]>,
                C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            >
            SelectSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>,
            >
        {
            // Creates a new selection structure over a [`RankSmall`] structure.
            pub fn new(rank_small: RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>) -> Self {
                let num_ones = rank_small.num_ones();

                let inventory_size = num_ones.div_ceil(Self::ONES_PER_INVENTORY);
                let mut inventory = Vec::<u32>::with_capacity(inventory_size + 1);

                let mut past_ones: usize = 0;
                let mut next_quantum: usize = 0;

                for superblock in rank_small
                    .bits
                    .as_ref()
                    .chunks(Self::SUPERBLOCK_SIZE / usize::BITS as usize)
                {
                    for (i, word) in superblock.iter().copied().enumerate() {
                        let ones_in_word = word.count_ones() as usize;

                        while past_ones + ones_in_word > next_quantum {
                            let in_word_index = word.select_in_word(next_quantum - past_ones);
                            let in_superblock_index = i * usize::BITS as usize + in_word_index;
                            inventory.push(in_superblock_index as u32);
                            next_quantum += Self::ONES_PER_INVENTORY;
                        }

                        past_ones += ones_in_word;
                    }
                }
                assert_eq!(num_ones, past_ones);

                if inventory.is_empty() {
                    inventory.push(0);
                } else {
                    inventory.push(inventory[inventory.len() - 1]);
                }

                assert_eq!(inventory.len(), inventory_size + 1);

                let inventory = inventory.into_boxed_slice();

                Self {
                    rank_small,
                    inventory,
                }
            }
        }

        impl<
                const LOG2_ONES_PER_INVENTORY: usize,
                B: AsRef<[usize]> + BitLength + SelectHinted,
                C1: AsRef<[usize]>,
                C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            > SelectUnchecked
            for SelectSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>,
            >
        {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                let upper_counts = self.rank_small.upper_counts.as_ref();
                let counts = self.rank_small.counts.as_ref();
                let mut upper_block_idx = 0;
                let mut next_upper_block_idx;
                let mut last_upper_block_idx = self.rank_small.upper_counts.len() - 1;
                let mut upper_rank = *upper_counts.get_unchecked(upper_block_idx) as usize;
                loop {
                    if last_upper_block_idx - upper_block_idx <= 1 {
                        break;
                    }
                    next_upper_block_idx = (upper_block_idx + last_upper_block_idx) / 2;
                    upper_rank = *upper_counts.get_unchecked(next_upper_block_idx) as usize;
                    if rank >= upper_rank {
                        upper_block_idx = next_upper_block_idx;
                    } else {
                        last_upper_block_idx = next_upper_block_idx;
                    }
                }

                let inventory = self.inventory.as_ref();
                let rel_inv_pos =
                    *inventory.get_unchecked(rank / Self::ONES_PER_INVENTORY) as usize;
                let inv_pos = rel_inv_pos + upper_block_idx * (1 << 32);

                let next_rel_inv_pos =
                    *inventory.get_unchecked(rank / Self::ONES_PER_INVENTORY + 1) as usize;
                let next_inv_pos = match next_rel_inv_pos.cmp(&rel_inv_pos) {
                    std::cmp::Ordering::Greater => next_rel_inv_pos + upper_block_idx * (1 << 32),
                    std::cmp::Ordering::Less => (upper_block_idx + 1) * (1 << 32),
                    // the two last elements of the inventory are the same
                    // because at construction time we add the last element twice
                    std::cmp::Ordering::Equal => self.rank_small.bits.len(),
                };
                let mut last_block_idx = next_inv_pos / Self::BLOCK_SIZE;

                let jump = (rank % Self::ONES_PER_INVENTORY) / Self::BLOCK_SIZE;
                let mut block_idx = inv_pos / Self::BLOCK_SIZE + jump;

                let mut hint_rank = upper_rank + counts.get_unchecked(block_idx).absolute as usize;
                let mut next_rank;
                let mut next_block_idx;

                debug_assert!(block_idx <= last_block_idx);

                loop {
                    if last_block_idx - block_idx <= 1 {
                        break;
                    }
                    next_block_idx = (block_idx + last_block_idx) / 2;
                    next_rank = upper_rank + counts.get_unchecked(next_block_idx).absolute as usize;
                    if rank >= next_rank {
                        block_idx = next_block_idx;
                        hint_rank = next_rank;
                    } else {
                        last_block_idx = next_block_idx;
                    }
                }

                let hint_pos;
                // second sub block
                let b1 = counts.get_unchecked(block_idx).rel(1);
                if hint_rank + b1 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE;
                    return self
                        .rank_small
                        .bits
                        .select_hinted(rank, hint_pos, hint_rank);
                }
                // third sub block
                let b2 = counts.get_unchecked(block_idx).rel(2);
                if hint_rank + b2 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE + Self::SUBBLOCK_SIZE;
                    return self
                        .rank_small
                        .bits
                        .select_hinted(rank, hint_pos, hint_rank + b1);
                }
                // fourth sub block
                let b3 = counts.get_unchecked(block_idx).rel(3);
                if hint_rank + b3 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE + 2 * Self::SUBBLOCK_SIZE;
                    return self
                        .rank_small
                        .bits
                        .select_hinted(rank, hint_pos, hint_rank + b2);
                }

                hint_pos = block_idx * Self::BLOCK_SIZE + 3 * Self::SUBBLOCK_SIZE;
                self.rank_small
                    .bits
                    .select_hinted(rank, hint_pos, hint_rank + b3)
            }
        }

        impl<
                const LOG2_ONES_PER_INVENTORY: usize,
                B: AsRef<[usize]> + BitLength + SelectHinted,
                C1: AsRef<[usize]>,
                C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            > Select
            for SelectSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>,
            >
        {
        }
    };
}

impl_rank_small_sel!(2; 9);
impl_rank_small_sel!(1; 9);
impl_rank_small_sel!(1; 10);
impl_rank_small_sel!(1; 11);
impl_rank_small_sel!(3; 13);
