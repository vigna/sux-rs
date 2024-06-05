/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::ops::Index;

use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

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
/// convenient and faster to use [`SimpleSelect`], even with `M` set to 1 (in
/// which case the additional space is 1.5-3% of the original bit vector).
///
/// # Examples
///
/// ```rust
/// use sux::{rank_small, bit_vec};
/// use sux::rank_sel::SelectSmall;
///
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let rank_small = rank_small![1; bits];
/// let select_small = SelectSmall::new(rank_small);
///
/// assert_eq!(select_small.select(0), Some(0));
/// assert_eq!(select_small.select(1), Some(2));
/// assert_eq!(select_small.select(2), Some(3));
/// assert_eq!(select_small.select(3), Some(5));
/// assert_eq!(select_small.select(4), Some(7));
/// assert_eq!(select_small.select(5), None);
/// ```

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SelectSmall<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    const LOG2_ONES_PER_INVENTORY: usize = 12,
    R = RankSmall<NUM_U32S, COUNTER_WIDTH>,
    I = Vec<u32>,
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
    const WORDS_PER_BLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_BLOCK;
    const WORDS_PER_SUBBLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_SUBBLOCK;
    const BLOCK_SIZE: usize = (Self::WORDS_PER_BLOCK * usize::BITS as usize);
    const SUBBLOCK_SIZE: usize = (Self::WORDS_PER_SUBBLOCK * usize::BITS as usize);
    const ONES_PER_INVENTORY: usize = 1 << LOG2_ONES_PER_INVENTORY;
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B,
        I,
    > SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    pub fn into_inner(self) -> B {
        self.rank_small
    }
}

macro_rules! impl_rank_small_sel {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        impl<
                const LOG2_ONES_PER_INVENTORY: usize,
                B: RankHinted<64> + BitCount + AsRef<[usize]> + BitLength,
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
            pub fn new(rank_small: RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>) -> Self {
                let num_ones = rank_small.bits.count_ones();

                let inventory_size = num_ones.div_ceil(Self::ONES_PER_INVENTORY);
                let mut inventory = Vec::<u32>::with_capacity(inventory_size + 1);

                let mut curr_num_ones: usize = 0;
                let mut next_quantum: usize = 0;
                let mut upper_counts_idx;

                for (i, word) in rank_small.bits.as_ref().iter().copied().enumerate() {
                    let ones_in_word = word.count_ones() as usize;

                    upper_counts_idx = i / (1 << 26);

                    while curr_num_ones + ones_in_word > next_quantum {
                        let in_word_index = word.select_in_word(next_quantum - curr_num_ones);
                        let index = ((i * usize::BITS as usize) + in_word_index);

                        inventory.push(
                            (index - rank_small.upper_counts.as_ref()[upper_counts_idx]) as u32,
                        );

                        next_quantum += Self::ONES_PER_INVENTORY;
                    }
                    curr_num_ones += ones_in_word;
                }
                assert_eq!(num_ones, curr_num_ones);

                if !inventory.is_empty() {
                    inventory.push(inventory[inventory.len() - 1]);
                } else {
                    inventory.push(0);
                }

                Self {
                    rank_small,
                    inventory,
                }
            }
        }

        impl<const LOG2_ONES_PER_INVENTORY: usize> Select
            for SelectSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH>,
            >
        {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                let upper_counts_ref =
                    <Vec<_> as AsRef<[_]>>::as_ref(&self.rank_small.upper_counts);
                let counts_ref = <Vec<_> as AsRef<[_]>>::as_ref(&self.rank_small.counts);
                let mut upper_block_idx = 0;
                let mut next_upper_block_idx;
                let mut last_upper_block_idx = self.rank_small.upper_counts.len() - 1;
                let mut upper_rank = *upper_counts_ref.get_unchecked(upper_block_idx) as usize;
                loop {
                    if last_upper_block_idx - upper_block_idx <= 1 {
                        break;
                    }
                    next_upper_block_idx = (upper_block_idx + last_upper_block_idx) / 2;
                    upper_rank = *upper_counts_ref.get_unchecked(next_upper_block_idx) as usize;
                    if rank >= upper_rank {
                        upper_block_idx = next_upper_block_idx;
                    } else {
                        last_upper_block_idx = next_upper_block_idx;
                    }
                }

                let inv_ref = <Vec<u32> as AsRef<[u32]>>::as_ref(&self.inventory);
                let rel_inv_pos = *inv_ref.get_unchecked(rank / Self::ONES_PER_INVENTORY) as usize;
                let inv_pos = rel_inv_pos + upper_block_idx * (1 << 32);

                let next_rel_inv_pos =
                    *inv_ref.get_unchecked(rank / Self::ONES_PER_INVENTORY + 1) as usize;
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

                let mut hint_rank =
                    upper_rank + counts_ref.get_unchecked(block_idx).absolute as usize;
                let mut next_rank;
                let mut next_block_idx;

                debug_assert!(block_idx <= last_block_idx);

                loop {
                    if last_block_idx - block_idx <= 1 {
                        break;
                    }
                    next_block_idx = (block_idx + last_block_idx) / 2;
                    next_rank =
                        upper_rank + counts_ref.get_unchecked(next_block_idx).absolute as usize;
                    if rank >= next_rank {
                        block_idx = next_block_idx;
                        hint_rank = next_rank;
                    } else {
                        last_block_idx = next_block_idx;
                    }
                }

                let hint_pos;
                // second sub block
                let b1 = counts_ref.get_unchecked(block_idx).rel(1);
                if hint_rank + b1 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE;
                    return self
                        .rank_small
                        .bits
                        .select_hinted_unchecked(rank, hint_pos, hint_rank);
                }
                // third sub block
                let b2 = counts_ref.get_unchecked(block_idx).rel(2);
                if hint_rank + b2 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE + Self::SUBBLOCK_SIZE;
                    return self.rank_small.bits.select_hinted_unchecked(
                        rank,
                        hint_pos,
                        hint_rank + b1,
                    );
                }
                // fourth sub block
                let b3 = counts_ref.get_unchecked(block_idx).rel(3);
                if hint_rank + b3 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE + 2 * Self::SUBBLOCK_SIZE;
                    return self.rank_small.bits.select_hinted_unchecked(
                        rank,
                        hint_pos,
                        hint_rank + b2,
                    );
                }

                hint_pos = block_idx * Self::BLOCK_SIZE + 3 * Self::SUBBLOCK_SIZE;
                self.rank_small
                    .bits
                    .select_hinted_unchecked(rank, hint_pos, hint_rank + b3)
            }
        }

        /// Forward [`Rank`] to the underlying implementation.
        impl<const LOG2_ONES_PER_INVENTORY: usize> Rank
            for SelectSmall<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH>,
            >
        {
            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> usize {
                self.rank_small.rank_unchecked(pos)
            }

            #[inline(always)]
            fn rank(&self, pos: usize) -> usize {
                self.rank_small.rank(pos)
            }
        }
    };
}

impl_rank_small_sel!(2; 9);
impl_rank_small_sel!(1; 9);
impl_rank_small_sel!(1; 10);
impl_rank_small_sel!(1; 11);
impl_rank_small_sel!(3; 13);

/// Forward [`BitCount`] to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: BitCount,
        I,
    > BitCount for SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.rank_small.count_ones()
    }
    #[inline(always)]
    fn count_zeros(&self) -> usize {
        self.rank_small.count_zeros()
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: BitLength,
        I,
    > BitLength for SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.rank_small.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: AsRef<[usize]>,
        I,
    > AsRef<[usize]> for SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.rank_small.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: Index<usize, Output = bool>,
        I,
    > Index<usize> for SelectSmall<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    type Output = bool;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.rank_small[index]
    }
}
