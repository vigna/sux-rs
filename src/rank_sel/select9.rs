/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use super::rank9::BlockCounters;
use super::Rank9;
use crate::{
    prelude::SelectUnchecked,
    traits::{BitLength, NumBits, Select},
};
use ambassador::Delegate;
use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

const ONES_STEP_9: usize = (1usize << 0)
    | (1usize << 9)
    | (1usize << 18)
    | (1usize << 27)
    | (1usize << 36)
    | (1usize << 45)
    | (1usize << 54);

const MSBS_STEP_9: usize = 0x100usize * ONES_STEP_9;

const ONES_STEP_16: usize = (1usize << 0) | (1usize << 16) | (1usize << 32) | (1usize << 48);
const MSBS_STEP_16: usize = 0x8000usize * ONES_STEP_16;

macro_rules! ULEQ_STEP_9 {
    ($x:ident, $y:ident) => {
        (((((($y) | MSBS_STEP_9) - (($x) & !MSBS_STEP_9)) | ($x ^ $y)) ^ ($x & !$y)) & MSBS_STEP_9)
    };
}

macro_rules! ULEQ_STEP_16 {
    ($x:ident, $y:ident) => {
        (((((($y) | MSBS_STEP_16) - (($x) & !MSBS_STEP_16)) | ($x ^ $y)) ^ ($x & !$y))
            & MSBS_STEP_16)
    };
}

use crate::ambassador_impl_AsRef;
use crate::ambassador_impl_Index;
use crate::traits::rank_sel::ambassador_impl_BitCount;
use crate::traits::rank_sel::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_NumBits;
use crate::traits::rank_sel::ambassador_impl_Rank;
use crate::traits::rank_sel::ambassador_impl_RankHinted;
use crate::traits::rank_sel::ambassador_impl_RankUnchecked;
use crate::traits::rank_sel::ambassador_impl_RankZero;
use crate::traits::rank_sel::ambassador_impl_SelectHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZero;
use crate::traits::rank_sel::ambassador_impl_SelectZeroHinted;
use crate::traits::rank_sel::ambassador_impl_SelectZeroUnchecked;
use std::ops::Index;

/// A selection structure over [`Rank9`] using 25%–37.5% additional space and
/// providing constant-time selection.
///
/// [`Select9`] uses an absolute inventory and a relative subinventory to locate
/// the [`Rank9`] block containing the desired bit, and then perform broadword
/// operations using the [`Rank9`] counters.
///
/// Note that the additional space is in addition to [`Rank9`], so the overall
/// cost of the selection structure is 50%–62.5% of the original bit vector. Due
/// to the large space, unless the bit vector has a pathologically irregular bit
/// distribution [`SelectAdapt`](super::SelectAdapt) is usually a better choice.
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
/// use sux::bit_vec;
/// use sux::prelude::{Rank, Rank9, Select, Select9};
/// // A Select9 structure is built on a Rank9 structure
/// let select9 = Select9::new(Rank9::new(bit_vec![1, 0, 1, 1, 0, 1, 0, 1]));
///
/// assert_eq!(select9.select(0), Some(0));
/// assert_eq!(select9.select(1), Some(2));
/// assert_eq!(select9.select(2), Some(3));
/// assert_eq!(select9.select(3), Some(5));
/// assert_eq!(select9.select(4), Some(7));
/// assert_eq!(select9.select(5), None);
///
/// // Rank methods are forwarded
/// assert_eq!(select9.rank(0), 0);
/// assert_eq!(select9.rank(1), 1);
/// assert_eq!(select9.rank(2), 1);
/// assert_eq!(select9.rank(3), 2);
/// assert_eq!(select9.rank(4), 3);
/// assert_eq!(select9.rank(5), 3);
/// assert_eq!(select9.rank(6), 4);
/// assert_eq!(select9.rank(7), 4);
/// assert_eq!(select9.rank(8), 5);
///
/// // Access to the underlying bit vector is forwarded, too
/// assert_eq!(select9[0], true);
/// assert_eq!(select9[1], false);
/// assert_eq!(select9[2], true);
/// assert_eq!(select9[3], true);
/// assert_eq!(select9[4], false);
/// assert_eq!(select9[5], true);
/// assert_eq!(select9[6], false);
/// assert_eq!(select9[7], true);
/// ```

#[derive(Epserde, Debug, Clone, MemDbg, MemSize, Delegate)]
#[delegate(AsRef<[usize]>, target = "rank9")]
#[delegate(Index<usize>, target = "rank9")]
#[delegate(crate::traits::rank_sel::BitCount, target = "rank9")]
#[delegate(crate::traits::rank_sel::BitLength, target = "rank9")]
#[delegate(crate::traits::rank_sel::NumBits, target = "rank9")]
#[delegate(crate::traits::rank_sel::Rank, target = "rank9")]
#[delegate(crate::traits::rank_sel::RankHinted<64>, target = "rank9")]
#[delegate(crate::traits::rank_sel::RankUnchecked, target = "rank9")]
#[delegate(crate::traits::rank_sel::RankZero, target = "rank9")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "rank9")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "rank9")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "rank9")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "rank9")]
pub struct Select9<R = Rank9, I = Box<[usize]>> {
    rank9: R,
    inventory: I,
    subinventory: I,
    inventory_size: usize,
    subinventory_size: usize,
}

impl<R, I> Select9<R, I> {
    pub fn into_inner(self) -> R {
        self.rank9
    }

    const LOG2_ZEROS_PER_INVENTORY: usize = 9;
    const ONES_PER_INVENTORY: usize = 1 << Self::LOG2_ZEROS_PER_INVENTORY;
}

impl<R: BitLength, I> Select9<R, I> {
    /// Returns the number of bits in the underlying bit vector.
    ///
    /// This method is equivalent to
    /// [`BitLength::len`](crate::traits::BitLength::len), but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>> Select9<Rank9<B, C>, Box<[usize]>> {
    pub fn new(rank9: Rank9<B, C>) -> Self {
        let num_bits = rank9.len();
        let num_words = (num_bits + 63) / 64;
        let inventory_size = rank9.num_ones().div_ceil(Self::ONES_PER_INVENTORY);

        let u64_per_subinventory = 4;
        let subinventory_size = num_words.div_ceil(u64_per_subinventory);

        let mut inventory = Vec::with_capacity(inventory_size + 1);
        let mut subinventory = vec![0; subinventory_size].into_boxed_slice();

        // construct the inventory
        let mut curr_num_ones = 0;
        let mut next_quantum = 0;
        for (i, word) in rank9.bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as usize;

            while curr_num_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word(next_quantum - curr_num_ones);
                let index = (i * u64::BITS as usize) + in_word_index;

                inventory.push(index);

                next_quantum += Self::ONES_PER_INVENTORY;
            }
            curr_num_ones += ones_in_word;
        }
        inventory.push(((num_words + 3) & !3) * 64);
        assert!(inventory.len() == inventory_size + 1);
        let inventory = inventory.into_boxed_slice();

        let iter = 0..inventory_size;
        let counts = rank9.counts.as_ref();

        // construct the subinventory
        iter.for_each(|inventory_idx| {
            let subinv_start = (inventory[inventory_idx] / 64) / u64_per_subinventory;
            let subinv_end = (inventory[inventory_idx + 1] / 64) / u64_per_subinventory;
            let span = subinv_end - subinv_start;
            let block_left = (inventory[inventory_idx] / 64) / 8;
            let block_span = (inventory[inventory_idx + 1] / 64) / 8 - block_left;
            let counts_at_start = counts[block_left].absolute;

            let mut state = -1;
            let s16: &mut [u16] =
                unsafe { subinventory[subinv_start..subinv_end].align_to_mut().1 };
            match span {
                0..=1 => {}
                2..=15 => {
                    debug_assert!(((block_span + 8) & !7) <= span * 4);
                    for (k, v) in s16.iter_mut().enumerate().take(block_span) {
                        debug_assert!(*v == 0);
                        *v = (counts[block_left + k + 1].absolute - counts_at_start) as u16;
                    }
                    for v in s16.iter_mut().take((block_span + 8) & !7).skip(block_span) {
                        debug_assert!(*v == 0);
                        *v = 0xFFFFu16;
                    }
                }
                16..=127 => {
                    debug_assert!(((block_span + 8) & !7) + 8 <= span * 4);
                    debug_assert!(block_span / 8 <= 8);
                    for k in 0..block_span {
                        debug_assert!(s16[k + 8] == 0);
                        s16[k + 8] = (counts[block_left + k + 1].absolute - counts_at_start) as u16;
                    }
                    for k in block_span..((block_span + 8) & !7) {
                        debug_assert!(s16[k + 8] == 0);
                        s16[k + 8] = 0xFFFFu16;
                    }
                    for (k, v) in s16.iter_mut().enumerate().take(block_span / 8) {
                        debug_assert!(*v == 0);
                        *v = (counts[block_left + (k + 1) * 8].absolute - counts_at_start) as u16;
                    }
                    for v in s16.iter_mut().take(8).skip(block_span / 8) {
                        debug_assert!(*v == 0);
                        *v = 0xFFFFu16;
                    }
                }
                128..=255 => {
                    state = 2;
                }
                256..=511 => {
                    state = 1;
                }
                _ => {
                    state = 0;
                }
            }

            if state != -1 {
                // clean up the lower bits
                let mut word_idx = inventory[inventory_idx] / usize::BITS as usize;
                let bit_idx = inventory[inventory_idx] % usize::BITS as usize;
                let mut word = (rank9.bits.as_ref()[word_idx] >> bit_idx) << bit_idx;

                let start_bit_idx = inventory[inventory_idx];
                let end_bit_idx = inventory[inventory_idx + 1];
                let end_word_idx = end_bit_idx.div_ceil(u64::BITS as usize);
                let mut subinventory_idx = 0;
                'outer: loop {
                    while word != 0 {
                        let in_word_index = word.trailing_zeros() as usize;
                        let bit_index = (word_idx * u64::BITS as usize) + in_word_index;
                        let sub_offset = bit_index - start_bit_idx;
                        match state {
                            0 => {
                                debug_assert!(subinventory[subinv_start + subinventory_idx] == 0);
                                subinventory[subinv_start + subinventory_idx] = bit_index;
                            }
                            1 => {
                                let s32: &mut [u32] = unsafe {
                                    subinventory[subinv_start..subinv_end].align_to_mut().1
                                };
                                debug_assert!(s32[subinventory_idx] == 0);
                                debug_assert!((bit_index - start_bit_idx) < (1 << 32));
                                s32[subinventory_idx] = sub_offset as u32;
                            }
                            2 => {
                                let s16: &mut [u16] = unsafe {
                                    subinventory[subinv_start..subinv_end].align_to_mut().1
                                };
                                debug_assert!(s16[subinventory_idx] == 0);
                                debug_assert!(bit_index - start_bit_idx < (1 << 16));
                                s16[subinventory_idx] = (bit_index - start_bit_idx) as u16;
                            }
                            _ => unreachable!(),
                        }

                        subinventory_idx += 1;
                        if subinventory_idx == Self::ONES_PER_INVENTORY {
                            break 'outer;
                        }

                        word &= word - 1;
                    }

                    // move to the next word and boundcheck
                    word_idx += 1;
                    if word_idx == end_word_idx {
                        break;
                    }

                    // read the next word
                    word = rank9.bits.as_ref()[word_idx];
                }
            }
        });

        Self {
            rank9,
            inventory,
            subinventory,
            inventory_size,
            subinventory_size,
        }
    }
}

impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>, I: AsRef<[usize]>> SelectUnchecked
    for Select9<Rank9<B, C>, I>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let inventory_index_left = rank >> Self::LOG2_ZEROS_PER_INVENTORY;

        debug_assert!(inventory_index_left <= self.inventory_size);
        let inventory_left = *self.inventory.as_ref().get_unchecked(inventory_index_left);

        let block_right = (*self
            .inventory
            .as_ref()
            .get_unchecked(inventory_index_left + 1))
            / 64;
        let mut block_left = inventory_left / 64;
        let span = block_right / 4 - block_left / 4;

        let subinv_pos = block_left / 4;
        let subinv_ref = self.subinventory.as_ref();

        let counts = self.rank9.counts.as_ref();

        let mut count_left;
        let rank_in_block;

        match span {
            0..=1 => {
                block_left &= !7;
                count_left = block_left / Rank9::<B, C>::WORDS_PER_BLOCK;

                debug_assert!(rank < counts.get_unchecked(count_left + 1).absolute);
                rank_in_block = rank - counts.get_unchecked(count_left).absolute;
            }
            2..=15 => {
                block_left &= !7;
                count_left = block_left / Rank9::<B, C>::WORDS_PER_BLOCK;
                let rank_in_superblock = rank - counts.get_unchecked(count_left).absolute;

                let rank_in_superblock_step_16 = rank_in_superblock * ONES_STEP_16;

                let first = *subinv_ref.get_unchecked(subinv_pos);
                let second = *subinv_ref.get_unchecked(subinv_pos + 1);

                let where_: usize = (ULEQ_STEP_16!(first, rank_in_superblock_step_16).count_ones()
                    as usize
                    + ULEQ_STEP_16!(second, rank_in_superblock_step_16).count_ones() as usize)
                    * 2;

                debug_assert!(where_ <= 16);

                block_left += where_ * 4;
                count_left += where_ / 2;

                rank_in_block = rank - counts.get_unchecked(count_left).absolute;
                debug_assert!(rank_in_block < 512);
            }
            16..=127 => {
                block_left &= !7;
                count_left = block_left / Rank9::<B, C>::WORDS_PER_BLOCK;
                let rank_in_superblock = rank - counts.get_unchecked(count_left).absolute;
                let rank_in_superblock_step_16 = rank_in_superblock * ONES_STEP_16;

                let first = *subinv_ref.get_unchecked(subinv_pos);
                let second = *subinv_ref.get_unchecked(subinv_pos + 1);

                let where0 = (ULEQ_STEP_16!(first, rank_in_superblock_step_16).count_ones()
                    as usize
                    + ULEQ_STEP_16!(second, rank_in_superblock_step_16).count_ones() as usize)
                    * 2;

                debug_assert!(where0 <= 16);

                let first_bis = *self
                    .subinventory
                    .as_ref()
                    .get_unchecked(subinv_pos + where0 + 2);
                let second_bis = *self
                    .subinventory
                    .as_ref()
                    .get_unchecked(subinv_pos + where0 + 2 + 1);

                let where1 = where0 * 8
                    + (ULEQ_STEP_16!(first_bis, rank_in_superblock_step_16).count_ones() as usize
                        + ULEQ_STEP_16!(second_bis, rank_in_superblock_step_16).count_ones()
                            as usize)
                        * 2;

                block_left += where1 * 4;
                count_left += where1 / 2;
                rank_in_block = rank - counts.get_unchecked(count_left).absolute;

                debug_assert!(rank_in_block < 512);
            }
            128..=255 => {
                let (_, s, _) = subinv_ref
                    .get_unchecked(subinv_pos..self.subinventory_size)
                    .align_to::<u16>();
                return *s.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize + inventory_left;
            }
            256..=511 => {
                let (_, s, _) = subinv_ref
                    .get_unchecked(subinv_pos..self.subinventory_size)
                    .align_to::<u32>();
                return *s.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize + inventory_left;
            }
            _ => {
                return *subinv_ref.get_unchecked(rank % Self::ONES_PER_INVENTORY);
            }
        }

        let rank_in_block_step_9 = rank_in_block * ONES_STEP_9;
        let relative = counts.get_unchecked(count_left).relative;

        let offset_in_block = ULEQ_STEP_9!(relative, rank_in_block_step_9).count_ones() as usize;
        debug_assert!(offset_in_block <= 7);

        let word = block_left + offset_in_block;
        let rank_in_word = rank_in_block - counts.get_unchecked(count_left).rel(offset_in_block);

        word * 64
            + self
                .rank9
                .bits
                .as_ref()
                .get_unchecked(word)
                .select_in_word(rank_in_word)
    }
}

impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>, I: AsRef<[usize]>> Select
    for Select9<Rank9<B, C>, I>
{
}
