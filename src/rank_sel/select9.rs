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
use common_traits::{SelectInWord, Sequence};
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

const ONES_STEP_9: usize = 1usize << 0
    | 1usize << 9
    | 1usize << 18
    | 1usize << 27
    | 1usize << 36
    | 1usize << 45
    | 1usize << 54;

const MSBS_STEP_9: usize = 0x100usize * ONES_STEP_9;

const ONES_STEP_16: usize = 1usize << 0 | 1usize << 16 | 1usize << 32 | 1usize << 48;
const MSBS_STEP_16: usize = 0x8000usize * ONES_STEP_16;

macro_rules! ULEQ_STEP_9 {
    ($x:ident, $y:ident) => {
        ((((((($y) | MSBS_STEP_9) - (($x) & !MSBS_STEP_9)) | ($x ^ $y)) ^ ($x & !$y))
            & MSBS_STEP_9)
            >> 8)
    };
}

macro_rules! ULEQ_STEP_16 {
    ($x:ident, $y:ident) => {
        ((((((($y) | MSBS_STEP_16) - (($x) & !MSBS_STEP_16)) | ($x ^ $y)) ^ ($x & !$y))
            & MSBS_STEP_16)
            >> 15)
    };
}

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
/// distribution [`SimpleSelect`](super::SimpleSelect) is usually a better
/// choice.
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

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Select9<R = Rank9, I = Vec<usize>> {
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

    const LOG2_ONES_PER_INVENTORY: usize = 9;
    const ONES_PER_INVENTORY: usize = 1 << Self::LOG2_ONES_PER_INVENTORY;
}

impl<B: AsRef<[usize]> + BitLength, C: AsRef<[BlockCounters]>> Select9<Rank9<B, C>, Vec<usize>> {
    pub fn new(rank9: Rank9<B, C>) -> Self {
        let num_bits = rank9.len();
        let num_words = (num_bits + 63) / 64;
        let inventory_size =
            (rank9.num_ones() + Self::ONES_PER_INVENTORY - 1) / Self::ONES_PER_INVENTORY;

        let u64_per_subinventory = 4;
        let subinventory_size = (num_words + u64_per_subinventory - 1) / u64_per_subinventory;

        let mut inventory: Vec<usize> = Vec::with_capacity(inventory_size + 1);
        let mut subinventory: Vec<usize> = vec![0; subinventory_size];

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
                    assert!(((block_span + 8) & !7) <= span * 4);
                    for (k, v) in s16.iter_mut().enumerate().take(block_span) {
                        assert!(*v == 0);
                        *v = (counts[block_left + k + 1].absolute - counts_at_start) as u16;
                    }
                    for v in s16.iter_mut().take((block_span + 8) & !7).skip(block_span) {
                        assert!(*v == 0);
                        *v = 0xFFFFu16;
                    }
                }
                16..=127 => {
                    assert!(((block_span + 8) & !7) + 8 <= span * 4);
                    assert!(block_span / 8 <= 8);
                    for k in 0..block_span {
                        assert!(s16[k + 8] == 0);
                        s16[k + 8] = (counts[block_left + k + 1].absolute - counts_at_start) as u16;
                    }
                    for k in block_span..((block_span + 8) & !7) {
                        assert!(s16[k + 8] == 0);
                        s16[k + 8] = 0xFFFFu16;
                    }
                    for (k, v) in s16.iter_mut().enumerate().take(block_span / 8) {
                        assert!(*v == 0);
                        *v = (counts[block_left + (k + 1) * 8].absolute - counts_at_start) as u16;
                    }
                    for v in s16.iter_mut().take(8).skip(block_span / 8) {
                        assert!(*v == 0);
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
                                assert!(subinventory[subinv_start + subinventory_idx] == 0);
                                subinventory[subinv_start + subinventory_idx] = bit_index;
                            }
                            1 => {
                                let s32: &mut [u32] = unsafe {
                                    subinventory[subinv_start..subinv_end].align_to_mut().1
                                };
                                assert!(s32[subinventory_idx] == 0);
                                assert!((bit_index - start_bit_idx) < (1 << 32));
                                s32[subinventory_idx] = sub_offset as u32;
                            }
                            2 => {
                                let s16: &mut [u16] = unsafe {
                                    subinventory[subinv_start..subinv_end].align_to_mut().1
                                };
                                assert!(s16[subinventory_idx] == 0);
                                assert!(bit_index - start_bit_idx < (1 << 16));
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
        let inventory_index_left = rank >> Self::LOG2_ONES_PER_INVENTORY;

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

                let where_: usize = (ULEQ_STEP_16!(first, rank_in_superblock_step_16)
                    + ULEQ_STEP_16!(second, rank_in_superblock_step_16))
                .wrapping_mul(ONES_STEP_16)
                    >> 47;

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

                let where0 = ((ULEQ_STEP_16!(first, rank_in_superblock_step_16)
                    + ULEQ_STEP_16!(second, rank_in_superblock_step_16))
                .wrapping_mul(ONES_STEP_16))
                    >> 47;

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
                    + ((ULEQ_STEP_16!(first_bis, rank_in_superblock_step_16)
                        + ULEQ_STEP_16!(second_bis, rank_in_superblock_step_16))
                    .wrapping_mul(ONES_STEP_16)
                        >> 47);

                block_left += where1 * 4;
                count_left += where1 / 2;
                rank_in_block = rank - counts.get_unchecked(count_left).absolute;

                debug_assert!(rank_in_block < 512);
            }
            128..=255 => {
                let (_, s, _) = subinv_ref
                    .get_range_unchecked(subinv_pos..self.subinventory_size)
                    .align_to::<u16>();
                return *s.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize + inventory_left;
            }
            256..=511 => {
                let (_, s, _) = subinv_ref
                    .get_range_unchecked(subinv_pos..self.subinventory_size)
                    .align_to::<u32>();
                return *s.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize + inventory_left;
            }
            _ => {
                return *subinv_ref.get_unchecked(rank % Self::ONES_PER_INVENTORY);
            }
        }

        let rank_in_block_step_9 = rank_in_block * ONES_STEP_9;
        let relative = counts.get_unchecked(count_left).relative;

        let offset_in_block =
            (ULEQ_STEP_9!(relative, rank_in_block_step_9).wrapping_mul(ONES_STEP_9) >> 54u64) & 0x7;
        debug_assert!(offset_in_block <= 7);

        let word = block_left + offset_in_block;

        let rank_in_word = rank_in_block - counts.get_unchecked(count_left).rel(offset_in_block);
        debug_assert!(rank_in_word < 64);

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

crate::forward_mult![Select9<R, I>; R; rank9;
    crate::forward_as_ref_slice_usize,
    crate::forward_index_bool,
    crate::traits::rank_sel::forward_bit_length,
    crate::traits::rank_sel::forward_bit_count,
    crate::traits::rank_sel::forward_num_bits,
    crate::traits::rank_sel::forward_rank,
    crate::traits::rank_sel::forward_rank_hinted,
    crate::traits::rank_sel::forward_rank_zero,
    crate::traits::rank_sel::forward_select_zero,
    crate::traits::rank_sel::forward_select_hinted,
    crate::traits::rank_sel::forward_select_zero_hinted
];

#[cfg(test)]
mod test_select9 {
    use super::*;
    use crate::traits::{Rank, Select};
    use crate::{prelude::BitVec, traits::BitCount};
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn test_select9() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in (1..1000).chain((1000..10000).step_by(100)) {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let select9 = Select9::new(Rank9::new(bits.clone()));

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(select9.select(i), Some(pos[i]));
            }
            assert_eq!(select9.select(ones + 1), None);
        }
    }

    #[test]
    fn test_select9_mult_usize() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in (1 << 10..1 << 15).step_by(usize::BITS as _) {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let select9 = Select9::new(Rank9::new(bits.clone()));

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(select9.select(i), Some(pos[i]));
            }
            assert_eq!(select9.select(ones + 1), None);
        }
    }

    #[test]
    fn test_select9_empty() {
        let bits = BitVec::new(0);
        let select9 = Select9::new(Rank9::new(bits.clone()));
        assert_eq!(select9.count_ones(), 0);
        assert_eq!(select9.len(), 0);
        assert_eq!(select9.select(0), None);
    }

    #[test]
    fn test_select9_ones() {
        let len = 300_000;
        let bits = (0..len).map(|_| true).collect::<BitVec>();
        let select9 = Select9::new(Rank9::new(bits));
        assert_eq!(select9.count_ones(), len);
        assert_eq!(select9.len(), len);
        for i in 0..len {
            assert_eq!(select9.select(i), Some(i));
        }
    }

    #[test]
    fn test_select9_zeros() {
        let len = 300_000;
        let bits = (0..len).map(|_| false).collect::<BitVec>();
        let select9 = Select9::new(Rank9::new(bits));
        assert_eq!(select9.count_ones(), 0);
        assert_eq!(select9.len(), len);
        assert_eq!(select9.select(0), None);
    }

    #[test]
    fn test_select9_few_ones() {
        let lens = [1 << 18, 1 << 19, 1 << 20];
        for len in lens {
            for num_ones in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
                let bits = (0..len)
                    .map(|i| i % (len / num_ones) == 0)
                    .collect::<BitVec>();
                let select9 = Select9::new(Rank9::new(bits));
                assert_eq!(select9.count_ones(), num_ones);
                assert_eq!(select9.len(), len);
                for i in 0..num_ones {
                    assert_eq!(select9.select(i), Some(i * (len / num_ones)));
                }
            }
        }
    }

    #[test]
    fn test_select9_non_uniform() {
        let lens = [1 << 18, 1 << 19, 1 << 20, 1 << 25];

        let mut rng = SmallRng::seed_from_u64(0);
        for len in lens {
            for density in [0.5] {
                let density0 = density * 0.01;
                let density1 = density * 0.99;

                let len1;
                let len2;
                if len % 2 != 0 {
                    len1 = len / 2 + 1;
                    len2 = len / 2;
                } else {
                    len1 = len / 2;
                    len2 = len / 2;
                }

                let first_half = loop {
                    let b = (0..len1)
                        .map(|_| rng.gen_bool(density0))
                        .collect::<BitVec>();
                    if b.count_ones() > 0 {
                        break b;
                    }
                };
                let num_ones_first_half = first_half.count_ones();
                let second_half = (0..len2)
                    .map(|_| rng.gen_bool(density1))
                    .collect::<BitVec>();
                let num_ones_second_half = second_half.count_ones();

                assert!(num_ones_first_half > 0);
                assert!(num_ones_second_half > 0);

                let bits = first_half
                    .into_iter()
                    .chain(second_half.into_iter())
                    .collect::<BitVec>();

                assert_eq!(
                    num_ones_first_half + num_ones_second_half,
                    bits.count_ones()
                );

                assert_eq!(bits.len(), len as usize);

                let ones = bits.count_ones();
                let mut pos = Vec::with_capacity(ones);
                for i in 0..(len as usize) {
                    if bits[i] {
                        pos.push(i);
                    }
                }

                let select9 = Select9::new(Rank9::new(bits));

                for i in 0..(ones) {
                    assert!(select9.select(i) == Some(pos[i]));
                }
                assert_eq!(select9.select(ones + 1), None);
            }
        }
    }

    #[test]
    fn test_select9_rank() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in (10_000..100_000).step_by(1000) {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let select9 = Select9::new(Rank9::new(bits.clone()));

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..len {
                assert_eq!(select9.rank(i), ranks[i]);
            }
            assert_eq!(select9.rank(len + 1), select9.count_ones());
        }
    }
}
