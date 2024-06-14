/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::*;
use common_traits::SelectInWord;
use epserde::*;
use mem_dbg::*;

/// The [`SelectZero`] version of [`SimpleSelect`](crate::rank_sel::SimpleSelect).

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SimpleSelectZeroConst<
    B,
    I,
    const LOG2_ZEROS_PER_INVENTORY: usize = 10,
    const LOG2_U64_PER_SUBINVENTORY: usize = 2,
> {
    bits: B,
    inventory: I,
}

/// constants used throughout the code
impl<B, I, const LOG2_ZEROS_PER_INVENTORY: usize, const LOG2_U64_PER_SUBINVENTORY: usize>
    SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    const ZEROS_PER_INVENTORY: usize = 1 << LOG2_ZEROS_PER_INVENTORY;
    const U64_PER_SUBINVENTORY: usize = 1 << LOG2_U64_PER_SUBINVENTORY;
    const U64_PER_INVENTORY: usize = 1 + Self::U64_PER_SUBINVENTORY;

    const LOG2_ZEROS_PER_SUB64: usize = LOG2_ZEROS_PER_INVENTORY - LOG2_U64_PER_SUBINVENTORY;
    const ZEROS_PER_SUB64: usize = 1 << Self::LOG2_ZEROS_PER_SUB64;

    const LOG2_ZEROS_PER_SUB16: usize = Self::LOG2_ZEROS_PER_SUB64 - 2;
    const ZEROS_PER_SUB16: usize = 1 << Self::LOG2_ZEROS_PER_SUB16;

    /// We use the sign bit to store the type of the subinventory (u16 vs. u64).
    const INVENTORY_MASK: usize = (1 << 63) - 1;

    /// Return the inner BitVector
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Replaces the backend with a new one implementing [`SelectZeroHinted`].
    pub unsafe fn map<C>(
        self,
        f: impl FnOnce(B) -> C,
    ) -> SimpleSelectZeroConst<C, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
    where
        C: SelectZeroHinted,
    {
        SimpleSelectZeroConst {
            bits: f(self.bits),
            inventory: self.inventory,
        }
    }
}

impl<
        B: SelectZeroHinted + NumBits + AsRef<[usize]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SimpleSelectZeroConst<B, Vec<usize>, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    pub fn new(bitvec: B) -> Self {
        let num_zeros = bitvec.num_zeros();
        // number of inventories we will create
        let inventory_size = num_zeros.div_ceil(Self::ZEROS_PER_INVENTORY);

        // A u64 for the inventory, and Self::U64_PER_SUBINVENTORY u64's for the subinventory
        let inventory_words = inventory_size * Self::U64_PER_INVENTORY + 1;
        let mut inventory = Vec::with_capacity(inventory_words);

        let mut past_zeros = 0;
        let mut next_quantum = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        'outer: for (i, word) in bitvec.as_ref().iter().copied().map(|x| !x).enumerate() {
            let zeros_in_word = word.count_ones() as usize;
            // skip the word if we can
            while (past_zeros + zeros_in_word).min(num_zeros) > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - past_zeros) as usize);
                let index = (i * u64::BITS as usize) + in_word_index;

                // write the position of the one in the inventory
                inventory.push(index);
                // make space for the subinventory
                inventory.resize(inventory.len() + Self::U64_PER_SUBINVENTORY, 0);

                next_quantum += Self::ZEROS_PER_INVENTORY;
                if next_quantum >= num_zeros {
                    break 'outer;
                }
            }
            past_zeros += zeros_in_word;
        }

        // TODO assert_eq!(num_zeros, past_zeros.min(num_zeros));
        // in the last inventory write the number of bits
        inventory.push(BitLength::len(&bitvec));
        debug_assert_eq!(inventory_words, inventory.len());

        // fill the second layer of the index
        for inventory_idx in 0..inventory_size {
            // get the start and end index of the current inventory
            let start_idx = inventory_idx * Self::U64_PER_INVENTORY;
            let end_idx = start_idx + Self::U64_PER_INVENTORY;
            // read the first level index to get the start and end bit index
            let start_bit_idx = inventory[start_idx];
            let end_bit_idx = inventory[end_idx];
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            // compute were we should the word boundaries of where we should
            // scan
            let mut word_idx = start_bit_idx / u64::BITS as usize;

            // cleanup the lower bits
            let bit_idx = start_bit_idx % u64::BITS as usize;
            let mut word = !(bitvec.as_ref()[word_idx as usize]) >> bit_idx << bit_idx;
            // compute the global number of zeros up to the current inventory
            let mut past_zeros = inventory_idx * Self::ZEROS_PER_INVENTORY;
            // and what's the next bit rank which index should log in the sub
            // inventory (the first subinventory element is always 0)
            let mut next_quantum = past_zeros;
            let quantum;

            if span <= u16::MAX as usize {
                quantum = Self::ZEROS_PER_SUB16;
            } else {
                quantum = Self::ZEROS_PER_SUB64;
                inventory[start_idx] |= 1 << 63;
            }

            let end_word_idx = end_bit_idx.div_ceil(u64::BITS as usize);

            // the first subinventory element is always 0
            let mut subinventory_idx = 1;
            next_quantum += quantum;

            'outer: loop {
                let zeros_in_word = word.count_ones() as usize;

                // if the quantum is in this word, write it in the subinventory
                // this can happen multiple times if the quantum is small
                while (past_zeros + zeros_in_word).min(num_zeros) > next_quantum {
                    debug_assert!(next_quantum <= end_bit_idx as _);
                    // find the quantum bit in the word
                    let in_word_index = word.select_in_word(next_quantum - past_zeros);
                    // compute the global index of the quantum bit in the bitvec
                    let bit_index = (word_idx * usize::BITS as usize) + in_word_index;
                    // compute the offset of the quantum bit
                    // from the start of the subinventory
                    let sub_offset = bit_index - start_bit_idx;

                    if span <= u16::MAX as usize {
                        let subinventory: &mut [u16] =
                            unsafe { inventory[start_idx + 1..end_idx].align_to_mut().1 };

                        subinventory[subinventory_idx] = sub_offset as u16;
                    } else {
                        inventory[start_idx + 1 + subinventory_idx] = sub_offset;
                    }

                    // update the subinventory index and the next quantum
                    subinventory_idx += 1;
                    if subinventory_idx == (1 << LOG2_ZEROS_PER_INVENTORY) / quantum {
                        break 'outer;
                    }
                    next_quantum += quantum;
                }

                // we are done with the word, so update the number of zeros
                past_zeros += zeros_in_word;
                // move to the next word and boundcheck
                word_idx += 1;
                if word_idx == end_word_idx {
                    break;
                }
                // read the next word
                word = !bitvec.as_ref()[word_idx as usize];
            }
        }

        Self {
            bits: bitvec,
            inventory,
        }
    }
}

/// Provide the hint to the underlying structure.
impl<
        B: SelectZeroHinted + NumBits,
        I: AsRef<[usize]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZeroUnchecked
    for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        // find the index of the first level inventory
        let inventory_index = rank / Self::ZEROS_PER_INVENTORY;
        // find the index of the second level inventory
        let subrank = rank % Self::ZEROS_PER_INVENTORY;
        // find the position of the first index value in the interleaved inventory
        let start_idx = inventory_index * (1 + Self::U64_PER_SUBINVENTORY);
        // read the potentially unaliged i64 (i.e. the first index value)
        let inventory_rank = *self.inventory.as_ref().get_unchecked(start_idx);
        // get a reference to the u64s in this subinventory
        let u64s = self
            .inventory
            .as_ref()
            .get_unchecked(start_idx + 1..start_idx + 1 + Self::U64_PER_SUBINVENTORY);

        // if the inventory_rank is positive, the subranks are u16s otherwise they are u64s
        let (pos, residual) = if inventory_rank as isize >= 0 {
            let (_pre, u16s, _post) = u64s.align_to::<u16>();
            (
                inventory_rank + *u16s.get_unchecked(subrank / Self::ZEROS_PER_SUB16) as usize,
                subrank % Self::ZEROS_PER_SUB16,
            )
        } else {
            (
                (inventory_rank & Self::INVENTORY_MASK)
                    + u64s.get_unchecked(subrank / Self::ZEROS_PER_SUB64),
                subrank % Self::ZEROS_PER_SUB64,
            )
        };

        // linear scan to finish the search
        self.bits
            .select_zero_hinted_unchecked(rank, pos as usize, rank - residual)
    }
}

impl<
        B: SelectZeroHinted + NumBits,
        I: AsRef<[usize]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZero
    for SimpleSelectZeroConst<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
}

crate::forward_mult![
    SimpleSelectZeroConst<B, I, [const] LOG2_ZEROS_PER_INVENTORY: usize, [const] LOG2_U64_PER_SUBINVENTORY: usize>; B; bits;
    crate::forward_as_ref_slice_usize,
    crate::forward_index_bool,
    crate::traits::rank_sel::forward_bit_length,
    crate::traits::rank_sel::forward_bit_count,
    crate::traits::rank_sel::forward_num_bits,
    crate::traits::rank_sel::forward_rank,
    crate::traits::rank_sel::forward_rank_hinted,
    crate::traits::rank_sel::forward_rank_zero,
    crate::traits::rank_sel::forward_select_unchecked,
    crate::traits::rank_sel::forward_select,
    crate::traits::rank_sel::forward_select_hinted,
    crate::traits::rank_sel::forward_select_zero_hinted
];
