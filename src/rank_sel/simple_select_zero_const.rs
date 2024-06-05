/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::{bits::CountBitVec, traits::*};
use common_traits::SelectInWord;
use epserde::*;
use mem_dbg::*;

/**

A selection structure for zeros based on a two-level index.

See [`SimpleSelectConst`](crate::rank_sel::SimpleSelectConst).

*/
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SimpleSelectZeroConst<
    B: SelectZeroHinted = CountBitVec,
    I: AsRef<[u64]> = Vec<u64>,
    const LOG2_ZEROS_PER_INVENTORY: usize = 10,
    const LOG2_U64_PER_SUBINVENTORY: usize = 2,
> {
    bits: B,
    inventory: I,
}

/// constants used throughout the code
impl<
        B: SelectZeroHinted,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    const ONES_PER_INVENTORY: usize = 1 << LOG2_ZEROS_PER_INVENTORY;
    const U64_PER_SUBINVENTORY: usize = 1 << LOG2_U64_PER_SUBINVENTORY;
    const U64_PER_INVENTORY: usize = 1 + Self::U64_PER_SUBINVENTORY;

    const LOG2_ZEROS_PER_SUB64: usize = LOG2_ZEROS_PER_INVENTORY - LOG2_U64_PER_SUBINVENTORY;
    const ONES_PER_SUB64: usize = 1 << Self::LOG2_ZEROS_PER_SUB64;

    const LOG2_ZEROS_PER_SUB16: usize = Self::LOG2_ZEROS_PER_SUB64 - 2;
    const ONES_PER_SUB16: usize = 1 << Self::LOG2_ZEROS_PER_SUB16;

    /// We use the sign bit to store the type of the subinventory (u16 vs. u64).
    const INVENTORY_MASK: u64 = (1 << 63) - 1;

    /// Return the inner BitVector
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Return raw bits and inventory
    pub fn into_raw_parts(self) -> (B, I) {
        (self.bits, self.inventory)
    }

    /// Create a new instance from raw bits and inventory
    ///
    /// # Safety
    /// The inventory must be consistent with the bits otherwise you will get
    /// wrong results, and possibly memory corruption.
    pub unsafe fn from_raw_parts(bits: B, inventory: I) -> Self {
        SimpleSelectZeroConst { bits, inventory }
    }

    /// Modify the inner BitVector with possibly another type
    pub fn map_bits<B2>(
        self,
        f: impl FnOnce(B) -> B2,
    ) -> SimpleSelectZeroConst<B2, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
    where
        B2: SelectZeroHinted,
    {
        SimpleSelectZeroConst {
            bits: f(self.bits),
            inventory: self.inventory,
        }
    }

    /// Modify the inner inventory with possibly another type
    pub fn map_inventory<I2>(
        self,
        f: impl FnOnce(I) -> I2,
    ) -> SimpleSelectZeroConst<B, I2, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
    where
        I2: AsRef<[u64]>,
    {
        SimpleSelectZeroConst {
            bits: self.bits,
            inventory: f(self.inventory),
        }
    }

    /// Modify the inner BitVector and inventory with possibly other types
    pub fn map<B2, I2>(
        self,
        f: impl FnOnce(B, I) -> (B2, I2),
    ) -> SimpleSelectZeroConst<B2, I2, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
    where
        B2: SelectZeroHinted,
        I2: AsRef<[u64]>,
    {
        let (bits, inventory) = f(self.bits, self.inventory);
        SimpleSelectZeroConst { bits, inventory }
    }
}

impl<
        B: SelectZeroHinted + BitLength + BitCount + AsRef<[usize]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SimpleSelectZeroConst<B, Vec<u64>, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    pub fn new(bitvec: B) -> Self {
        let number_of_zeros = bitvec.count_zeros();
        // number of inventories we will create
        let inventory_size = (number_of_zeros).div_ceil(Self::ONES_PER_INVENTORY);
        let inventory_len = inventory_size * Self::U64_PER_INVENTORY + 1;
        // inventory_size, an u64 for the first layer index, and Self::U64_PER_SUBINVENTORY for the sub layer
        let mut inventory = Vec::with_capacity(inventory_len);
        // scan the bitvec and fill the first layer of the inventory
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        'outer: for (i, word) in bitvec.as_ref().iter().copied().map(|x| !x).enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while (number_of_ones + ones_in_word).min(number_of_zeros as u64) > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * u64::BITS as usize) + in_word_index;

                // write the position of the one in the inventory
                inventory.push(index as u64);
                // make space for the subinventory
                inventory.resize(inventory.len() + Self::U64_PER_SUBINVENTORY, 0);

                next_quantum += Self::ONES_PER_INVENTORY as u64;
                if next_quantum >= number_of_zeros as u64 {
                    break 'outer;
                }
            }
            number_of_ones += ones_in_word;
        }
        // in the last inventory write the number of bits
        inventory.push(BitLength::len(&bitvec) as u64);
        debug_assert_eq!(inventory_len, inventory.len());
        let iter = 0..inventory_size;

        // fill the second layer of the index
        iter.for_each(|inventory_idx| {
            // get the start and end u64 index of the current inventory
            let start_idx = inventory_idx * Self::U64_PER_INVENTORY;
            let end_idx = start_idx + Self::U64_PER_INVENTORY;
            // read the first level index to get the start and end bit index
            let start_bit_idx = inventory[start_idx];
            let end_bit_idx = inventory[end_idx];
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            // compute were we should the word boundaries of where we should
            // scan
            let mut word_idx = start_bit_idx / u64::BITS as u64;

            // cleanup the lower bits
            let bit_idx = start_bit_idx % u64::BITS as u64;
            let mut word = !(bitvec.as_ref()[word_idx as usize]) >> bit_idx << bit_idx;
            // compute the global number of ones
            let mut number_of_ones = inventory_idx * Self::ONES_PER_INVENTORY;
            // and what's the next bit rank which index should log in the sub
            // inventory (the first subinventory element is always 0)
            let mut next_quantum = number_of_ones;
            let quantum;

            if span <= u16::MAX as u64 {
                quantum = Self::ONES_PER_SUB16;
            } else {
                quantum = Self::ONES_PER_SUB64;
                inventory[start_idx] |= 1_u64 << 63;
            }

            let end_word_idx = end_bit_idx.div_ceil(u64::BITS as u64);

            // the first subinventory element is always 0
            let mut subinventory_idx = 1;
            next_quantum += quantum;

            'outer: loop {
                let ones_in_word = word.count_ones() as usize;

                // if the quantum is in this word, write it in the subinventory
                // this can happen multiple times if the quantum is small
                while (number_of_ones + ones_in_word).min(number_of_zeros) > next_quantum {
                    debug_assert!(next_quantum <= end_bit_idx as _);
                    // find the quantum bit in the word
                    let in_word_index = word.select_in_word(next_quantum - number_of_ones);
                    // compute the global index of the quantum bit in the bitvec
                    let bit_index = (word_idx * u64::BITS as u64) + in_word_index as u64;
                    // compute the offset of the quantum bit
                    // from the start of the subinventory
                    let sub_offset = bit_index - start_bit_idx;

                    if span <= u16::MAX as u64 {
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

                // we are done with the word, so update the number of ones
                number_of_ones += ones_in_word;
                // move to the next word and boundcheck
                word_idx += 1;
                if word_idx == end_word_idx {
                    break;
                }
                // read the next word
                word = !bitvec.as_ref()[word_idx as usize];
            }
        });

        Self {
            bits: bitvec,
            inventory,
        }
    }
}

/// Provide the hint to the underlying structure.
impl<
        B: SelectZeroHinted + BitCount + BitLength,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZero
    for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        // find the index of the first level inventory
        let inventory_index = rank / Self::ONES_PER_INVENTORY;
        // find the index of the second level inventory
        let subrank = rank % Self::ONES_PER_INVENTORY;
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
                inventory_rank + u16s[subrank / Self::ONES_PER_SUB16] as u64,
                subrank % Self::ONES_PER_SUB16,
            )
        } else {
            (
                (inventory_rank & Self::INVENTORY_MASK) + u64s[subrank / Self::ONES_PER_SUB64],
                subrank % Self::ONES_PER_SUB64,
            )
        };

        // linear scan to finish the search
        self.bits
            .select_zero_hinted_unchecked(rank, pos as usize, rank - residual)
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        B: SelectZeroHinted + BitLength,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > BitLength
    for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`BitCount`] to the underlying implementation.
impl<
        B: SelectZeroHinted + BitCount,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > BitCount
    for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.bits.count_ones()
    }

    #[inline(always)]
    fn count_zeros(&self) -> usize {
        self.bits.count_zeros()
    }
}

/// Forward [`Select`] to the underlying implementation.
impl<
        B: SelectZeroHinted + Select,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > Select for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    fn select(&self, rank: usize) -> Option<usize> {
        self.bits.select(rank)
    }
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        self.bits.select_unchecked(rank)
    }
}

/// Forward [`Rank`] to the underlying implementation.
impl<
        B: SelectZeroHinted + Rank,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > Rank for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    fn rank(&self, pos: usize) -> usize {
        self.bits.rank(pos)
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_unchecked(pos)
    }
}

/// Forward [`RankZero`] to the underlying implementation.
impl<
        B: SelectZeroHinted + RankZero,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > RankZero
    for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    fn rank_zero(&self, pos: usize) -> usize {
        self.bits.rank_zero(pos)
    }

    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_zero_unchecked(pos)
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<
        B: SelectZeroHinted + AsRef<[usize]>,
        I: AsRef<[u64]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > AsRef<[usize]>
    for SimpleSelectZeroConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
