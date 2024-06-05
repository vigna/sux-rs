/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};
use std::{
    cmp::{max, min},
    ops::Index,
};

use crate::prelude::{BitCount, BitFieldSlice, BitLength, BitVec, Rank, Select, SelectHinted};

/// A simple select implementation based on a two-level inventory.
///
/// The structure has been described by Sebastiano Vigna in “[Broadword
/// Implementation of Rank/Select
/// Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
/// _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
/// 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
/// Springer, 2008.
///
/// # Implementation Details
///
/// The structure is based on a first-level inventory and a second-level
/// subinventory. Similarly to [`Rank9`](super::Rank9), the two levels are
/// interleaved to reduce the number of cache misses.
///
/// The inventory is sized so that the distance between two indexed ones is on
/// average a given target value `L`. For each indexed one in the inventory (for
/// which we use a 64-bit integer), we allocate at most `M` (a power of 2)
/// 64-bit integers for the subinventory. The relative target space occupancy of
/// the selection structure is thus at most `64 * (1 + M) / L`. However, since
/// the number of ones per inventory has to be a power of two, the _actual_
/// value of `L` might be smaller by a factor of 2, doubling the actual space
/// occupancy with respect to the target space occupancy.
///
/// For example, using [the default value of
/// `L`](SimpleSelect::DEFAULT_TARGET_INVENTORY_SPAN) and `M = 8`, the space
/// occupancy is between 7% and 14%. The space might be smaller for very sparse
/// vectors as less than `M` subinventory words per inventory might be used.
///
/// Given a specific indexed one in the inventory, if the distance to the next
/// indexed one is smaller than 2¹⁶ we use the 'M' words associated to the
/// subinventory to store `4M` 16-bit integers, representing the offsets of
/// regularly spaced ones inside the inventory. Otherwise, we store the exact
/// position of all such ones either using the 'M' available words, or, if they
/// are not sufficient, using a spill vector.
///
/// Note that is is possible to build pathological cases  (e.g., half of the bit
/// vector extremely dense, half of the vector extremely sparse) in which the
/// spill vector uses a very large amount of space (more than 50%). In these
/// cases, [`Select9`](super::Select9) might be a better choice.
///
/// In the 16-bit case, the average distance between two ones indexed by the
/// subinventories is `L/4M` (again, the actual value might be twice as large
/// because of rounding). Within this range, we perform a sequential broadword
/// search, which has a linear cost.
///
/// # Choosing Parameters
///
/// The value `L` should be chosen so that the distance between two indexed ones
/// in the inventory is always smaller than 2¹⁶. The [default suggested
/// value](SimpleSelect::DEFAULT_TARGET_INVENTORY_SPAN) is a reasonable choice
/// for vectors that reasonably uniform, but smaller values can be used for more
/// irregular vectors, at the cost of a larger space occupancy.
///
/// The value `M` should be as high as possible, compatibly with the desired
/// space occupancy, but values resulting in linear searches shorter than a
/// couple of words will not generally improve performance; moreover,
/// interleaving inventories is not useful if `M` is so large that the
/// subinventory takes several cache lines. For example, using [default value
/// for `L`](SimpleSelect::DEFAULT_TARGET_INVENTORY_SPAN) a reasonable choice
/// for `M` is between 2 and 5, corrisponding to worst-case linear searches
/// between 1024 and 128 bits.

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SimpleSelect<B: SelectHinted = BitVec, I: AsRef<[usize]> = Vec<usize>> {
    bits: B,
    inventory: I,
    exact_spill: I,
    num_ones: usize,
    log2_ones_per_inventory: usize,
    log2_ones_per_sub16: usize,
    log2_u64_per_subinventory: usize,
    ones_per_sub64: usize,
    u64_per_inventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
    inventory_size: usize,
    exact_spill_size: usize,
}

impl<B: SelectHinted, I: AsRef<[usize]>> SimpleSelect<B, I> {
    pub const DEFAULT_TARGET_INVENTORY_SPAN: usize = 8192;
}

impl SimpleSelect<BitVec, Vec<usize>> {
    /// Creates a new selection structure over a bit vector using a [default
    /// target inventory span](SimpleSelect::DEFAULT_TARGET_INVENTORY_SPAN).
    ///
    /// # Arguments
    ///
    /// * `bits`: A bit vector.
    ///
    /// * `max_log2_u64_per_subinventory`: The base-2 logarithm of the maximum
    ///   number [`M`](SimpleSelect) of 64-bit words in each subinventory.
    ///   Increasing by one this value approximately doubles the space occupancy
    ///   and halves the length of sequential broadword searches. Typical values
    ///   are 3 and 4.
    ///
    pub fn new(bits: BitVec, max_log2_u64_per_subinventory: usize) -> Self {
        Self::with_span(
            bits,
            Self::DEFAULT_TARGET_INVENTORY_SPAN,
            max_log2_u64_per_subinventory,
        )
    }

    /// Creates a new selection structure over a bit vector with a specified
    /// target inventory span.
    ///
    /// # Arguments
    ///
    /// * `bits`: A bit vector.
    ///
    /// * `target_inventory_span`: The target span [`L`](SimpleSelect) of a
    ///   first-level inventory entry. The actual span might be smaller by a
    ///   factor of 2.
    ///
    /// * `max_log2_u64_per_subinventory`: The base-2 logarithm of the maximum
    ///   number [`M`](SimpleSelect) of 64-bit words in each subinventory.
    ///   Increasing by one this value approximately doubles the space occupancy
    ///   and halves the length of sequential broadword searches. Typical values
    ///   are 3 and 4.
    ///
    pub fn with_span(
        bits: BitVec,
        target_inventory_span: usize,
        max_log2_u64_per_subinventory: usize,
    ) -> Self {
        let num_bits = max(1usize, bits.len());
        let num_ones = bits.count_ones();

        let log2_ones_per_inventory = (num_ones * target_inventory_span)
            .div_ceil(num_bits)
            .max(1)
            .ilog2() as usize;

        Self::with_inv(
            bits,
            num_ones,
            log2_ones_per_inventory,
            max_log2_u64_per_subinventory,
        )
    }

    /// Creates a new selection structure over a bit vector with a specified
    /// distance between indexed ones.
    ///
    /// This low-level constructor should be used with care, as the parameter
    /// `log2_ones_per_inventory` is usually computed as the floor of the base-2
    /// logarithm of ceiling of the target inventory span multiplied by the
    /// density of ones in the bit vector. Thus, this constructor makes sense
    /// only if the density is known in advance.
    ///
    /// Unless you understand all the implications, it is preferrable to use the
    /// [standard constructor](SimpleSelect::new).
    ///
    /// # Arguments
    ///
    /// * `bits`: A bit vector.
    ///
    /// * `log2_ones_per_inventory`: The base-2 logarithm of the distance
    ///   between two indexed ones.
    ///
    /// * `max_log2_u64_per_subinventory`: The base-2 logarithm of the maximum
    ///   number [`M`](SimpleSelect) of 64-bit words in each subinventory.
    ///   Increasing by one this value approximately doubles the space occupancy
    ///   and halves the length of sequential broadword searches. Typical values
    ///   are 3 and 4.

    pub fn with_inv(
        bits: BitVec,
        num_ones: usize,
        log2_ones_per_inventory: usize,
        max_log2_u64_per_subinventory: usize,
    ) -> Self {
        let num_bits = max(1usize, bits.len());
        let ones_per_inventory = 1usize << log2_ones_per_inventory;
        let ones_per_inventory_mask = ones_per_inventory - 1;
        let inventory_size = num_ones.div_ceil(ones_per_inventory);

        let log2_u64_per_subinventory = min(
            max_log2_u64_per_subinventory as i32,
            max(0, log2_ones_per_inventory as i32 - 2),
        ) as usize;

        let u64_per_subinventory = 1usize << log2_u64_per_subinventory;
        let u64_per_inventory = u64_per_subinventory + 1;

        let log2_ones_per_sub64 = max(
            0,
            log2_ones_per_inventory as i32 - log2_u64_per_subinventory as i32,
        ) as usize;
        let log2_ones_per_sub16 = max(0, (log2_ones_per_sub64 as i32) - 2) as usize;
        let ones_per_sub64 = 1usize << log2_ones_per_sub64;
        let ones_per_sub16 = 1usize << log2_ones_per_sub16;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let mut inventory = Vec::with_capacity(inventory_size * u64_per_inventory + 1);

        let mut curr_num_ones: usize = 0;
        let mut next_quantum: usize = 0;
        let mut spilled = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        for (i, word) in bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as usize;

            while curr_num_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word(next_quantum - curr_num_ones);
                let index = (i * usize::BITS as usize) + in_word_index;

                inventory.push(index);
                inventory.resize(inventory.len() + u64_per_subinventory, 0);

                next_quantum += ones_per_inventory;
            }
            curr_num_ones += ones_in_word;
        }
        assert_eq!(num_ones, curr_num_ones);
        // in the last inventory write the number of bits
        inventory.push(num_bits);
        assert_eq!(inventory.len(), inventory_size * u64_per_inventory + 1);

        // We estimate the subinventory and exact spill size
        for (i, inv) in inventory[..inventory_size * u64_per_inventory]
            .iter()
            .copied()
            .enumerate()
            .step_by(u64_per_inventory)
        {
            let start = inv;
            let span = inventory[i + u64_per_inventory] as usize - start;
            curr_num_ones = (i / u64_per_inventory) * ones_per_inventory;
            let ones = min(num_ones - curr_num_ones, ones_per_inventory);

            assert!(start + span == num_bits || ones == ones_per_inventory);

            // We accumulate space for exact pointers only if necessary.
            if span >= (1 << 16) && ones_per_sub64 > 1 {
                spilled += ones;
            }
        }

        let exact_spill_size = spilled;
        let mut exact_spill = vec![0; exact_spill_size];

        spilled = 0;
        let iter = 0..inventory_size;

        // Second phase: we fill the subinventories and the exact spill
        iter.for_each(|inventory_idx| {
            // get the start and end u64 index of the current inventory
            let start_idx = inventory_idx * u64_per_inventory;
            let end_idx = start_idx + u64_per_inventory;
            // read the first level index to get the start and end bit index
            let start_bit_idx = inventory[start_idx];
            let end_bit_idx = inventory[end_idx];
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            let mut word_idx = start_bit_idx / usize::BITS as usize;

            // cleanup the lower bits
            let bit_idx = start_bit_idx % usize::BITS as usize;
            let mut word = (bits.as_ref()[word_idx as usize] >> bit_idx) << bit_idx;

            // compute the global number of ones
            let mut number_of_ones = inventory_idx * ones_per_inventory;
            let mut next_quantum = number_of_ones;
            let quantum;

            if span <= u16::MAX as usize {
                quantum = ones_per_sub16;
            } else {
                quantum = 1;
                inventory[start_idx] |= 1_usize << 63;
                inventory[start_idx + 1] = spilled;
            }

            let end_word_idx = end_bit_idx.div_ceil(usize::BITS as usize);

            // the first subinventory element is always 0
            let mut subinventory_idx = 1;

            // pre increment the next quantum only when using the subinventories
            if span <= u16::MAX as usize {
                next_quantum += quantum;
            }

            'outer: loop {
                let ones_in_word = word.count_ones() as usize;

                // if the quantum is in this word, write it in the subinventory
                // this can happen multiple times if the quantum is small
                while number_of_ones + ones_in_word > next_quantum {
                    assert!(next_quantum <= end_bit_idx as _);
                    // find the quantum bit in the word
                    let in_word_index = word.select_in_word(next_quantum - number_of_ones);
                    // compute the global index of the quantum bit in the bitvec
                    let bit_index = (word_idx * usize::BITS as usize) + in_word_index;
                    if bit_index >= end_bit_idx {
                        break 'outer;
                    }
                    // compute the offset of the quantum bit
                    // from the start of the subinventory
                    let sub_offset = bit_index - start_bit_idx;

                    if span <= u16::MAX as usize {
                        let subinventory: &mut [u16] =
                            unsafe { inventory[start_idx + 1..end_idx].align_to_mut().1 };

                        subinventory[subinventory_idx] = sub_offset as u16;
                    } else if ones_per_sub64 != 1 {
                        assert!(spilled < exact_spill_size);
                        exact_spill[spilled] = bit_index;
                        spilled += 1;
                    }

                    // update the subinventory index if used
                    if span <= u16::MAX as usize {
                        subinventory_idx += 1;
                        if subinventory_idx == (1 << log2_ones_per_inventory) / quantum {
                            break 'outer;
                        }
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
                word = bits.as_ref()[word_idx as usize];
            }
        });

        assert_eq!(spilled, exact_spill_size);

        Self {
            bits,
            inventory,
            exact_spill,
            num_ones,
            log2_ones_per_inventory,
            log2_ones_per_sub16,
            log2_u64_per_subinventory,
            ones_per_sub64,
            u64_per_inventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
            inventory_size,
            exact_spill_size,
        }
    }

    pub fn log2_ones_per_inventory(&self) -> usize {
        self.log2_ones_per_inventory
    }
    pub fn log2_u64_per_subinventory(&self) -> usize {
        self.log2_u64_per_subinventory
    }
}

impl<B: SelectHinted + BitLength + AsRef<[usize]>, I: AsRef<[usize]>> Select
    for SimpleSelect<B, I>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let inventory_ref = self.inventory.as_ref();
        let inventory_index = rank >> self.log2_ones_per_inventory;
        let inventory_start_pos =
            (inventory_index << self.log2_u64_per_subinventory) + inventory_index;
        debug_assert!(inventory_index <= self.inventory_size);

        let inventory_rank = { *inventory_ref.get_unchecked(inventory_start_pos) };
        let subrank = rank & self.ones_per_inventory_mask;

        if (inventory_rank as isize) < 0 {
            if subrank == 0 {
                return inventory_rank & !(1usize << 63);
            }
            debug_assert!(
                { *inventory_ref.get_unchecked(inventory_start_pos + 1) } + subrank
                    < self.exact_spill_size
            );
            return self.exact_spill.get_unchecked(
                { *inventory_ref.get_unchecked(inventory_start_pos + 1) } + subrank,
            );
        }

        let (_, u16s, _) = inventory_ref
            .get_unchecked(inventory_start_pos + 1..(self.inventory_size * self.u64_per_inventory))
            .align_to::<u16>();

        let hint_pos =
            inventory_rank + *u16s.get_unchecked(subrank >> self.log2_ones_per_sub16) as usize;
        let residual = subrank & self.ones_per_sub16_mask;

        self.bits
            .select_hinted_unchecked(rank, hint_pos, rank - residual)
    }
}

impl<B: SelectHinted + BitLength + AsRef<[usize]>, I: AsRef<[usize]>> BitCount
    for SimpleSelect<B, I>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<B: SelectHinted + BitLength + AsRef<[usize]>, I: AsRef<[usize]>> BitLength
    for SimpleSelect<B, I>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`Rank`] to the underlying implementation.
impl<B: Rank + SelectHinted + AsRef<[usize]>, I: AsRef<[usize]>> Rank for SimpleSelect<B, I> {
    #[inline(always)]
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_unchecked(pos)
    }

    #[inline(always)]
    fn rank(&self, pos: usize) -> usize {
        self.bits.rank(pos)
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<B: SelectHinted + AsRef<[usize]>, I: AsRef<[usize]>> AsRef<[usize]> for SimpleSelect<B, I> {
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<B: SelectHinted + AsRef<[usize]> + Index<usize, Output = bool>, I: AsRef<[usize]>> Index<usize>
    for SimpleSelect<B, I>
{
    type Output = bool;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.bits[index]
    }
}

#[cfg(test)]
mod test_simple_select {
    use super::*;
    use crate::prelude::BitVec;

    #[test]
    fn test_simple_select_ones_per_sub64() {
        let len = 1 << 18;
        let bits = (0..len / 2)
            .map(|_| false)
            .chain([true])
            .chain((0..1 << 17).map(|_| false))
            .chain([true, true])
            .chain((0..1 << 18).map(|_| false))
            .chain([true])
            .chain((0..len / 2).map(|_| false))
            .collect::<BitVec>();
        let simple: SimpleSelect = SimpleSelect::new(bits, 3);

        assert_eq!(simple.ones_per_sub64, 1);
        assert_eq!(simple.count_ones(), 4);
        assert_eq!(simple.select(0), Some(len / 2));
        assert_eq!(simple.select(1), Some(len / 2 + (1 << 17) + 1));
        assert_eq!(simple.select(2), Some(len / 2 + (1 << 17) + 2));
    }
}
