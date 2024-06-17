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
use std::cmp::{max, min};

use crate::{
    prelude::{BitCount, BitFieldSlice, BitLength, SelectHinted},
    traits::{NumBits, SelectZero, SelectZeroHinted, SelectZeroUnchecked},
};

/// A selection structure based on an adaptive two-level inventory.
///
///
/// # Examples
/// ```rust
/// use sux::bit_vec;
/// use sux::traits::{Rank, Select};
/// use sux::rank_sel::{SelectAdapt, Rank9};
///
/// // Standalone select
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let select = SelectAdapt::new(bits, 3);
///
/// assert_eq!(select.select(0), Some(0));
/// assert_eq!(select.select(1), Some(2));
/// assert_eq!(select.select(2), Some(3));
/// assert_eq!(select.select(3), Some(5));
/// assert_eq!(select.select(4), Some(7));
/// assert_eq!(select.select(5), None);
///
/// // Access to the underlying bit vector is forwarded, too
/// assert_eq!(select[0], true);
/// assert_eq!(select[1], false);
/// assert_eq!(select[2], true);
/// assert_eq!(select[3], true);
/// assert_eq!(select[4], false);
/// assert_eq!(select[5], true);
/// assert_eq!(select[6], false);
/// assert_eq!(select[7], true);
///
/// // Map the backend to a different structure
/// let sel_rank9 = select.map(Rank9::new);
///
/// // Rank methods are forwarded
/// assert_eq!(sel_rank9.rank(0), 0);
/// assert_eq!(sel_rank9.rank(1), 1);
/// assert_eq!(sel_rank9.rank(2), 1);
/// assert_eq!(sel_rank9.rank(3), 2);
/// assert_eq!(sel_rank9.rank(4), 3);
/// assert_eq!(sel_rank9.rank(5), 3);
/// assert_eq!(sel_rank9.rank(6), 4);
/// assert_eq!(sel_rank9.rank(7), 4);
/// assert_eq!(sel_rank9.rank(8), 5);
///
/// // Select over a Rank9 structure
/// let rank9 = Rank9::new(sel_rank9.into_inner().into_inner());
/// let rank9_sel = SelectAdapt::new(rank9, 3);
///
/// assert_eq!(rank9_sel.select(0), Some(0));
/// assert_eq!(rank9_sel.select(1), Some(2));
/// assert_eq!(rank9_sel.select(2), Some(3));
/// assert_eq!(rank9_sel.select(3), Some(5));
/// assert_eq!(rank9_sel.select(4), Some(7));
/// assert_eq!(rank9_sel.select(5), None);
///
/// // Rank methods are forwarded
/// assert_eq!(rank9_sel.rank(0), 0);
/// assert_eq!(rank9_sel.rank(1), 1);
/// assert_eq!(rank9_sel.rank(2), 1);
/// assert_eq!(rank9_sel.rank(3), 2);
/// assert_eq!(rank9_sel.rank(4), 3);
/// assert_eq!(rank9_sel.rank(5), 3);
/// assert_eq!(rank9_sel.rank(6), 4);
/// assert_eq!(rank9_sel.rank(7), 4);
/// assert_eq!(rank9_sel.rank(8), 5);
///
/// // Access to the underlying bit vector is forwarded, too
/// assert_eq!(rank9_sel[0], true);
/// assert_eq!(rank9_sel[1], false);
/// assert_eq!(rank9_sel[2], true);
/// assert_eq!(rank9_sel[3], true);
/// assert_eq!(rank9_sel[4], false);
/// assert_eq!(rank9_sel[5], true);
/// assert_eq!(rank9_sel[6], false);
/// assert_eq!(rank9_sel[7], true);
/// ```

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SelectZeroAdaptConst<
    B,
    I = Box<[usize]>,
    const LOG2_ONES_PER_INVENTORY: usize = 10,
    const MAX_LOG2_U64_PER_SUBINVENTORY: usize = 2,
> {
    bits: B,
    inventory: I,
    spill: I,
    log2_ones_per_sub16: usize,
    u64_per_inventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
}

// Convenience trait to handle the information packed in the two upper bits of
// an inventory entry.
trait Inventory {
    fn is_u16_span(&self) -> bool;
    fn is_u32_span(&self) -> bool;
    fn is_u64_span(&self) -> bool;
    fn set_u16_span(&mut self);
    fn set_u32_span(&mut self);
    fn set_u64_span(&mut self);
    fn get(&self) -> usize;
}

impl Inventory for usize {
    #[inline(always)]
    fn is_u16_span(&self) -> bool {
        // This test is optimized for speed as it is the common case
        *self as isize >= 0
    }

    #[inline(always)]
    fn is_u32_span(&self) -> bool {
        *self >> 62 == 2
    }

    #[inline(always)]
    fn is_u64_span(&self) -> bool {
        *self >> 62 == 3
    }

    #[inline(always)]
    fn set_u16_span(&mut self) {}

    #[inline(always)]
    fn set_u32_span(&mut self) {
        *self |= 1 << 63;
    }

    #[inline(always)]
    fn set_u64_span(&mut self) {
        *self |= 3 << 62;
    }

    #[inline(always)]
    fn get(&self) -> usize {
        *self & 0x3FFF_FFFF_FFFF_FFFF
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanType {
    U16,
    U32,
    U64,
}

fn span_type(x: usize) -> SpanType {
    match x {
        0..=0x10000 => SpanType::U16,
        0x10001..=0x100000000 => SpanType::U32,
        _ => SpanType::U64,
    }
}

// Compute adaptively the number of 32-bit subinventory entries
#[inline(always)]
fn log2_ones_per_sub32(span: usize, log2_ones_per_sub16: usize) -> usize {
    debug_assert!(span >= 1 << 16);
    // Since span >= 2^16, (span >> 15).ilog2() >= 0, which implies in any case
    // at least doubling the frequency of the subinventory with respect to the
    // 16-bit case, unless log2_ones_per_u16 = 0, that is, we are recording the
    // position of every one in the subinventory.
    log2_ones_per_sub16.saturating_sub((span >> 15).ilog2() as usize + 1)
}

impl<B, I, const LOG2_ONES_PER_INVENTORY: usize, const LOG2_U64_PER_SUBINVENTORY: usize>
    SelectZeroAdaptConst<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Replaces the backend with a new one implementing [`SelectHinted`].
    pub unsafe fn map<C>(self, f: impl FnOnce(B) -> C) -> SelectZeroAdaptConst<C, I>
    where
        C: SelectHinted,
    {
        SelectZeroAdaptConst {
            bits: f(self.bits),
            inventory: self.inventory,
            spill: self.spill,
            log2_ones_per_sub16: self.log2_ones_per_sub16,
            u64_per_inventory: self.u64_per_inventory,
            ones_per_inventory_mask: self.ones_per_inventory_mask,
            ones_per_sub16_mask: self.ones_per_sub16_mask,
        }
    }

    pub const DEFAULT_TARGET_INVENTORY_SPAN: usize = 8192;
}

impl<
        B: AsRef<[usize]> + BitLength + BitCount + SelectHinted,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZeroAdaptConst<B, Box<[usize]>, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    /// Creates a new selection structure over a [`SelectHinted`] with a specified
    /// distance between indexed ones.
    ///
    /// This low-level constructor should be used with care, as the parameter
    /// `log2_ones_per_inventory` is usually computed as the floor of the base-2
    /// logarithm of ceiling of the target inventory span multiplied by the
    /// density of ones in the bit vector. Thus, this constructor makes sense
    /// only if the density is known in advance.
    ///
    /// Unless you understand all the implications, it is preferrable to use the
    /// [standard constructor](SelectAdapt::new).
    ///
    /// # Arguments
    ///
    /// * `bits`: A [`SelectHinted`].
    ///
    /// * `log2_ones_per_inventory`: The base-2 logarithm of the distance
    ///   between two indexed ones.
    ///
    /// * `max_log2_u64_per_subinventory`: The base-2 logarithm of the maximum
    ///   number [*M*](SelectAdapt) of 64-bit words in each subinventory.
    ///   Increasing by one this value approximately doubles the space occupancy
    ///   and halves the length of sequential broadword searches. Typical values
    ///   are 3 and 4.

    pub fn new(bits: B) -> Self {
        let num_ones = bits.count_ones();
        let num_bits = max(1, bits.len());
        let ones_per_inventory = 1 << LOG2_ONES_PER_INVENTORY;
        let ones_per_inventory_mask = ones_per_inventory - 1;
        let inventory_size = num_ones.div_ceil(ones_per_inventory);

        // We use a smaller value than max_log2_u64_per_subinventory when with a
        // smaller value we can still index, in the 16-bit case, all bits the
        // subinventory. This can happen only in extremely sparse vectors, or
        // if a very small value of log2_ones_per_inventory is set directly.

        let log2_u64_per_subinventory =
            LOG2_U64_PER_SUBINVENTORY.min(LOG2_ONES_PER_INVENTORY.saturating_sub(2));

        let u64_per_subinventory = 1 << log2_u64_per_subinventory;
        // A u64 for the inventory, and u64_per_inventory for the subinventory
        let u64_per_inventory = u64_per_subinventory + 1;

        let log2_ones_per_sub16 =
            LOG2_ONES_PER_INVENTORY.saturating_sub(log2_u64_per_subinventory + 2);
        let ones_per_sub16 = 1 << log2_ones_per_sub16;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let inventory_words = inventory_size * u64_per_inventory + 1;
        let mut inventory = Vec::with_capacity(inventory_words);

        let mut past_ones = 0;
        let mut next_quantum = 0;
        let mut spilled = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        for (i, word) in bits.as_ref().iter().copied().map(|x| !x).enumerate() {
            let ones_in_word = word.count_ones() as usize;

            while past_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word(next_quantum - past_ones);
                let index = (i * usize::BITS as usize) + in_word_index;

                // write the position of the one in the inventory
                inventory.push(index);
                // make space for the subinventory
                inventory.resize(inventory.len() + u64_per_subinventory, 0);

                next_quantum += ones_per_inventory;
            }
            past_ones += ones_in_word;
        }

        assert_eq!(num_ones, past_ones);
        // in the last inventory write the number of bits
        inventory.push(num_bits);
        assert_eq!(inventory.len(), inventory_words);

        // We estimate the subinventory and exact spill size
        for (i, inv) in inventory[..inventory_size * u64_per_inventory]
            .iter()
            .copied()
            .step_by(u64_per_inventory)
            .enumerate()
        {
            let start = inv;
            let span = inventory[i * u64_per_inventory + u64_per_inventory] - start;
            past_ones = i * ones_per_inventory;
            let ones = min(num_ones - past_ones, ones_per_inventory);

            debug_assert!(start + span == num_bits || ones == ones_per_inventory);

            match span_type(span) {
                // We store the entries first in the subinventory and then in
                // the spill buffer. The first u64 word will be used to store
                // the position of the entry in the spill buffer. Using the
                // first word gives a cache advantage to entries that will need
                // another cache miss to be read from the spill buffer.
                SpanType::U32 => {
                    // We store an inventory entry each 1 << log2_ones_per_sub32 ones.
                    let log2_ones_per_sub32 = log2_ones_per_sub32(span, log2_ones_per_sub16);
                    let num_u32s = ones.div_ceil(1 << log2_ones_per_sub32);
                    let num_u64s = num_u32s.div_ceil(2);
                    let spilled_u64s = num_u64s - u64_per_subinventory + 1;
                    spilled += spilled_u64s;
                }
                SpanType::U64 => {
                    // We store an inventory entry for each one after the first.
                    spilled += (ones - 1).saturating_sub(u64_per_subinventory - 1);
                }
                _ => {}
            }
        }

        let spill_size = spilled;

        let mut inventory: Box<[usize]> = inventory.into();
        let mut spill: Box<[usize]> = vec![0; spill_size].into();

        spilled = 0;

        // Second phase: we fill the subinventories and the spill.
        for inventory_idx in 0..inventory_size {
            // Get the start and end indices of the current inventory
            let start_inv_idx = inventory_idx * u64_per_inventory;
            let end_inv_idx = start_inv_idx + u64_per_inventory;
            // Read the first-level index to get the start and end bit indices
            let start_bit_idx = inventory[start_inv_idx];
            let end_bit_idx = inventory[end_inv_idx];
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            let span_type = span_type(span);

            // Compute the number of ones before the current inventory
            let mut past_ones = inventory_idx * ones_per_inventory;
            let mut next_quantum = past_ones;
            let log2_quantum;

            match span_type {
                SpanType::U16 => {
                    log2_quantum = log2_ones_per_sub16;
                    inventory[start_inv_idx].set_u16_span();
                }
                SpanType::U32 => {
                    log2_quantum = log2_ones_per_sub32(span, log2_ones_per_sub16);
                    inventory[start_inv_idx].set_u32_span();
                    // The first word of the subinventory is used to store the spill index.
                    inventory[start_inv_idx + 1] = spilled;
                }
                SpanType::U64 => {
                    log2_quantum = 0;
                    inventory[start_inv_idx].set_u64_span();
                    // The first word of the subinventory is used to store the spill index.
                    inventory[start_inv_idx + 1] = spilled;
                }
            }

            let quantum = 1 << log2_quantum;

            // If the span is 16-bit or 32-bit the first subinventory element is
            // always zero, so we don't write it explicitly. Moreover, in the
            // U64 case we don't write it at all.
            let mut subinventory_idx = 1;
            next_quantum += quantum;

            // This is used only when span_type == SpanType::U32
            let mut u32_odd_spill = false;

            let mut word_idx = start_bit_idx / usize::BITS as usize;
            let end_word_idx = end_bit_idx.div_ceil(usize::BITS as usize);
            let bit_idx = start_bit_idx % usize::BITS as usize;

            // Clear the lower bits
            let mut word = !((bits.as_ref()[word_idx] >> bit_idx) << bit_idx);

            'outer: loop {
                let ones_in_word = word.count_ones() as usize;

                // If the quantum is in this word, write it in the subinventory.
                // Note that this can happen multiple times in the same word if
                // the quantum is small, hence the following loop.
                while past_ones + ones_in_word > next_quantum {
                    debug_assert!(next_quantum <= end_bit_idx);
                    // find the quantum bit in the word
                    let in_word_index = word.select_in_word(next_quantum - past_ones);
                    // compute the global index of the quantum bit in the bitvec
                    let bit_index = (word_idx * usize::BITS as usize) + in_word_index;

                    // This exit is necessary in case the number of ones per
                    // inventory is larger than the number of available
                    // subinventory entries, which can happen if the bit vector
                    // is very sparse, or if we are in the last inventory entry.
                    if bit_index >= end_bit_idx {
                        break 'outer;
                    }

                    // Compute the offset of the quantum bit from the start of
                    // the subinventory
                    let sub_offset = bit_index - start_bit_idx;

                    match span_type {
                        SpanType::U16 => {
                            let subinventory: &mut [u16] = unsafe {
                                inventory[start_inv_idx + 1..end_inv_idx].align_to_mut().1
                            };

                            subinventory[subinventory_idx] = sub_offset as u16;
                            subinventory_idx += 1;
                            // This exit is not necessary for correctness, but
                            // it avoids the additional loop iterations that
                            // would be necessary to find the position of the
                            // next one (i.e., end_bit_idx).
                            if subinventory_idx << log2_quantum == (1 << LOG2_ONES_PER_INVENTORY) {
                                break 'outer;
                            }
                        }
                        SpanType::U32 => {
                            if subinventory_idx < 2 * (u64_per_subinventory - 1) {
                                let subinventory: &mut [u32] = unsafe {
                                    inventory[start_inv_idx + 2..end_inv_idx].align_to_mut().1
                                };

                                debug_assert_eq!(subinventory[subinventory_idx], 0);
                                subinventory[subinventory_idx] = sub_offset as u32;
                                subinventory_idx += 1;
                            } else {
                                debug_assert!(spilled < spill_size);
                                // Maybe pointer dereferencing?
                                let u32_spill: &mut [u32] = unsafe { spill.align_to_mut().1 };
                                debug_assert_eq!(
                                    u32_spill[spilled * 2 + u32_odd_spill as usize],
                                    0
                                );
                                u32_spill[spilled * 2 + u32_odd_spill as usize] = sub_offset as u32;
                                spilled += u32_odd_spill as usize;
                                u32_odd_spill = !u32_odd_spill;
                                subinventory_idx += 1;
                            }
                            // This exit is not necessary for correctness, but
                            // it avoids the additional loop iterations that
                            // would be necessary to find the position of the
                            // next one (i.e., end_bit_idx).
                            if subinventory_idx << log2_quantum == (1 << LOG2_ONES_PER_INVENTORY) {
                                break 'outer;
                            }
                        }
                        SpanType::U64 => {
                            if subinventory_idx < u64_per_subinventory {
                                inventory[start_inv_idx + 1 + subinventory_idx] = bit_index;
                                subinventory_idx += 1;
                            } else {
                                assert!(spilled < spill_size);
                                spill[spilled] = bit_index;
                                spilled += 1;
                            }
                        }
                    }

                    next_quantum += quantum;
                }

                // We are done with the word, so update the number of ones
                past_ones += ones_in_word;
                // Move to the next word and check whether it is the last one
                word_idx += 1;
                if word_idx == end_word_idx {
                    break;
                }

                // Read the next word
                word = !bits.as_ref()[word_idx];
            }

            // This test is necessary to handle two corner cases for 32-bit
            // spans:
            // - the case in which the last inventory entry has a 32-bit span
            // and contains an odd number of ones greater than one, as spilled
            // is not incremented by the loop code (note that u32_odd_spill =>
            // span_type == SpanType::U32);
            // - the case in which an inventory entry contains a single one, as
            // its zero subinventory entry is written implicitly, but we still
            // need to increment spilled to allocate space for it.
            if u32_odd_spill || span_type == SpanType::U32 && subinventory_idx == 1 {
                spilled += 1;
            }
        }

        debug_assert_eq!(spilled, spill_size);

        Self {
            bits,
            inventory,
            spill,
            log2_ones_per_sub16,
            u64_per_inventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
        }
    }
}

impl<
        B: SelectZeroHinted + AsRef<[usize]> + BitLength,
        I: AsRef<[usize]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZeroUnchecked
    for SelectZeroAdaptConst<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        let inventory = self.inventory.as_ref();
        let inventory_index = rank >> LOG2_ONES_PER_INVENTORY;
        let inventory_start_pos = (inventory_index << LOG2_U64_PER_SUBINVENTORY) + inventory_index;

        let inventory_rank = { *inventory.get_unchecked(inventory_start_pos) };
        let subrank = rank & self.ones_per_inventory_mask;

        if inventory_rank.is_u16_span() {
            let subinventory = inventory
                .get_unchecked(inventory_start_pos + 1..)
                .align_to::<u16>()
                .1;

            debug_assert!(subrank >> self.log2_ones_per_sub16 < subinventory.len());

            let hint_pos = inventory_rank
                + *subinventory.get_unchecked(subrank >> self.log2_ones_per_sub16) as usize;
            let residual = subrank & self.ones_per_sub16_mask;

            return self
                .bits
                .select_zero_hinted_unchecked(rank, hint_pos, rank - residual);
        }

        let u64_per_subinventory = 1 << LOG2_U64_PER_SUBINVENTORY;

        if inventory_rank.is_u32_span() {
            let inventory_rank = inventory_rank.get();

            let span = (*inventory.get_unchecked(inventory_start_pos + self.u64_per_inventory))
                .get()
                - inventory_rank;
            let log2_ones_per_sub32 = log2_ones_per_sub32(span, self.log2_ones_per_sub16);
            let hint_pos = if subrank >> log2_ones_per_sub32 < (u64_per_subinventory - 1) * 2 {
                let u32s = inventory
                    .get_unchecked(inventory_start_pos + 2..)
                    .align_to::<u32>()
                    .1;

                inventory_rank + *u32s.get_unchecked(subrank >> log2_ones_per_sub32) as usize
            } else {
                let inventory_rank = inventory_rank.get();
                let start_spill_idx = *inventory.get_unchecked(inventory_start_pos + 1);

                let spilled_u32s = self
                    .spill
                    .as_ref()
                    .get_unchecked(start_spill_idx..)
                    .align_to::<u32>()
                    .1;

                inventory_rank
                    + *spilled_u32s.get_unchecked(
                        (subrank >> log2_ones_per_sub32) - (u64_per_subinventory - 1) * 2,
                    ) as usize
            };
            let residual = subrank & ((1 << log2_ones_per_sub32) - 1);
            return self
                .bits
                .select_zero_hinted_unchecked(rank, hint_pos, rank - residual);
        }

        debug_assert!(inventory_rank.is_u64_span());
        let inventory_rank = inventory_rank.get();

        if subrank < u64_per_subinventory {
            if subrank == 0 {
                return inventory_rank;
            }
            return *inventory.get_unchecked(inventory_start_pos + 1 + subrank);
        }
        let spill_idx =
            { *inventory.get_unchecked(inventory_start_pos + 1) } + subrank - u64_per_subinventory;
        debug_assert!(spill_idx < self.spill.as_ref().len());
        self.spill.get_unchecked(spill_idx)
    }
}

impl<
        B: SelectZeroHinted + AsRef<[usize]> + NumBits,
        I: AsRef<[usize]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZero
    for SelectZeroAdaptConst<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
}

crate::forward_mult![
    SelectZeroAdaptConst<B, I, [const] LOG2_ONES_PER_INVENTORY: usize, [const] LOG2_U64_PER_SUBINVENTORY: usize>; B; bits;
    crate::forward_as_ref_slice_usize,
    crate::forward_index_bool,
    crate::traits::rank_sel::forward_bit_length,
    crate::traits::rank_sel::forward_bit_count,
    crate::traits::rank_sel::forward_num_bits,
    crate::traits::rank_sel::forward_rank,
    crate::traits::rank_sel::forward_rank_hinted,
    crate::traits::rank_sel::forward_rank_zero,
    crate::traits::rank_sel::forward_select,
    crate::traits::rank_sel::forward_select_unchecked,
    crate::traits::rank_sel::forward_select_hinted,
    crate::traits::rank_sel::forward_select_zero_hinted
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::bits::BitVec;
    use crate::traits::AddNumBits;
    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_extremely_sparse() {
        let len = 1 << 18;
        let bits: AddNumBits<BitVec> = (0..len / 2)
            .map(|_| false)
            .chain([true])
            .chain((0..1 << 17).map(|_| false))
            .chain([true, true])
            .chain((0..1 << 18).map(|_| false))
            .chain([true])
            .chain((0..len / 2).map(|_| false))
            .map(|x| !x)
            .collect::<BitVec>()
            .into();
        let simple = SelectZeroAdaptConst::<_, _, 13, 0>::new(bits);

        assert_eq!(simple.count_ones(), 4);
        assert_eq!(simple.select_zero(0), Some(len / 2));
        assert_eq!(simple.select_zero(1), Some(len / 2 + (1 << 17) + 1));
        assert_eq!(simple.select_zero(2), Some(len / 2 + (1 << 17) + 2));
    }

    #[test]
    fn test_sub32s() {
        let lens = [1_000_000];
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.1;
        for len in lens {
            let bits: AddNumBits<BitVec> = (0..len)
                .map(|_| rng.gen_bool(density))
                .map(|x| !x)
                .collect::<BitVec>()
                .into();
            let simple = SelectZeroAdaptConst::<_, _, 13, 3>::new(bits.clone());

            let ones = simple.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(simple.select_zero(i), Some(pos[i]));
            }
            assert_eq!(simple.select_zero(ones + 1), None);
        }
    }

    #[test]
    fn test_sub64s() {
        let len = 5_000_000_000;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut bits = BitVec::new(len);
        let mut pos = BTreeSet::new();
        for _ in 0..(1 << 13) / 4 * 3 {
            let p = rng.gen_range(0..len);
            if pos.insert(p) {
                bits.set(p, true);
            }
        }
        bits.flip();
        let bits: AddNumBits<BitVec> = bits.into();

        let simple = SelectZeroAdaptConst::<_, _, 13, 0>::new(&bits);
        assert!(simple.inventory[0].is_u64_span());

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p));
        }
        assert_eq!(simple.select_zero(pos.len()), None);
        let simple = SelectZeroAdaptConst::<_, _, 13, 3>::new(&bits);
        assert!(simple.inventory[0].is_u64_span());

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p));
        }
        assert_eq!(simple.select_zero(pos.len()), None);
        let simple = SelectZeroAdaptConst::<_, _, 13, 16>::new(&bits);
        assert!(simple.inventory[0].is_u64_span());

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p));
        }
        assert_eq!(simple.select_zero(pos.len()), None);
    }
}
