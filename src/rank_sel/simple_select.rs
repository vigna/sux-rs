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

use crate::prelude::{BitCount, BitFieldSlice, BitLength, Select, SelectHinted};

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
/// subinventories is `L/4M`, (again, the actual value might be twice as large
/// because of rounding). However, the worst-case distance might as high as
/// 2¹⁶/4M, as we use `4M` 16-bit integers until the width of the inventory span
/// makes it possible. Within this range, we perform a sequential broadword
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
/// for `M` is between 4 and 32, corrisponding to worst-case linear searches
/// between 1024 and 128 bits (note that the constructors take the base-2
/// logarithm of `M`)
///
/// # Examples
/// ```rust
/// use sux::bit_vec;
/// use sux::traits::{Rank, Select};
/// use sux::rank_sel::{SimpleSelect, Rank9};
///
/// // Standalone select
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let select = SimpleSelect::new(bits, 3);
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
/// let rank9_sel = SimpleSelect::new(rank9, 3);
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
pub struct SimpleSelect<B, I = Box<[usize]>> {
    bits: B,
    inventory: I,
    spill: I,
    num_ones: usize,
    log2_ones_per_inventory: usize,
    log2_ones_per_sub16: usize,
    log2_u64_per_subinventory: usize,
    u64_per_inventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
}

trait Inventory {
    fn is_16_bit_span(&self) -> bool;
    fn is_32_bit_span(&self) -> bool;
    fn is_64_bit_span(&self) -> bool;
    fn set_16_bit_span(&mut self);
    fn set_32_bit_span(&mut self);
    fn set_64_bit_span(&mut self);
    fn get(&self) -> usize;
}

impl Inventory for usize {
    #[inline(always)]
    fn is_16_bit_span(&self) -> bool {
        *self as isize >= 0
    }

    #[inline(always)]
    fn is_32_bit_span(&self) -> bool {
        *self >> 62 == 2
    }

    #[inline(always)]
    fn is_64_bit_span(&self) -> bool {
        *self >> 62 == 3
    }

    #[inline(always)]
    fn set_16_bit_span(&mut self) {}

    #[inline(always)]
    fn set_32_bit_span(&mut self) {
        *self |= 1 << 63;
    }

    #[inline(always)]
    fn set_64_bit_span(&mut self) {
        *self |= 3 << 62;
    }

    #[inline(always)]
    fn get(&self) -> usize {
        *self & 0x3FFF_FFFF_FFFF_FFFF
    }
}

#[inline(always)]
fn calc_log2_ones_per_sub32(span: usize, log2_ones_per_sub16: usize) -> usize {
    debug_assert!(span >= 1 << 16);
    // Since span >= 2^16, (span >> 15).ilog2() >= 1, which implies in any case
    // at least doubling the frequency of the subinventory, unless
    // log2_ones_per_u16 = 0, that is, we are inventoring the whole span.
    log2_ones_per_sub16 - ((span >> 15).ilog2() as usize).min(log2_ones_per_sub16)
}

impl<B, I> SimpleSelect<B, I> {
    pub fn into_inner(self) -> B {
        self.bits
    }

    /// Replaces the backend with a new one implementing [`SelectHinted`].
    pub fn map<C>(self, f: impl FnOnce(B) -> C) -> SimpleSelect<C, I>
    where
        C: SelectHinted,
    {
        SimpleSelect {
            bits: f(self.bits),
            inventory: self.inventory,
            spill: self.spill,
            num_ones: self.num_ones,
            log2_ones_per_inventory: self.log2_ones_per_inventory,
            log2_ones_per_sub16: self.log2_ones_per_sub16,
            log2_u64_per_subinventory: self.log2_u64_per_subinventory,
            u64_per_inventory: self.u64_per_inventory,
            ones_per_inventory_mask: self.ones_per_inventory_mask,
            ones_per_sub16_mask: self.ones_per_sub16_mask,
        }
    }

    pub const DEFAULT_TARGET_INVENTORY_SPAN: usize = 8192;
}

impl<B: BitLength, I> BitCount for SimpleSelect<B, I> {
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.num_ones
    }
}

impl<B: AsRef<[usize]> + BitLength + BitCount + SelectHinted> SimpleSelect<B, Box<[usize]>> {
    /// Creates a new selection structure over a [`SelectHinted`] using a
    /// [default target inventory
    /// span](SimpleSelect::DEFAULT_TARGET_INVENTORY_SPAN).
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
    pub fn new(bits: B, max_log2_u64_per_subinventory: usize) -> Self {
        Self::with_span(
            bits,
            Self::DEFAULT_TARGET_INVENTORY_SPAN,
            max_log2_u64_per_subinventory,
        )
    }

    /// Creates a new selection structure over a [`SelectHinted`] with a specified
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
        bits: B,
        target_inventory_span: usize,
        max_log2_u64_per_subinventory: usize,
    ) -> Self {
        // TODO: is this necessary? (everywhere)
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
    /// [standard constructor](SimpleSelect::new).
    ///
    /// # Arguments
    ///
    /// * `bits`: A [`SelectHinted`].
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
        bits: B,
        num_ones: usize,
        log2_ones_per_inventory: usize,
        max_log2_u64_per_subinventory: usize,
    ) -> Self {
        let num_bits = max(1, bits.len());
        let ones_per_inventory = 1 << log2_ones_per_inventory;
        let ones_per_inventory_mask = ones_per_inventory - 1;
        let inventory_size = num_ones.div_ceil(ones_per_inventory);

        // We use a smaller value than max_log2_u64_per_subinventory when with a
        // smaller value we can still index, in the 16-bit case, all bits the
        // subinventory. This can happen only in extremely sparse vectors, or
        // if a very small value of log2_ones_per_inventory is set directly.
        //
        // Note that as a consequence the subinventory space is never large
        // enough to hold all 32-bit or 64-bit entries, so we always need a
        // spill pointer, and we can avoid handling the case without one.

        // log2_u64_per_subinventory = max_log2_u64_per_subinventory.min(max(0,
        //    log2_ones_per_inventory - 2));
        let log2_u64_per_subinventory =
            max_log2_u64_per_subinventory.min(max(2, log2_ones_per_inventory) - 2);

        let u64_per_subinventory = 1 << log2_u64_per_subinventory;
        // A u64 for the inventory, and u64_per_inventory for the subinventory
        let u64_per_inventory = u64_per_subinventory + 1;

        //let log2_ones_per_sub16 = (log2_ones_per_inventory -
        //    (log2_u64_per_subinventory + 2)).max(0);
        let log2_ones_per_sub16 = (log2_ones_per_inventory - log2_u64_per_subinventory).max(2) - 2;
        let ones_per_sub16 = 1 << log2_ones_per_sub16;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let inventory_words = inventory_size * u64_per_inventory + 1;
        let mut inventory = Vec::with_capacity(inventory_words);

        let mut past_ones: usize = 0;
        let mut next_quantum: usize = 0;
        let mut spilled = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        for (i, word) in bits.as_ref().iter().copied().enumerate() {
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

            if span > u16::MAX as usize + 1 {
                if span <= u32::MAX as usize + 1 {
                    let log2_ones_per_sub32 = calc_log2_ones_per_sub32(span, log2_ones_per_sub16);
                    let num_u64s =
                        u64_per_subinventory << (1 + log2_ones_per_sub16 - log2_ones_per_sub32);
                    let spilled_u64s = num_u64s - u64_per_subinventory + 1;
                    spilled += spilled_u64s;
                } else {
                    // we store the exact positions of the ones first in
                    // the subinventory and then in the spill buffer.
                    // the first u64 word is used to store the spill index
                    spilled += max(0, ones - (u64_per_subinventory - 1));
                }
            }
        }

        let spill_size = spilled;

        let mut inventory: Box<[usize]> = inventory.into();
        let mut spill: Box<[usize]> = vec![0; spill_size].into();

        spilled = 0;

        // Second phase: we fill the subinventories and the exact spill
        for inventory_idx in 0..inventory_size {
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

            if span <= u16::MAX as usize + 1 {
                quantum = ones_per_sub16;
                inventory[start_idx].set_16_bit_span();
            } else if span <= u32::MAX as usize + 1 {
                let log2_ones_per_sub32 = calc_log2_ones_per_sub32(span, log2_ones_per_sub16);
                let ones_per_sub32 = 1 << log2_ones_per_sub32;
                quantum = ones_per_sub32;
                inventory[start_idx].set_32_bit_span();
                // the last word of the subinventory is used to store the spill index
                inventory[end_idx - 1] = spilled;
            } else {
                quantum = 1;
                inventory[start_idx].set_64_bit_span();
                // the first word of the subinventory is used to store the spill index
                inventory[start_idx + 1] = spilled;
            }

            let end_word_idx = end_bit_idx.div_ceil(usize::BITS as usize);

            // the first subinventory element is always 0 if the span is less than 2^32
            // otherwise it is the spill index
            // TODO this should be always true
            let mut subinventory_idx = 1;

            next_quantum += quantum;

            let mut u32_odd_spill = false;

            'outer: loop {
                let ones_in_word = word.count_ones() as usize;

                // if the quantum is in this word, write it in the subinventory
                // this can happen multiple times if the quantum is small
                while number_of_ones + ones_in_word > next_quantum {
                    debug_assert!(next_quantum <= end_bit_idx as _);
                    // find the quantum bit in the word
                    let in_word_index = word.select_in_word(next_quantum - number_of_ones);
                    // compute the global index of the quantum bit in the bitvec
                    let bit_index = (word_idx * usize::BITS as usize) + in_word_index;
                    // TODO: Is this necessary?
                    if bit_index >= end_bit_idx {
                        break 'outer;
                    }
                    // compute the offset of the quantum bit
                    // from the start of the subinventory
                    let sub_offset = bit_index - start_bit_idx;

                    if span <= u16::MAX as usize + 1 {
                        let subinventory: &mut [u16] =
                            unsafe { inventory[start_idx + 1..end_idx].align_to_mut().1 };

                        subinventory[subinventory_idx] = sub_offset as u16;

                        subinventory_idx += 1;
                        // TODO avoid division
                        if subinventory_idx == (1 << log2_ones_per_inventory) / quantum {
                            break 'outer;
                        }
                    } else if span <= u32::MAX as usize + 1 {
                        if subinventory_idx < 2 * (u64_per_subinventory - 1) {
                            let subinventory: &mut [u32] =
                                unsafe { inventory[start_idx + 1..(end_idx - 1)].align_to_mut().1 };

                            debug_assert_eq!(subinventory[subinventory_idx], 0);
                            subinventory[subinventory_idx] = sub_offset as u32;
                            subinventory_idx += 1;
                        } else {
                            debug_assert!(spilled < spill_size);
                            // Maybe pointer dereferencing?
                            let u32_spill: &mut [u32] = unsafe { spill.align_to_mut().1 };
                            debug_assert_eq!(u32_spill[spilled * 2 + u32_odd_spill as usize], 0);
                            u32_spill[spilled * 2 + u32_odd_spill as usize] = sub_offset as u32;
                            spilled += u32_odd_spill as usize;
                            u32_odd_spill = !u32_odd_spill;
                            subinventory_idx += 1;
                        }
                        if subinventory_idx == (1 << log2_ones_per_inventory) / quantum {
                            break 'outer;
                        }
                    } else {
                        if subinventory_idx < u64_per_subinventory {
                            inventory[start_idx + 1 + subinventory_idx] = bit_index;
                            subinventory_idx += 1;
                        } else {
                            assert!(spilled < spill_size);
                            spill[spilled] = bit_index;
                            spilled += 1;
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
        }

        // TODO: why does this assertion fail?
        // debug_assert_eq!(spilled, spill_size);

        Self {
            bits,
            inventory,
            spill,
            num_ones,
            log2_ones_per_inventory,
            log2_ones_per_sub16,
            log2_u64_per_subinventory,
            u64_per_inventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
        }
    }

    pub fn log2_ones_per_inventory(&self) -> usize {
        self.log2_ones_per_inventory
    }
    pub fn log2_u64_per_subinventory(&self) -> usize {
        self.log2_u64_per_subinventory
    }
}

impl<B: SelectHinted + AsRef<[usize]> + BitLength + BitCount, I: AsRef<[usize]>> Select
    for SimpleSelect<B, I>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let inventory = self.inventory.as_ref();
        let inventory_index = rank >> self.log2_ones_per_inventory;
        let inventory_start_pos =
            (inventory_index << self.log2_u64_per_subinventory) + inventory_index;

        let inventory_rank = { *inventory.get_unchecked(inventory_start_pos) };
        let subrank = rank & self.ones_per_inventory_mask;

        if inventory_rank.is_16_bit_span() {
            let subinventory = inventory
                .get_unchecked(
                    inventory_start_pos + 1..inventory_start_pos + self.u64_per_inventory,
                )
                .align_to::<u16>()
                .1;

            debug_assert!(subrank >> self.log2_ones_per_sub16 < subinventory.len());

            let hint_pos = inventory_rank
                + *subinventory.get_unchecked(subrank >> self.log2_ones_per_sub16) as usize;
            let residual = subrank & self.ones_per_sub16_mask;

            return self
                .bits
                .select_hinted_unchecked(rank, hint_pos, rank - residual);
        }

        let u64_per_subinventory = 1 << self.log2_u64_per_subinventory;

        if inventory_rank.is_32_bit_span() {
            let inventory_rank = inventory_rank.get();
            let hint_pos;
            let span = (*inventory.get_unchecked(inventory_start_pos + self.u64_per_inventory))
                .get()
                - inventory_rank;
            let log2_ones_per_sub32 = calc_log2_ones_per_sub32(span, self.log2_ones_per_sub16);
            let ones_per_sub32 = 1 << log2_ones_per_sub32;
            if subrank < ones_per_sub32 * (u64_per_subinventory - 1) * 2 {
                let u32s = inventory
                    .get_unchecked(
                        inventory_start_pos + 1..(inventory_start_pos + 1 + u64_per_subinventory),
                    )
                    .align_to::<u32>()
                    .1;

                hint_pos =
                    inventory_rank + *u32s.get_unchecked(subrank >> log2_ones_per_sub32) as usize;
            } else {
                let inventory_rank = inventory_rank.get();
                let start_spill_idx =
                    *inventory.get_unchecked(inventory_start_pos + u64_per_subinventory);

                let spilled_u32s = self
                    .spill
                    .as_ref()
                    .get_unchecked(start_spill_idx..self.spill.as_ref().len()) // TODO: Maybe exact value?
                    .align_to::<u32>()
                    .1;

                hint_pos = inventory_rank
                    + *spilled_u32s.get_unchecked(
                        (subrank >> log2_ones_per_sub32) - (u64_per_subinventory - 1) * 2,
                    ) as usize;
            }
            let residual = subrank & ((1 << log2_ones_per_sub32) - 1);
            return self
                .bits
                .select_hinted_unchecked(rank, hint_pos, rank - residual);
        }

        debug_assert!(inventory_rank.is_64_bit_span());
        let inventory_rank = inventory_rank.get();

        if subrank == 0 {
            return inventory_rank;
        }
        if subrank < u64_per_subinventory {
            return *inventory.get_unchecked(inventory_start_pos + 1 + subrank);
        }
        let spill_idx =
            { *inventory.get_unchecked(inventory_start_pos + 1) } + subrank - u64_per_subinventory;
        debug_assert!(spill_idx < self.spill.as_ref().len());
        self.spill.get_unchecked(spill_idx)
    }
}

crate::forward_mult![
    SimpleSelect<B, I>; B; bits;
    crate::forward_as_ref_slice_usize,
    crate::forward_index_bool,
    crate::traits::rank_sel::forward_bit_length,
    crate::traits::rank_sel::forward_rank,
    crate::traits::rank_sel::forward_rank_hinted,
    crate::traits::rank_sel::forward_rank_zero,
    crate::traits::rank_sel::forward_select_zero,
    crate::traits::rank_sel::forward_select_hinted,
    crate::traits::rank_sel::forward_select_zero_hinted
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::bits::BitVec;
    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_simple_select_ones_per_sub64() {
        // TODO: What are we testing here?
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
        let simple = SimpleSelect::new(bits, 3);

        assert_eq!(simple.count_ones(), 4);
        assert_eq!(simple.select(0), Some(len / 2));
        assert_eq!(simple.select(1), Some(len / 2 + (1 << 17) + 1));
        assert_eq!(simple.select(2), Some(len / 2 + (1 << 17) + 2));
    }

    #[test]
    fn test_sub32s() {
        let lens = [1_000_000];
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.1;
        for len in lens {
            let bits: BitVec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let simple = SimpleSelect::with_inv(bits.clone(), bits.count_ones(), 13, 3);

            let ones = simple.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(simple.select(i), Some(pos[i]));
            }
            assert_eq!(simple.select(ones + 1), None);
        }
    }

    #[test]
    fn test_sub64s() {
        let len = 5_000_000_000;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut bits: BitVec = BitVec::new(len);
        let mut pos = BTreeSet::new();
        for _ in 0..(1 << 13) / 4 * 3 {
            let p = rng.gen_range(0..len);
            if pos.insert(p) {
                bits.set(p, true);
            }
        }

        for m in [0, 3, 16] {
            let simple = SimpleSelect::with_inv(&bits, pos.len(), 13, m);
            assert!(simple.inventory[0].is_64_bit_span());

            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(simple.select(i), Some(p));
            }
            assert_eq!(simple.select(pos.len()), None);
        }
    }
}
