/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use super::{Inventory, LOG2_U16_PER_USIZE, SpanType, U32_PER_USIZE, assert_inventory_length};
use crate::utils::SelectInWord;
use ambassador::Delegate;
use mem_dbg::{MemDbg, MemSize};
use num_primitive::PrimitiveInteger;
use std::{
    cmp::{max, min},
    ops::Deref,
};

use crate::{
    prelude::{BitCount, BitLength, SelectZeroHinted},
    traits::{
        Backend, NumBits, Rank, RankHinted, RankUnchecked, RankZero, Select, SelectHinted,
        SelectUnchecked, SelectZero, SelectZeroUnchecked, Word,
    },
};

use crate::ambassador_impl_Index;
use crate::traits::ambassador_impl_Backend;
use crate::traits::bal_paren::{BalParen, ambassador_impl_BalParen};
use crate::traits::bit_vec_ops::ambassador_impl_BitLength;
use crate::traits::rank_sel::ambassador_impl_BitCount;
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

// NOTE: to make parallel modifications with SelectAdapt as easy as possible,
// "ones" are considered to be zeros in the following code.

/// A version of [`SelectAdapt`](super::SelectAdapt) implementing [selection on
/// zeros](crate::traits::SelectZero).
///
/// # Examples
/// ```rust
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// # use sux::bit_vec;
/// # use sux::traits::{Rank, SelectZero, SelectZeroUnchecked, AddNumBits};
/// # use sux::rank_sel::{SelectZeroAdapt, Rank9};
/// // Standalone select
/// let bits = bit_vec![0, 1, 0, 0, 1, 0, 1, 0];
/// let select = SelectZeroAdapt::new(bits);
///
/// // If the backend does not implement NumBits
/// // we just get SelectZeroUnchecked
/// unsafe {
///     assert_eq!(select.select_zero_unchecked(0), 0);
///     assert_eq!(select.select_zero_unchecked(1), 2);
///     assert_eq!(select.select_zero_unchecked(2), 3);
///     assert_eq!(select.select_zero_unchecked(3), 5);
///     assert_eq!(select.select_zero_unchecked(4), 7);
/// }
///
/// // Let's add NumBits to the backend
/// let bits: AddNumBits<_> = bit_vec![0, 1, 0, 0, 1, 0, 1, 0].into();
/// let select = SelectZeroAdapt::new(bits);
///
/// assert_eq!(select.select_zero(0), Some(0));
/// assert_eq!(select.select_zero(1), Some(2));
/// assert_eq!(select.select_zero(2), Some(3));
/// assert_eq!(select.select_zero(3), Some(5));
/// assert_eq!(select.select_zero(4), Some(7));
/// assert_eq!(select.select_zero(5), None);
///
/// // Access to the underlying bit vector is forwarded, too
/// assert_eq!(select[0], false);
/// assert_eq!(select[1], true);
/// assert_eq!(select[2], false);
/// assert_eq!(select[3], false);
/// assert_eq!(select[4], true);
/// assert_eq!(select[5], false);
/// assert_eq!(select[6], true);
/// assert_eq!(select[7], false);
///
/// // Map the backend to a different structure
/// let sel_rank9 = unsafe { select.map(Rank9::new) };
///
/// // Rank methods are forwarded
/// assert_eq!(sel_rank9.rank(0), 0);
/// assert_eq!(sel_rank9.rank(1), 0);
/// assert_eq!(sel_rank9.rank(2), 1);
/// assert_eq!(sel_rank9.rank(3), 1);
/// assert_eq!(sel_rank9.rank(4), 1);
/// assert_eq!(sel_rank9.rank(5), 2);
/// assert_eq!(sel_rank9.rank(6), 2);
/// assert_eq!(sel_rank9.rank(7), 3);
/// assert_eq!(sel_rank9.rank(8), 3);
///
/// // Select over a Rank9 structure
/// let rank9 = Rank9::new(sel_rank9.into_inner().into_inner());
/// let rank9_sel = SelectZeroAdapt::new(rank9);
///
/// assert_eq!(rank9_sel.select_zero(0), Some(0));
/// assert_eq!(rank9_sel.select_zero(1), Some(2));
/// assert_eq!(rank9_sel.select_zero(2), Some(3));
/// assert_eq!(rank9_sel.select_zero(3), Some(5));
/// assert_eq!(rank9_sel.select_zero(4), Some(7));
/// assert_eq!(rank9_sel.select_zero(5), None);
///
/// // Rank methods are forwarded
/// assert_eq!(rank9_sel.rank(0), 0);
/// assert_eq!(rank9_sel.rank(1), 0);
/// assert_eq!(rank9_sel.rank(2), 1);
/// assert_eq!(rank9_sel.rank(3), 1);
/// assert_eq!(rank9_sel.rank(4), 1);
/// assert_eq!(rank9_sel.rank(5), 2);
/// assert_eq!(rank9_sel.rank(6), 2);
/// assert_eq!(rank9_sel.rank(7), 3);
/// assert_eq!(rank9_sel.rank(8), 3);
///
/// // Access to the underlying bit vector is forwarded, too
/// assert_eq!(rank9_sel[0], false);
/// assert_eq!(rank9_sel[1], true);
/// assert_eq!(rank9_sel[2], false);
/// assert_eq!(rank9_sel[3], false);
/// assert_eq!(rank9_sel[4], true);
/// assert_eq!(rank9_sel[5], false);
/// assert_eq!(rank9_sel[6], true);
/// assert_eq!(rank9_sel[7], false);
/// # }
/// ```

#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(crate::traits::Backend, target = "bits")]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::rank_sel::BitCount, target = "bits")]
#[delegate(crate::traits::bit_vec_ops::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::NumBits, target = "bits")]
#[delegate(crate::traits::rank_sel::Rank, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::RankUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::RankZero, target = "bits")]
#[delegate(crate::traits::rank_sel::Select, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
#[delegate(crate::bal_paren::BalParen, target = "bits")]
pub struct SelectZeroAdapt<B, I = Box<[usize]>> {
    bits: B,
    inventory: I,
    spill: I,
    log2_ones_per_inventory: usize,
    log2_ones_per_sub16: usize,
    log2_words_per_subinventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
}

impl<B: Backend + AsRef<[B::Word]>, I> AsRef<[B::Word]> for SelectZeroAdapt<B, I> {
    #[inline(always)]
    fn as_ref(&self) -> &[B::Word] {
        self.bits.as_ref()
    }
}

impl<B, I> Deref for SelectZeroAdapt<B, I> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

impl<B, I> SelectZeroAdapt<B, I> {
    /// Returns the underlying bit vector, consuming this structure.
    pub fn into_inner(self) -> B {
        self.bits
    }

    // Compute adaptively the number of 32-bit subinventory entries
    #[inline(always)]
    const fn log2_ones_per_sub32(span: usize, log2_ones_per_sub16: usize) -> usize {
        debug_assert!(span > 1 << 16);
        // Since span > 2^16, (span >> 15).ilog2() >= 0, which implies in any case
        // at least doubling the frequency of the subinventory with respect to the
        // 16-bit case, unless log2_ones_per_u16 = 0, that is, we are recording the
        // position of every one in the subinventory.
        log2_ones_per_sub16.saturating_sub((span >> 15).ilog2() as usize + 1)
    }
    /// Replaces the backend with a new one implementing [`SelectZeroHinted`].
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new backend is identical to the old one as a bit vector.
    pub unsafe fn map<C: SelectZeroHinted>(self, f: impl FnOnce(B) -> C) -> SelectZeroAdapt<C, I> {
        SelectZeroAdapt {
            bits: f(self.bits),
            inventory: self.inventory,
            spill: self.spill,
            log2_ones_per_inventory: self.log2_ones_per_inventory,
            log2_ones_per_sub16: self.log2_ones_per_sub16,
            log2_words_per_subinventory: self.log2_words_per_subinventory,
            ones_per_inventory_mask: self.ones_per_inventory_mask,
            ones_per_sub16_mask: self.ones_per_sub16_mask,
        }
    }
}

impl<B: BitLength, C> SelectZeroAdapt<B, C> {
    /// Returns the number of bits in the underlying bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + BitCount>
    SelectZeroAdapt<B, Box<[usize]>>
{
    /// See [`SelectAdapt::new`](super::SelectAdapt::new).
    pub fn new(bits: B) -> Self {
        Self::with_span(
            bits,
            super::select_adapt::default_target_inventory_span(
                super::select_adapt::DEFAULT_LOG2_WORDS_PER_SUBINVENTORY,
            ),
            super::select_adapt::DEFAULT_LOG2_WORDS_PER_SUBINVENTORY,
        )
    }

    /// See [`SelectAdapt::with_span`](super::SelectAdapt::with_span).
    pub fn with_span(
        bits: B,
        target_inventory_span: usize,
        max_log2_words_per_subinventory: usize,
    ) -> Self {
        assert_inventory_length(bits.len());
        let num_bits = max(1usize, bits.len());
        let num_ones = bits.count_zeros();

        let log2_ones_per_inventory = (num_ones as u128 * target_inventory_span as u128)
            .div_ceil(num_bits as u128)
            .max(1)
            .ilog2() as usize;

        Self::_new(
            bits,
            num_ones,
            log2_ones_per_inventory,
            max_log2_words_per_subinventory,
        )
    }

    /// See [`SelectAdapt::with_inv`](super::SelectAdapt::with_inv).
    pub fn with_inv(
        bits: B,
        log2_ones_per_inventory: usize,
        max_log2_words_per_subinventory: usize,
    ) -> Self {
        assert_inventory_length(bits.len());
        let num_ones = bits.count_zeros();
        Self::_new(
            bits,
            num_ones,
            log2_ones_per_inventory,
            max_log2_words_per_subinventory,
        )
    }

    /// See [`SelectAdapt::with_overhead`](super::SelectAdapt::with_overhead).
    pub fn with_overhead(
        bits: B,
        overhead_percentage: f64,
        max_log2_words_per_subinv: usize,
    ) -> Self {
        assert!(
            overhead_percentage > 0.0,
            "overhead_percentage must be positive"
        );
        let m = 1usize << max_log2_words_per_subinv;

        let target_span =
            ((1 + m) as f64 * usize::BITS as f64 * 100.0 / overhead_percentage) as usize;
        let min_span = m * (usize::BITS as usize * usize::BITS as usize) / 16;

        Self::with_span(bits, target_span.max(min_span), max_log2_words_per_subinv)
    }

    fn _new(
        bits: B,
        num_ones: usize,
        log2_ones_per_inventory: usize,
        max_log2_words_per_subinventory: usize,
    ) -> Self {
        assert_inventory_length(bits.len());
        let num_bits = max(1, bits.len());
        let ones_per_inventory = 1 << log2_ones_per_inventory;
        let ones_per_inventory_mask = ones_per_inventory - 1;
        let inventory_size = num_ones.div_ceil(ones_per_inventory);

        // We use a smaller value than max_log2_words_per_subinventory when with a
        // smaller value we can still index, in the 16-bit case, all bits the
        // subinventory. This can happen only in extremely sparse vectors, or
        // if a very small value of log2_ones_per_inventory is set directly.

        let log2_words_per_subinventory =
            max_log2_words_per_subinventory.min(log2_ones_per_inventory.saturating_sub(2));

        let words_per_subinventory = 1 << log2_words_per_subinventory;
        // A u64 for the inventory, and words_per_inventory for the subinventory
        let words_per_inventory = words_per_subinventory + 1;

        let log2_ones_per_sub16 = log2_ones_per_inventory
            .saturating_sub(log2_words_per_subinventory + LOG2_U16_PER_USIZE);
        let ones_per_sub16 = 1 << log2_ones_per_sub16;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let inventory_words = inventory_size * words_per_inventory + 1;
        let mut inventory: Vec<usize> = Vec::with_capacity(inventory_words);

        let bits_per_word = B::Word::BITS as usize;

        let mut past_ones = 0;
        let mut next_quantum = 0;
        let mut spilled = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        for (i, word) in bits.as_ref().iter().copied().map(|b| !b).enumerate() {
            let ones_in_word = (word.count_ones() as usize).min(num_ones - past_ones);

            while past_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word(next_quantum - past_ones);
                let index = (i * bits_per_word) + in_word_index;

                // write the position of the one in the inventory
                inventory.push(index);
                // make space for the subinventory
                inventory.resize(inventory.len() + words_per_subinventory, 0);

                next_quantum += ones_per_inventory;
            }
            past_ones += ones_in_word;
        }

        assert_eq!(past_ones, num_ones);
        // in the last inventory write the number of bits
        inventory.push(num_bits);
        assert_eq!(inventory.len(), inventory_words);

        // We estimate the subinventory and exact spill size
        for (i, inv) in inventory[..inventory_size * words_per_inventory]
            .iter()
            .copied()
            .step_by(words_per_inventory)
            .enumerate()
        {
            let start = inv;
            let span = inventory[i * words_per_inventory + words_per_inventory] - start;
            past_ones = i * ones_per_inventory;
            let ones = min(num_ones - past_ones, ones_per_inventory);

            debug_assert!(start + span == num_bits || ones == ones_per_inventory);

            match SpanType::from_span(span) {
                // We store the entries first in the subinventory and then in
                // the spill buffer. The first u64 word will be used to store
                // the position of the entry in the spill buffer. Using the
                // first word gives a cache advantage to entries that will need
                // another cache miss to be read from the spill buffer.
                SpanType::U32 => {
                    // We store an inventory entry each 1 << log2_ones_per_sub32 ones.
                    let log2_ones_per_sub32 = Self::log2_ones_per_sub32(span, log2_ones_per_sub16);
                    let num_u32s = ones.div_ceil(1 << log2_ones_per_sub32);
                    let num_words = num_u32s.div_ceil(U32_PER_USIZE);
                    let spilled_u64s = num_words.saturating_sub(words_per_subinventory - 1);
                    spilled += spilled_u64s;
                }
                #[cfg(target_pointer_width = "64")]
                SpanType::U64 => {
                    // We store an inventory entry for each one after the first.
                    spilled += (ones - 1).saturating_sub(words_per_subinventory - 1);
                }
                _ => {}
            }
        }

        let spill_size = spilled;

        let mut inventory: Box<[usize]> = inventory.into();
        let mut spill: Box<[usize]> = vec![0; spill_size].into();

        spilled = 0;
        let locally_stored_u32s = U32_PER_USIZE * (words_per_subinventory - 1);

        // Second phase: we fill the subinventories and the spill.
        for inventory_idx in 0..inventory_size {
            // Get the start and end indices of the current inventory
            let start_inv_idx = inventory_idx * words_per_inventory;
            let end_inv_idx = start_inv_idx + words_per_inventory;
            // Read the first-level index to get the start and end bit indices
            let start_bit_idx = inventory[start_inv_idx];
            let end_bit_idx = inventory[end_inv_idx];
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            let span_type = SpanType::from_span(span);

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
                    log2_quantum = Self::log2_ones_per_sub32(span, log2_ones_per_sub16);
                    inventory[start_inv_idx].set_u32_span();
                    // The first word of the subinventory is used to store the spill index.
                    inventory[start_inv_idx + 1] = spilled;
                }
                #[cfg(target_pointer_width = "64")]
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

            let mut word_idx = start_bit_idx / bits_per_word;
            let end_word_idx = end_bit_idx.div_ceil(bits_per_word);
            let bit_idx = start_bit_idx % bits_per_word;

            // Clear the lower bits
            let mut word = (!bits.as_ref()[word_idx] >> bit_idx) << bit_idx;

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
                    let bit_index = (word_idx * bits_per_word) + in_word_index;

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
                            if subinventory_idx << log2_quantum == ones_per_inventory {
                                break 'outer;
                            }
                        }
                        SpanType::U32 => {
                            if subinventory_idx < locally_stored_u32s {
                                let subinventory: &mut [u32] = unsafe {
                                    inventory[start_inv_idx + 2..end_inv_idx].align_to_mut().1
                                };

                                debug_assert_eq!(subinventory[subinventory_idx], 0);
                                subinventory[subinventory_idx] = sub_offset as u32;
                            } else {
                                let u32_spill: &mut [u32] =
                                    unsafe { spill[spilled..].align_to_mut().1 };
                                debug_assert_eq!(
                                    u32_spill[subinventory_idx - locally_stored_u32s],
                                    0
                                );
                                u32_spill[subinventory_idx - locally_stored_u32s] =
                                    sub_offset as u32;
                            }

                            subinventory_idx += 1;
                            // This exit is not necessary for correctness, but
                            // it avoids the additional loop iterations that
                            // would be necessary to find the position of the
                            // next one (i.e., end_bit_idx).
                            if subinventory_idx << log2_quantum == ones_per_inventory {
                                break 'outer;
                            }
                        }
                        #[cfg(target_pointer_width = "64")]
                        SpanType::U64 => {
                            if subinventory_idx < words_per_subinventory {
                                inventory[start_inv_idx + 1 + subinventory_idx] = bit_index;
                                subinventory_idx += 1;
                            } else {
                                assert!(spilled < spill_size);
                                spill[spilled] = bit_index;
                                spilled += 1;
                            }
                            // This exit is not necessary for correctness, but
                            // it avoids the additional loop iterations that
                            // would be necessary to find the position of the
                            // next one (i.e., end_bit_idx). Note that here
                            // log2_quantum == 0.
                            if subinventory_idx == ones_per_inventory {
                                break 'outer;
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

            // If we are in the U32 case, we need to update the number of used
            // element in the spill buffer. The update must be done after the
            // loop, as for the last inventory entry only at this point we know
            // the actual number of elements in the subinventory.
            if span_type == SpanType::U32 {
                spilled += subinventory_idx
                    .saturating_sub(locally_stored_u32s)
                    .div_ceil(U32_PER_USIZE);
            }
        }

        assert_eq!(spilled, spill_size);

        Self {
            bits,
            inventory,
            spill,
            log2_ones_per_inventory,
            log2_ones_per_sub16,
            log2_words_per_subinventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
        }
    }
}

impl<
    B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + BitLength + SelectZeroHinted,
    I: AsRef<[usize]>,
> SelectZeroUnchecked for SelectZeroAdapt<B, I>
{
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        unsafe {
            let inventory = self.inventory.as_ref();
            let inventory_index = rank >> self.log2_ones_per_inventory;
            let inventory_start_pos =
                (inventory_index << self.log2_words_per_subinventory) + inventory_index;

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
                    .select_zero_hinted(rank, hint_pos, rank - residual);
            }

            let words_per_subinventory = 1 << self.log2_words_per_subinventory;

            if inventory_rank.is_u32_span() {
                let inventory_rank = inventory_rank.get();

                let span = (*inventory
                    .get_unchecked(inventory_start_pos + words_per_subinventory + 1))
                .get()
                    - inventory_rank;
                let log2_ones_per_sub32 = Self::log2_ones_per_sub32(span, self.log2_ones_per_sub16);
                let hint_pos = if subrank >> log2_ones_per_sub32
                    < (words_per_subinventory - 1) * U32_PER_USIZE
                {
                    let u32s = inventory
                        .get_unchecked(inventory_start_pos + 2..)
                        .align_to::<u32>()
                        .1;

                    inventory_rank + *u32s.get_unchecked(subrank >> log2_ones_per_sub32) as usize
                } else {
                    let start_spill_idx = *inventory.get_unchecked(inventory_start_pos + 1);

                    let spilled_u32s = self
                        .spill
                        .as_ref()
                        .get_unchecked(start_spill_idx..)
                        .align_to::<u32>()
                        .1;

                    inventory_rank
                        + *spilled_u32s.get_unchecked(
                            (subrank >> log2_ones_per_sub32)
                                - (words_per_subinventory - 1) * U32_PER_USIZE,
                        ) as usize
                };
                let residual = subrank & ((1 << log2_ones_per_sub32) - 1);
                return self
                    .bits
                    .select_zero_hinted(rank, hint_pos, rank - residual);
            }

            #[cfg(target_pointer_width = "64")]
            debug_assert!(inventory_rank.is_u64_span());
            let inventory_rank = inventory_rank.get();

            if subrank < words_per_subinventory {
                if subrank == 0 {
                    return inventory_rank;
                }
                return *inventory.get_unchecked(inventory_start_pos + 1 + subrank);
            }
            let spill_idx = { *inventory.get_unchecked(inventory_start_pos + 1) } + subrank
                - words_per_subinventory;
            debug_assert!(spill_idx < self.spill.as_ref().len());
            *self.spill.as_ref().get_unchecked(spill_idx)
        }
    }
}

impl<
    B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + NumBits + SelectZeroHinted,
    I: AsRef<[usize]>,
> SelectZero for SelectZeroAdapt<B, I>
{
}

#[cfg(test)]
#[cfg(target_pointer_width = "64")]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::bits::BitVec;
    use crate::traits::AddNumBits;
    use crate::traits::BitVecOpsMut;

    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_sub64s() {
        let len = 5_000_000_000;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut bits = BitVec::new(len);
        let mut pos = BTreeSet::new();
        for _ in 0..(1 << 13) / 4 * 3 {
            let p = rng.random_range(0..len);
            if pos.insert(p) {
                bits.set(p, true);
            }
        }
        bits.flip();
        let bits: AddNumBits<BitVec> = bits.into();

        for m in [0, 3, 16] {
            let simple = SelectZeroAdapt::with_inv(&bits, 13, m);
            assert!(simple.inventory[0].is_u64_span());

            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(simple.select_zero(i), Some(p));
            }
            assert_eq!(simple.select_zero(pos.len()), None);
        }
    }
}
