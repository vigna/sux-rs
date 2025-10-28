/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use ambassador::Delegate;
use common_traits::SelectInWord;
use mem_dbg::{MemDbg, MemSize};
use std::cmp::{max, min};

use crate::{
    prelude::{BitCount, BitLength, Select, SelectHinted},
    traits::{NumBits, SelectUnchecked},
};

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

/// A selection structure based on an adaptive two-level inventory.
///
/// The design of this selection structure starts from the `simple` structure
/// described by Sebastiano Vigna in “[Broadword Implementation of Rank/Select
/// Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
/// _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
/// 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
/// Springer, 2008, but adds adaptivity of the second-level inventory, using
/// thus significantly less space than `simple` for bit vectors with uneven
/// distribution.
///
/// [`SelectZeroAdapt`](super::SelectZeroAdapt) is a variant of this structure
/// that provides the same functionality for zero bits.
/// [`SelectAdaptConst`](super::SelectAdaptConst) provides similar functionality
/// but with const parameters.
///
/// # Implementation Details
///
/// The structure is based on a first-level inventory and a second-level
/// subinventory. Similarly to [`Rank9`](super::Rank9), the two levels are
/// interleaved to reduce the number of cache misses.
///
/// The inventory is sized so that the distance between two indexed ones is on
/// average a given target value *L*. For each indexed one in the inventory (for
/// which we use a 64-bit integer), we allocate at most *M* (a power of 2)
/// 64-bit integers for the subinventory. The relative target space occupancy of
/// the selection structure is thus at most 64(1 + *M*)/*L*. However, since the
/// number of ones per inventory has to be a power of two, the _actual_ value of
/// *L* might be smaller by a factor of 2, doubling the actual space occupancy
/// with respect to the target space occupancy.
///
/// For example, using [the default value of
/// *L*](SelectAdapt::DEFAULT_TARGET_INVENTORY_SPAN) and *M* = 8, the space
/// occupancy is between 7% and 14%. The space might be smaller for very sparse
/// vectors as less than *M* subinventory words per inventory might be used.
///
/// Given a specific indexed one in the inventory, if the distance to the next
/// indexed one is smaller than 2¹⁶ we use the *M* words associated to the
/// subinventory to store 4*M* 16-bit integers, representing the offsets of
/// regularly spaced ones inside the inventory.
///
/// Otherwise, if the distance is smaller than or equal to 2³², we use the *M*
/// words plus some additional words in a spill buffer to store the offsets of
/// regularly spaced ones inside the inventory using 32-bit integers (the first
/// of the *M* words points inside the spill buffer). The number of such
/// integers is chosen adaptively so that the average distance between two
/// indexed ones is comparable to the worst case of a 16-bit subinventory.
///
/// Finally, if the distance is larger than 2³², we use the *M* words plus some
/// additional words in a spill buffer to store exactly the position of every
/// bit in the subinventory using 64-bit integers.
///
/// Note that is is possible to build pathological cases  (e.g., half of the bit
/// vector extremely dense, half of the vector extremely sparse) in which the
/// structure has a different performance depending on the selected bit. In
/// these cases, [`Select9`](super::Select9) might be a better choice.
///
/// In the 16-bit case, the average distance between two ones indexed by the
/// subinventories is *L*/4*M*, (again, the actual value might be twice as large
/// because of rounding). However, the worst-case distance might as high as
/// 2¹⁶/4*M*, as we use 4*M* 16-bit integers until the width of the inventory
/// span makes it possible. Within this range, we perform a sequential broadword
/// search, which has a linear cost. In the 32-bit case, the average distance
/// between two ones is kept within 2¹⁶/4*M*.
///
/// # Choosing Parameters
///
/// The value *L* should be chosen so that the distance between two indexed ones
/// in the inventory is always smaller than 2¹⁶. The [default suggested
/// value](SelectAdapt::DEFAULT_TARGET_INVENTORY_SPAN) is a reasonable choice
/// for vectors that reasonably uniform, but smaller values can be used for more
/// irregular vectors, at the cost of a larger space occupancy. Moreover, a
/// smaller value of *L* might be provide faster selection in exchange for more
/// space occupancy for small vectors (a few million bits), as the inventory
/// would still fit the cache. Note that halving (or doubling) at the same time
/// the value of *L* and *M* will give a structure with essentially the same
/// space usage.
///
/// The value *M* should be as high as possible, compatibly with the desired
/// space occupancy, but values resulting in linear searches shorter than a
/// couple of words will not generally improve performance; moreover,
/// interleaving inventories is not useful if *M* is so large that the
/// subinventory takes several cache lines. For example, using [default value
/// for *L*](SelectAdapt::DEFAULT_TARGET_INVENTORY_SPAN) a reasonable choice for
/// *M* is between 4 and 32, corresponding to worst-case linear searches between
/// 1024 and 128 bits, typical choices being 8 and 16 (note that the
/// constructors take the base-2 logarithm of *M*).
///
/// # Examples
/// ```rust
/// # use sux::bit_vec;
/// # use sux::traits::{Rank, Select, SelectUnchecked, AddNumBits};
/// # use sux::rank_sel::{SelectAdapt, Rank9};
/// // Standalone select
/// let bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 1];
/// let select = SelectAdapt::new(bits, 3);
///
/// // If the backend does not implement NumBits
/// // we just get SelectUnchecked
/// unsafe {
///     assert_eq!(select.select_unchecked(0), 0);
///     assert_eq!(select.select_unchecked(1), 2);
///     assert_eq!(select.select_unchecked(2), 3);
///     assert_eq!(select.select_unchecked(3), 5);
///     assert_eq!(select.select_unchecked(4), 7);
/// }
///
/// // Let's add NumBits to the backend
/// let bits: AddNumBits<_> = bit_vec![1, 0, 1, 1, 0, 1, 0, 1].into();
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
/// let sel_rank9 = unsafe { select.map(Rank9::new) };
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

#[derive(Debug, Clone, Copy, MemDbg, MemSize, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(AsRef<[usize]>, target = "bits")]
#[delegate(Index<usize>, target = "bits")]
#[delegate(crate::traits::rank_sel::BitCount, target = "bits")]
#[delegate(crate::traits::rank_sel::BitLength, target = "bits")]
#[delegate(crate::traits::rank_sel::NumBits, target = "bits")]
#[delegate(crate::traits::rank_sel::Rank, target = "bits")]
#[delegate(crate::traits::rank_sel::RankHinted<64>, target = "bits")]
#[delegate(crate::traits::rank_sel::RankUnchecked, target = "bits")]
#[delegate(crate::traits::rank_sel::RankZero, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZero, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroHinted, target = "bits")]
#[delegate(crate::traits::rank_sel::SelectZeroUnchecked, target = "bits")]
pub struct SelectAdapt<B, I = Box<[usize]>> {
    bits: B,
    inventory: I,
    spill: I,
    log2_ones_per_inventory: usize,
    log2_ones_per_sub16: usize,
    log2_u64_per_subinventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
}

// Convenience trait to handle the information packed in the two upper bits of
// an inventory entry. It is used by all variants.

pub(super) trait Inventory {
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

// The type subinventory entries for a span. It is used by all variants.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpanType {
    U16,
    U32,
    U64,
}

impl SpanType {
    pub fn from_span(x: usize) -> SpanType {
        match x {
            0..=0x10000 => SpanType::U16,
            0x10001..=0x100000000 => SpanType::U32,
            _ => SpanType::U64,
        }
    }
}

impl<B, I> SelectAdapt<B, I> {
    pub fn into_inner(self) -> B {
        self.bits
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
    /// Replaces the backend with a new one implementing [`SelectHinted`].
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new backend is identical to the old one as a bit vector.
    pub unsafe fn map<C>(self, f: impl FnOnce(B) -> C) -> SelectAdapt<C, I>
    where
        C: SelectHinted,
    {
        SelectAdapt {
            bits: f(self.bits),
            inventory: self.inventory,
            spill: self.spill,
            log2_ones_per_inventory: self.log2_ones_per_inventory,
            log2_ones_per_sub16: self.log2_ones_per_sub16,
            log2_u64_per_subinventory: self.log2_u64_per_subinventory,
            ones_per_inventory_mask: self.ones_per_inventory_mask,
            ones_per_sub16_mask: self.ones_per_sub16_mask,
        }
    }

    pub const DEFAULT_TARGET_INVENTORY_SPAN: usize = 8192;
}

impl<B: BitLength, C> SelectAdapt<B, C> {
    /// Returns the number of bits in the bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<B: AsRef<[usize]> + BitCount> SelectAdapt<B, Box<[usize]>> {
    /// Creates a new selection structure over a bit vector using a
    /// [default target inventory
    /// span](SelectAdapt::DEFAULT_TARGET_INVENTORY_SPAN).
    ///
    /// # Arguments
    ///
    /// * `bits`: A bit vector.
    ///
    /// * `max_log2_u64_per_subinv`: The base-2 logarithm of the maximum
    ///   number [*M*](SelectAdapt) of 64-bit words in each subinventory.
    ///   Increasing by one this value approximately doubles the space occupancy
    ///   and halves the length of sequential broadword searches. Typical values
    ///   are 3 and 4.
    ///
    pub fn new(bits: B, max_log2_u64_per_subinv: usize) -> Self {
        Self::with_span(
            bits,
            Self::DEFAULT_TARGET_INVENTORY_SPAN,
            max_log2_u64_per_subinv,
        )
    }

    /// Creates a new selection structure over a bit vector with a specified
    /// target inventory span.
    ///
    /// # Arguments
    ///
    /// * `bits`: A bit vector.
    ///
    /// * `target_inventory_span`: The target span [*L*](SelectAdapt) of a
    ///   first-level inventory entry. The actual span might be smaller by a
    ///   factor of 2.
    ///
    /// * `max_log2_u64_per_subinventory`: The base-2 logarithm of the maximum
    ///   number [*M*](SelectAdapt) of 64-bit words in each subinventory.
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

        Self::_new(
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
    /// Unless you understand all the implications, it is preferable to use the
    /// [standard constructor](SelectAdapt::new).
    ///
    /// # Arguments
    ///
    /// * `bits`: A bit vector.
    ///
    /// * `log2_ones_per_inventory`: The base-2 logarithm of the distance
    ///   between two indexed ones.
    ///
    /// * `max_log2_u64_per_subinventory`: The base-2 logarithm of the maximum
    ///   number [*M*](SelectAdapt) of 64-bit words in each subinventory.
    ///   Increasing by one this value approximately doubles the space occupancy
    ///   and halves the length of sequential broadword searches. Typical values
    ///   are 3 and 4.
    pub fn with_inv(
        bits: B,
        log2_ones_per_inventory: usize,
        max_log2_u64_per_subinventory: usize,
    ) -> Self {
        let num_ones = bits.count_ones();
        Self::_new(
            bits,
            num_ones,
            log2_ones_per_inventory,
            max_log2_u64_per_subinventory,
        )
    }

    fn _new(
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

        let log2_u64_per_subinventory =
            max_log2_u64_per_subinventory.min(log2_ones_per_inventory.saturating_sub(2));

        let u64_per_subinventory = 1 << log2_u64_per_subinventory;
        // A u64 for the inventory, and u64_per_inventory for the subinventory
        let u64_per_inventory = u64_per_subinventory + 1;

        let log2_ones_per_sub16 =
            log2_ones_per_inventory.saturating_sub(log2_u64_per_subinventory + 2);
        let ones_per_sub16 = 1 << log2_ones_per_sub16;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let inventory_words = inventory_size * u64_per_inventory + 1;
        let mut inventory = Vec::with_capacity(inventory_words);

        let mut past_ones = 0;
        let mut next_quantum = 0;
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

        assert_eq!(past_ones, num_ones);
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
                    let num_u64s = num_u32s.div_ceil(2);
                    let spilled_u64s = num_u64s.saturating_sub(u64_per_subinventory - 1);
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
        let locally_stored_u32s = 2 * (u64_per_subinventory - 1);

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

            let mut word_idx = start_bit_idx / usize::BITS as usize;
            let end_word_idx = end_bit_idx.div_ceil(usize::BITS as usize);
            let bit_idx = start_bit_idx % usize::BITS as usize;

            // Clear the lower bits
            let mut word = (bits.as_ref()[word_idx] >> bit_idx) << bit_idx;

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
                        SpanType::U64 => {
                            if subinventory_idx < u64_per_subinventory {
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
                word = bits.as_ref()[word_idx];
            }

            // If we are in the U32 case, we need to update the number of used
            // element in the spill buffer. The update must be done after the
            // loop, as for the last inventory entry only at this point we know
            // the actual number of elements in the subinventory.
            if span_type == SpanType::U32 {
                spilled += subinventory_idx
                    .saturating_sub(locally_stored_u32s)
                    .div_ceil(2);
            }
        }

        assert_eq!(spilled, spill_size);

        Self {
            bits,
            inventory,
            spill,
            log2_ones_per_inventory,
            log2_ones_per_sub16,
            log2_u64_per_subinventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
        }
    }
}

impl<B: AsRef<[usize]> + BitLength + SelectHinted, I: AsRef<[usize]>> SelectUnchecked
    for SelectAdapt<B, I>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        unsafe {
            let inventory = self.inventory.as_ref();
            let inventory_index = rank >> self.log2_ones_per_inventory;
            let inventory_start_pos =
                (inventory_index << self.log2_u64_per_subinventory) + inventory_index;

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

                return self.bits.select_hinted(rank, hint_pos, rank - residual);
            }

            let u64_per_subinventory = 1 << self.log2_u64_per_subinventory;

            if inventory_rank.is_u32_span() {
                let inventory_rank = inventory_rank.get();

                let span = (*inventory
                    .get_unchecked(inventory_start_pos + u64_per_subinventory + 1))
                .get()
                    - inventory_rank;
                let log2_ones_per_sub32 = Self::log2_ones_per_sub32(span, self.log2_ones_per_sub16);
                let hint_pos = if subrank >> log2_ones_per_sub32 < (u64_per_subinventory - 1) * 2 {
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
                            (subrank >> log2_ones_per_sub32) - (u64_per_subinventory - 1) * 2,
                        ) as usize
                };
                let residual = subrank & ((1 << log2_ones_per_sub32) - 1);
                return self.bits.select_hinted(rank, hint_pos, rank - residual);
            }

            debug_assert!(inventory_rank.is_u64_span());
            let inventory_rank = inventory_rank.get();

            if subrank < u64_per_subinventory {
                if subrank == 0 {
                    return inventory_rank;
                }
                return *inventory.get_unchecked(inventory_start_pos + 1 + subrank);
            }
            let spill_idx = { *inventory.get_unchecked(inventory_start_pos + 1) } + subrank
                - u64_per_subinventory;
            debug_assert!(spill_idx < self.spill.as_ref().len());
            *self.spill.as_ref().get_unchecked(spill_idx)
        }
    }
}

impl<B: SelectHinted + AsRef<[usize]> + NumBits, I: AsRef<[usize]>> Select for SelectAdapt<B, I> {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::bits::BitVec;
    use crate::traits::AddNumBits;
    use crate::traits::BitVecOpsMut;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

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
        let bits: AddNumBits<BitVec> = bits.into();

        for m in [0, 3, 16] {
            let simple = SelectAdapt::with_inv(&bits, 13, m);
            assert!(simple.inventory[0].is_u64_span());

            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(simple.select(i), Some(p));
            }
            assert_eq!(simple.select(pos.len()), None);
        }
    }
}
