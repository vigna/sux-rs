/*
 *
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use super::{
    Inventory, LOG2_U16_PER_USIZE, U32_PER_USIZE, assert_inventory_length, build_inventory,
    log2_ones_per_sub32,
};
use crate::utils::SelectInWord;
use ambassador::Delegate;
use mem_dbg::{MemDbg, MemSize};
use std::{cmp::max, ops::Deref};

use crate::traits::{Backend, BitLength, Word};

use crate::ambassador_impl_Index;
use crate::traits::BitVecOps;
use crate::traits::ambassador_impl_Backend;
use crate::traits::bal_paren::{BalParen, ambassador_impl_BalParen};
use crate::traits::bit_vec_ops::ambassador_impl_BitLength;
use crate::traits::rank_sel::*;
use std::ops::Index;

// NOTE: to make parallel modifications with SelectAdapt as easy as possible,
// "ones" are considered to be zeros in the following code.

/// A version of [`SelectAdapt`] implementing [selection on zeros].
///
/// # Examples
/// ```rust
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// use sux::bit_vec;
/// use sux::rank_sel::{Rank9, SelectZeroAdapt};
/// use sux::traits::{AddNumBits, Rank, SelectZero, SelectZeroUnchecked};
///
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
///
/// [`SelectAdapt`]: super::SelectAdapt
/// [selection on zeros]: crate::traits::SelectZero
#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(crate::traits::Backend, target = "bits")]
#[delegate(Index<usize>, target = "bits")]
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

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

impl<B, I> SelectZeroAdapt<B, I> {
    /// Returns the underlying bit vector, consuming this structure.
    pub fn into_inner(self) -> B {
        self.bits
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

impl<B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + BitLength>
    SelectZeroAdapt<B, Box<[usize]>>
{
    /// See [`SelectAdapt::new`].
    ///
    /// [`SelectAdapt::new`]: super::SelectAdapt::new
    #[must_use]
    pub fn new(bits: B) -> Self {
        Self::with_span(
            bits,
            super::select_adapt::default_target_inventory_span(
                super::select_adapt::DEFAULT_LOG2_WORDS_PER_SUBINVENTORY,
            ),
            super::select_adapt::DEFAULT_LOG2_WORDS_PER_SUBINVENTORY,
        )
    }

    /// See [`SelectAdapt::with_span`].
    ///
    /// [`SelectAdapt::with_span`]: super::SelectAdapt::with_span
    #[must_use]
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

    /// See [`SelectAdapt::with_inv`].
    ///
    /// [`SelectAdapt::with_inv`]: super::SelectAdapt::with_inv
    #[must_use]
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

    /// See [`SelectAdapt::with_overhead`].
    ///
    /// [`SelectAdapt::with_overhead`]: super::SelectAdapt::with_overhead
    #[must_use]
    pub fn with_overhead(
        bits: B,
        overhead_percentage: f64,
        max_log2_words_per_subinv: usize,
    ) -> Self {
        let target_span = super::select_adapt::target_span_from_overhead(
            overhead_percentage,
            max_log2_words_per_subinv,
        );

        Self::with_span(bits, target_span, max_log2_words_per_subinv)
    }

    fn _new(
        bits: B,
        num_ones: usize,
        log2_ones_per_inventory: usize,
        max_log2_words_per_subinventory: usize,
    ) -> Self {
        assert_inventory_length(bits.len());
        assert!(
            log2_ones_per_inventory < usize::BITS as usize,
            "log2_ones_per_inventory ({log2_ones_per_inventory}) must be less than {}",
            usize::BITS
        );
        assert!(
            max_log2_words_per_subinventory <= super::select_adapt::MAX_LOG2_WORDS_PER_SUBINVENTORY,
            "max_log2_words_per_subinventory ({max_log2_words_per_subinventory}) exceeds the supported maximum ({})",
            super::select_adapt::MAX_LOG2_WORDS_PER_SUBINVENTORY
        );
        let ones_per_inventory_mask = (1 << log2_ones_per_inventory) - 1;

        // We use a smaller value than max_log2_words_per_subinventory when with a
        // smaller value we can still index, in the 16-bit case, all bits the
        // subinventory. This can happen only in extremely sparse vectors, or
        // if a very small value of log2_ones_per_inventory is set directly.

        let log2_words_per_subinventory = max_log2_words_per_subinventory
            .min(log2_ones_per_inventory.saturating_sub(2))
            // A subinventory never needs more entries than there are
            // ones: without this cap a very sparse vector with a large
            // span would allocate a huge, almost empty subinventory
            .min(
                num_ones
                    .max(1)
                    .next_power_of_two()
                    .ilog2()
                    .saturating_sub(2) as usize,
            );

        let log2_ones_per_sub16 = log2_ones_per_inventory
            .saturating_sub(log2_words_per_subinventory + LOG2_U16_PER_USIZE);
        let ones_per_sub16_mask = (1 << log2_ones_per_sub16) - 1;

        let (inventory, spill) = build_inventory::<true, _>(
            bits.as_ref(),
            bits.len(),
            num_ones,
            log2_ones_per_inventory,
            log2_words_per_subinventory,
            log2_ones_per_sub16,
        );

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
    #[inline]
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

                return self.bits.select_zero_hinted::<{ usize::MAX }>(
                    rank,
                    hint_pos,
                    rank - residual,
                );
            }

            self.select_zero_unchecked_cold(rank, inventory_start_pos, inventory_rank, subrank)
        }
    }
}

impl<
    B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + BitLength + SelectZeroHinted,
    I: AsRef<[usize]>,
> SelectZeroAdapt<B, I>
{
    #[inline(never)]
    unsafe fn select_zero_unchecked_cold(
        &self,
        rank: usize,
        inventory_start_pos: usize,
        inventory_rank: usize,
        subrank: usize,
    ) -> usize {
        unsafe {
            let inventory = self.inventory.as_ref();
            let words_per_subinventory = 1 << self.log2_words_per_subinventory;

            if inventory_rank.is_u32_span() {
                let inventory_rank = inventory_rank.get();

                let span = (*inventory
                    .get_unchecked(inventory_start_pos + words_per_subinventory + 1))
                .get()
                    - inventory_rank;
                let log2_ones_per_sub32 = log2_ones_per_sub32(span, self.log2_ones_per_sub16);
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
                return self.bits.select_zero_hinted::<{ usize::MAX }>(
                    rank,
                    hint_pos,
                    rank - residual,
                );
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

#[cfg(all(test, feature = "slow_tests", target_pointer_width = "64"))]
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
