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
use crate::rank_sel::select_adapt;
use crate::traits::{Backend, BitLength, Word};
use crate::utils::SelectInWord;
use ambassador::Delegate;
use mem_dbg::{MemDbg, MemSize};
use std::ops::Deref;

use crate::ambassador_impl_Index;
use crate::traits::BitVecOps;
use crate::traits::ambassador_impl_Backend;
use crate::traits::bal_paren::{BalParen, ambassador_impl_BalParen};
use crate::traits::bit_vec_ops::ambassador_impl_BitLength;
use crate::traits::rank_sel::*;
use std::ops::Index;

// NOTE: to make parallel modifications with SelectAdaptConst as easy as
// possible, "ones" are considered to be zeros in the following code.

/// A version of [`SelectAdaptConst`] implementing [selection on zeros].
///
/// # Examples
/// ```rust
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// use sux::bit_vec;
/// use sux::rank_sel::{Rank9, SelectZeroAdaptConst};
/// use sux::traits::{AddNumBits, Rank, SelectZero, SelectZeroUnchecked};
///
/// // Standalone select (default values)
/// let bits = bit_vec![0, 1, 0, 0, 1, 0, 1, 0];
/// let select = SelectZeroAdaptConst::<_>::new(bits);
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
/// let select = SelectZeroAdaptConst::<_>::new(bits);
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
/// let rank9_sel = SelectZeroAdaptConst::<_>::new(rank9);
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
/// [`SelectAdaptConst`]: super::SelectAdaptConst
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
pub struct SelectZeroAdaptConst<
    B,
    I = Box<[usize]>,
    const LOG2_ZEROS_PER_INVENTORY: usize = {
        (select_adapt::default_target_inventory_span(
            select_adapt::DEFAULT_LOG2_WORDS_PER_SUBINVENTORY,
        )
        .ilog2() as usize)
            .saturating_sub(1) // density 50%
    },
    const LOG2_WORDS_PER_SUBINVENTORY: usize = {
        select_adapt::DEFAULT_LOG2_WORDS_PER_SUBINVENTORY
    },
> {
    bits: B,
    inventory: I,
    spill: I,
}

impl<
    B: Backend + AsRef<[B::Word]>,
    I,
    const LOG2_ZEROS_PER_INVENTORY: usize,
    const LOG2_WORDS_PER_SUBINVENTORY: usize,
> AsRef<[B::Word]>
    for SelectZeroAdaptConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
{
    #[inline(always)]
    fn as_ref(&self) -> &[B::Word] {
        self.bits.as_ref()
    }
}

impl<B, I, const LOG2_ZEROS_PER_INVENTORY: usize, const LOG2_WORDS_PER_SUBINVENTORY: usize> Deref
    for SelectZeroAdaptConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
{
    type Target = B;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

impl<B, I, const LOG2_ZEROS_PER_INVENTORY: usize, const LOG2_WORDS_PER_SUBINVENTORY: usize>
    SelectZeroAdaptConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
{
    const LOG2_ONES_PER_SUB16: usize =
        LOG2_ZEROS_PER_INVENTORY.saturating_sub(LOG2_WORDS_PER_SUBINVENTORY + LOG2_U16_PER_USIZE);
    const ONES_PER_SUB16_MASK: usize = (1 << Self::LOG2_ONES_PER_SUB16) - 1;
    const ONES_PER_INVENTORY_MASK: usize = (1 << LOG2_ZEROS_PER_INVENTORY) - 1;
    /// Compile-time check that the const parameters are valid shift amounts.
    /// `usize::BITS as usize` is a lossless u32->usize widening.
    const PARAMS_OK: () = assert!(
        LOG2_ZEROS_PER_INVENTORY < usize::BITS as usize
            && LOG2_WORDS_PER_SUBINVENTORY
                <= super::select_adapt_const::MAX_CONST_LOG2_WORDS_PER_SUBINVENTORY,
        "LOG2_ZEROS_PER_INVENTORY must be less than the word width and LOG2_WORDS_PER_SUBINVENTORY must not exceed the supported maximum"
    );

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
    pub unsafe fn map<C: SelectZeroHinted>(
        self,
        f: impl FnOnce(B) -> C,
    ) -> SelectZeroAdaptConst<C, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY> {
        SelectZeroAdaptConst {
            bits: f(self.bits),
            inventory: self.inventory,
            spill: self.spill,
        }
    }
}

impl<
    B: BitLength,
    C,
    const LOG2_ZEROS_PER_INVENTORY: usize,
    const LOG2_WORDS_PER_SUBINVENTORY: usize,
> SelectZeroAdaptConst<B, C, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
{
    /// Returns the number of bits in the underlying bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }
}

impl<
    B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + BitLength,
    const LOG2_ZEROS_PER_INVENTORY: usize,
    const LOG2_WORDS_PER_SUBINVENTORY: usize,
> SelectZeroAdaptConst<B, Box<[usize]>, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
{
    /// See [`SelectAdaptConst::new`].
    ///
    /// [`SelectAdaptConst::new`]: super::SelectAdaptConst::new
    #[must_use]
    pub fn new(bits: B) -> Self {
        assert_inventory_length(bits.len());
        // Force the const-parameter validity check.
        let () = Self::PARAMS_OK;
        let num_ones = bits.count_zeros();
        let (inventory, spill) = build_inventory::<true, _>(
            bits.as_ref(),
            bits.len(),
            num_ones,
            LOG2_ZEROS_PER_INVENTORY,
            LOG2_WORDS_PER_SUBINVENTORY,
            Self::LOG2_ONES_PER_SUB16,
        );

        Self {
            bits,
            inventory,
            spill,
        }
    }
}

impl<
    B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]> + BitLength + SelectZeroHinted,
    I: AsRef<[usize]>,
    const LOG2_ZEROS_PER_INVENTORY: usize,
    const LOG2_WORDS_PER_SUBINVENTORY: usize,
> SelectZeroUnchecked
    for SelectZeroAdaptConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
{
    #[inline]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        unsafe {
            let inventory = self.inventory.as_ref();
            let inventory_index = rank >> LOG2_ZEROS_PER_INVENTORY;
            let inventory_start_pos =
                (inventory_index << LOG2_WORDS_PER_SUBINVENTORY) + inventory_index;

            let inventory_rank = { *inventory.get_unchecked(inventory_start_pos) };
            let subrank = rank & Self::ONES_PER_INVENTORY_MASK;

            if inventory_rank.is_u16_span() {
                let subinventory = inventory
                    .get_unchecked(inventory_start_pos + 1..)
                    .align_to::<u16>()
                    .1;

                debug_assert!(subrank >> Self::LOG2_ONES_PER_SUB16 < subinventory.len());

                let hint_pos = inventory_rank
                    + *subinventory.get_unchecked(subrank >> Self::LOG2_ONES_PER_SUB16) as usize;
                let residual = subrank & Self::ONES_PER_SUB16_MASK;

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
    const LOG2_ZEROS_PER_INVENTORY: usize,
    const LOG2_WORDS_PER_SUBINVENTORY: usize,
> SelectZeroAdaptConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
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
            let words_per_subinventory = 1 << LOG2_WORDS_PER_SUBINVENTORY;

            if inventory_rank.is_u32_span() {
                let inventory_rank = inventory_rank.get();

                let span = (*inventory
                    .get_unchecked(inventory_start_pos + words_per_subinventory + 1))
                .get()
                    - inventory_rank;
                let log2_ones_per_sub32 = log2_ones_per_sub32(span, Self::LOG2_ONES_PER_SUB16);
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
    const LOG2_ZEROS_PER_INVENTORY: usize,
    const LOG2_WORDS_PER_SUBINVENTORY: usize,
> SelectZero for SelectZeroAdaptConst<B, I, LOG2_ZEROS_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY>
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
