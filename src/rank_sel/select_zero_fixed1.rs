/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use crate::traits::bit_field_slice::BitFieldSlice;
use crate::traits::bit_field_slice::BitFieldSliceMut;
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;
use mem_dbg::*;

/**
A selection structure for zeros based on a one-level index.

See [`SelectFixed1`](crate::rank_sel::SelectFixed1).

*/
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SelectZeroFixed1<
    B: SelectZeroHinted = CountBitVec,
    O: BitFieldSlice<usize> = Vec<usize>,
    const LOG2_ZEROS_PER_INVENTORY: usize = 8,
> {
    bits: B,
    inventory: O,
    _marker: core::marker::PhantomData<[(); LOG2_ZEROS_PER_INVENTORY]>,
}

impl<
        B: SelectZeroHinted + BitLength + BitCount + AsRef<[usize]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > SelectZeroFixed1<B, Vec<usize>, LOG2_ZEROS_PER_INVENTORY>
{
    pub fn new(bitvec: B) -> Self {
        let mut res = SelectZeroFixed1 {
            inventory: vec![
                0;
                (bitvec.len() - bitvec.count() + (1 << LOG2_ZEROS_PER_INVENTORY) - 1)
                    >> LOG2_ZEROS_PER_INVENTORY
            ],
            bits: bitvec,
            _marker: core::marker::PhantomData,
        };
        res.build_zeros();
        res
    }
}

impl<
        B: SelectZeroHinted + BitLength + AsRef<[usize]>,
        O: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
{
    fn build_zeros(&mut self) {
        let mut number_of_zeros = 0;
        let mut next_quantum = 0;
        let mut zeros_index = 0;
        for (i, mut word) in self.bits.as_ref().iter().copied().enumerate() {
            word = !word;
            let zeros_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_zeros + zeros_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_zeros) as usize);
                let index = (i * usize::BITS as usize) + in_word_index;
                if index >= <Self as BitLength>::len(self) as _ {
                    return;
                }
                self.inventory.set(zeros_index, index);
                next_quantum += 1 << LOG2_ZEROS_PER_INVENTORY;
                zeros_index += 1;
            }

            number_of_zeros += zeros_in_word;
        }
    }
}

/// Provide the hint to the underlying structure.
impl<
        B: SelectZeroHinted + BitLength + BitCount,
        O: BitFieldSlice<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > SelectZero for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        let index = rank >> LOG2_ZEROS_PER_INVENTORY;
        let pos = self.inventory.get_unchecked(index);
        let rank_at_pos = index << LOG2_ZEROS_PER_INVENTORY;

        self.bits
            .select_zero_hinted_unchecked(rank, pos, rank_at_pos)
    }
}

/// Forget the index.
impl<B: SelectZeroHinted, const LOG2_ZEROS_PER_INVENTORY: usize> ConvertTo<B>
    for SelectZeroFixed1<B, Vec<usize>, LOG2_ZEROS_PER_INVENTORY>
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a selection structure.
impl<
        B: SelectZeroHinted + BitLength + BitCount + AsRef<[usize]>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > ConvertTo<SelectZeroFixed1<B, Vec<usize>, LOG2_ZEROS_PER_INVENTORY>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SelectZeroFixed1<B, Vec<usize>, LOG2_ZEROS_PER_INVENTORY>> {
        Ok(SelectZeroFixed1::new(self))
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        B: SelectZeroHinted + BitLength,
        O: BitFieldSlice<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > BitLength for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`BitCount`] to the underlying implementation.
impl<
        B: SelectZeroHinted + BitCount,
        O: BitFieldSlice<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > BitCount for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Forward [`Select`] to the underlying implementation.
impl<
        B: SelectZeroHinted + Select,
        O: BitFieldSlice<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > Select for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
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
        O: BitFieldSlice<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > Rank for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
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
        O: BitFieldSlice<usize>,
        const LOG2_ZEROS_PER_INVENTORY: usize,
    > RankZero for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
{
    fn rank_zero(&self, pos: usize) -> usize {
        self.bits.rank_zero(pos)
    }

    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_zero_unchecked(pos)
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<B, O, const LOG2_ZEROS_PER_INVENTORY: usize> AsRef<[usize]>
    for SelectZeroFixed1<B, O, LOG2_ZEROS_PER_INVENTORY>
where
    B: AsRef<[usize]> + SelectZeroHinted,
    O: BitFieldSlice<usize>,
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
