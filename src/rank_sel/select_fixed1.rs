/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::prelude::*;
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;
use mem_dbg::*;

/**

A selection structure based on a one-level index.

More precisely, given a constant `LOG2_ONES_PER_INVENTORY`, this index records the position
of one every 2<sup>`LOG2_ONES_PER_INVENTORY`</sup>.
The positions are recorded in a [`BitFieldSlice`] whose [bit width](BitFieldSliceCore::bit_width)
must be sufficient to record all the positions.

Note that [`SelectFixed2`] has usually better performance than this structure, which is mainly
useful for testing, benchmarking and debugging.

The current implementation uses a [`Vec<usize>`] as the underlying [`BitFieldSlice`]. Thus,
the overhead of the structure is [`BitCount::count()`] * [`usize::BITS`] / 2<sup>`LOG2_ONES_PER_INVENTORY`</sup> bits.

The index takes a backend parameter `B` that can be any type that implements
[`SelectHinted`]. This will usually be something like [`CountBitVec`](crate::bits::bit_vec::CountBitVec), or possibly
a [`CountBitVec`](crate::bits::bit_vec::CountBitVec) wrapped in another index structure for which
this structure has delegation (e.g., [`SelectZeroFixed1`](crate::rank_sel::SelectZeroFixed1)). See the documentation
of [`EliasFano`](crate::dict::elias_fano::EliasFano) for an example of this approach.

See [`SelectZeroFixed1`](crate::rank_sel::SelectZeroFixed1) for the same structure for zeros.

*/
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SelectFixed1<
    B: SelectHinted = CountBitVec,
    O: BitFieldSlice<usize> = Vec<usize>,
    const LOG2_ONES_PER_INVENTORY: usize = 8,
> {
    bits: B,
    inventory: O,
    _marker: core::marker::PhantomData<[(); LOG2_ONES_PER_INVENTORY]>,
}

impl<B: SelectHinted + AsRef<[usize]>, const LOG2_ONES_PER_INVENTORY: usize>
    SelectFixed1<B, Vec<usize>, LOG2_ONES_PER_INVENTORY>
{
    pub fn new(bitvec: B, number_of_ones: usize) -> Self {
        let mut res = SelectFixed1 {
            inventory: vec![
                0;
                (number_of_ones + (1 << LOG2_ONES_PER_INVENTORY) - 1)
                    >> LOG2_ONES_PER_INVENTORY
            ],
            bits: bitvec,
            _marker: core::marker::PhantomData,
        };
        res.build_ones();
        res
    }
}

impl<
        B: SelectHinted + AsRef<[usize]>,
        O: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    fn build_ones(&mut self) {
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ones_index = 0;

        for (i, word) in self.bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * usize::BITS as usize) + in_word_index;
                self.inventory.set(ones_index, index);
                next_quantum += 1 << LOG2_ONES_PER_INVENTORY;
                ones_index += 1;
            }

            number_of_ones += ones_in_word;
        }
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectHinted + BitCount, O: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize>
    Select for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let index = rank >> LOG2_ONES_PER_INVENTORY;
        let pos = self.inventory.get_unchecked(index);
        let rank_at_pos = index << LOG2_ONES_PER_INVENTORY;

        self.bits.select_hinted_unchecked(rank, pos, rank_at_pos)
    }
}

/// Forget the index.
impl<B: SelectHinted, T, const LOG2_ONES_PER_INVENTORY: usize> ConvertTo<B>
    for SelectFixed1<B, T, LOG2_ONES_PER_INVENTORY>
where
    T: AsRef<[usize]>,
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a selection structure.
impl<B: SelectHinted + BitCount + AsRef<[usize]>, const LOG2_ONES_PER_INVENTORY: usize>
    ConvertTo<SelectFixed1<B, Vec<usize>, LOG2_ONES_PER_INVENTORY>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SelectFixed1<B, Vec<usize>, LOG2_ONES_PER_INVENTORY>> {
        let count = self.count();
        Ok(SelectFixed1::new(self, count))
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        B: SelectHinted + BitLength,
        O: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > BitLength for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`BitCount`] to the underlying implementation.
impl<B: SelectHinted + BitCount, O: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize>
    BitCount for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Forward [`SelectZero`] to the underlying implementation.
impl<
        B: SelectHinted + SelectZero,
        O: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > SelectZero for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    fn select_zero(&self, rank: usize) -> Option<usize> {
        self.bits.select_zero(rank)
    }
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.bits.select_zero_unchecked(rank)
    }
}

/// Forward [`SelectZeroHinted`] to the underlying implementation.
impl<
        B: SelectHinted + SelectZeroHinted,
        O: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > SelectZeroHinted for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    unsafe fn select_zero_hinted_unchecked(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        self.bits
            .select_zero_hinted_unchecked(rank, pos, rank_at_pos)
    }

    #[inline(always)]
    fn select_zero_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> Option<usize> {
        self.bits.select_zero_hinted(rank, pos, rank_at_pos)
    }
}

/// Forward [`Rank`] to the underlying implementation.
impl<B: SelectHinted + Rank, O: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize> Rank
    for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    fn rank(&self, pos: usize) -> usize {
        self.bits.rank(pos)
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_unchecked(pos)
    }
}

/// Forward [`RankZero`] to the underlying implementation.
impl<B: SelectHinted + RankZero, O: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize>
    RankZero for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
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
        B: SelectHinted + AsRef<[usize]>,
        O: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > AsRef<[usize]> for SelectFixed1<B, O, LOG2_ONES_PER_INVENTORY>
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
