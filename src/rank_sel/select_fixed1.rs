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
use common_traits::SelectInWord;
use epserde::*;
use mem_dbg::*;

/**

A selection structure based on a one-level index.

More precisely, given a constant `LOG2_ONES_PER_INVENTORY`, this index records the position
of one every 2<sup>`LOG2_ONES_PER_INVENTORY`</sup>.
The positions are recorded in a [`BitFieldSlice`] whose [bit width](BitFieldSliceCore::bit_width)
must be sufficient to record all the positions.

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
    I: BitFieldSlice<usize> = Vec<usize>,
    const LOG2_ONES_PER_INVENTORY: usize = 8,
> {
    bits: B,
    inventory: I,
    _marker: core::marker::PhantomData<[(); LOG2_ONES_PER_INVENTORY]>,
}

impl<B: SelectHinted + BitCount + AsRef<[usize]>, const LOG2_ONES_PER_INVENTORY: usize>
    SelectFixed1<B, Vec<usize>, LOG2_ONES_PER_INVENTORY>
{
    pub fn new(bitvec: B) -> Self {
        let mut res = SelectFixed1 {
            inventory: vec![
                0;
                (bitvec.count_ones() + (1 << LOG2_ONES_PER_INVENTORY) - 1)
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
        I: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
{
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
        SelectFixed1 {
            bits,
            inventory,
            _marker: core::marker::PhantomData,
        }
    }

    /// Modify the inner BitVector with possibly another type
    pub fn map_bits<B2>(
        self,
        f: impl FnOnce(B) -> B2,
    ) -> SelectFixed1<B2, I, LOG2_ONES_PER_INVENTORY>
    where
        B2: SelectHinted + BitLength + BitCount + AsRef<[usize]>,
    {
        SelectFixed1 {
            bits: f(self.bits),
            inventory: self.inventory,
            _marker: core::marker::PhantomData,
        }
    }

    /// Modify the inner inventory with possibly another type
    pub fn map_inventory<I2>(
        self,
        f: impl FnOnce(I) -> I2,
    ) -> SelectFixed1<B, I2, LOG2_ONES_PER_INVENTORY>
    where
        I2: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
    {
        SelectFixed1 {
            bits: self.bits,
            inventory: f(self.inventory),
            _marker: core::marker::PhantomData,
        }
    }

    /// Modify the inner BitVector and inventory with possibly other types
    pub fn map<B2, I2>(
        self,
        f: impl FnOnce(B, I) -> (B2, I2),
    ) -> SelectFixed1<B2, I2, LOG2_ONES_PER_INVENTORY>
    where
        B2: SelectHinted + BitLength + BitCount + AsRef<[usize]>,
        I2: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
    {
        let (bits, inventory) = f(self.bits, self.inventory);
        SelectFixed1 {
            bits,
            inventory,
            _marker: core::marker::PhantomData,
        }
    }

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
impl<B: SelectHinted + BitCount, I: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize>
    Select for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let index = rank >> LOG2_ONES_PER_INVENTORY;
        let pos = self.inventory.get_unchecked(index);
        let rank_at_pos = index << LOG2_ONES_PER_INVENTORY;

        self.bits.select_hinted_unchecked(rank, pos, rank_at_pos)
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        B: SelectHinted + BitLength,
        I: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > BitLength for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`BitCount`] to the underlying implementation.
impl<B: SelectHinted + BitCount, I: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize>
    BitCount for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
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

/// Forward [`SelectZero`] to the underlying implementation.
impl<
        B: SelectHinted + SelectZero,
        I: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > SelectZero for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
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
        I: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > SelectZeroHinted for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
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
impl<B: SelectHinted + Rank, I: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize> Rank
    for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
{
    fn rank(&self, pos: usize) -> usize {
        self.bits.rank(pos)
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_unchecked(pos)
    }
}

/// Forward [`RankZero`] to the underlying implementation.
impl<B: SelectHinted + RankZero, I: BitFieldSlice<usize>, const LOG2_ONES_PER_INVENTORY: usize>
    RankZero for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
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
        I: BitFieldSlice<usize>,
        const LOG2_ONES_PER_INVENTORY: usize,
    > AsRef<[usize]> for SelectFixed1<B, I, LOG2_ONES_PER_INVENTORY>
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
