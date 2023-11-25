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

/// An index that records the position of the ones in a bit vector at a fixed
/// set of positions.
///
/// More precisely, given a constant quantum <var>q</var>, this index records the position
/// of the ones at positions 0, <var>q</var>, <var>2q</var>, &hellip;, and so on.
/// The positions are recorded in a provided [`BitFieldSliceMut`] whose [bit width](BitFieldSliceCore::bit_width)
/// must be sufficient to record all the positions.
///
/// The index takes a backend parameter `B` that can be any type that implements
/// [`SelectHinted`]. This will usually be something like [`CountBitVec`](crate::bits::bit_vec::CountBitVec), or possibly
/// a [`CountBitVec`](crate::bits::bit_vec::CountBitVec) wrapped in another index structure for which
/// this structure has delegation (e.g., [`SelectZeroFixed1`](crate::rank_sel::SelectZeroFixed1)). See the documentation
/// of [`EliasFano`](crate::dict::elias_fano::EliasFano) for an example of this approach.
///
/// See [`SelectZeroFixed1`](crate::rank_sel::SelectZeroFixed1) for the same index for zeros.
#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SelectFixed1<
    B: SelectHinted = CountBitVec,
    O: BitFieldSlice<usize> = Vec<usize>,
    const QUANTUM_LOG2: usize = 8,
> {
    bits: B,
    ones: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectHinted + AsRef<[usize]>, const QUANTUM_LOG2: usize>
    SelectFixed1<B, Vec<usize>, QUANTUM_LOG2>
{
    pub fn new(bitvec: B, number_of_ones: usize) -> Result<Self> {
        let mut res = SelectFixed1 {
            ones: vec![0; (number_of_ones + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2],
            bits: bitvec,
            _marker: core::marker::PhantomData,
        };
        res.build_ones()?;
        Ok(res)
    }
}

impl<
        B: SelectHinted + AsRef<[usize]>,
        O: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
        const QUANTUM_LOG2: usize,
    > SelectFixed1<B, O, QUANTUM_LOG2>
{
    fn build_ones(&mut self) -> Result<()> {
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ones_index = 0;

        for (i, word) in self.bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * usize::BITS as usize) + in_word_index;
                self.ones.set(ones_index, index);
                next_quantum += 1 << QUANTUM_LOG2;
                ones_index += 1;
            }

            number_of_ones += ones_in_word;
        }
        Ok(())
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectHinted, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> Select
    for SelectFixed1<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let pos = self.ones.get_unchecked(index);
        let rank_at_pos = index << QUANTUM_LOG2;

        self.bits.select_hinted_unchecked(rank, pos, rank_at_pos)
    }
}

/// Forget the index.
impl<B: SelectHinted, T, const QUANTUM_LOG2: usize> ConvertTo<B>
    for SelectFixed1<B, T, QUANTUM_LOG2>
where
    T: AsRef<[usize]>,
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a selection structure.
impl<B: SelectHinted + AsRef<[usize]>, const QUANTUM_LOG2: usize>
    ConvertTo<SelectFixed1<B, Vec<usize>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SelectFixed1<B, Vec<usize>, QUANTUM_LOG2>> {
        let count = self.count();
        SelectFixed1::new(self, count)
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<B: SelectHinted + BitLength, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> BitLength
    for SelectFixed1<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`BitCount`] to the underlying implementation.
impl<B: SelectHinted, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> BitCount
    for SelectFixed1<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Forward [`SelectZero`] to the underlying implementation.
impl<B: SelectHinted + SelectZero, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> SelectZero
    for SelectFixed1<B, O, QUANTUM_LOG2>
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
impl<B: SelectHinted + SelectZeroHinted, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize>
    SelectZeroHinted for SelectFixed1<B, O, QUANTUM_LOG2>
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

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<B: SelectHinted + AsRef<[usize]>, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize>
    AsRef<[usize]> for SelectFixed1<B, O, QUANTUM_LOG2>
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
