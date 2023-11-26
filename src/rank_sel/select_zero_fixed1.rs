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

/// An index that records the position of the zeros in a bit vector at a fixed
/// set of positions.
///
/// More precisely, given a constant quantum <var>q</var>, this index records the position
/// of the zeros at positions 0, <var>q</var>, <var>2q</var>, &hellip;, and so on.
/// The positions are recorded in a provided [`BitFieldSliceMut`] whose [bit width](BitFieldSliceCore::bit_width)
/// must be sufficient to record all the positions.
///
/// The index takes a backend parameter `B` that can be any type that implements
/// [`SelectHinted`]. This will usually be something like [`CountBitVec`](crate::bits::bit_vec::CountBitVec), or possibly
/// a [`CountBitVec`](crate::bits::bit_vec::CountBitVec) wrapped in another index structure for which
/// this structure has delegation (e.g., [`SelectFixed1`](crate::rank_sel::SelectFixed1)). See the documentation
/// of [`EliasFano`](crate::dict::elias_fano::EliasFano) for an example of this approach.
///
/// See [`SelectFixed1`](crate::rank_sel::SelectFixed1) for the same index for ones.
#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SelectZeroFixed1<
    B: SelectZeroHinted = CountBitVec,
    O: BitFieldSlice<usize> = Vec<usize>,
    const QUANTUM_LOG2: usize = 8,
> {
    bits: B,
    zeros: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectZeroHinted + BitLength + BitCount + AsRef<[usize]>, const QUANTUM_LOG2: usize>
    SelectZeroFixed1<B, Vec<usize>, QUANTUM_LOG2>
{
    pub fn new(bitvec: B) -> Result<Self> {
        let mut res = SelectZeroFixed1 {
            zeros: vec![
                0;
                (bitvec.len() - bitvec.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2
            ],
            bits: bitvec,
            _marker: core::marker::PhantomData,
        };
        res.build_zeros()?;
        Ok(res)
    }
}

impl<
        B: SelectZeroHinted + BitLength + AsRef<[usize]>,
        O: BitFieldSlice<usize> + BitFieldSliceMut<usize>,
        const QUANTUM_LOG2: usize,
    > SelectZeroFixed1<B, O, QUANTUM_LOG2>
{
    fn build_zeros(&mut self) -> Result<()> {
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
                    return Ok(());
                }
                self.zeros.set(zeros_index, index);
                next_quantum += 1 << QUANTUM_LOG2;
                zeros_index += 1;
            }

            number_of_zeros += zeros_in_word;
        }

        Ok(())
    }
}

/// Provide the hint to the underlying structure.
impl<
        B: SelectZeroHinted + BitLength + BitCount,
        O: BitFieldSlice<usize>,
        const QUANTUM_LOG2: usize,
    > SelectZero for SelectZeroFixed1<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let pos = self.zeros.get_unchecked(index);
        let rank_at_pos = index << QUANTUM_LOG2;

        self.bits
            .select_zero_hinted_unchecked(rank, pos, rank_at_pos)
    }
}

/// Forget the index.
impl<B: SelectZeroHinted, const QUANTUM_LOG2: usize> ConvertTo<B>
    for SelectZeroFixed1<B, Vec<usize>, QUANTUM_LOG2>
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a selection structure.
impl<B: SelectZeroHinted + BitLength + BitCount + AsRef<[usize]>, const QUANTUM_LOG2: usize>
    ConvertTo<SelectZeroFixed1<B, Vec<usize>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SelectZeroFixed1<B, Vec<usize>, QUANTUM_LOG2>> {
        SelectZeroFixed1::new(self)
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<B: SelectZeroHinted + BitLength, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> BitLength
    for SelectZeroFixed1<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Forward [`BitCount`] to the underlying implementation.
impl<B: SelectZeroHinted + BitCount, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> BitCount
    for SelectZeroFixed1<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Forward [`Select`] to the underlying implementation.
impl<B: SelectZeroHinted + Select, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> Select
    for SelectZeroFixed1<B, O, QUANTUM_LOG2>
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
impl<B: SelectZeroHinted + Rank, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> Rank
    for SelectZeroFixed1<B, O, QUANTUM_LOG2>
{
    fn rank(&self, pos: usize) -> usize {
        self.bits.rank(pos)
    }

    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_unchecked(pos)
    }
}

/// Forward [`RankZero`] to the underlying implementation.
impl<B: SelectZeroHinted + RankZero, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> RankZero
    for SelectZeroFixed1<B, O, QUANTUM_LOG2>
{
    fn rank_zero(&self, pos: usize) -> usize {
        self.bits.rank_zero(pos)
    }

    unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
        self.bits.rank_zero_unchecked(pos)
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<B, O, const QUANTUM_LOG2: usize> AsRef<[usize]> for SelectZeroFixed1<B, O, QUANTUM_LOG2>
where
    B: AsRef<[usize]> + SelectZeroHinted,
    O: BitFieldSlice<usize>,
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
