/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::prelude::*;
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;

/// An index that records the position of the ones in a bit vector at a fixed
/// set of positions.
///
/// More precisely, given a constant quantum <var>q</var>, this index records the position
/// of the ones at positions 0, <var>q</var>, <var>2q</var>, &hellip;, and so on.
/// The positions are recorded in a provided [`VSliceMut`] whose [bit width](VSliceCore::bit_width)
/// must be sufficient to record all the positions.
///
/// The index takes a backend parameter `B` that can be any type that implements
/// [`SelectHinted`]. This will usually be something like [`CountBitVec`](crate::bits::bit_vec::CountBitVec), or possibly
/// a [`CountBitVec`](crate::bits::bit_vec::CountBitVec) wrapped in another index structure for which
/// this structure has delegation (e.g., [`QuantumZeroIndex`](crate::rank_sel::QuantumZeroIndex)). See the documentation
/// of [`EliasFano`](crate::dict::elias_fano::EliasFano) for an example of this approach.
///
/// See [`QuantumZeroIndex`](crate::rank_sel::QuantumZeroIndex) for the same index for zeros.
#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantumIndex<B: SelectHinted, O: VSlice = Vec<usize>, const QUANTUM_LOG2: usize = 8> {
    bits: B,
    ones: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectHinted + AsRef<[usize]>, O: VSlice + VSliceMut, const QUANTUM_LOG2: usize>
    QuantumIndex<B, O, QUANTUM_LOG2>
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
                let index = (i * 64) + in_word_index;
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
impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> Select
    for QuantumIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let pos = self.ones.get_unchecked(index);
        let rank_at_pos = index << QUANTUM_LOG2;

        self.bits.select_unchecked_hinted(rank, pos, rank_at_pos)
    }
}

/// If the underlying implementation has select zero, forward the methods.
impl<B: SelectHinted + SelectZero, O: VSlice, const QUANTUM_LOG2: usize> SelectZero
    for QuantumIndex<B, O, QUANTUM_LOG2>
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

/// If the underlying implementation has hint for select zero, forward the methods.
impl<B: SelectHinted + SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZeroHinted
    for QuantumIndex<B, O, QUANTUM_LOG2>
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
}

impl<B: SelectHinted + BitLength, O: VSlice, const QUANTUM_LOG2: usize> BitLength
    for QuantumIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> BitCount
    for QuantumIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Forget the index.
impl<B: SelectHinted, T, const QUANTUM_LOG2: usize> ConvertTo<B>
    for QuantumIndex<B, Vec<T>, QUANTUM_LOG2>
where
    Vec<T>: VSlice,
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a quantum index.
impl<B: SelectHinted + AsRef<[usize]>, const QUANTUM_LOG2: usize>
    ConvertTo<QuantumIndex<B, Vec<usize>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<QuantumIndex<B, Vec<usize>, QUANTUM_LOG2>> {
        let mut res = QuantumIndex {
            ones: vec![0; (self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2],
            bits: self,
            _marker: core::marker::PhantomData,
        };
        res.build_ones()?;
        Ok(res)
    }
}

impl<B, O, const QUANTUM_LOG2: usize> AsRef<[usize]> for QuantumIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[usize]> + SelectHinted,
    O: VSlice,
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
