/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::bit_field_slice::BitFieldSlice;
use crate::traits::bit_field_slice::BitFieldSliceMut;
use crate::{bits::prelude::CountBitVec, traits::prelude::*};
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;

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
/// this structure has delegation (e.g., [`QuantumIndex`](crate::rank_sel::QuantumIndex)). See the documentation
/// of [`EliasFano`](crate::dict::elias_fano::EliasFano) for an example of this approach.
///
/// See [`QuantumIndex`](crate::rank_sel::QuantumIndex) for the same index for ones.
#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantumZeroIndex<
    B: SelectZeroHinted = CountBitVec,
    O: BitFieldSlice<usize> = Vec<usize>,
    const QUANTUM_LOG2: usize = 8,
> {
    bits: B,
    zeros: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<
        B: SelectZeroHinted + AsRef<[usize]>,
        O: BitFieldSlice<usize> + BitFieldSliceMut<usize> + core::fmt::Debug,
        const QUANTUM_LOG2: usize,
    > QuantumZeroIndex<B, O, QUANTUM_LOG2>
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

/// Provide the hint to the underlying structure
impl<B: SelectZeroHinted, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> SelectZero
    for QuantumZeroIndex<B, O, QUANTUM_LOG2>
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

/// If the underlying implementation has select, forward the methods
impl<B: SelectZeroHinted + Select, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> Select
    for QuantumZeroIndex<B, O, QUANTUM_LOG2>
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

impl<B: SelectZeroHinted, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> BitLength
    for QuantumZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<B: SelectZeroHinted, O: BitFieldSlice<usize>, const QUANTUM_LOG2: usize> BitCount
    for QuantumZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Forget the index.
impl<B: SelectZeroHinted, const QUANTUM_LOG2: usize> ConvertTo<B>
    for QuantumZeroIndex<B, Vec<usize>, QUANTUM_LOG2>
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a quantum index.
impl<B: SelectZeroHinted + AsRef<[usize]>, const QUANTUM_LOG2: usize>
    ConvertTo<QuantumZeroIndex<B, Vec<usize>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<QuantumZeroIndex<B, Vec<usize>, QUANTUM_LOG2>> {
        let mut res = QuantumZeroIndex {
            zeros: vec![0; (self.len() - self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2],
            bits: self,
            _marker: core::marker::PhantomData,
        };
        res.build_zeros()?;
        Ok(res)
    }
}

impl<B, O, const QUANTUM_LOG2: usize> AsRef<[usize]> for QuantumZeroIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[usize]> + SelectZeroHinted,
    O: BitFieldSlice<usize>,
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}
