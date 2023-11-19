/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::{bits::CountBitVec, traits::*};
use anyhow::Result;
use bitvec::view::AsBits;
use common_traits::SelectInWord;
use epserde::*;
//#[cfg(feature = "rayon")]
//use rayon::prelude::*;

/// Two layer index (with interleaved layers) optimized for
/// a bitmap with approximately half ones and half zeros.
/// This is meant for elias-fano high-bits.
#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimpleSelectHalf<
    B: SelectHinted = CountBitVec,
    I: AsRef<[u64]> = Vec<u64>,
    const LOG2_ONES_PER_INVENTORY: usize = 10,
    const LOG2_U64_PER_SUBINVENTORY: usize = 2,
> {
    bits: B,
    inventory: I,
}

/// constants used throughout the code
impl<
        B: SelectHinted,
        I: AsRef<[u64]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SimpleSelectHalf<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    const ONES_PER_INVENTORY: usize = 1 << LOG2_ONES_PER_INVENTORY;
    const U64_PER_SUBINVENTORY: usize = 1 << LOG2_U64_PER_SUBINVENTORY;
    const U64_PER_INVENTORY: usize = 1 + Self::U64_PER_SUBINVENTORY;

    const LOG2_ONES_PER_SUB64: usize = LOG2_ONES_PER_INVENTORY - LOG2_U64_PER_SUBINVENTORY;
    const ONES_PER_SUB64: usize = 1 << Self::LOG2_ONES_PER_SUB64;

    const LOG2_ONES_PER_SUB16: usize = Self::LOG2_ONES_PER_SUB64 - 2;
    const ONES_PER_SUB16: usize = 1 << Self::LOG2_ONES_PER_SUB16;
}

impl<
        B: SelectHinted + AsRef<[usize]> + BitLength,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SimpleSelectHalf<B, Vec<u64>, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    pub fn new(bitvec: B) -> Self {
        // estimate the number of ones with our core assumption!
        let expected_ones = BitLength::len(&bitvec) / 2;
        // number of inventories we will create
        let inventory_size = 1 + expected_ones.div_ceil(Self::ONES_PER_INVENTORY);
        let inventory_len = inventory_size * Self::U64_PER_INVENTORY;
        // inventory_size, an u64 for the first layer index, and Self::U64_PER_SUBINVENTORY for the sub layer
        let mut inventory = Vec::with_capacity(inventory_len);
        // scan the bitvec and fill the first layer of the inventory
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        for (i, word) in bitvec.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * u64::BITS as usize) + in_word_index;

                // write the one in the inventory
                inventory.push(index as u64);
                for _ in 0..Self::U64_PER_SUBINVENTORY {
                    inventory.push(0);
                }

                next_quantum += Self::ONES_PER_INVENTORY as u64;
            }
            number_of_ones += ones_in_word;
        }
        // in the last inventory write the number of bits
        inventory.push(BitLength::len(&bitvec) as u64);

        // build the index (in parallel if rayon enabled)
        let iter = 0..inventory.len().div_ceil(Self::U64_PER_INVENTORY) - 1;
        //#[cfg(feature = "rayon")]
        //let iter = iter.into_par_iter();

        // fill the second layer of the index
        iter.for_each(|inventory_idx| {
            // get the start and end u64 index of the current inventory
            let start_idx = inventory_idx * Self::U64_PER_INVENTORY;
            let end_idx = start_idx + Self::U64_PER_INVENTORY;
            // read the first level index to get the start and end bit index
            let start_bit_idx = inventory[start_idx];
            let end_bit_idx = inventory[end_idx];
            dbg!(start_bit_idx, end_bit_idx);
            debug_assert!(
                end_bit_idx >= start_bit_idx,
                "Start: {} End: {}",
                start_bit_idx,
                end_bit_idx
            );
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            // compute were we should the word boundaries of where we should
            // scan
            let mut word_idx = start_bit_idx / u64::BITS as u64;

            // cleanup the lower bits
            let bit_idx = start_bit_idx % u64::BITS as u64;
            let mut word = (bitvec.as_ref()[word_idx as usize] >> bit_idx) << bit_idx;
            // compute the global number of ones
            let mut number_of_ones = inventory_idx * Self::ONES_PER_INVENTORY;
            // and what's the next bit rank which index should log in the sub
            // inventory (the first subinventory element is always 0)
            let mut next_quantum = number_of_ones;

            if span < u16::MAX as u64 {
                let quantum = Self::ONES_PER_SUB16;
                let end_word_idx = (end_bit_idx - quantum as u64).div_ceil(u64::BITS as u64);
                // get a mutable reference to the subinventory as u16s, since
                // it's a u64 slice it's always aligned
                // + 1 to skip the first index value
                let subinventory: &mut [u16] =
                    unsafe { inventory[start_idx + 1..end_idx].align_to_mut().1 };

                // the first subinventory element is always 0
                let mut subinventory_idx = 1;
                next_quantum += quantum;

                loop {
                    let ones_in_word = word.count_ones() as usize;

                    // if the quantum is in this word, write it in the subinventory
                    // this can happen multiple times if the quantum is small
                    while number_of_ones + ones_in_word > next_quantum {
                        debug_assert!(next_quantum <= end_bit_idx as _);
                        // find the quantum bit in the word
                        let in_word_index = word.select_in_word(next_quantum - number_of_ones);
                        // compute the global index of the quantum bit in the bitvec
                        let bit_index = (word_idx * u64::BITS as u64) + in_word_index as u64;
                        // compute the offset of the quantum bit
                        // from the start of the subinventory
                        let sub_offset = (bit_index - start_bit_idx) as u16;

                        // write the offset in the subinventory
                        subinventory[subinventory_idx] = sub_offset;

                        // update the subinventory index and the next quantum
                        subinventory_idx += 1;
                        next_quantum += quantum;
                    }

                    // we are done with the word, so update the number of ones
                    number_of_ones += ones_in_word;
                    // move to the next word and boundcheck
                    word_idx += 1;
                    if word_idx == end_word_idx {
                        break;
                    }

                    // read the next word
                    word = bitvec.as_ref()[word_idx as usize];
                }
            } else {
                let quantum = Self::ONES_PER_SUB64;
                panic!();
            };
        });
        Self {
            bits: bitvec,
            inventory,
        }
    }
}

/// Provide the hint to the underlying structure
impl<
        B: SelectHinted,
        I: AsRef<[u64]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > Select for SimpleSelectHalf<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        // find the index of the first level inventory
        let inventory_index = rank / Self::ONES_PER_INVENTORY;
        // find the index of the second level inventory
        let subrank = rank % Self::ONES_PER_INVENTORY;
        // find the position of the first index value in the interleaved inventory
        let start_idx = inventory_index * (1 + Self::U64_PER_SUBINVENTORY);
        // read the potentially unaliged i64 (i.e. the first index value)
        let inventory_rank = *self.inventory.as_ref().get_unchecked(start_idx) as i64;
        // get a reference to the u64s in this subinventory
        let u64s = self
            .inventory
            .as_ref()
            .get_unchecked(start_idx + 1..start_idx + 1 + Self::U64_PER_SUBINVENTORY);

        // if the inventory_rank is positive, the subranks are u16s otherwise they are u64s
        let (pos, residual) = if inventory_rank >= 0 {
            // dense case, read the u16s
            let (pre, u16s, post) = u64s.align_to::<u16>();
            // u16 should always be aligned with u64s ...
            debug_assert!(pre.is_empty());
            debug_assert!(post.is_empty());
            (
                inventory_rank as u64 + u16s[subrank / Self::ONES_PER_SUB16] as u64,
                subrank % Self::ONES_PER_SUB16,
            )
        } else {
            debug_assert!(subrank < Self::U64_PER_SUBINVENTORY);
            // sparse case, read the u64s
            (
                (-inventory_rank - 1) as u64 + u64s[subrank / Self::ONES_PER_SUB16],
                subrank % Self::ONES_PER_SUB64,
            )
        };

        // linear scan to finish the search
        self.bits
            .select_hinted_unchecked(rank, pos as usize, rank - residual)
    }
}

/// If the underlying implementation has select zero, forward the methods.
impl<
        B: SelectHinted + SelectZero,
        I: AsRef<[u64]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SelectZero for SimpleSelectHalf<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
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

/// If the underlying implementation has BitLength, forward the methods.
impl<
        B: SelectHinted + BitLength,
        I: AsRef<[u64]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > BitLength for SimpleSelectHalf<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// If the underlying implementation has BitCount, forward the methods.
impl<
        B: SelectHinted + BitCount,
        I: AsRef<[u64]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > BitCount for SimpleSelectHalf<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// If the underlying implementation has AsRef<[usize]>, forward the methods.
impl<
        B: SelectHinted + AsRef<[usize]>,
        I: AsRef<[u64]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > AsRef<[usize]>
    for SimpleSelectHalf<B, I, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    fn as_ref(&self) -> &[usize] {
        self.bits.as_ref()
    }
}

/// Forget the index.
impl<B: SelectHinted, T, const QUANTUM_LOG2: usize> ConvertTo<B>
    for SimpleSelectHalf<B, T, QUANTUM_LOG2>
where
    T: AsRef<[u64]>,
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

/// Create and add a quantum index.
impl<B: SelectHinted + AsRef<[usize]> + BitLength, const QUANTUM_LOG2: usize>
    ConvertTo<SimpleSelectHalf<B, Vec<u64>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SimpleSelectHalf<B, Vec<u64>, QUANTUM_LOG2>> {
        Ok(SimpleSelectHalf::new(self))
    }
}
