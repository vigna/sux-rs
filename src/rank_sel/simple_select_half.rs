/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::{bits::CountBitVec, traits::*};
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
    bitvec: B,
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
        let inventory_size =
            1 + (expected_ones + Self::ONES_PER_INVENTORY - 1) / Self::ONES_PER_INVENTORY;
        let inventory_len = inventory_size * (Self::U64_PER_SUBINVENTORY + 1);
        // inventory_size, an u64 for the first layer index, and Self::U64_PER_SUBINVENTORY for the sub layer
        let mut inventory = vec![0; inventory_len];
        // scan the bitvec and fill the first layer of the inventory
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ptr = 0;
        for (i, word) in bitvec.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * u64::BITS as usize) + in_word_index;

                // write the one in the inventory
                inventory[ptr] = index as u64;

                ptr += Self::U64_PER_SUBINVENTORY + 1;
                next_quantum += Self::ONES_PER_INVENTORY as u64;
            }
            number_of_ones += ones_in_word;
        }
        // in the last inventory write the number of bits
        inventory[ptr] = BitLength::len(&bitvec) as u64;

        // build the index (in parallel if rayon enabled)
        let iter = 0..inventory_size - 1;
        //#[cfg(feature = "rayon")]
        //let iter = iter.into_par_iter();

        // fill the second layer of the index
        iter.for_each(|inventory_idx| {
            let start_idx = inventory_idx * (1 + Self::U64_PER_SUBINVENTORY);
            let end_idx = start_idx + 1 + Self::U64_PER_SUBINVENTORY;

            let start_bit_idx = inventory[start_idx];
            let end_bit_idx = inventory[start_idx + Self::U64_PER_SUBINVENTORY + 1];
            dbg!(start_bit_idx, end_bit_idx);
            let end_word_idx = end_bit_idx.div_ceil(u64::BITS as u64);
            let span = end_bit_idx - start_bit_idx;

            let mut word_idx = start_bit_idx / u64::BITS as u64;
            let bit_idx = start_bit_idx % u64::BITS as u64;

            // cleanup the lower bits
            let mut word = (bitvec.as_ref()[word_idx as usize] >> bit_idx) << bit_idx;
            //
            let mut number_of_ones = inventory_idx * Self::ONES_PER_INVENTORY;
            let mut next_quantum = number_of_ones;

            let subinventory = unsafe { inventory[start_idx + 1..end_idx].align_to_mut().1 };

            let (quantum, size) = if span < u16::MAX as u64 {
                (Self::ONES_PER_SUB16, core::mem::size_of::<u16>())
            } else {
                (Self::ONES_PER_SUB64, core::mem::size_of::<u64>())
            };

            dbg!(quantum, size, Self::ONES_PER_SUB16, Self::ONES_PER_SUB64);
            assert_eq!(quantum, Self::ONES_PER_SUB16);
            let mut subinventory_idx = 0;
            loop {
                let ones_in_word = word.count_ones() as usize;
                dbg!(next_quantum, number_of_ones, ones_in_word);
                while number_of_ones + ones_in_word > next_quantum {
                    let in_word_index = word.select_in_word(next_quantum - number_of_ones);
                    let index = (word_idx * u64::BITS as u64) + in_word_index as u64;
                    let sub_offset = (index - start_bit_idx) as u16;
                    dbg!(sub_offset);
                    subinventory[subinventory_idx..subinventory_idx + size]
                        .copy_from_slice(&sub_offset.to_ne_bytes());

                    subinventory_idx += size;
                    next_quantum += quantum;
                }
                number_of_ones += ones_in_word;

                word_idx += 1;
                if word_idx == end_word_idx {
                    break;
                }

                dbg!(word_idx, end_word_idx, bitvec.as_ref().len());
                word = bitvec.as_ref()[word_idx as usize];
            }
        });

        dbg!(&inventory);

        Self { bitvec, inventory }
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

        dbg!(inventory_rank);
        eprintln!("{:x?}", &u64s);
        // if the inventory_rank is positive, the subranks are u16s otherwise they are u64s
        let (pos, residual) = if inventory_rank >= 0 {
            // dense case, read the u16s
            let (pre, u16s, post) = u64s.align_to::<u16>();
            // u16 should always be aligned with u64s ...
            debug_assert!(pre.is_empty());
            debug_assert!(post.is_empty());
            dbg!(u16s[subrank / Self::ONES_PER_SUB16] as u64);
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

        dbg!(rank, pos, residual);
        // linear scan to finish the search
        self.bitvec
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
        self.bitvec.select_zero(rank)
    }
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.bitvec.select_zero_unchecked(rank)
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
        self.bitvec.len()
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
        self.bitvec.count()
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
        self.bitvec.as_ref()
    }
}
