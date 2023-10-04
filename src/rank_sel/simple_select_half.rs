use crate::{bits::prelude::CountBitVec, traits::prelude::*};
use common_traits::SelectInWord;
use epserde::*;
#[cfg(feature="rayon")]
use rayon::prelude::*;

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

    const LOG2_ONES_PER_SUB64: usize =
        LOG2_ONES_PER_INVENTORY - LOG2_U64_PER_SUBINVENTORY;
    const ONES_PER_SUB64: usize = 1 << Self::LOG2_ONES_PER_SUB64;

    const LOG2_ONES_PER_SUB16: usize = Self::LOG2_ONES_PER_SUB64 - 2;
    const ONES_PER_SUB16: usize = 1 << Self::LOG2_ONES_PER_SUB16;
}

impl<
        B: SelectHinted + AsRef<[usize]>,
        const LOG2_ONES_PER_INVENTORY: usize,
        const LOG2_U64_PER_SUBINVENTORY: usize,
    > SimpleSelectHalf<B, Vec<u64>, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY>
{
    pub fn new(bitvec: B) -> Self {
        // estimate the number of ones with our core assumption!
        let expected_ones = bitvec.len() / 2; 
        // number of inventories we will create
		let inventory_size = 1 + (expected_ones + Self::ONES_PER_INVENTORY - 1) / Self::ONES_PER_INVENTORY;
        // inventory_size, an u64 for the first layer index, and Self::U64_PER_SUBINVENTORY for the sub layer
        let mut inventory = vec![0; inventory_size];
        // scan the bitvec and fill the first layer of the inventory
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ptr = 0;
        for (i, word) in bitvec.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * usize::BITS as usize) + in_word_index;

                // write the one in the inventory
                inventory[ptr] = index as u64;

                ptr += 1;
                next_quantum += Self::ONES_PER_INVENTORY as u64;
            }
            number_of_ones += ones_in_word;
        }
        // in the last inventory write the number of bits
        inventory[ptr] = bitvec.len() as u64;

        // build the index (in parallel if rayon enabled)
        #[cfg(feature="rayon")]
        let iter = (0..inventory_size).into_par_iter();
        #[cfg(not(feature="rayon"))]
        let iter = (0..inventory_size);
    
        // fill the second layer of the index
        let data = bitvec.as_ref();
        iter.for_each(|inventory_idx| {
            let start_idx = inventory_idx * (1 + Self::U64_PER_SUBINVENTORY);
            let span = inventory[start_idx + 1] - inventory[start_idx];
            if span < u16::MAX as u64 {
                // dense

            } else {
                // sparse
                for word in data.iter().enumerate() {
                    
                }
            }
        });

        Self {
            bitvec,
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
        let sub_start_idx = start_idx + 1 + subrank;
        let u64s = self.inventory.as_ref().get_unchecked(sub_start_idx..sub_start_idx + Self::U64_PER_SUBINVENTORY);
        // if the inventory_rank is positive, the subranks are u16s otherwise they are u64s
        let (pos, residual) = if inventory_rank >= 0 {
            // dense case, read the u16s
            let (pre, u16s, post) = u64s.align_to::<u16>();
            // u16 should always be aligned with u64s ...
            debug_assert!(pre.is_empty());
            debug_assert!(post.is_empty());
            (
                inventory_rank as u64 + *u16s.get_unchecked(subrank / Self::ONES_PER_SUB16) as u64,
                subrank % Self::ONES_PER_SUB16,
            )
        } else {
			debug_assert!(subrank < Self::U64_PER_SUBINVENTORY);
            // sparse case, read the u64s
            (
                (-inventory_rank - 1) as u64 + *u64s.get_unchecked(subrank / Self::ONES_PER_SUB16),
                subrank % Self::ONES_PER_SUB64,
            )
        };
        // linear scan to finish the search
        self.bitvec.select_hinted_unchecked(rank, pos as usize, rank - residual)
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
