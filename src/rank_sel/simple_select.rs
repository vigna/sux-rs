use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};
use std::cmp::{max, min};

use crate::prelude::{BitCount, BitFieldSlice, BitLength, BitVec, Select, SelectHinted};

fn most_significant_one(word: usize) -> i32 {
    if word == 0 {
        -1
    } else {
        63 ^ word.leading_zeros() as i32
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct SimpleSelect<B: SelectHinted = BitVec, I: AsRef<[u64]> = Vec<u64>> {
    bits: B,
    inventory: I,
    exact_spill: I,
    num_ones: usize,
    log2_ones_per_inventory: usize,
    log2_ones_per_sub16: usize,
    log2_u64_per_subinventory: usize,
    ones_per_sub64: usize,
    u64_per_inventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
    inventory_size: usize,
    exact_spill_size: usize,
}

impl<B: SelectHinted, I: AsRef<[u64]>> SimpleSelect<B, I> {
    const MAX_ONES_PER_INVENTORY: usize = 8192;
}

impl SimpleSelect<BitVec, Vec<u64>> {
    pub fn new(bits: BitVec, max_log2_u64_per_subinventory: u64) -> Self {
        let num_bits = max(1usize, bits.len() as usize);
        let num_ones = bits.count();

        let log2_ones_per_inventory = (num_ones * Self::MAX_ONES_PER_INVENTORY)
            .div_ceil(num_bits)
            .max(1)
            .ilog2() as usize;

        let ones_per_inventory = 1usize << log2_ones_per_inventory;
        let ones_per_inventory_mask = ones_per_inventory - 1;
        let inventory_size = num_ones.div_ceil(ones_per_inventory - 1);

        let log2_u64_per_subinventory = min(
            max_log2_u64_per_subinventory as i32,
            max(0, log2_ones_per_inventory as i32 - 2),
        ) as usize;

        let u64_per_subinventory = 1usize << log2_u64_per_subinventory;
        let u64_per_inventory = u64_per_subinventory + 1;

        let log2_ones_per_sub64 = max(
            0,
            log2_ones_per_inventory as i32 - log2_u64_per_subinventory as i32,
        ) as usize;
        let log2_ones_per_sub16 = max(0, (log2_ones_per_sub64 as i32) - 2) as usize;
        let ones_per_sub64 = 1usize << log2_ones_per_sub64;
        let ones_per_sub16 = 1usize << log2_ones_per_sub16;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let mut inventory = Vec::with_capacity(inventory_size * u64_per_inventory + 1);

        let mut curr_num_ones: usize = 0;
        let mut next_quantum: usize = 0;
        let mut spilled = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        for (i, word) in bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as usize;

            while curr_num_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - curr_num_ones) as usize);
                let index = (i * u64::BITS as usize) + in_word_index;

                inventory.push(index as u64);
                inventory.resize(inventory.len() + u64_per_subinventory, 0);

                next_quantum += ones_per_inventory;
            }
            curr_num_ones += ones_in_word;
        }
        assert_eq!(num_ones, curr_num_ones);
        // in the last inventory write the number of bits
        inventory.push(num_bits as u64);
        assert_eq!(inventory.len(), inventory_size * u64_per_inventory + 1);

        // We estimate the subinventory and exact spill size
        for (i, inv) in inventory[..inventory_size * u64_per_inventory]
            .iter()
            .copied()
            .enumerate()
            .step_by(u64_per_inventory)
        {
            let start = inv as usize;
            let span = inventory[i + u64_per_inventory] as usize - start;
            curr_num_ones = (i / u64_per_inventory) * ones_per_inventory;
            let ones = min(num_ones - curr_num_ones, ones_per_inventory);

            assert!(start + span == num_bits || ones == ones_per_inventory);

            // We accumulate space for exact pointers only if necessary.
            if span >= (1 << 16) && ones_per_sub64 > 1 {
                spilled += ones;
            }
        }

        let exact_spill_size = spilled;
        let mut exact_spill = vec![0u64; exact_spill_size];

        spilled = 0;
        let iter = 0..inventory_size;

        // Second phase: we fill the subinventories and the exact spill
        iter.for_each(|inventory_idx| {
            // get the start and end u64 index of the current inventory
            let start_idx = inventory_idx * u64_per_inventory;
            let end_idx = start_idx + u64_per_inventory;
            // read the first level index to get the start and end bit index
            let start_bit_idx = inventory[start_idx];
            let end_bit_idx = inventory[end_idx];
            // compute the span of the inventory
            let span = end_bit_idx - start_bit_idx;
            let mut word_idx = start_bit_idx / u64::BITS as u64;

            // cleanup the lower bits
            let bit_idx = start_bit_idx % u64::BITS as u64;
            let mut word = (bits.as_ref()[word_idx as usize] >> bit_idx) << bit_idx;

            // compute the global number of ones
            let mut number_of_ones = inventory_idx * ones_per_inventory;
            let mut next_quantum = number_of_ones;
            let quantum;

            if span <= u16::MAX as u64 {
                quantum = ones_per_sub16;
            } else {
                quantum = 1;
                inventory[start_idx] |= 1u64 << 63;
                inventory[start_idx + 1] = spilled as u64;
            }

            let end_word_idx = end_bit_idx.div_ceil(u64::BITS as u64);

            // the first subinventory element is always 0
            let mut subinventory_idx = 1;

            // pre increment the next quantum only when using the subinventories
            if span <= u16::MAX as u64 {
                next_quantum += quantum;
            }

            'outer: loop {
                let ones_in_word = word.count_ones() as usize;

                // if the quantum is in this word, write it in the subinventory
                // this can happen multiple times if the quantum is small
                while number_of_ones + ones_in_word > next_quantum {
                    assert!(next_quantum <= end_bit_idx as _);
                    // find the quantum bit in the word
                    let in_word_index = word.select_in_word(next_quantum - number_of_ones);
                    // compute the global index of the quantum bit in the bitvec
                    let bit_index = (word_idx * u64::BITS as u64) + in_word_index as u64;
                    if bit_index >= end_bit_idx as u64 {
                        break 'outer;
                    }
                    // compute the offset of the quantum bit
                    // from the start of the subinventory
                    let sub_offset = bit_index - start_bit_idx;

                    if span <= u16::MAX as u64 {
                        let subinventory: &mut [u16] =
                            unsafe { inventory[start_idx + 1..end_idx].align_to_mut().1 };

                        subinventory[subinventory_idx] = sub_offset as u16;
                    } else if ones_per_sub64 != 1 {
                        assert!(spilled < exact_spill_size);
                        exact_spill[spilled] = bit_index;
                        spilled += 1;
                    }

                    // update the subinventory index if used
                    if span <= u16::MAX as u64 {
                        subinventory_idx += 1;
                        if subinventory_idx == (1 << log2_ones_per_inventory) / quantum {
                            break 'outer;
                        }
                    }

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
                word = bits.as_ref()[word_idx as usize];
            }
        });

        assert_eq!(spilled, exact_spill_size);

        Self {
            bits,
            inventory,
            exact_spill,
            num_ones,
            log2_ones_per_inventory,
            log2_ones_per_sub16,
            log2_u64_per_subinventory,
            ones_per_sub64,
            u64_per_inventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
            inventory_size,
            exact_spill_size,
        }
    }
}

impl<B: SelectHinted + Select + AsRef<[usize]>, I: AsRef<[u64]>> BitCount for SimpleSelect<B, I> {
    fn count(&self) -> usize {
        self.num_ones
    }
}

impl<B: SelectHinted + Select + BitLength + AsRef<[usize]>, I: AsRef<[u64]>> BitLength
    for SimpleSelect<B, I>
{
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<B: SelectHinted + Select + BitLength + AsRef<[usize]>, I: AsRef<[u64]>> Select
    for SimpleSelect<B, I>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let inventory_ref = self.inventory.as_ref();
        let inventory_index = rank >> self.log2_ones_per_inventory;
        let inventory_start_pos =
            (inventory_index << self.log2_u64_per_subinventory) + inventory_index;
        debug_assert!(inventory_index <= self.inventory_size);

        let inventory_rank = *inventory_ref.get_unchecked(inventory_start_pos) as usize;
        let subrank = rank & self.ones_per_inventory_mask;

        if (inventory_rank as isize) < 0 {
            if subrank == 0 {
                return inventory_rank & !(1usize << 63);
            }
            debug_assert!(
                *inventory_ref.get_unchecked(inventory_start_pos + 1) as usize + subrank
                    < self.exact_spill_size
            );
            return self.exact_spill.get_unchecked(
                *inventory_ref.get_unchecked(inventory_start_pos + 1) as usize + subrank,
            ) as usize;
        }

        let (_, u16s, _) = inventory_ref
            .get_unchecked(inventory_start_pos + 1..(self.inventory_size * self.u64_per_inventory))
            .align_to::<u16>();

        let hint_pos =
            inventory_rank + *u16s.get_unchecked(subrank >> self.log2_ones_per_sub16) as usize;
        let residual = subrank & self.ones_per_sub16_mask;

        self.bits
            .select_hinted_unchecked(rank, hint_pos, rank - residual)
    }

    fn select(&self, rank: usize) -> Option<usize> {
        if rank >= self.count() {
            None
        } else {
            Some(unsafe { self.select_unchecked(rank) })
        }
    }
}

#[cfg(test)]
mod test_simple_select {
    use super::*;
    use crate::prelude::BitVec;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_simple_select() {
        let lens = (100_000..1_000_000).step_by(100_000);
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in lens {
            let bits: BitVec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();

            let simple: SimpleSelect<BitVec, Vec<u64>> = SimpleSelect::new(bits.clone(), 3);

            let ones = simple.count();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(simple.select(i), Some(pos[i]));
            }
            assert_eq!(simple.select(ones + 1), None);
        }
    }

    #[test]
    fn test_simple_select_mult_usize() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in (1 << 10..1 << 15).step_by(usize::BITS as _) {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let simple: SimpleSelect = SimpleSelect::new(bits.clone(), 3);

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(simple.select(i), Some(pos[i]));
            }
            assert_eq!(simple.select(ones + 1), None);
        }
    }

    #[test]
    fn test_simple_select_empty() {
        let bits = BitVec::new(0);
        let simple: SimpleSelect = SimpleSelect::new(bits.clone(), 3);
        assert_eq!(simple.count(), 0);
        assert_eq!(simple.len(), 0);
        assert_eq!(simple.select(0), None);
    }

    #[test]
    fn test_simple_select_ones() {
        let len = 300_000;
        let bits = (0..len).map(|_| true).collect::<BitVec>();
        let simple: SimpleSelect = SimpleSelect::new(bits, 3);
        assert_eq!(simple.count(), len);
        assert_eq!(simple.len(), len);
        for i in 0..len {
            assert_eq!(simple.select(i), Some(i));
        }
    }

    #[test]
    fn test_simple_select_zeros() {
        let len = 300_000;
        let bits = (0..len).map(|_| false).collect::<BitVec>();
        let simple: SimpleSelect = SimpleSelect::new(bits, 3);
        assert_eq!(simple.count(), 0);
        assert_eq!(simple.len(), len);
        assert_eq!(simple.select(0), None);
    }

    #[test]
    fn test_simple_select_few_ones() {
        let lens = [1 << 18, 1 << 19, 1 << 20];
        for len in lens {
            for num_ones in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
                let bits = (0..len)
                    .map(|i| i % (len / num_ones) == 0)
                    .collect::<BitVec>();
                let simple: SimpleSelect = SimpleSelect::new(bits, 3);
                assert_eq!(simple.count(), num_ones);
                assert_eq!(simple.len(), len);
                for i in 0..num_ones {
                    assert_eq!(simple.select(i), Some(i * (len / num_ones)));
                }
            }
        }
    }

    #[test]
    fn test_simple_select_ones_per_sub64() {
        let len = 1 << 18;
        let bits = (0..len / 2)
            .map(|_| false)
            .chain([true])
            .chain((0..1 << 17).map(|_| false))
            .chain([true, true])
            .chain((0..1 << 18).map(|_| false))
            .chain([true])
            .chain((0..len / 2).map(|_| false))
            .collect::<BitVec>();
        let simple: SimpleSelect = SimpleSelect::new(bits, 3);

        assert_eq!(simple.ones_per_sub64, 1);
        assert_eq!(simple.count(), 4);
        assert_eq!(simple.select(0), Some(len / 2));
        assert_eq!(simple.select(1), Some(len / 2 + (1 << 17) + 1));
        assert_eq!(simple.select(2), Some(len / 2 + (1 << 17) + 2));
    }

    #[test]
    fn test_simple_non_uniform() {
        let lens = [1 << 18, 1 << 19, 1 << 20, 1 << 25];

        let mut rng = SmallRng::seed_from_u64(0);
        for len in lens {
            for density in [0.5] {
                let density0 = density * 0.01;
                let density1 = density * 0.99;

                let len1;
                let len2;
                if len % 2 != 0 {
                    len1 = len / 2 + 1;
                    len2 = len / 2;
                } else {
                    len1 = len / 2;
                    len2 = len / 2;
                }

                let first_half = loop {
                    let b = (0..len1)
                        .map(|_| rng.gen_bool(density0))
                        .collect::<BitVec>();
                    if b.count_ones() > 0 {
                        break b;
                    }
                };
                let num_ones_first_half = first_half.count_ones();
                let second_half = (0..len2)
                    .map(|_| rng.gen_bool(density1))
                    .collect::<BitVec>();
                let num_ones_second_half = second_half.count_ones();

                assert!(num_ones_first_half > 0);
                assert!(num_ones_second_half > 0);

                let bits = first_half
                    .into_iter()
                    .chain(second_half.into_iter())
                    .collect::<BitVec>();

                assert_eq!(
                    num_ones_first_half + num_ones_second_half,
                    bits.count_ones()
                );

                assert_eq!(bits.len(), len as usize);

                let ones = bits.count_ones();
                let mut pos = Vec::with_capacity(ones);
                for i in 0..(len as usize) {
                    if bits[i] {
                        pos.push(i);
                    }
                }

                let simple: SimpleSelect = SimpleSelect::new(bits, 3);
                for i in 0..(ones) {
                    assert_eq!(simple.select(i), Some(pos[i]));
                }
                assert_eq!(simple.select(ones + 1), None);
            }
        }
    }
}
