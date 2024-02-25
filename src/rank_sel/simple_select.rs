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
    log2_ones_per_inventory: usize,
    log2_ones_per_sub16: usize,
    log2_ones_per_sub64: usize,
    log2_u64_per_subinventory: usize,
    ones_per_inventory: usize,
    ones_per_sub16: usize,
    ones_per_sub64: usize,
    u64_per_subinventory: usize,
    u64_per_inventory: usize,
    ones_per_inventory_mask: usize,
    ones_per_sub16_mask: usize,
    ones_per_sub64_mask: usize,
    num_words: usize,
    inventory_size: usize,
    exact_spill_size: usize,
    num_ones: usize,
}

impl<B: SelectHinted, I: AsRef<[u64]>> SimpleSelect<B, I> {
    const MAX_ONES_PER_INVENTORY: usize = 8192;
}

impl SimpleSelect<BitVec, Vec<u64>> {
    pub fn new(bits: BitVec, max_log2_u64_per_subinventory: u64) -> Self {
        let num_bits = bits.len();
        let num_words = (num_bits + 63) / 64;
        let num_ones = bits.count();

        let ones_per_inventory = if num_bits == 0 {
            0
        } else {
            (num_ones * Self::MAX_ONES_PER_INVENTORY + num_bits - 1) / num_bits
        };

        // Make ones_per_inventory into a power of 2
        let log2_ones_per_inventory = max(0, most_significant_one(ones_per_inventory)) as usize;

        let ones_per_inventory = 1usize << log2_ones_per_inventory;
        let ones_per_inventory_mask = ones_per_inventory - 1;
        let inventory_size = (num_ones + ones_per_inventory - 1) / ones_per_inventory;

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
        let ones_per_sub64_mask = ones_per_sub64 - 1;
        let ones_per_sub16_mask = ones_per_sub16 - 1;

        let mut inventory = vec![0u64; inventory_size * u64_per_inventory + 1];
        let end_of_inventory_pos = inventory_size * u64_per_inventory + 1;

        let mut d = 0;

        // First phase: we build an inventory for each one out of ones_per_inventory.
        for (i, bit) in bits.into_iter().enumerate() {
            if !bit {
                continue;
            }
            if (d & ones_per_inventory_mask) == 0 {
                inventory[(d >> log2_ones_per_inventory) * u64_per_inventory] = i as u64;
            }
            d += 1;
        }
        assert_eq!(num_ones, d);

        inventory[(inventory_size * u64_per_inventory) as usize] = num_bits as u64;

        if ones_per_inventory <= 1 {
            todo!("early return");
        }

        d = 0;
        let mut ones;
        let mut spilled = 0;
        let mut start = 0usize;
        let mut span = 0usize;
        let mut inventory_index = 0usize;

        // We estimate the subinventory and exact spill size
        for bit in bits.into_iter() {
            if !bit {
                continue;
            }
            if (d & ones_per_inventory_mask) == 0 {
                inventory_index = d >> log2_ones_per_inventory;
                start = inventory[inventory_index * u64_per_inventory] as usize;
                span = inventory[(inventory_index + 1) * u64_per_inventory] as usize - start;
                ones = min(num_ones - d, ones_per_inventory);

                assert!(start + span == num_bits || ones == ones_per_inventory);

                // We accumulate space for exact pointers ONLY if necessary.
                if span >= (1 << 16) && ones_per_sub64 > 1 {
                    spilled += ones;
                }
            }
            d += 1;
        }

        let exact_spill_size = spilled;
        let mut exact_spill = vec![0u64; exact_spill_size];

        let mut offset = 0usize;
        let mut inv_pos = 0usize;
        spilled = 0;
        d = 0;

        for (i, bit) in bits.into_iter().enumerate() {
            if !bit {
                continue;
            }
            if (d & ones_per_inventory_mask) == 0 {
                inventory_index = d >> log2_ones_per_inventory;
                start = inventory[inventory_index * u64_per_inventory] as usize;
                span = inventory[(inventory_index + 1) * u64_per_inventory] as usize - start;
                inv_pos = inventory_index * u64_per_inventory + 1;
                offset = 0;
            }
            if span < (1 << 16) {
                assert!(i - start <= (1 << 16));
                if (d & ones_per_sub16_mask) == 0 {
                    assert!(offset < u64_per_subinventory * 4);
                    //assert!(p16 + offset < (uint16_t *)end_of_inventory);
                    let (_, p16, _) = unsafe { inventory[inv_pos..].align_to_mut::<u16>() };
                    p16[offset] = (i - start) as u16;
                    offset += 1;
                }
            } else if ones_per_sub64 == 1 {
                assert!(inv_pos + offset < end_of_inventory_pos);
                inventory[inv_pos + offset] = i as u64;
                offset += 1;
            } else {
                assert!(inv_pos < end_of_inventory_pos);
                if (d & ones_per_inventory_mask) == 0 {
                    inventory[inventory_index * u64_per_inventory] |= 1u64 << 63;
                    inventory[inv_pos] = spilled as u64;
                }
                assert!(spilled < exact_spill_size);
                exact_spill[spilled] = i as u64;
                spilled += 1;
            }

            d += 1;
        }

        Self {
            bits,
            inventory,
            exact_spill,
            log2_ones_per_inventory,
            log2_ones_per_sub16,
            log2_ones_per_sub64,
            log2_u64_per_subinventory,
            ones_per_inventory,
            ones_per_sub16,
            ones_per_sub64,
            u64_per_subinventory,
            u64_per_inventory,
            ones_per_inventory_mask,
            ones_per_sub16_mask,
            ones_per_sub64_mask,
            num_words,
            inventory_size,
            exact_spill_size,
            num_ones,
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
        assert!(inventory_index <= self.inventory_size);

        let inventory_rank = *inventory_ref.get_unchecked(inventory_start_pos) as usize;
        let subrank = rank & self.ones_per_inventory_mask;

        if subrank == 0 {
            return inventory_rank & !(1usize << 63);
        }

        if (inventory_rank as isize) < 0 {
            if self.ones_per_sub64 == 1 {
                return *inventory_ref.get_unchecked(inventory_start_pos + 1 + subrank) as usize;
            }
            assert!(
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
}

#[cfg(test)]
mod test_simple_select {
    use super::*;
    use crate::prelude::BitVec;
    use mem_dbg::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_simple_select() {
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in (1..10000).chain((10000..100000).step_by(100)) {
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
    fn show_mem() {
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        let len = 1_000_000_000;
        let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();

        let simple: SimpleSelect = SimpleSelect::new(bits, 3);

        println!("size:     {}", simple.mem_size(SizeFlags::default()));
        println!("capacity: {}", simple.mem_size(SizeFlags::CAPACITY));

        simple.mem_dbg(DbgFlags::default()).unwrap();
    }

    #[test]
    fn show_mem_non_uniform() {
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        let len = 100_000_000;

        let density0 = density * 0.01;
        let density1 = density * 0.99;

        let first_half = loop {
            let b = (0..len / 2)
                .map(|_| rng.gen_bool(density0))
                .collect::<BitVec>();
            if b.count_ones() > 0 {
                break b;
            }
        };
        let second_half = (0..len / 2)
            .map(|_| rng.gen_bool(density1))
            .collect::<BitVec>();

        let bits = first_half
            .into_iter()
            .chain(second_half.into_iter())
            .collect::<BitVec>();

        let simple: SimpleSelect = SimpleSelect::new(bits, 3);

        println!("size:     {}", simple.mem_size(SizeFlags::default()));
        println!("capacity: {}", simple.mem_size(SizeFlags::CAPACITY));

        simple.mem_dbg(DbgFlags::default()).unwrap();
    }
}
