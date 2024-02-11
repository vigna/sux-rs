use common_traits::{SelectInWord, Sequence};

use crate::prelude::{BitCount, BitLength, BitVec, Rank, RankHinted, Select, SelectHinted};

use super::Rank9;

const ONES_STEP_9: u64 =
    1u64 << 0 | 1u64 << 9 | 1u64 << 18 | 1u64 << 27 | 1u64 << 36 | 1u64 << 45 | 1u64 << 54;
const MSBS_STEP_9: u64 = 0x100u64 * ONES_STEP_9;

const ONES_STEP_16: u64 = 1u64 << 0 | 1u64 << 16 | 1u64 << 32 | 1u64 << 48;
const MSBS_STEP_16: u64 = 0x8000u64 * ONES_STEP_16;

macro_rules! ULEQ_STEP_9 {
    ($x:ident, $y:ident) => {
        ((((((($y) | MSBS_STEP_9) - (($x) & !MSBS_STEP_9)) | ($x ^ $y)) ^ ($x & !$y))
            & MSBS_STEP_9)
            >> 8)
    };
}

macro_rules! ULEQ_STEP_16 {
    ($x:ident, $y:ident) => {
        ((((((($y) | MSBS_STEP_16) - (($x) & !MSBS_STEP_16)) | ($x ^ $y)) ^ ($x & !$y))
            & MSBS_STEP_16)
            >> 15)
    };
}

pub struct Rank9Sel<
    B: RankHinted<HINT_BIT_SIZE> + Rank + BitCount + AsRef<[usize]> = BitVec,
    I: AsRef<[u64]> = Vec<u64>,
    const HINT_BIT_SIZE: usize = 64,
> {
    rank9: Rank9<B, I, HINT_BIT_SIZE>,
    inventory: I,
    subinventory: I,
    inventory_size: usize,
    subinventory_size: usize,
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + Select + AsRef<[usize]>,
        I: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > Rank9Sel<B, I, HINT_BIT_SIZE>
{
    const LOG2_ONES_PER_INVENTORY: usize = 9;
    const ONES_PER_INVENTORY: usize = 1 << Self::LOG2_ONES_PER_INVENTORY;
    const INVENTORY_MASK: usize = Self::ONES_PER_INVENTORY - 1;
}

impl<const HINT_BIT_SIZE: usize> Rank9Sel<BitVec, Vec<u64>, HINT_BIT_SIZE> {
    pub fn new(bits: BitVec) -> Self {
        let rank9 = Rank9::new(bits);

        let num_bits = rank9.len();
        let num_words = (num_bits + 63) / 64;
        let inventory_size =
            (rank9.count() + Self::ONES_PER_INVENTORY - 1) / Self::ONES_PER_INVENTORY;
        let subinventory_size = (num_words + 3) / 4;

        let mut inventory: Vec<u64> = vec![0; inventory_size + 1];
        let mut subinventory: Vec<u64> = vec![0; subinventory_size];

        // construct the inventory
        let mut d = 0;
        for (i, bit) in rank9.bits.into_iter().enumerate() {
            if bit {
                if d & Self::INVENTORY_MASK == 0 {
                    inventory[d >> Self::LOG2_ONES_PER_INVENTORY] = i as u64;
                }
                d += 1;
            }
        }
        assert!(rank9.count() == d);
        inventory[inventory_size] = ((num_words as u64 + 3) & !3u64) * 64;

        // construct the subinventory
        d = 0;
        let mut state = -1i32;
        let mut index: usize;
        let mut span: u64;
        let mut block_span: u64;
        let mut block_left: u64;
        let mut counts_at_start: u64;
        let mut subinv_pos = 0usize;
        let mut first_bit = 0usize;

        for (i, bit) in rank9.bits.into_iter().enumerate() {
            if !bit {
                continue;
            }
            if d & Self::INVENTORY_MASK == 0 {
                first_bit = i;
                index = d >> Self::LOG2_ONES_PER_INVENTORY;
                assert!(inventory[index] == first_bit as u64);

                subinv_pos = (inventory[index] as usize / 64) / 4;

                span = (inventory[index + 1] / 64) / 4 - (inventory[index] / 64) / 4;
                state = -1;
                counts_at_start = rank9.counts[((inventory[index] as usize / 64) / 8) * 2];
                block_span = (inventory[index + 1] / 64) / 8 - (inventory[index] / 64) / 8;
                block_left = (inventory[index] / 64) / 8;

                if span >= 512 {
                    state = 0;
                } else if span >= 256 {
                    state = 1;
                } else if span >= 128 {
                    state = 2;
                } else if span >= 16 {
                    assert!(((block_span + 8) & !7u64) + 8 <= span * 4);
                    let (_, s, _) = unsafe { subinventory[subinv_pos..].align_to_mut::<u16>() };

                    for k in 0..(block_span as usize) {
                        assert!(s[k + 8] == 0);
                        s[k + 8] = (rank9.counts[(block_left as usize + k + 1) * 2]
                            - counts_at_start) as u16;
                    }

                    for k in (block_span as usize)..(((block_span + 8) & !7u64) as usize) {
                        assert!(s[k + 8] == 0);
                        s[k + 8] = 0xFFFFu16;
                    }

                    assert!(block_span / 8 <= 8);

                    for k in 0..(block_span as usize / 8) {
                        assert!(s[k] == 0);
                        s[k] = (rank9.counts[(block_left as usize + (k + 1) * 8) * 2]
                            - counts_at_start) as u16;
                    }

                    for k in (block_span as usize / 8)..8 {
                        assert!(s[k] == 0);
                        s[k] = 0xFFFFu16;
                    }
                } else if span >= 2 {
                    assert!(((block_span + 8) & !7u64) <= span * 4);

                    let (_, s, _) = unsafe { subinventory[subinv_pos..].align_to_mut::<u16>() };

                    for k in 0..(block_span as usize) {
                        assert!(s[k] == 0);
                        s[k] = (rank9.counts[(block_left as usize + k + 1) * 2] - counts_at_start)
                            as u16;
                    }

                    for k in (block_span as usize)..(((block_span + 8) & !7u64) as usize) {
                        assert!(s[k] == 0);
                        s[k] = 0xFFFFu16;
                    }
                }
            }

            match state {
                0 => {
                    assert!(subinventory[subinv_pos + (d & Self::INVENTORY_MASK)] == 0);
                    subinventory[subinv_pos + (d & Self::INVENTORY_MASK)] = i as u64;
                }
                1 => {
                    let (_, s, _) = unsafe { subinventory[subinv_pos..].align_to_mut::<u32>() };
                    assert!(s[d & Self::INVENTORY_MASK] == 0);
                    assert!((i as u64 - first_bit as u64) < (1u64 << 32));
                    s[d & Self::INVENTORY_MASK] = (i - first_bit) as u32;
                }
                2 => {
                    let (_, s, _) = unsafe { subinventory[subinv_pos..].align_to_mut::<u16>() };
                    assert!(s[d & Self::INVENTORY_MASK] == 0);
                    assert!(i - first_bit < (1 << 16));
                    s[d & Self::INVENTORY_MASK] = (i - first_bit) as u16;
                }
                _ => {}
            }

            d += 1;
        }

        Self {
            rank9,
            inventory,
            subinventory,
            inventory_size,
            subinventory_size,
        }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > Select for Rank9Sel<B, C, HINT_BIT_SIZE>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let inventory_index_left = rank >> Self::LOG2_ONES_PER_INVENTORY;

        assert!(inventory_index_left <= self.inventory_size);
        let inventory_left = *self.inventory.as_ref().get_unchecked(inventory_index_left);

        let block_right = (*self
            .inventory
            .as_ref()
            .get_unchecked(inventory_index_left + 1))
            / 64;
        let mut block_left = inventory_left / 64;
        let span = block_right / 4 - block_left / 4;

        let subinv_pos = block_left as usize / 4;
        let subinv_ref = self.subinventory.as_ref();

        let mut count_left: u64;
        let rank_in_block: u64;

        if span < 2 {
            block_left &= !7;
            count_left = (block_left / 4) & !1;

            assert!(
                rank < *self
                    .rank9
                    .counts
                    .as_ref()
                    .get_unchecked(count_left as usize + 2) as usize
            );

            rank_in_block = rank as u64
                - *self
                    .rank9
                    .counts
                    .as_ref()
                    .get_unchecked(count_left as usize);
        } else if span < 16 {
            block_left &= !7;
            count_left = (block_left / 4) & !1;
            let rank_in_superblock = rank as u64
                - *self
                    .rank9
                    .counts
                    .as_ref()
                    .get_unchecked(count_left as usize);

            let rank_in_superblock_step_16 = rank_in_superblock * ONES_STEP_16;

            let first = *subinv_ref.get_unchecked(subinv_pos);
            let second = *subinv_ref.get_unchecked(subinv_pos + 1);

            let where_: u64 = (ULEQ_STEP_16!(first, rank_in_superblock_step_16)
                + ULEQ_STEP_16!(second, rank_in_superblock_step_16))
            .wrapping_mul(ONES_STEP_16)
                >> 47;

            assert!(where_ <= 16);

            block_left += where_ * 4;
            count_left += where_;

            rank_in_block = rank as u64
                - *self
                    .rank9
                    .counts
                    .as_ref()
                    .get_unchecked(count_left as usize);

            assert!(rank_in_block < 512);
        } else if span < 128 {
            block_left &= !7;
            count_left = (block_left / 4) & !1;
            let rank_in_superblock = rank as u64
                - *self
                    .rank9
                    .counts
                    .as_ref()
                    .get_unchecked(count_left as usize);
            let rank_in_superblock_step_16 = rank_in_superblock * ONES_STEP_16;

            let first = *subinv_ref.get_unchecked(subinv_pos);
            let second = *subinv_ref.get_unchecked(subinv_pos + 1);

            let where0 = ((ULEQ_STEP_16!(first, rank_in_superblock_step_16)
                + ULEQ_STEP_16!(second, rank_in_superblock_step_16))
                * ONES_STEP_16)
                >> 47;

            assert!(where0 <= 16);

            let first_bis = *self
                .subinventory
                .as_ref()
                .get_unchecked(subinv_pos + where0 as usize + 2);
            let second_bis = *self
                .subinventory
                .as_ref()
                .get_unchecked(subinv_pos + where0 as usize + 2 + 1);

            let where1 = where0 * 8
                + ((ULEQ_STEP_16!(first_bis, rank_in_superblock_step_16)
                    + ULEQ_STEP_16!(second_bis, rank_in_superblock_step_16))
                .wrapping_mul(ONES_STEP_16)
                    >> 47);

            block_left += where1 * 4;
            count_left += where1;
            rank_in_block = rank as u64
                - *self
                    .rank9
                    .counts
                    .as_ref()
                    .get_unchecked(count_left as usize);

            assert!(rank_in_block < 512);
        } else if span < 256 {
            let (_, s, _) = subinv_ref
                .get_range_unchecked(subinv_pos..self.subinventory_size)
                .align_to::<u16>();
            return *s.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize
                + inventory_left as usize;
        } else if span < 512 {
            let (_, s, _) = subinv_ref
                .get_range_unchecked(subinv_pos..self.subinventory_size)
                .align_to::<u32>();
            return *s.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize
                + inventory_left as usize;
        } else {
            return *subinv_ref.get_unchecked(rank % Self::ONES_PER_INVENTORY) as usize;
        }

        let rank_in_block_step_9 = rank_in_block * ONES_STEP_9;

        let subcounts = *self
            .rank9
            .counts
            .as_ref()
            .get_unchecked(count_left as usize + 1);

        let offset_in_block =
            (ULEQ_STEP_9!(subcounts, rank_in_block_step_9).wrapping_mul(ONES_STEP_9) >> 54u64)
                & 0x7u64;
        assert!(offset_in_block <= 7);

        let word = block_left + offset_in_block;
        //assert!(word <= (self.rank9.bits.len() as u64 + 63) / 64);

        let rank_in_word = rank_in_block
            - ((subcounts >> (((offset_in_block.wrapping_sub(1)) & 7) * 9u64)) & 0x1FF);
        assert!(rank_in_word < 64);

        word as usize * 64
            + self
                .rank9
                .bits
                .as_ref()
                .get_unchecked(word as usize)
                .select_in_word(rank_in_word as usize)
    }

    fn select(&self, rank: usize) -> Option<usize> {
        if rank >= self.count() {
            None
        } else {
            Some(unsafe { self.select_unchecked(rank) })
        }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + Select + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > Rank for Rank9Sel<B, C, HINT_BIT_SIZE>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.rank9.rank_unchecked(pos)
    }

    fn rank(&self, pos: usize) -> usize {
        unsafe { self.rank_unchecked(pos.min(self.len())) }
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + Select + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > BitCount for Rank9Sel<B, C, HINT_BIT_SIZE>
{
    fn count(&self) -> usize {
        self.rank9.count()
    }
}

impl<
        B: RankHinted<HINT_BIT_SIZE> + Rank + Select + AsRef<[usize]>,
        C: AsRef<[u64]>,
        const HINT_BIT_SIZE: usize,
    > BitLength for Rank9Sel<B, C, HINT_BIT_SIZE>
{
    fn len(&self) -> usize {
        self.rank9.len()
    }
}

#[cfg(test)]
mod test_rank9sel {
    use super::*;
    use crate::prelude::BitVec;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_rank9sel() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in 1..10000 {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank9sel: Rank9Sel<BitVec, Vec<u64>, 64> = Rank9Sel::new(bits.clone());

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(rank9sel.select(i), Some(pos[i]));
            }
            assert_eq!(rank9sel.select(ones + 1), None);
        }
    }
}
