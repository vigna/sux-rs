use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct Rank10Sel<
    const UPPER_BLOCK_SIZE: usize,
    const HINT_BIT_SIZE: usize = 64,
    B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]> = BitVec,
> {
    rank10: Rank10<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>,
    inventory: Vec<u64>,
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]>,
    > Rank10Sel<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    const MAJOR_BLOCK_SIZE: usize = Rank10::<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>::MAJOR_BLOCK_SIZE;
    const LOWER_BLOCK_SIZE: usize = Rank10::<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>::LOWER_BLOCK_SIZE;
    const ONES_PER_INVENTORY: usize = 8192;
}

impl<const UPPER_BLOCK_SIZE: usize, const HINT_BIT_SIZE: usize>
    Rank10Sel<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, BitVec>
{
    pub fn new(bits: BitVec) -> Self {
        let rank10 = Rank10::<UPPER_BLOCK_SIZE, HINT_BIT_SIZE>::new(bits);

        let num_bits = rank10.bits.len();
        let num_ones = rank10.bits.count_ones() as usize;

        let inventory_size = (num_ones + Self::ONES_PER_INVENTORY - 1) / Self::ONES_PER_INVENTORY;
        let mut inventory = Vec::<u64>::with_capacity(inventory_size + 1);

        // let ones_per_major_inventory = 1 << 32;
        // let major_inventory_size =
        //     (num_ones + ones_per_major_inventory - 1) / ones_per_major_inventory;
        // let mut major_inventory = Vec::<u64>::with_capacity(major_inventory_size);

        let mut curr_num_ones: usize = 0;
        let mut next_quantum: usize = 0;

        for (i, word) in rank10.bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as usize;

            while curr_num_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - curr_num_ones) as usize);
                let index = (i * u64::BITS as usize) + in_word_index;

                inventory.push(index as u64);

                next_quantum += Self::ONES_PER_INVENTORY;
            }
            curr_num_ones += ones_in_word;
        }
        assert_eq!(num_ones, curr_num_ones);
        inventory.push(num_bits as u64);
        assert_eq!(inventory.len(), inventory_size + 1);

        Self { rank10, inventory }
    }
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]>,
    > Select for Rank10Sel<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let inventory_index = rank / Self::ONES_PER_INVENTORY;
        let jump = (rank % Self::ONES_PER_INVENTORY) - (rank % UPPER_BLOCK_SIZE);

        let inv_pos = self.inventory[inventory_index] as usize;
        let next_inv_pos = self.inventory[inventory_index + 1] as usize;
        let mut hint_pos = inv_pos - (inv_pos % UPPER_BLOCK_SIZE) + jump;
        let block = hint_pos / UPPER_BLOCK_SIZE;
        let major_block = hint_pos / Self::MAJOR_BLOCK_SIZE;

        let mut hint_rank = self.rank10.counts.major(major_block) + self.rank10.counts.upper(block);

        let mut next_rank;
        let mut next_pos;
        loop {
            if hint_pos + UPPER_BLOCK_SIZE >= next_inv_pos {
                break;
            }
            next_pos = hint_pos + UPPER_BLOCK_SIZE;
            let next_major_block = next_pos / Self::MAJOR_BLOCK_SIZE;
            let next_block = next_pos / UPPER_BLOCK_SIZE;
            next_rank =
                self.rank10.counts.major(next_major_block) + self.rank10.counts.upper(next_block);
            if next_rank >= rank as u64 {
                break;
            }
            hint_rank = next_rank;
            hint_pos = next_pos;
        }

        let idx = hint_pos % Self::LOWER_BLOCK_SIZE;
        hint_rank += self.rank10.counts.lower(block, idx);
        hint_pos += idx * Self::LOWER_BLOCK_SIZE;

        self.rank10
            .bits
            .select_hinted_unchecked(rank, hint_pos, hint_rank as usize)
    }
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]>,
    > Rank for Rank10Sel<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    unsafe fn rank_unchecked(&self, pos: usize) -> usize {
        self.rank10.rank_unchecked(pos)
    }
    fn rank(&self, pos: usize) -> usize {
        self.rank10.rank(pos)
    }
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]>,
    > BitCount for Rank10Sel<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    fn count(&self) -> usize {
        self.rank10.count()
    }
}

impl<
        const UPPER_BLOCK_SIZE: usize,
        const HINT_BIT_SIZE: usize,
        B: RankHinted<HINT_BIT_SIZE> + SelectHinted + Rank + Select + BitCount + AsRef<[usize]>,
    > BitLength for Rank10Sel<UPPER_BLOCK_SIZE, HINT_BIT_SIZE, B>
{
    fn len(&self) -> usize {
        self.rank10.len()
    }
}

#[cfg(test)]
mod test_rank10sel {
    use super::*;
    use crate::prelude::BitVec;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    const TEST_UPPER_BLOCK_SIZE: usize = 512;

    #[test]
    fn test_rank10sel() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        let lens = (1..1000).chain((1000..10000).step_by(100));
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits.clone());

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(rank10sel.select(i), Some(pos[i]));
            }
            assert_eq!(rank10sel.select(ones + 1), None);
        }
    }

    #[test]
    fn test_rank10sel_mult_usize() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let density = 0.5;
        for len in (1 << 10..1 << 15).step_by(usize::BITS as _) {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits.clone());

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(rank10sel.select(i), Some(pos[i]));
            }
            assert_eq!(rank10sel.select(ones + 1), None);
        }
    }

    #[test]
    fn test_rank10sel_empty() {
        let bits = BitVec::new(0);
        let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits.clone());
        assert_eq!(rank10sel.count(), 0);
        assert_eq!(rank10sel.len(), 0);
        assert_eq!(rank10sel.select(0), None);
    }

    #[test]
    fn test_rank10sel_ones() {
        let len = 300_000;
        let bits = (0..len).map(|_| true).collect::<BitVec>();
        let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits);
        assert_eq!(rank10sel.count(), len);
        assert_eq!(rank10sel.len(), len);
        for i in 0..len {
            assert_eq!(rank10sel.select(i), Some(i));
        }
    }

    #[test]
    fn test_rank10sel_zeros() {
        let len = 300_000;
        let bits = (0..len).map(|_| false).collect::<BitVec>();
        let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits);
        assert_eq!(rank10sel.count(), 0);
        assert_eq!(rank10sel.len(), len);
        assert_eq!(rank10sel.select(0), None);
    }

    #[test]
    fn test_rank10sel_few_ones() {
        let lens = [1 << 18, 1 << 19, 1 << 20];
        for len in lens {
            for num_ones in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
                let bits = (0..len)
                    .map(|i| i % (len / num_ones) == 0)
                    .collect::<BitVec>();
                let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits);
                assert_eq!(rank10sel.count(), num_ones);
                assert_eq!(rank10sel.len(), len);
                for i in 0..num_ones {
                    assert_eq!(rank10sel.select(i), Some(i * (len / num_ones)));
                }
            }
        }
    }

    #[test]
    fn test_rank10sel_non_uniform() {
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

                let rank10sel: Rank10Sel<TEST_UPPER_BLOCK_SIZE> = Rank10Sel::new(bits);

                for i in 0..ones {
                    assert!(rank10sel.select(i) == Some(pos[i]));
                }
                assert_eq!(rank10sel.select(ones + 1), None);
            }
        }
    }
}
