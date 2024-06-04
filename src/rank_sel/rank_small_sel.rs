use std::ops::Index;

use common_traits::SelectInWord;
use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};

use crate::prelude::*;

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct RankSmallSel<
    const NUM_U32S: usize,
    const COUNTER_WIDTH: usize,
    const LOG2_ONES_PER_INVENTORY: usize = 12,
    R = RankSmall<NUM_U32S, COUNTER_WIDTH>,
    I = Vec<u32>,
> {
    rank_small: R,
    inventory: I,
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        R,
        I,
    > RankSmallSel<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, R, I>
{
    const WORDS_PER_BLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_BLOCK;
    const WORDS_PER_SUBBLOCK: usize = RankSmall::<NUM_U32S, COUNTER_WIDTH>::WORDS_PER_SUBBLOCK;
    const BLOCK_SIZE: usize = (Self::WORDS_PER_BLOCK * usize::BITS as usize);
    const SUBBLOCK_SIZE: usize = (Self::WORDS_PER_SUBBLOCK * usize::BITS as usize);
    const ONES_PER_INVENTORY: usize = 1 << LOG2_ONES_PER_INVENTORY;
}

impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B,
        I,
    > RankSmallSel<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    pub fn into_inner(self) -> B {
        self.rank_small
    }
}

macro_rules! impl_rank_small_sel {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        impl<
                const LOG2_ONES_PER_INVENTORY: usize,
                B: RankHinted<64> + BitCount + BitLength + AsRef<[usize]>,
                C1: AsRef<[usize]>,
                C2: AsRef<[Block32Counters<$NUM_U32S, $COUNTER_WIDTH>]>,
            >
            RankSmallSel<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>,
            >
        {
            pub fn new(rank_small: RankSmall<$NUM_U32S, $COUNTER_WIDTH, B, C1, C2>) -> Self {
                let num_ones = rank_small.bits.count_ones();

                let inventory_size = num_ones.div_ceil(Self::ONES_PER_INVENTORY);
                let mut inventory = Vec::<u32>::with_capacity(inventory_size + 1);

                let mut curr_num_ones: usize = 0;
                let mut next_quantum: usize = 0;
                let mut upper_counts_idx;

                for (i, word) in rank_small.bits.as_ref().iter().copied().enumerate() {
                    let ones_in_word = word.count_ones() as usize;

                    upper_counts_idx = i / (1 << 26);

                    while curr_num_ones + ones_in_word > next_quantum {
                        let in_word_index = word.select_in_word(next_quantum - curr_num_ones);
                        let index = ((i * usize::BITS as usize) + in_word_index);

                        inventory.push(
                            (index - rank_small.upper_counts.as_ref()[upper_counts_idx]) as u32,
                        );

                        next_quantum += Self::ONES_PER_INVENTORY;
                    }
                    curr_num_ones += ones_in_word;
                }
                assert_eq!(num_ones, curr_num_ones);

                if !inventory.is_empty() {
                    inventory.push(inventory[inventory.len() - 1]);
                } else {
                    inventory.push(0);
                }

                Self {
                    rank_small,
                    inventory,
                }
            }
        }

        impl<const LOG2_ONES_PER_INVENTORY: usize> Select
            for RankSmallSel<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH>,
            >
        {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                let upper_counts_ref =
                    <Vec<_> as AsRef<[_]>>::as_ref(&self.rank_small.upper_counts);
                let counts_ref = <Vec<_> as AsRef<[_]>>::as_ref(&self.rank_small.counts);
                let mut upper_block_idx = 0;
                let mut next_upper_block_idx;
                let mut last_upper_block_idx = self.rank_small.upper_counts.len() - 1;
                let mut upper_rank = *upper_counts_ref.get_unchecked(upper_block_idx) as usize;
                loop {
                    if last_upper_block_idx - upper_block_idx <= 1 {
                        break;
                    }
                    next_upper_block_idx = (upper_block_idx + last_upper_block_idx) / 2;
                    upper_rank = *upper_counts_ref.get_unchecked(next_upper_block_idx) as usize;
                    if rank >= upper_rank {
                        upper_block_idx = next_upper_block_idx;
                    } else {
                        last_upper_block_idx = next_upper_block_idx;
                    }
                }

                let inv_ref = <Vec<u32> as AsRef<[u32]>>::as_ref(&self.inventory);
                let rel_inv_pos = *inv_ref.get_unchecked(rank / Self::ONES_PER_INVENTORY) as usize;
                let inv_pos = rel_inv_pos + upper_block_idx * (1 << 32);

                let next_rel_inv_pos =
                    *inv_ref.get_unchecked(rank / Self::ONES_PER_INVENTORY + 1) as usize;
                let next_inv_pos = match next_rel_inv_pos.cmp(&rel_inv_pos) {
                    std::cmp::Ordering::Greater => next_rel_inv_pos + upper_block_idx * (1 << 32),
                    std::cmp::Ordering::Less => (upper_block_idx + 1) * (1 << 32),
                    // the two last elements of the inventory are the same
                    // because at construction time we add the last element twice
                    std::cmp::Ordering::Equal => self.rank_small.bits.len(),
                };
                let mut last_block_idx = next_inv_pos / Self::BLOCK_SIZE;

                let jump = (rank % Self::ONES_PER_INVENTORY) / Self::BLOCK_SIZE;
                let mut block_idx = inv_pos / Self::BLOCK_SIZE + jump;

                let mut hint_rank =
                    upper_rank + counts_ref.get_unchecked(block_idx).absolute as usize;
                let mut next_rank;
                let mut next_block_idx;

                debug_assert!(block_idx <= last_block_idx);

                loop {
                    if last_block_idx - block_idx <= 1 {
                        break;
                    }
                    next_block_idx = (block_idx + last_block_idx) / 2;
                    next_rank =
                        upper_rank + counts_ref.get_unchecked(next_block_idx).absolute as usize;
                    if rank >= next_rank {
                        block_idx = next_block_idx;
                        hint_rank = next_rank;
                    } else {
                        last_block_idx = next_block_idx;
                    }
                }

                let hint_pos;
                // second sub block
                let b1 = counts_ref.get_unchecked(block_idx).rel(1);
                if hint_rank + b1 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE;
                    return self
                        .rank_small
                        .bits
                        .select_hinted_unchecked(rank, hint_pos, hint_rank);
                }
                // third sub block
                let b2 = counts_ref.get_unchecked(block_idx).rel(2);
                if hint_rank + b2 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE + Self::SUBBLOCK_SIZE;
                    return self.rank_small.bits.select_hinted_unchecked(
                        rank,
                        hint_pos,
                        hint_rank + b1,
                    );
                }
                // fourth sub block
                let b3 = counts_ref.get_unchecked(block_idx).rel(3);
                if hint_rank + b3 > rank {
                    hint_pos = block_idx * Self::BLOCK_SIZE + 2 * Self::SUBBLOCK_SIZE;
                    return self.rank_small.bits.select_hinted_unchecked(
                        rank,
                        hint_pos,
                        hint_rank + b2,
                    );
                }

                hint_pos = block_idx * Self::BLOCK_SIZE + 3 * Self::SUBBLOCK_SIZE;
                self.rank_small
                    .bits
                    .select_hinted_unchecked(rank, hint_pos, hint_rank + b3)
            }
        }

        /// Forward [`Rank`] to the underlying implementation.
        impl<const LOG2_ONES_PER_INVENTORY: usize> Rank
            for RankSmallSel<
                $NUM_U32S,
                $COUNTER_WIDTH,
                LOG2_ONES_PER_INVENTORY,
                RankSmall<$NUM_U32S, $COUNTER_WIDTH>,
            >
        {
            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> usize {
                self.rank_small.rank_unchecked(pos)
            }

            #[inline(always)]
            fn rank(&self, pos: usize) -> usize {
                self.rank_small.rank(pos)
            }
        }
    };
}

impl_rank_small_sel!(2; 9);
impl_rank_small_sel!(1; 9);
impl_rank_small_sel!(1; 10);
impl_rank_small_sel!(1; 11);
impl_rank_small_sel!(3; 13);

/// Forward [`BitCount`] to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: BitCount,
        I,
    > BitCount for RankSmallSel<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    #[inline(always)]
    fn count_ones(&self) -> usize {
        self.rank_small.count_ones()
    }
    #[inline(always)]
    fn count_zeros(&self) -> usize {
        self.rank_small.count_zeros()
    }
}

/// Forward [`BitLength`] to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: BitLength,
        I,
    > BitLength for RankSmallSel<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.rank_small.len()
    }
}

/// Forward `AsRef<[usize]>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: AsRef<[usize]>,
        I,
    > AsRef<[usize]> for RankSmallSel<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    #[inline(always)]
    fn as_ref(&self) -> &[usize] {
        self.rank_small.as_ref()
    }
}

/// Forward `Index<usize, Output = bool>` to the underlying implementation.
impl<
        const NUM_U32S: usize,
        const COUNTER_WIDTH: usize,
        const LOG2_ONES_PER_INVENTORY: usize,
        B: Index<usize, Output = bool>,
        I,
    > Index<usize> for RankSmallSel<NUM_U32S, COUNTER_WIDTH, LOG2_ONES_PER_INVENTORY, B, I>
{
    type Output = bool;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        // TODO: why is & necessary?
        &self.rank_small[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::{bit_vec, prelude::BitVec};
    use epserde::deser::DeserializeInner;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    use super::RankSmallSel;

    macro_rules! test_rank_small_sel {
        ($NUM_U32S: literal; $COUNTER_WIDTH: literal; $LOG2_ONES_PER_INVENTORY: literal) => {
            use $crate::traits::Select;
            let mut rng = SmallRng::seed_from_u64(0);
            let density = 0.5;
            let lens = (1..1000)
                .chain((1000..10000).step_by(100))
                .chain([1 << 20, 1 << 24]);
            for len in lens {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let rank_small_sel =
                    RankSmallSel::<$NUM_U32S, $COUNTER_WIDTH, $LOG2_ONES_PER_INVENTORY>::new(
                        $crate::prelude::RankSmall::<$NUM_U32S, $COUNTER_WIDTH>::new(bits.clone()),
                    );

                let ones = bits.count_ones();
                let mut pos = Vec::with_capacity(ones);
                for i in 0..len {
                    if bits[i] {
                        pos.push(i);
                    }
                }

                for i in 0..ones {
                    assert_eq!(rank_small_sel.select(i), Some(pos[i]));
                }
                assert_eq!(rank_small_sel.select(ones + 1), None);
            }
        };
    }

    #[test]
    fn rank_small_sel0() {
        test_rank_small_sel!(2; 9; 13);
    }

    #[test]
    fn rank_small_sel1() {
        test_rank_small_sel!(1; 9; 13);
    }

    #[test]
    fn rank_small_sel2() {
        test_rank_small_sel!(1; 10; 13);
    }

    #[test]
    fn rank_small_sel3() {
        test_rank_small_sel!(1; 11; 13);
    }

    #[test]
    fn rank_small_sel4() {
        test_rank_small_sel!(3; 13; 13);
    }
}
