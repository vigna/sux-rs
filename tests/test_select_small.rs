/*
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::bits::BitVec;
use sux::rank_sel::{RankSmall, SelectSmall};
use sux::traits::BitCount;

macro_rules! test_rank_small_sel {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal; $LOG2_ONES_PER_INVENTORY: literal) => {
        use sux::traits::Select;
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        let lens = (1..1000)
            .chain((1000..10000).step_by(100))
            .chain([1 << 20, 1 << 24]);
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small_sel =
                SelectSmall::<$NUM_U32S, $COUNTER_WIDTH, $LOG2_ONES_PER_INVENTORY, _>::new(
                    RankSmall::<$NUM_U32S, $COUNTER_WIDTH, _>::new(bits.clone()),
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
