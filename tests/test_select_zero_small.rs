/*
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::prelude::{BitCount, BitLength, BitVec, RankSmall, SelectZero, SelectZeroSmall};

macro_rules! test_rank_small_sel_zero {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal; $LOG2_ZEROS_PER_INVENTORY: literal) => {
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        let lens = (1..1000)
            .chain((1000..10000).step_by(100))
            .chain([1 << 20, 1 << 24]);
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small_sel =
                SelectZeroSmall::<$NUM_U32S, $COUNTER_WIDTH, $LOG2_ZEROS_PER_INVENTORY, _>::new(
                    RankSmall::<$NUM_U32S, $COUNTER_WIDTH, _>::new(bits.clone()),
                );

            let zeros = bits.len() - bits.count_ones();
            let mut pos = Vec::with_capacity(zeros);
            for i in 0..len {
                if !bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..zeros {
                assert_eq!(rank_small_sel.select_zero(i), Some(pos[i]));
            }
            assert_eq!(rank_small_sel.select_zero(zeros + 1), None);
        }
    };
}

#[test]
fn rank_small_sel_zero0() {
    test_rank_small_sel_zero!(2; 9; 13);
}

#[test]
fn rank_small_sel_zero1() {
    test_rank_small_sel_zero!(1; 9; 13);
}

#[test]
fn rank_small_sel_zero2() {
    test_rank_small_sel_zero!(1; 10; 13);
}

#[test]
fn rank_small_sel_zero3() {
    test_rank_small_sel_zero!(1; 11; 13);
}

#[test]
fn rank_small_sel_zero4() {
    test_rank_small_sel_zero!(3; 13; 13);
}
