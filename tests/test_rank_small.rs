/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::prelude::*;
use sux::rank_small;

macro_rules! test_rank_small {
    ($n: tt) => {
        let mut rng = SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small = rank_small![$n; bits.clone()];

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in bits.into_iter() {
                ranks.push(r);
                if bit {
                    r += 1;
                }
            }

            for i in 0..bits.len() {
                assert_eq!(
                    rank_small.rank(i),
                    ranks[i],
                    "i = {}, len = {}, left = {}, right = {}",
                    i,
                    len,
                    rank_small.rank(i),
                    ranks[i]
                );
            }
            assert_eq!(rank_small.rank(bits.len() + 1), bits.count_ones());
        }
    };
}

#[test]
fn test_rank_small0() {
    test_rank_small![0];
}

#[test]
fn test_rank_small1() {
    test_rank_small![1];
}

#[test]
fn test_rank_small2() {
    test_rank_small![2];
}

#[test]
fn test_rank_small3() {
    test_rank_small![3];
}

#[test]
fn test_rank_small4() {
    test_rank_small![4];
}
