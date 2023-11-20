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

#[test]
fn test_selects() {
    for size in [0, 1, 2, 3, 4, 7, 8, 9, 10, 100, 10000, 100000] {
        let mut rng = SmallRng::seed_from_u64(0);
        let bitvec = (0..size).map(|_| rng.gen_bool(0.5)).collect::<BitVec>();
        let ones = bitvec.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..size {
            if bitvec[i] {
                pos.push(i);
            }
        }

        let simple = <SimpleSelectHalf<_, _, 10, 2>>::new(&bitvec);

        for i in 0..ones {
            assert_eq!(simple.select(i), Some(pos[i]), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }

    for size in [1000000] {
        let mut rng = SmallRng::seed_from_u64(0);
        let bitvec = (0..size).map(|_| rng.gen_bool(0.001)).collect::<BitVec>();
        let ones = bitvec.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..size {
            if bitvec[i] {
                pos.push(i);
            }
        }

        let simple = <SimpleSelectHalf<_, _, 10, 2>>::new(&bitvec);

        for i in 0..ones {
            assert_eq!(simple.select(i), Some(pos[i]), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }

    for size in [6_000_000_000] {
        let mut bitvec = BitVec::new(size);
        let mut pos = vec![];
        for idx in 0..1024 {
            pos.push(idx * size / 1024);
            bitvec.set(idx * size / 1024, true);
        }
        let ones = pos.len();

        let simple = <SimpleSelectHalf<_, _, 10, 2>>::new(&bitvec);

        for i in 0..ones {
            assert_eq!(simple.select(i), Some(pos[i]), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}
