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
    const MAX: usize = 100_000;
    let mut rng = SmallRng::seed_from_u64(0);
    let bitvec = (0..MAX).map(|_| rng.gen_bool(0.5)).collect::<BitVec>();
    println!("{}", &bitvec);
    let ones = bitvec.count_ones();

    let simple = <SimpleSelectHalf<_, _, 10, 2>>::new(&bitvec);
    let quantum = <QuantumIndex<_, _, 8>>::new(&bitvec, ones).unwrap();

    for i in 0..ones {
        dbg!(i);
        assert_eq!(bitvec.select(i), simple.select(i));
        assert_eq!(bitvec.select(i), quantum.select(i));
    }
}
