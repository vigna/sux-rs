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
fn test_select_fixed1() {
    for len in [0, 1, 2, 3, 4, 7, 8, 9, 10, 100]
        .into_iter()
        .chain(1000..1200)
        .chain([10000, 100000])
    {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut bitvec = (0..len).map(|_| rng.gen_bool(0.5)).collect::<BitVec>();
        bitvec.flip();
        let ones = bitvec.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bitvec[i] {
                pos.push(i);
            }
        }

        let simple =
            SelectFixed1::<CountBitVec, Vec<usize>, 8>::new(CountBitVec::from(bitvec), ones);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[cfg(feature = "slow")]
#[test]
fn test_select_fixed1_slow() {
    for len in [6_000_000_000] {
        let mut bitvec = BitVec::new(len);
        let mut pos = vec![];
        for idx in 0..1024 {
            pos.push(idx * len / 1024);
            bitvec.set(idx * len / 1024, true);
        }
        let ones = pos.len();

        let simple =
            SelectFixed1::<CountBitVec, Vec<usize>, 8>::new(CountBitVec::from(bitvec), ones);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[test]
fn test_select_zero_fixed1() {
    for len in [0, 1, 2, 3, 4, 7, 8, 9, 10, 100]
        .into_iter()
        .chain(1000..1200)
        .chain([10000, 100000])
    {
        let mut rng = SmallRng::seed_from_u64(0);
        let bitvec = (0..len).map(|_| rng.gen_bool(0.5)).collect::<BitVec>();
        let zeros = bitvec.len() - bitvec.count_ones();
        let mut pos = Vec::with_capacity(zeros);
        for i in 0..len {
            if !bitvec[i] {
                pos.push(i);
            }
        }

        let mut bitvec_clone = bitvec.clone();

        let simple = SelectZeroFixed1::<CountBitVec, Vec<usize>, 8>::new(CountBitVec::from(bitvec));

        bitvec_clone.flip();
        let simple_ones = <SelectFixed2<_, _, 10, 2>>::new(&bitvec_clone);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple_ones.select(i), Some(p), "i: {} ones : {}", i, zeros);
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p), "i: {} ones : {}", i, zeros);
        }

        assert_eq!(simple.select_zero(zeros + 1), None);
    }
}

#[cfg(feature = "slow")]
#[test]
fn test_select_zero_fixed1_slow() {
    for len in [6_000_000_000] {
        let mut bitvec = BitVec::new(len);
        bitvec.fill(true);
        let mut pos = vec![];
        for idx in 0..1024 {
            pos.push(idx * len / 1024);
            bitvec.set(idx * len / 1024, false);
        }
        let zeros = pos.len();

        let simple = <SelectZeroFixed2<_, _, 10, 2>>::new(&bitvec);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p), "i: {} zeros : {}", i, zeros);
        }
        assert_eq!(simple.select_zero(zeros + 1), None);
    }
}

#[test]
fn test_select_fixed2() {
    for len in [0, 1, 2, 3, 4, 7, 8, 9, 10, 100]
        .into_iter()
        .chain(1000..1200)
        .chain([10000, 100000])
    {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut bitvec = (0..len).map(|_| rng.gen_bool(0.5)).collect::<BitVec>();
        bitvec.flip();
        let ones = bitvec.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bitvec[i] {
                pos.push(i);
            }
        }

        let simple = <SelectFixed2<_, _, 10, 2>>::new(&bitvec);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[cfg(feature = "slow")]
#[test]
fn test_select_fixed2_slow() {
    for len in [6_000_000_000] {
        let mut bitvec = BitVec::new(len);
        let mut pos = vec![];
        for idx in 0..1024 {
            pos.push(idx * len / 1024);
            bitvec.set(idx * len / 1024, true);
        }
        let ones = pos.len();

        let simple = <SelectFixed2<_, _, 10, 2>>::new(&bitvec);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p), "i: {} ones : {}", i, ones);
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[test]
fn test_select_zero_fixed2() {
    for len in [0, 1, 2, 3, 4, 7, 8, 9, 10, 100]
        .into_iter()
        .chain(1000..1200)
        .chain([10000, 100000])
    {
        let mut rng = SmallRng::seed_from_u64(0);
        let bitvec = (0..len).map(|_| rng.gen_bool(0.5)).collect::<BitVec>();
        let zeros = bitvec.len() - bitvec.count_ones();
        let mut pos = Vec::with_capacity(zeros);
        for i in 0..len {
            if !bitvec[i] {
                pos.push(i);
            }
        }

        let mut bitvec_clone = bitvec.clone();

        let simple = <SelectZeroFixed2<_, _, 10, 2>>::new(&bitvec);

        bitvec_clone.flip();
        let simple_ones = <SelectFixed2<_, _, 10, 2>>::new(&bitvec_clone);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple_ones.select(i), Some(p), "i: {} ones : {}", i, zeros);
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p), "i: {} ones : {}", i, zeros);
        }

        assert_eq!(simple.select_zero(zeros + 1), None);
    }
}

#[cfg(feature = "slow")]
#[test]
fn test_select_zero_fixed2_slow() {
    for len in [6_000_000_000] {
        let mut bitvec = BitVec::new(len);
        bitvec.fill(true);
        let mut pos = vec![];
        for idx in 0..1024 {
            pos.push(idx * len / 1024);
            bitvec.set(idx * len / 1024, false);
        }
        let zeros = pos.len();

        let simple = <SelectZeroFixed2<_, _, 10, 2>>::new(&bitvec);

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p), "i: {} zeros : {}", i, zeros);
        }
        assert_eq!(simple.select_zero(zeros + 1), None);
    }
}
