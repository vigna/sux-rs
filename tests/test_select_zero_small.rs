/*
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::prelude::*;

macro_rules! test {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        let lens = (1..1000)
            .chain((1000..10000).step_by(100))
            .chain([1 << 20, 1 << 24]);
        for len in lens {
            let bits = (0..len)
                .map(|_| rng.random_bool(density))
                .collect::<BitVec>();
            let rank_small_sel = SelectZeroSmall::<$NUM_U32S, $COUNTER_WIDTH, _>::new(RankSmall::<
                $NUM_U32S,
                $COUNTER_WIDTH,
                _,
            >::new(
                bits.clone(),
            ));

            let zeros = bits.len() - bits.count_ones();
            let mut pos = Vec::with_capacity(zeros);
            for i in 0..len {
                if !bits[i] {
                    pos.push(i);
                }
            }

            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(rank_small_sel.select_zero(i), Some(p));
            }
            assert_eq!(rank_small_sel.select_zero(zeros + 1), None);
        }
    };
}

#[test]
fn test_rank_small0() {
    test!(2; 9);
}

#[test]
fn test_rank_small1() {
    test!(1; 9);
}

#[test]
fn test_rank_small2() {
    test!(1; 10);
}

#[test]
fn test_rank_small3() {
    test!(1; 11);
}

#[test]
fn test_rank_small4() {
    test!(3; 13);
}

#[test]
fn test_empty() {
    let bits = BitVec::new(0);
    let select = SelectZeroSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits.clone()));
    assert_eq!(select.count_ones(), 0);
    assert_eq!(select.len(), 0);
    assert_eq!(select.select_zero(0), None);

    let inner = select.into_inner();
    assert_eq!(inner.len(), 0);
    let inner = inner.into_inner();
    assert_eq!(inner.len(), 0);
}

#[test]
fn test_ones() {
    let len = 300_000;
    let bits = (0..len).map(|_| true).collect::<BitVec>();
    let select = SelectZeroSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits));

    assert_eq!(select.len(), len);
    assert_eq!(select.select_zero(0), None);
}

#[test]
fn test_zeros() {
    let len = 300_000;
    let bits = (0..len).map(|_| false).collect::<BitVec>();
    let select = SelectZeroSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits));
    assert_eq!(select.len(), len);
    for i in 0..len {
        assert_eq!(select.select_zero(i), Some(i));
    }
}

#[test]
fn test_few_zeros() {
    let lens = [1 << 18, 1 << 19, 1 << 20];
    for len in lens {
        for num_ones in [1, 2, 4, 8, 16, 32, 64, 128] {
            let bits = (0..len)
                .map(|i| i % (len / num_ones) != 0)
                .collect::<BitVec>();
            let select = SelectZeroSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits));
            assert_eq!(select.len(), len);
            for i in 0..num_ones {
                assert_eq!(select.select_zero(i), Some(i * (len / num_ones)));
            }
        }
    }
}

#[test]
fn test_non_uniform() {
    let lens = [1 << 18, 1 << 19, 1 << 20];

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
                    .map(|_| rng.random_bool(density0))
                    .collect::<BitVec>();
                if b.count_ones() > 0 {
                    break b;
                }
            };
            let num_ones_first_half = first_half.count_ones();
            let second_half = (0..len2)
                .map(|_| rng.random_bool(density1))
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

            let zeros = bits.len() - bits.count_ones();
            let mut pos = Vec::with_capacity(zeros);
            for i in 0..(len as usize) {
                if !bits[i] {
                    pos.push(i);
                }
            }

            let select = SelectZeroSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits));
            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(select.select_zero(i), Some(p));
            }
            assert_eq!(select.select_zero(zeros + 1), None);
        }
    }
}

#[test]
fn test_extremely_sparse() {
    let len = 1 << 18;
    let bits = (0..len / 2)
        .map(|_| true)
        .chain([false])
        .chain((0..1 << 17).map(|_| true))
        .chain([false, false])
        .chain((0..1 << 18).map(|_| true))
        .chain([false])
        .chain((0..len / 2).map(|_| true))
        .collect::<BitVec>();
    let select = SelectZeroSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits));

    assert_eq!(select.count_zeros(), 4);
    assert_eq!(select.select_zero(0), Some(len / 2));
    assert_eq!(select.select_zero(1), Some(len / 2 + (1 << 17) + 1));
    assert_eq!(select.select_zero(2), Some(len / 2 + (1 << 17) + 2));
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_extremely_sparse_and_large() {
    let num_words = 3 * (1 << 26) + 1;
    let len = num_words * 64;
    let mut data: Vec<usize> = Vec::with_capacity(num_words);
    data.push(!(1_usize));
    for _ in 0..((1 << 26) - 2) {
        data.push(!0);
    }
    data.push(!(1 << 63));
    for _ in 0..(1 << 26) {
        data.push(usize::MAX as usize);
    }
    for _ in 0..(1 << 26) {
        data.push(usize::MAX as usize);
    }
    data.push(!(1_usize));

    assert_eq!(data.len(), num_words);

    let bits = unsafe { BitVec::from_raw_parts(data, len) };
    let rank_small = RankSmall::<2, 9>::new(bits);
    let select = SelectZeroSmall::<2, 9, _>::new(rank_small);

    assert_eq!(select.count_zeros(), 3);

    assert_eq!(select.select_zero(0), Some(0));
    assert_eq!(select.select_zero(1), Some((1 << 32) - 1));
    assert_eq!(select.select_zero(2), Some(3 * (1 << 32)));
    assert_eq!(select.select_zero(3), None);
}

#[allow(unused_macros)]
macro_rules! test_large {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        const ONES_STEP_4: usize = 1usize << 0
            | 1 << 4
            | 1 << 8
            | 1 << 12
            | 1 << 16
            | 1 << 20
            | 1 << 24
            | 1 << 28
            | 1 << 32
            | 1 << 36
            | 1 << 40
            | 1 << 44
            | 1 << 48
            | 1 << 52
            | 1 << 56
            | 1 << 60;
        const ZEROS_STEP_4: usize = !ONES_STEP_4;

        let len = 3 * (1 << 32) + 64 * 1000;
        let num_words = len / 64;
        let mut data: Vec<usize> = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            data.push(ZEROS_STEP_4);
        }
        let bits = unsafe { BitVec::from_raw_parts(data, len) };

        let rank_small = RankSmall::<$NUM_U32S, $COUNTER_WIDTH>::new(bits);
        let select = SelectZeroSmall::<$NUM_U32S, $COUNTER_WIDTH, _>::new(rank_small);
        for i in (0..len).step_by(4) {
            assert_eq!(select.select_zero(i / 4), Some(i));
        }
    };
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_large0() {
    test_large!(2; 9);
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_large1() {
    test_large!(1; 9);
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_large2() {
    test_large!(1; 10);
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_large3() {
    test_large!(1; 11);
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_large4() {
    test_large!(3; 13);
}
