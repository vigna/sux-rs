/*
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::prelude::*;

macro_rules! test {
    ($NUM_U32S: literal; $COUNTER_WIDTH: literal) => {
        use sux::traits::Select;
        let mut rng = SmallRng::seed_from_u64(0);
        let density = 0.5;
        let lens = (1..1000)
            .chain((1000..10000).step_by(100))
            .chain([1 << 20, 1 << 24]);
        for len in lens {
            let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            let rank_small_sel =
                SelectSmall::<$NUM_U32S, $COUNTER_WIDTH, _>::new(RankSmall::<
                    $NUM_U32S,
                    $COUNTER_WIDTH,
                    _,
                >::new(bits.clone()));

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
    let select = SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits.clone()));
    assert_eq!(select.count_ones(), 0);
    assert_eq!(select.len(), 0);
    assert_eq!(select.select(0), None);

    let inner = select.into_inner();
    assert_eq!(inner.len(), 0);
    let inner = inner.into_inner();
    assert_eq!(inner.len(), 0);
}

#[test]
fn test_ones() {
    let len = 300_000;
    let bits = (0..len).map(|_| true).collect::<BitVec>();
    let select = SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits));
    assert_eq!(select.count_ones(), len);
    assert_eq!(select.len(), len);
    for i in 0..len {
        assert_eq!(select.select(i), Some(i));
    }
}

#[test]
fn test_zeros() {
    let len = 300_000;
    let bits = (0..len).map(|_| false).collect::<BitVec>();
    let select = SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits));
    assert_eq!(select.count_ones(), 0);
    assert_eq!(select.len(), len);
    assert_eq!(select.select(0), None);
}

#[test]
fn test_few_ones() {
    let lens = [1 << 18, 1 << 19, 1 << 20];
    for len in lens {
        for num_ones in [1, 2, 4, 8, 16, 32, 64, 128] {
            let bits = (0..len)
                .map(|i| i % (len / num_ones) == 0)
                .collect::<BitVec>();
            let select = SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits));
            assert_eq!(select.count_ones(), num_ones);
            assert_eq!(select.len(), len);
            for i in 0..num_ones {
                assert_eq!(select.select(i), Some(i * (len / num_ones)));
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

            let select = SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits));
            for i in 0..ones {
                assert_eq!(select.select(i), Some(pos[i]));
            }
            assert_eq!(select.select(ones + 1), None);
        }
    }
}

#[test]
fn test_extremely_sparse() {
    let len = 1 << 18;
    let bits = (0..len / 2)
        .map(|_| false)
        .chain([true])
        .chain((0..1 << 17).map(|_| false))
        .chain([true, true])
        .chain((0..1 << 18).map(|_| false))
        .chain([true])
        .chain((0..len / 2).map(|_| false))
        .collect::<BitVec>();
    let select = SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits));

    assert_eq!(select.count_ones(), 4);
    assert_eq!(select.select(0), Some(len / 2));
    assert_eq!(select.select(1), Some(len / 2 + (1 << 17) + 1));
    assert_eq!(select.select(2), Some(len / 2 + (1 << 17) + 2));
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_large() {
    let mut bits = BitVec::new(3 * (1 << 32) + 100000);
    for i in 0..bits.len() {
        if i % 5 == 0 {
            bits.set(i, true);
        };
    }
    let rank_small = RankSmall::<2, 9>::new(bits.clone());
    let select = SelectSmall::<2, 9>::new(rank_small);
    for i in (0..bits.len()).step_by(5) {
        assert_eq!(select.select(i / 5), Some(i));
    }
}
