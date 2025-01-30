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

const INV: usize = 13;
const SUB: usize = 0;

#[test]
fn test() {
    let lens = (1..100)
        .step_by(10)
        .chain((100_000..1_100_000).step_by(100_000));
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in [0.1, 0.5, 0.9] {
            let bits: AddNumBits<_> = (0..len)
                .map(|_| rng.random_bool(density))
                .map(|b| !b)
                .collect::<BitVec>()
                .into();

            let select = SelectZeroAdaptConst::<_, _, INV, SUB>::new(bits.clone());

            let zeros = select.num_zeros();
            let mut pos = Vec::with_capacity(zeros);
            for i in 0..len {
                if !bits[i] {
                    pos.push(i);
                }
            }

            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(select.select_zero(i), Some(p));
            }
            assert_eq!(select.select_zero(zeros + 1), None);
        }
    }
}

#[test]
fn test_one_u64() {
    let lens = [1_000_000];
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.1;
    for len in lens {
        let bits: AddNumBits<_> = (0..len)
            .map(|_| rng.random_bool(density))
            .map(|b| !b)
            .collect::<BitVec>()
            .into();
        let select = SelectZeroAdaptConst::<_, _, 13, 0>::new(bits.clone());

        let zeros = select.num_zeros();
        let mut pos = Vec::with_capacity(zeros);
        for i in 0..len {
            if !bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(select.select_zero(i), Some(p), "i = {}", i);
        }
        assert_eq!(select.select_zero(zeros + 1), None);
    }
}

#[test]
fn test_w_rank9() {
    let lens = (1..100)
        .step_by(10)
        .chain((100_000..1_000_000).step_by(100_000));
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in lens {
        let bits: BitVec = (0..len)
            .map(|_| rng.random_bool(density))
            .map(|b| !b)
            .collect::<BitVec>();

        let rank9 = Rank9::new(bits.clone());

        let select = SelectZeroAdaptConst::<_, _, INV, SUB>::new(rank9);

        let zeros = select.num_zeros();
        let mut pos = Vec::with_capacity(zeros);
        for i in 0..len {
            if !bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(select.select_zero(i), Some(p));
        }
        assert_eq!(select.select_zero(zeros + 1), None);
    }
}

#[test]
fn test_empty() {
    let bits: AddNumBits<_> = BitVec::new(0).into();
    let select = SelectZeroAdaptConst::<_, _, INV, SUB>::new(bits.clone());
    assert_eq!(select.num_zeros(), 0);
    assert_eq!(select.len(), 0);
    assert_eq!(select.select_zero(0), None);

    let inner = select.into_inner();
    assert_eq!(inner.len(), 0);
    let inner = inner.into_inner();
    assert_eq!(inner.len(), 0);
}

#[test]
fn test_zeros() {
    let len = 300_000;
    let bits: AddNumBits<_> = (0..len).map(|_| false).collect::<BitVec>().into();
    let select = SelectZeroAdaptConst::<_, _, INV, SUB>::new(bits);
    assert_eq!(select.num_zeros(), len);
    assert_eq!(select.len(), len);
    for i in 0..len {
        assert_eq!(select.select_zero(i), Some(i));
    }
}

#[test]
fn test_ones() {
    let len = 300_000;
    let bits: AddNumBits<_> = (0..len).map(|_| true).collect::<BitVec>().into();
    let select = SelectZeroAdaptConst::<_, _, INV, SUB>::new(bits);
    assert_eq!(select.num_zeros(), 0);
    assert_eq!(select.len(), len);
    assert_eq!(select.select_zero(0), None);
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
                    .map(|b| !b)
                    .collect::<BitVec>();
                if b.count_zeros() > 0 {
                    break b;
                }
            };
            let num_zeros_first_half = first_half.count_zeros();
            let second_half = (0..len2)
                .map(|_| rng.random_bool(density1))
                .map(|b| !b)
                .collect::<BitVec>();
            let num_zeros_second_half = second_half.count_zeros();

            assert!(num_zeros_first_half > 0);
            assert!(num_zeros_second_half > 0);

            let bits: AddNumBits<_> = first_half
                .into_iter()
                .chain(second_half.into_iter())
                .collect::<BitVec>()
                .into();

            assert_eq!(
                num_zeros_first_half + num_zeros_second_half,
                bits.num_zeros()
            );

            assert_eq!(bits.len(), len as usize);

            let zeros = bits.num_zeros();
            let mut pos = Vec::with_capacity(zeros);
            for i in 0..(len as usize) {
                if !bits[i] {
                    pos.push(i);
                }
            }

            let select = SelectZeroAdaptConst::<_, _, INV, SUB>::new(bits);
            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(select.select_zero(i), Some(p));
            }
            assert_eq!(select.select_zero(zeros + 1), None);
        }
    }
}

#[test]
fn test_map() {
    let bits: AddNumBits<_> = bit_vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1].into();
    let sel = SelectZeroAdaptConst::<_, _>::new(bits);
    let rank_sel = unsafe { sel.map(Rank9::new) };
    assert_eq!(rank_sel.rank(0), 0);
    assert_eq!(rank_sel.rank(1), 0);
    assert_eq!(rank_sel.rank(2), 1);
    assert_eq!(rank_sel.rank(10), 5);

    let rank_seol01 = unsafe { rank_sel.map(SelectZeroAdaptConst::<_, _>::new) };
    assert_eq!(rank_seol01.select_zero(0), Some(0));
    assert_eq!(rank_seol01.select_zero(1), Some(2));
    assert_eq!(rank_seol01.select_zero(2), Some(5));
    assert_eq!(rank_seol01.select_zero(3), Some(7));
    assert_eq!(rank_seol01.select_zero(4), Some(8));
    assert_eq!(rank_seol01.select_zero(5), None);
}

#[test]
fn test_extremely_sparse() {
    let len = 1 << 18;
    let bits: AddNumBits<_> = (0..len / 2)
        .map(|_| false)
        .chain([true])
        .chain((0..1 << 17).map(|_| false))
        .chain([true, true])
        .chain((0..1 << 18).map(|_| false))
        .chain([true])
        .chain((0..len / 2).map(|_| false))
        .map(|b| !b)
        .collect::<BitVec>()
        .into();
    let simple = SelectZeroAdaptConst::<_, _, 13, 0>::new(bits);

    assert_eq!(simple.count_zeros(), 4);
    assert_eq!(simple.select_zero(0), Some(len / 2));
    assert_eq!(simple.select_zero(1), Some(len / 2 + (1 << 17) + 1));
    assert_eq!(simple.select_zero(2), Some(len / 2 + (1 << 17) + 2));
}

#[test]
fn test_sub32s() {
    let lens = [1_000_000];
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.1;
    for len in lens {
        let bits: AddNumBits<BitVec> = (0..len)
            .map(|_| rng.random_bool(density))
            .map(|b| !b)
            .collect::<BitVec>()
            .into();
        let simple = SelectZeroAdaptConst::<_, _, 13, 3>::new(bits.clone());

        let zeros = simple.count_zeros();
        let mut pos = Vec::with_capacity(zeros);
        for i in 0..len {
            if !bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p));
        }
        assert_eq!(simple.select_zero(zeros + 1), None);
    }
}

#[test]
fn test_sub32s_last_small() {
    let lens = [1_000_000];
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.0001;
    for len in lens {
        let mut bits = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>();
        bits.flip();
        let bits: AddNumBits<_> = bits.into();
        let simple = SelectZeroAdaptConst::<_, _, 13, 16>::new(bits.clone());

        let zeros = simple.count_zeros();
        let mut pos = Vec::with_capacity(zeros);
        for i in 0..len {
            if !bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select_zero(i), Some(p));
        }
        assert_eq!(simple.select_zero(zeros + 1), None);
    }
}
