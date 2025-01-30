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

#[test]
fn test() {
    let lens = (1..100)
        .step_by(10)
        .chain((100_000..1_000_000).step_by(100_000));
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in lens {
        let bits: AddNumBits<_> = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();

        let select = SelectAdapt::new(bits.clone(), 3);

        let ones = select.num_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(select.select(i), Some(p));
        }
        assert_eq!(select.select(ones + 1), None);
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
            .collect::<BitVec>()
            .into();
        let simple = SelectAdapt::<_, _>::with_inv(bits.clone(), 13, 0);

        let ones = simple.num_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p));
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[test]
fn test_mult_usize() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in (1 << 10..1 << 15).step_by(usize::BITS as _) {
        let bits: AddNumBits<_> = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();
        let select = SelectAdapt::new(bits.clone(), 3);

        let ones = bits.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(select.select(i), Some(p));
        }
        assert_eq!(select.select(ones + 1), None);
    }
}

#[test]
fn test_empty() {
    let bits: AddNumBits<_> = BitVec::new(0).into();
    let select = SelectAdapt::new(bits.clone(), 3);
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
    let bits: AddNumBits<_> = (0..len).map(|_| true).collect::<BitVec>().into();
    let select = SelectAdapt::new(bits, 3);
    assert_eq!(select.count_ones(), len);
    assert_eq!(select.len(), len);
    for i in 0..len {
        assert_eq!(select.select(i), Some(i));
    }
}

#[test]
fn test_zeros() {
    let len = 300_000;
    let bits: AddNumBits<_> = (0..len).map(|_| false).collect::<BitVec>().into();
    let select = SelectAdapt::new(bits, 3);
    assert_eq!(select.count_ones(), 0);
    assert_eq!(select.len(), len);
    assert_eq!(select.select(0), None);
}

#[test]
fn test_few_ones() {
    let lens = [1 << 18, 1 << 19, 1 << 20];
    for len in lens {
        for num_ones in [1, 2, 4, 8, 16, 32, 64, 128] {
            let bits: AddNumBits<_> = (0..len)
                .map(|i| i % (len / num_ones) == 0)
                .collect::<BitVec>()
                .into();
            let select = SelectAdapt::new(bits, 3);
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

            let ones = bits.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..(len as usize) {
                if bits[i] {
                    pos.push(i);
                }
            }

            let bits: AddNumBits<_> = bits.into();

            let select = SelectAdapt::new(bits, 3);
            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(select.select(i), Some(p));
            }
            assert_eq!(select.select(ones + 1), None);
        }
    }
}

#[test]
fn test_map() {
    let bits: AddNumBits<_> = bit_vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1].into();
    let sel = SelectAdapt::<_, _>::new(bits, 3);
    let rank_sel = unsafe { sel.map(RankSmall::<1, 10, _>::new) };
    assert_eq!(rank_sel.rank(0), 0);
    assert_eq!(rank_sel.rank(1), 0);
    assert_eq!(rank_sel.rank(2), 1);
    assert_eq!(rank_sel.rank(10), 5);
}

#[test]
fn test_extremely_sparse() {
    let len = 1 << 18;
    let bits: AddNumBits<BitVec> = (0..len / 2)
        .map(|_| false)
        .chain([true])
        .chain((0..1 << 17).map(|_| false))
        .chain([true, true])
        .chain((0..1 << 18).map(|_| false))
        .chain([true])
        .chain((0..len / 2).map(|_| false))
        .collect::<BitVec>()
        .into();
    let simple = SelectAdapt::new(bits, 0);

    assert_eq!(simple.count_ones(), 4);
    assert_eq!(simple.select(0), Some(len / 2));
    assert_eq!(simple.select(1), Some(len / 2 + (1 << 17) + 1));
    assert_eq!(simple.select(2), Some(len / 2 + (1 << 17) + 2));
}

#[test]
fn test_sub32s() {
    let lens = [1_000_000];
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.1;
    for len in lens {
        let bits: AddNumBits<BitVec> = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();
        let simple = SelectAdapt::with_inv(bits.clone(), 13, 3);

        let ones = simple.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p));
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[test]
fn test_sub32s_last_small() {
    let lens = [1_000_000];
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.0001;
    for len in lens {
        let bits: AddNumBits<BitVec> = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();
        let simple = SelectAdapt::with_inv(bits.clone(), 13, 16);

        let ones = simple.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(simple.select(i), Some(p));
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}
