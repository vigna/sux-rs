/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::prelude::*;

#[test]
fn test() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in (1..1000).chain((1000..10000).step_by(100)) {
        let bits = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>();
        let select9 = Select9::new(Rank9::new(bits.clone()));

        let ones = bits.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(select9.select(i), Some(p));
        }
        assert_eq!(select9.select(ones + 1), None);
    }
}

#[test]
fn test_into_inner() {
    let bits = BitVec::new(0);
    let select = Select9::new(Rank9::new(bits));

    let inner = select.into_inner();
    assert_eq!(inner.len(), 0);
    let inner = inner.into_inner();
    assert_eq!(inner.len(), 0);
}

#[test]
fn test_mult_usize() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in (1 << 10..1 << 15).step_by(usize::BITS as _) {
        let bits = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>();
        let select9 = Select9::new(Rank9::new(bits.clone()));

        let ones = bits.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(select9.select(i), Some(p));
        }
        assert_eq!(select9.select(ones + 1), None);
    }
}

#[test]
fn test_empty() {
    let bits = BitVec::new(0);
    let select9 = Select9::new(Rank9::new(bits.clone()));
    assert_eq!(select9.count_ones(), 0);
    assert_eq!(select9.len(), 0);
    assert_eq!(select9.select(0), None);
}

#[test]
fn test_ones() {
    let len = 300_000;
    let bits = (0..len).map(|_| true).collect::<BitVec>();
    let select9 = Select9::new(Rank9::new(bits));
    assert_eq!(select9.count_ones(), len);
    assert_eq!(select9.len(), len);
    for i in 0..len {
        assert_eq!(select9.select(i), Some(i));
    }
}

#[test]
fn test_zeros() {
    let len = 300_000;
    let bits = (0..len).map(|_| false).collect::<BitVec>();
    let select9 = Select9::new(Rank9::new(bits));
    assert_eq!(select9.count_ones(), 0);
    assert_eq!(select9.len(), len);
    assert_eq!(select9.select(0), None);
}

#[test]
fn test_few_ones() {
    let lens = [1 << 18, 1 << 19, 1 << 20];
    for len in lens {
        for num_ones in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let bits = (0..len)
                .map(|i| i % (len / num_ones) == 0)
                .collect::<BitVec>();
            let select9 = Select9::new(Rank9::new(bits));
            assert_eq!(select9.count_ones(), num_ones);
            assert_eq!(select9.len(), len);
            for i in 0..num_ones {
                assert_eq!(select9.select(i), Some(i * (len / num_ones)));
            }
        }
    }
}

#[test]
fn test_non_uniform() {
    let lens = [1 << 18, 1 << 19, 1 << 20, 1 << 25];

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

            let select9 = Select9::new(Rank9::new(bits));

            for (i, &p) in pos.iter().enumerate() {
                assert_eq!(select9.select(i), Some(p));
            }
            assert_eq!(select9.select(ones + 1), None);
        }
    }
}

#[test]
fn test_rank() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in (10_000..100_000).step_by(1000) {
        let bits = (0..len)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>();
        let select9 = Select9::new(Rank9::new(bits.clone()));

        let mut ranks = Vec::with_capacity(len);
        let mut r = 0;
        for bit in bits.into_iter() {
            ranks.push(r);
            if bit {
                r += 1;
            }
        }

        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(select9.rank(i), r);
        }
        assert_eq!(select9.rank(len + 1), select9.count_ones());
    }
}
