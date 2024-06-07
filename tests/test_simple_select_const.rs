/*
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::bit_vec;
use sux::bits::BitVec;
use sux::bits::CountBitVec;
use sux::rank_sel::Rank9;
use sux::rank_sel::SimpleSelectConst;
use sux::traits::BitCount;
use sux::traits::BitLength;
use sux::traits::Rank;
use sux::traits::Select;
use sux::traits::SelectZero;

const INV: usize = 13;
const SUB: usize = 0;

#[test]
fn test_simple_select_const() {
    let lens = (1..100)
        .step_by(10)
        .chain((100_000..1_100_000).step_by(100_000));
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in [0.1, 0.5, 0.9] {
            let bits: CountBitVec = (0..len)
                .map(|_| rng.gen_bool(density))
                .collect::<BitVec>()
                .into();

            let simple = SimpleSelectConst::<_, _, INV, SUB>::new(bits.clone());

            let ones = simple.count_ones();
            let mut pos = Vec::with_capacity(ones);
            for i in 0..len {
                if bits[i] {
                    pos.push(i);
                }
            }

            for i in 0..ones {
                assert_eq!(simple.select(i), Some(pos[i]));
            }
            assert_eq!(simple.select(ones + 1), None);
        }
    }
}

#[test]
fn debug() {
    let lens = [1_000_000];
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.1;
    for len in lens {
        let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
        let simple = SimpleSelectConst::<_, _, 13, 0>::new(bits.clone());

        let ones = simple.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for i in 0..ones {
            assert_eq!(simple.select(i), Some(pos[i]));
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[test]
fn test_simple_select_const_w_rank9() {
    let lens = (1..100)
        .step_by(10)
        .chain((100_000..1_000_000).step_by(100_000));
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.5;
    for len in lens {
        let bits: BitVec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();

        let rank9 = Rank9::new(bits.clone());

        let simple = SimpleSelectConst::<_, _, INV, SUB>::new(rank9);

        let ones = simple.count_ones();
        let mut pos = Vec::with_capacity(ones);
        for i in 0..len {
            if bits[i] {
                pos.push(i);
            }
        }

        for i in 0..ones {
            assert_eq!(simple.select(i), Some(pos[i]));
        }
        assert_eq!(simple.select(ones + 1), None);
    }
}

#[test]
fn test_simple_select_const_empty() {
    let bits = BitVec::new(0);
    let simple = SimpleSelectConst::<_, _, INV, SUB>::new(bits.clone());
    assert_eq!(simple.count_ones(), 0);
    assert_eq!(simple.len(), 0);
    assert_eq!(simple.select(0), None);
}

#[test]
fn test_simple_select_const_ones() {
    let len = 300_000;
    let bits: CountBitVec = (0..len).map(|_| true).collect::<BitVec>().into();
    let simple = SimpleSelectConst::<_, _, INV, SUB>::new(bits);
    assert_eq!(simple.count_ones(), len);
    assert_eq!(simple.len(), len);
    for i in 0..len {
        assert_eq!(simple.select(i), Some(i));
    }
}

#[test]
fn test_simple_select_const_zeros() {
    let len = 300_000;
    let bits: CountBitVec = (0..len).map(|_| false).collect::<BitVec>().into();
    let simple = SimpleSelectConst::<_, _, INV, SUB>::new(bits);
    assert_eq!(simple.count_ones(), 0);
    assert_eq!(simple.len(), len);
    assert_eq!(simple.select(0), None);
}

#[test]
fn test_simple_select_const_non_uniform() {
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

            let bits: CountBitVec = first_half
                .into_iter()
                .chain(second_half.into_iter())
                .collect::<BitVec>()
                .into();

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

            let simple = SimpleSelectConst::<_, _, INV, SUB>::new(bits);
            for i in 0..(ones) {
                assert_eq!(simple.select(i), Some(pos[i]));
            }
            assert_eq!(simple.select(ones + 1), None);
        }
    }
}

#[test]
fn test_map() {
    let bits = bit_vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
    let sel = SimpleSelectConst::<_, _>::new(bits);
    let rank_sel = sel.map(Rank9::new);
    assert_eq!(rank_sel.rank(0), 0);
    assert_eq!(rank_sel.rank(1), 0);
    assert_eq!(rank_sel.rank(2), 1);
    assert_eq!(rank_sel.rank(10), 5);

    let rank_seol01 = rank_sel.map(SimpleSelectConst::<_, _>::new);
    assert_eq!(rank_seol01.select_zero(0), Some(0));
    assert_eq!(rank_seol01.select_zero(1), Some(2));
    assert_eq!(rank_seol01.select_zero(2), Some(5));
    assert_eq!(rank_seol01.select_zero(3), Some(7));
    assert_eq!(rank_seol01.select_zero(4), Some(8));
    assert_eq!(rank_seol01.select_zero(5), None);
}
