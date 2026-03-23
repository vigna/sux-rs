/*
 * SPDX-FileCopyrightText: 2024 Michele Andreata
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::prelude::*;
use sux::rank_sel::select_adapt;
use sux::traits::BitVecOps;

macro_rules! test_rank_small {
    ($W:tt : $n:tt) => {
        let mut rng = SmallRng::seed_from_u64(0);
        let lens = (1..1000)
            .chain((10_000..100_000).step_by(1000))
            .chain((100_000..1_000_000).step_by(100_000));
        let density = 0.5;
        for len in lens {
            let bits = (0..len)
                .map(|_| rng.random_bool(density))
                .collect::<BitVec<Vec<$W>>>();
            let rank_small = rank_small![$W : $n ; bits.clone()];

            let mut ranks = Vec::with_capacity(len);
            let mut r = 0;
            for bit in BitVecOps::<$W>::iter(&bits) {
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
            assert_eq!(
                rank_small.rank(bits.len() + 1),
                BitCount::count_ones(&bits)
            );
        }
    };
    ($n:tt) => {
        test_rank_small![u64 : $n];
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

#[test]
fn test_rank_small_0_u32() {
    test_rank_small![u32 : 0];
}

#[test]
fn test_rank_small_1_u32() {
    test_rank_small![u32 : 1];
}

#[test]
fn test_rank_small_last_u64() {
    let bits =
        unsafe { BitVec::from_raw_parts(vec![!1u64; 1 << 10], (1 << 10) * u64::BITS as usize) };

    let rank_small = rank_small![u64: 0; bits.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u64: 1; bits.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u64: 2; bits.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u64: 3; bits.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u64: 4; bits.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());
}

#[test]
fn test_rank_small_last_u32() {
    let bits32 =
        unsafe { BitVec::from_raw_parts(vec![!1u32; 1 << 10], (1 << 10) * u32::BITS as usize) };

    let rank_small = rank_small![u32: 0; bits32.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u32: 1; bits32.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u32: 2; bits32.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u32: 3; bits32.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u32: 4; bits32.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());

    let rank_small = rank_small![u32: 5; bits32.clone()];
    assert_eq!(rank_small.rank(rank_small.len()), rank_small.num_ones());
}

#[test]
fn test_rank_small_map() {
    let bits = bit_vec![u64: 0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
    let rank_small = rank_small![u64: 2; bits];
    let rank_small_sel = unsafe {
        rank_small.map(|b| {
            let b: AddNumBits<_> = b.into();
            SelectAdapt::with_span(b, select_adapt::DEFAULT_TARGET_INVENTORY_SPAN, 2)
        })
    };
    assert_eq!(rank_small_sel.rank(0), 0);
    assert_eq!(rank_small_sel.rank(1), 0);
    assert_eq!(rank_small_sel.rank(2), 1);
    assert_eq!(rank_small_sel.rank(10), 5);
    assert_eq!(rank_small_sel.select(0), Some(1));
    assert_eq!(rank_small_sel.select(1), Some(3));
    assert_eq!(rank_small_sel.select(6), None);
}

#[test]
fn test_rank_small_empty() {
    let bits = BitVec::<Vec<u64>>::new(0);
    let rank_small = RankSmall::<64, 2, 9, _>::new(bits);

    assert_eq!(rank_small.len(), 0);
    let inner = rank_small.into_inner();
    assert_eq!(inner.len(), 0);
}

#[cfg(feature = "slow_tests")]
#[test]
fn test_rank_small_large() {
    use sux::traits::BitVecOpsMut;

    let mut bits = BitVec::<Vec<usize>>::new(3 * (1 << 32) + 100000);
    for i in 0..bits.len() {
        if i % 5 == 0 {
            bits.set(i, true);
        };
    }
    let rank_small = RankSmall::<64, 2, 9, _>::new(bits.clone());
    for i in (0..bits.len()).step_by(5) {
        assert_eq!(rank_small.rank(i), i.div_ceil(5));
    }
}
