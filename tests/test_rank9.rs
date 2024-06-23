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
    let mut rng = SmallRng::seed_from_u64(0);
    let lens = (1..1000)
        .chain((10_000..100_000).step_by(1000))
        .chain((100_000..1_000_000).step_by(100_000));
    let density = 0.5;
    for len in lens {
        let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
        let rank9: Rank9 = Rank9::new(bits.clone());

        let mut ranks = Vec::with_capacity(len);
        let mut r = 0;
        for bit in bits.into_iter() {
            ranks.push(r);
            if bit {
                r += 1;
            }
        }

        for i in 0..bits.len() {
            assert_eq!(rank9.rank(i), ranks[i]);
        }
        assert_eq!(rank9.rank(bits.len() + 1), bits.count_ones());
    }
}

#[test]
fn test_rank_zero() {
    let bits = bit_vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
    let rank9 = Rank9::new(bits);
    assert_eq!(rank9.rank_zero(0), 0);
    assert_eq!(rank9.rank_zero(1), 1);
    assert_eq!(rank9.rank_zero(2), 1);
    assert_eq!(rank9.rank_zero(3), 2);
    assert_eq!(rank9.rank_zero(4), 2);
    assert_eq!(rank9.rank_zero(5), 2);
    assert_eq!(rank9.rank_zero(6), 3);
    assert_eq!(rank9.rank_zero(7), 3);
    assert_eq!(rank9.rank_zero(8), 4);
    assert_eq!(rank9.rank_zero(9), 5);
    assert_eq!(rank9.rank_zero(10), 5);
}

#[test]
fn test_map() {
    let bits = bit_vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
    let rank9 = Rank9::new(bits);
    let rank9_sel = unsafe {
        rank9.map(|x| {
            let x: AddNumBits<_> = x.into();
            SelectAdapt::<_, _>::new(x, 3)
        })
    };
    assert_eq!(rank9_sel.rank(0), 0);
    assert_eq!(rank9_sel.rank(1), 0);
    assert_eq!(rank9_sel.rank(2), 1);
    assert_eq!(rank9_sel.rank(10), 5);
    assert_eq!(rank9_sel.select(0), Some(1));
    assert_eq!(rank9_sel.select(1), Some(3));
    assert_eq!(rank9_sel.select(6), None);
}

#[test]
fn test_empty() {
    let bits = BitVec::new(0);
    let rank9 = Rank9::new(bits);

    assert_eq!(rank9.len(), 0);
    let inner = rank9.into_inner();
    assert_eq!(inner.len(), 0);
}
