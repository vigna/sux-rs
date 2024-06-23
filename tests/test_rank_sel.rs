/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::prelude::*;

#[test]
fn test_rank_sel_rank_zero() {
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
fn test_rank_sel_add_num_bits() {
    let bits = bit_vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
    let a: AddNumBits<_> = bits.clone().into();
    let (b, c) = a.into_raw_parts();
    assert_eq!(b, bits);
    assert_eq!(c, 5);
}
