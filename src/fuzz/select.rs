/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use crate::prelude::*;
use arbitrary::Arbitrary;
use std::collections::BTreeSet;

#[derive(Arbitrary, Debug)]
pub struct Data {
    /// (the positions of the bits to set to 1) % len
    ones: Vec<usize>,
    /// the length of the bitvec
    len: usize,
}

/// get random data and check that all selects behave correctly
pub fn harness(mut data: Data) {
    data.len %= 1 << 20; // avoid out of memory, 1MB should be enough to find errors
    data.len += 1; // avoid zero length
    let ones = data
        .ones
        .iter()
        .map(|value| value % data.len)
        .collect::<BTreeSet<_>>();

    let mut bitvec = BitVec::new(data.len);
    if data.len != 0 {
        ones.iter().for_each(|value| {
            bitvec.set(*value, true); // set bit to one
        });
    }

    let number_of_ones = bitvec.count_ones();
    let bitvec = unsafe { AddNumBits::from_raw_parts(bitvec, number_of_ones) };

    macro_rules! test_struct {
        ($ty:ty) => {
            let quantum = <$ty>::new(&bitvec);

            for (i, v) in ones.iter().enumerate() {
                assert_eq!(
                    *v,
                    quantum.select(i).unwrap(),
                    "Quantum select is wrong at idx {}",
                    i,
                );
            }
        };
    }
    test_struct!(SelectAdaptConst<_, _, 6>);
    test_struct!(SelectAdaptConst<_, _, 7>);
    test_struct!(SelectAdaptConst<_, _, 8>);
    test_struct!(SelectAdaptConst<_, _, 9>);
    test_struct!(SelectAdaptConst<_, _, 10>);
    test_struct!(SelectAdaptConst<_, _, 11>);
    test_struct!(SelectAdaptConst<_, _, 12>);
    test_struct!(SelectAdaptConst<_, _, 13>);
    test_struct!(SelectAdaptConst<_, _, 14>);
}
