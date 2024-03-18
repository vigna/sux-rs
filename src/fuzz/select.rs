/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use crate::prelude::*;
use arbitrary::Arbitrary;

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
    let mut bitvec = BitVec::new(data.len);
    if data.len != 0 {
        data.ones.iter_mut().for_each(|value| {
            *value %= data.len; // make them valid
            bitvec.set(*value, true); // set bit to one
        });
    }

    let number_of_ones = bitvec.count_ones();

    let quantum = <SelectFixed1<_, _, 8>>::new(&bitvec, number_of_ones);
    let simple = <SelectFixed2<_, _, 10, 2>>::new(&bitvec);

    for i in 0..number_of_ones {
        assert_eq!(
            bitvec.select(i),
            quantum.select(i),
            "Quantum select is wrong at idx {}",
            i,
        );
        assert_eq!(
            bitvec.select(i),
            simple.select(i),
            "Simple select is wrong at idx {}",
            i,
        );
    }
}
