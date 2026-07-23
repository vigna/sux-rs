/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */
use crate::prelude::*;
use crate::traits::BitVecOps;
use crate::traits::BitVecOpsMut;
use arbitrary::Arbitrary;
use std::collections::BTreeSet;

#[derive(Arbitrary, Debug)]
pub struct Data {
    /// (the positions of the bits to set to 1) % len
    ones: Vec<usize>,
    /// the length of the bitvec
    len: usize,
}

/// Get random data and check that every ones-select structure agrees with the
/// oracle on `select` (i-th one) and every zeros-select structure agrees on
/// `select_zero` (i-th zero).
///
/// `len` is only capped at `2^20` (not scaled by the number of ones), so a
/// `Data` with a large `len` and few `ones` produces the sparse, wide-span
/// bit vectors that exercise the adaptive builders' U64-span and spill paths.
pub fn harness(mut data: Data) {
    data.len %= 1 << 20; // avoid out of memory, 1Mbit is enough to find errors
    data.len += 1; // avoid zero length
    let ones = data
        .ones
        .iter()
        .map(|value| value % data.len)
        .collect::<BTreeSet<_>>();

    // usize-backed bit vector: the adaptive selects are built over a reference
    // to it, so a single copy serves every adaptive/const variant.
    let mut bitvec = BitVec::<Vec<usize>>::new(data.len);
    ones.iter().for_each(|value| {
        bitvec.set(*value, true);
    });
    let number_of_ones = bitvec.count_ones();
    // SAFETY: AddNumBits::from_raw_parts requires the supplied count to equal
    // the number of ones in the bit vector; number_of_ones was just computed
    // as bitvec.count_ones() on this exact value, so the invariant holds.
    let bitvec = unsafe { AddNumBits::from_raw_parts(bitvec, number_of_ones) };

    // u64-backed copy: Select9 and SelectSmall wrap rank backends that own the
    // bit vector, so they cannot borrow the shared bitvec above.
    let mut bitvec64 = BitVec::<Vec<u64>>::new(data.len);
    ones.iter().for_each(|value| {
        bitvec64.set(*value, true);
    });

    // Oracles: positions of the ones (ascending) and of the zeros (ascending).
    let ones_pos = ones.iter().copied().collect::<Vec<_>>();
    let zeros_pos = (0..data.len)
        .filter(|pos| !ones.contains(pos))
        .collect::<Vec<_>>();

    // select(i) must return the position of the i-th one.
    macro_rules! test_select {
        ($struct:expr) => {{
            let sel = $struct;
            for (i, &pos) in ones_pos.iter().enumerate() {
                assert_eq!(sel.select(i).unwrap(), pos, "select({i}) is wrong");
            }
            assert_eq!(sel.select(ones_pos.len()), None);
            assert_eq!(sel.select(usize::MAX), None);
        }};
    }
    test_select!(SelectAdaptConst::<_, _, 6>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 7>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 8>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 9>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 10>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 11>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 12>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 13>::new(&bitvec));
    test_select!(SelectAdaptConst::<_, _, 14>::new(&bitvec));
    // A minimal subinventory (log2 = 0) forces the spill path.
    test_select!(SelectAdaptConst::<_, _, 13, 0>::new(&bitvec));
    test_select!(SelectAdapt::new(&bitvec));
    test_select!(Select9::new(Rank9::new(bitvec64.clone())));
    test_select!(SelectSmall::<2, 9, _>::new(RankSmall::<64, 2, 9, _>::new(
        bitvec64.clone()
    )));

    // select_zero(i) must return the position of the i-th zero.
    macro_rules! test_select_zero {
        ($struct:expr) => {{
            let sel = $struct;
            for (i, &pos) in zeros_pos.iter().enumerate() {
                assert_eq!(
                    sel.select_zero(i).unwrap(),
                    pos,
                    "select_zero({i}) is wrong"
                );
            }
            assert_eq!(sel.select_zero(zeros_pos.len()), None);
            assert_eq!(sel.select_zero(usize::MAX), None);
        }};
    }
    test_select_zero!(SelectZeroAdaptConst::<_, _, 10>::new(&bitvec));
    test_select_zero!(SelectZeroAdaptConst::<_, _, 13>::new(&bitvec));
    // A minimal subinventory (log2 = 0) forces the spill path.
    test_select_zero!(SelectZeroAdaptConst::<_, _, 13, 0>::new(&bitvec));
    test_select_zero!(SelectZeroAdapt::new(&bitvec));
    test_select_zero!(SelectZeroSmall::<2, 9, _>::new(
        RankSmall::<64, 2, 9, _>::new(bitvec64.clone())
    ));
}

#[cfg(test)]
mod tests {
    use super::{Data, harness};

    /// Drive the harness over hand-picked shapes so a regression in any
    /// covered select structure surfaces without a full fuzz campaign.
    #[test]
    fn test_harness_covers_edge_shapes() {
        // All zeros: every position is a zero, no ones.
        harness(Data {
            ones: vec![],
            len: 1000,
        });
        // Near-all ones: 0..=998 set, one trailing zero.
        harness(Data {
            ones: (0..999).collect(),
            len: 999,
        });
        // Sparse wide span: three far-apart ones over ~200k bits exercise the
        // adaptive builders' wide-span and spill paths on both sides.
        harness(Data {
            ones: vec![0, 100_000, 199_999],
            len: 200_000,
        });
        // A denser irregular pattern.
        harness(Data {
            ones: vec![1, 2, 3, 64, 65, 4095, 4096, 8191, 12000],
            len: 12_345,
        });
    }
}
