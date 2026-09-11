/*
 *
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Tests that accessing an [rkyv]-archived Elias–Fano structure through
//! the lazy view gives the same results as accessing the in-memory structure.
//!
//! [rkyv]: https://crates.io/crates/rkyv

#![cfg(feature = "rkyv")]

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::dict::elias_fano::ArchivedEliasFano;
use sux::prelude::*;
use sux::traits::{IndexedSeq, Pred, PredUnchecked, Succ, SuccUnchecked, TryIntoUnaligned};

/// The high bits of the structures under test, with the same const parameters
/// used by the Elias–Fano benchmarks.
type High = SelectZeroAdaptConst<
    SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>,
    Box<[usize]>,
    12,
    3,
>;

type EfAligned = EliasFano<u64, High>;
type EfUnaligned = EliasFano<u64, High, BitFieldVecU<Box<[u64]>>>;

type ArchivedEfAligned = ArchivedEliasFano<u64, High, BitFieldVec<Box<[u64]>>>;
type ArchivedEfUnaligned = ArchivedEliasFano<u64, High, BitFieldVecU<Box<[u64]>>>;

/// Builds an Elias–Fano structure over `n` random values below `2^l * n`,
/// returning it together with the values.
fn build(n: usize, l: usize, seed: u64) -> (EfAligned, Vec<u64>) {
    let u = (1u64 << l) * n as u64;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
    values.sort_unstable();

    let mut builder = EliasFanoBuilder::new(n, u);
    for &v in &values {
        builder.push(v);
    }

    let ef = unsafe {
        builder
            .build()
            .map_high_bits(|h| SelectZeroAdaptConst::new(SelectAdaptConst::new(h)))
    };
    (ef, values)
}

/// Checks that `archived` answers every query exactly as `native` does.
macro_rules! check {
    ($native:expr, $archived:expr, $values:expr, $n:expr, $l:expr) => {{
        let native = &$native;
        let archived = $archived;
        let values: &[u64] = &$values;
        let n = $n;
        let u = (1u64 << $l) * n as u64;

        assert_eq!(IndexedSeq::len(archived), IndexedSeq::len(native));

        for i in 0..n {
            assert_eq!(
                IndexedSeq::get(archived, i),
                IndexedSeq::get(native, i),
                "get({i})"
            );
            assert_eq!(values[i], IndexedSeq::get(archived, i), "value {i}");
            assert_eq!(
                unsafe { IndexedSeq::get_unchecked(archived, i) },
                unsafe { IndexedSeq::get_unchecked(native, i) },
                "get_unchecked({i})"
            );
        }

        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..(4 * n) {
            let v = rng.random_range(0..u);
            assert_eq!(Succ::succ(archived, v), Succ::succ(native, v), "succ({v})");
            assert_eq!(
                Succ::succ_strict(archived, v),
                Succ::succ_strict(native, v),
                "succ_strict({v})"
            );
            assert_eq!(Pred::pred(archived, v), Pred::pred(native, v), "pred({v})");
            assert_eq!(
                Pred::pred_strict(archived, v),
                Pred::pred_strict(native, v),
                "pred_strict({v})"
            );
            assert_eq!(Pred::rank(archived, v), Pred::rank(native, v), "rank({v})");

            // Unchecked queries require the result to exist.
            if v >= values[0] {
                assert_eq!(
                    unsafe { PredUnchecked::pred_unchecked::<false>(archived, v) },
                    unsafe { PredUnchecked::pred_unchecked::<false>(native, v) },
                    "pred_unchecked({v})"
                );
            }
            if v > values[0] {
                assert_eq!(
                    unsafe { PredUnchecked::rank_unchecked(archived, v) },
                    unsafe { PredUnchecked::rank_unchecked(native, v) },
                    "rank_unchecked({v})"
                );
            }
            if v <= values[n - 1] {
                assert_eq!(
                    unsafe { SuccUnchecked::succ_unchecked::<false>(archived, v) },
                    unsafe { SuccUnchecked::succ_unchecked::<false>(native, v) },
                    "succ_unchecked({v})"
                );
            }
        }
    }};
}

#[test]
fn test_archived_aligned() {
    for (n, l) in [(1_usize, 2_usize), (2, 3), (100, 2), (1000, 8), (5000, 16)] {
        let (ef, values) = build(n, l, n as u64);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&ef).unwrap();
        // SAFETY: the bytes were just produced by serializing an `EfAligned`.
        let archived = unsafe { rkyv::access_unchecked::<ArchivedEfAligned>(&bytes) };
        check!(ef, archived, values, n, l);
    }
}

#[test]
fn test_archived_unaligned() {
    for (n, l) in [(1_usize, 2_usize), (2, 3), (100, 2), (1000, 8), (5000, 16)] {
        let (ef, values) = build(n, l, n as u64);
        let ef: EfUnaligned = ef.try_into_unaligned().unwrap();
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&ef).unwrap();
        // SAFETY: the bytes were just produced by serializing an `EfUnaligned`.
        let archived = unsafe { rkyv::access_unchecked::<ArchivedEfUnaligned>(&bytes) };
        check!(ef, archived, values, n, l);
    }
}

/// Checks that the native view borrows from the archive rather than copying it.
#[test]
fn test_view_borrows_archive() {
    let (ef, _) = build(1000, 4, 7);
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&ef).unwrap();
    // SAFETY: the bytes were just produced by serializing an `EfAligned`.
    let archived = unsafe { rkyv::access_unchecked::<ArchivedEfAligned>(&bytes) };
    let native = archived.lazy();

    let start = bytes.as_ptr() as usize;
    let end = start + bytes.len();
    let (_, _, _, _, high_bits, _, _) = native.into_parts();
    let high = high_bits.as_ref().as_ptr() as usize;
    assert!(
        (start..end).contains(&high),
        "the high bits should point inside the archive"
    );
}
