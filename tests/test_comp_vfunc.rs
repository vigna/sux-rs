/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]

use dsi_progress_logger::no_logging;
use sux::func::codec::Huffman;
use sux::func::{CompVFunc, VBuilder};
use sux::traits::TryIntoUnaligned;
use sux::utils::FromCloneableIntoIterator;
use sux::utils::lenders::FromSlice;

fn build_and_check(values: &[usize]) {
    let n = values.len();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64>::try_par_new(&keys, values, no_logging![]).expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_empty() {
    let values: Vec<usize> = vec![];
    let keys: Vec<u64> = vec![];
    let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
    assert!(func.is_empty());
    assert_eq!(func.len(), 0);
}

#[test]
fn test_streaming_construction() {
    // Exercises the lender-based `try_new` path. Uses
    // `FromCloneableIntoIterator` as the key lender and `FromSlice`
    // to wrap the value vector as a lender (mirrors the `-n` mode of
    // the `comp_vfunc` binary) so that both keys and values are
    // consumed one at a time, not stored as a slice by the
    // constructor.
    let n = 1000usize;
    let values: Vec<usize> = (0..n).map(|i| i % 5).collect();
    let func = CompVFunc::<usize>::try_new(
        FromCloneableIntoIterator::from(0..n),
        FromSlice::new(&values),
        no_logging![],
    )
    .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(i), v, "mismatch at key {i}");
    }
}

#[test]
fn test_single_value_distribution() {
    // 100 keys, all mapping to value 7. ZeroCodec-like behavior.
    let values = vec![7; 100];
    build_and_check(&values);
}

#[test]
fn test_skewed_small() {
    // 200 keys with a 3-symbol skewed distribution.
    let mut values = Vec::with_capacity(200);
    for i in 0..200 {
        values.push(match i % 10 {
            0..=6 => 0,
            7 | 8 => 1,
            _ => 2,
        });
    }
    build_and_check(&values);
}

#[test]
fn test_many_keys() {
    // Force a non-trivial number of keys and a moderately skewed
    // distribution. With FuseLge3Shards this is well above the
    // single-shard threshold.
    let n = 5_000usize;
    let values: Vec<usize> = (0..n)
        .map(|i| match i % 16 {
            0..=7 => 0,
            8..=11 => 1,
            12..=13 => 2,
            14 => 3,
            _ => 4,
        })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_string_keys() {
    // Verify the ToSig implementation works with string keys.
    let n = 300usize;
    let keys: Vec<String> = (0..n).map(|i| format!("key-{i:08}")).collect();
    let values: Vec<usize> = (0..n).map(|i| i % 5).collect();
    let func = CompVFunc::<String>::try_par_new(&keys, &values, no_logging![]).expect("build");
    for (k, &v) in keys.iter().zip(values.iter()) {
        assert_eq!(func.get(k), v, "mismatch at {k}");
    }
}

#[test]
fn test_u8_output_type() {
    // Exercise the W type parameter with a narrow output type.
    // Values are in 0..5 so they comfortably fit in a u8.
    let n = 500usize;
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<u8> = (0..n).map(|i| (i % 5) as u8).collect();
    let func = CompVFunc::<u64, u8>::try_par_new(&keys, &values, no_logging![]).expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u128_output_type() {
    // Exercise the W type parameter with large u128 values (all
    // above 2^100). Kept within a handful of distinct symbols so
    // the Huffman table holds them all and no escapes are needed —
    // the literal read path is bounded to 64 bits by the BitVec
    // backend, so u128 only works when escapes fit in a u64.
    let n = 400usize;
    let keys: Vec<u64> = (0..n as u64).collect();
    let base: u128 = 1u128 << 100;
    let values: Vec<u128> = (0..n).map(|i| base + (i % 5) as u128).collect();
    let func = CompVFunc::<u64, u128>::try_par_new(&keys, &values, no_logging![]).expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u8_forced_escape() {
    // Force the escape path with `length_limited(1, ...)`: only one
    // distinct codeword length survives the truncation, so the most
    // frequent symbols get a 1-bit codeword and the rest escape.
    // The escape literal is read back through the W → u64 cast in
    // `encode_val`, which is the only branch where the cast happens.
    let n = 600usize;
    // Strongly skewed: value 0 dominates, 1..=10 are rare. Huffman
    // gives 0 a length-1 codeword; with `max_decoding_table_length =
    // 1` everything else escapes.
    let values: Vec<u8> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { (i % 11) as u8 })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, u8>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.escaped_symbol_length() > 0,
        "test must actually exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u128_forced_escape() {
    // Same forced-escape scenario as `test_u8_forced_escape`, but
    // with u128 values: the escaped literals are stored under the W
    // → u64 cast in `encode_val`, so this proves that the escape
    // path round-trips u128 values whose bit width fits in 64 bits.
    // We pick a small `base` (2^16) to keep the per-shard literal
    // width — and therefore the multi-edge codeword layout —
    // tractable for a unit test.
    let n = 600usize;
    let base: u128 = 1u128 << 16;
    let values: Vec<u128> = (0..n)
        .map(|i| {
            if i % 12 < 9 {
                base
            } else {
                base + (i % 11) as u128
            }
        })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, u128>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.escaped_symbol_length() > 0,
        "test must actually exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_try_into_unaligned() {
    // Build a normal CompVFunc, convert to the unaligned variant, and
    // check that queries still match.
    let n = 1500usize;
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<usize> = (0..n).map(|i| i % 7).collect();
    let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
    let unaligned = func.try_into_unaligned().expect("convert");
    for (k, &v) in keys.iter().zip(values.iter()) {
        assert_eq!(unaligned.get(*k), v, "mismatch at {k}");
    }
    // Round-trip back.
    let back: CompVFunc<u64> = unaligned.into();
    for (k, &v) in keys.iter().zip(values.iter()) {
        assert_eq!(back.get(*k), v, "mismatch at {k} after round-trip");
    }
}

#[test]
fn test_with_escapes() {
    // 16 distinct values with very skewed distribution; the codec
    // should escape the rare ones.
    let mut values: Vec<usize> = Vec::with_capacity(2000);
    for i in 0..2000 {
        // Pareto-ish: most are 0, exponential tail.
        let v = if i % 100 < 50 {
            0
        } else if i % 100 < 75 {
            1
        } else if i % 100 < 88 {
            2
        } else if i % 100 < 94 {
            3
        } else if i % 100 < 97 {
            4
        } else {
            5 + (i % 11)
        };
        values.push(v);
    }
    let keys: Vec<u64> = (0..values.len() as u64).collect();
    let func = CompVFunc::<u64>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(8, 0.95),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_i32_output_type() {
    // Signed i32 values including negatives and extreme values.
    let n = 1000usize;
    let values: Vec<i32> = (0..n)
        .map(|i| match i % 6 {
            0 => -1,
            1 => 0,
            2 => 42,
            3 => -(i as i32),
            4 => i32::MAX,
            _ => i32::MIN,
        })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, i32>::try_par_new(&keys, &values, no_logging![]).expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_i8_with_negatives_and_escapes() {
    // i8 values spanning the full signed range, with escapes forced
    // by a tight length limit.
    let n = 800usize;
    let values: Vec<i8> = (0..n)
        .map(|i| {
            if i % 10 < 7 {
                -1i8 // dominant value
            } else {
                // Rare values spanning the full i8 range.
                (i % 256) as i8
            }
        })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, i8>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(3, 0.7),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u64_forced_escape() {
    // Force escaped storage of full 64-bit values. The dominant value
    // gets a short Huffman codeword; the rare ones span the entire
    // u64 range and must survive the escape + literal round-trip.
    let n = 600usize;
    let rare: [u64; 5] = [
        u64::MAX,
        0xDEAD_BEEF_CAFE_BABE,
        1,
        (1u64 << 63) | 1,
        0x0123_4567_89AB_CDEF,
    ];
    let values: Vec<u64> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, u64>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.escaped_symbol_length() > 0,
        "test must exercise the escape path"
    );
    assert!(
        func.escaped_symbol_length() >= 64,
        "escaped symbol length must cover full 64-bit values, got {}",
        func.escaped_symbol_length()
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}
