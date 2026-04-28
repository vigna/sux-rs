/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]

use dsi_progress_logger::no_logging;
use sux::bits::BitVec;
use sux::func::codec::{Decoder, Huffman};
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
    let values = vec![7; 100];
    build_and_check(&values);
}

#[test]
fn test_skewed_small() {
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
    let n = 300usize;
    let keys: Vec<String> = (0..n).map(|i| format!("key-{i:08}")).collect();
    let values: Vec<usize> = (0..n).map(|i| i % 5).collect();
    let func = CompVFunc::<String>::try_par_new(&keys, &values, no_logging![]).expect("build");
    for (k, &v) in keys.iter().zip(values.iter()) {
        assert_eq!(func.get(k), v, "mismatch at {k}");
    }
}

#[test]
fn test_try_into_unaligned() {
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
    let mut values: Vec<usize> = Vec::with_capacity(2000);
    for i in 0..2000 {
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
fn test_forced_escape() {
    // Force the escape path with `length_limited(1, ...)`: only one
    // distinct codeword length survives the truncation, so the most
    // frequent symbols get a 1-bit codeword and the rest escape.
    let n = 600usize;
    let values: Vec<usize> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { i % 11 })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must actually exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_large_values_forced_escape() {
    // Force escaped storage of values spanning the full usize range.
    let n = 600usize;
    let rare: [usize; 5] = [
        usize::MAX,
        0xDEAD_BEEF_CAFE_BABE_u64 as usize,
        1,
        (1usize << (usize::BITS - 1)) | 1,
        0x0123_4567_89AB_CDEF_u64 as usize,
    ];
    let values: Vec<usize> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
    assert!(
        func.try_into_unaligned().is_err(),
        "try_into_unaligned must fail for full-range usize escaped values"
    );
}

// ── Tests with u16 backend word type ──────────────────────────────

#[test]
fn test_u16_basic() {
    let n = 500usize;
    let values: Vec<u16> = (0..n as u16).map(|i| i % 7).collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u16]>>>::try_par_new(&keys, &values, no_logging![])
        .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u16_full_range_forced_escape() {
    let n = 600usize;
    let rare: [u16; 6] = [u16::MAX, 0x8000, 0x8001, 0xFFFF, 0xDEAD, 0xBEEF];
    let values: Vec<u16> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u16]>>>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
    assert!(
        func.try_into_unaligned().is_err(),
        "try_into_unaligned must fail for full-range u16 escaped values"
    );
}

// ── Tests with u32 backend word type ──────────────────────────────

#[test]
fn test_u32_basic() {
    let n = 500usize;
    let values: Vec<u32> = (0..n as u32).map(|i| i % 7).collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u32]>>>::try_par_new(&keys, &values, no_logging![])
        .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u32_full_range_forced_escape() {
    let n = 600usize;
    let rare: [u32; 6] = [
        u32::MAX,
        0x8000_0000,
        0x8000_0001,
        0xFFFF_FFFF,
        0xDEAD_BEEF,
        0x8765_4321,
    ];
    let values: Vec<u32> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u32]>>>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
    assert!(
        func.try_into_unaligned().is_err(),
        "try_into_unaligned must fail for full-range u32 escaped values"
    );
}

#[test]
fn test_u32_try_into_unaligned() {
    let n = 1500usize;
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<u32> = (0..n as u32).map(|i| i % 7).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u32]>>>::try_par_new(&keys, &values, no_logging![])
        .expect("build");
    let unaligned = func.try_into_unaligned().expect("convert");
    for (k, &v) in keys.iter().zip(values.iter()) {
        assert_eq!(unaligned.get(*k), v, "mismatch at {k}");
    }
}

// ── Tests with u64 backend word type ──────────────────────────────

#[test]
fn test_u64_basic() {
    let n = 500usize;
    let values: Vec<u64> = (0..n as u64).map(|i| i % 5).collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u64]>>>::try_par_new(&keys, &values, no_logging![])
        .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u64_full_range_forced_escape() {
    let n = 600usize;
    let rare: [u64; 6] = [
        u64::MAX,
        1u64 << 63,
        (1u64 << 63) | 1,
        0xDEAD_BEEF_CAFE_BABE,
        0x0123_4567_89AB_CDEF,
        0xFFFF_FFFF_FFFF_FFFF,
    ];
    let values: Vec<u64> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u64]>>>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
    assert!(
        func.try_into_unaligned().is_err(),
        "try_into_unaligned must fail for full-range u64 escaped values"
    );
}

// ── Tests with u128 backend word type ─────────────────────────────

#[test]
fn test_u128_basic() {
    let n = 500usize;
    let values: Vec<u128> = (0..n as u128).map(|i| i % 5).collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u128]>>>::try_par_new(&keys, &values, no_logging![])
        .expect("build");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
}

#[test]
fn test_u128_full_range_forced_escape() {
    let n = 600usize;
    let rare: [u128; 6] = [
        u128::MAX,
        1u128 << 127,
        (1u128 << 127) | 1,
        0xDEAD_BEEF_CAFE_BABE_0123_4567_89AB_CDEF,
        u128::MAX - 1,
        0xAAAA_AAAA_AAAA_AAAA_5555_5555_5555_5555,
    ];
    let values: Vec<u128> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64, BitVec<Box<[u128]>>>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
    assert!(
        func.try_into_unaligned().is_err(),
        "try_into_unaligned must fail for full-range u128 escaped values"
    );
}

// ── Tests with usize full-range values (MSB set) ─────────────────

#[test]
fn test_usize_msb_values() {
    let n = 600usize;
    let rare: [usize; 6] = [
        usize::MAX,
        1usize << (usize::BITS - 1),
        (1usize << (usize::BITS - 1)) | 0x42,
        usize::MAX - 1,
        0xAAAA_AAAA_AAAA_AAAA_u64 as usize,
        0x5555_5555_5555_5555_u64 as usize,
    ];
    let values: Vec<usize> = (0..n)
        .map(|i| if i % 12 < 9 { 0 } else { rare[i % rare.len()] })
        .collect();
    let keys: Vec<u64> = (0..n as u64).collect();
    let func = CompVFunc::<u64>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::length_limited(1, 1.0),
        VBuilder::default(),
        no_logging![],
    )
    .expect("build");
    assert!(
        func.decoder().escaped_symbols_len() > 0,
        "test must exercise the escape path"
    );
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
    }
    assert!(
        func.try_into_unaligned().is_err(),
        "try_into_unaligned must fail for full-range usize escaped values"
    );
}
