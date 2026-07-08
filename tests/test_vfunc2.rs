/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::func::VFunc2;

const NUM_SHARDED_KEYS: usize = 100_000;

fn vfunc2_value(i: usize) -> usize {
    if i % 5 == 0 {
        10_000 + i % 4_093
    } else {
        i % 11
    }
}

#[test]
fn test_parallel_sharded_vfunc2_preserves_store() -> Result<()> {
    let keys: Vec<u64> = (0..NUM_SHARDED_KEYS)
        .map(u64::try_from)
        .collect::<std::result::Result<_, _>>()?;
    let values: Vec<usize> = (0..NUM_SHARDED_KEYS).map(vfunc2_value).collect();

    let func: VFunc2<u64, BitFieldVec<Box<[usize]>>> =
        VFunc2::try_par_new(&keys, &values, no_logging![])?;

    for (&key, &expected) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), expected, "key {key}");
    }

    Ok(())
}

/// Two `u128` values sharing their low 64 bits but differing in the high bits
/// must be stored distinctly. Before the HybridMap key-truncation fix these
/// collided on the same flat-array slot, so half the keys returned the wrong
/// value.
#[test]
fn test_vfunc2_wide_u128_values() -> Result<()> {
    let low: u128 = 5;
    let high: u128 = (1u128 << 64) | 5; // same low 64 bits as `low`
    let n = 2000usize;
    let keys: Vec<u64> = (0..n)
        .map(u64::try_from)
        .collect::<std::result::Result<_, _>>()?;
    let values: Vec<u128> = (0..n)
        .map(|i| if i % 2 == 0 { low } else { high })
        .collect();

    let func: VFunc2<u64, BitFieldVec<Box<[u128]>>> =
        VFunc2::try_par_new(&keys, &values, no_logging![])?;
    assert!(func.validate().is_ok(), "wide-u128 VFunc2 must validate");

    for (&key, &expected) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), expected, "key {key}");
    }
    Ok(())
}

/// A uniform distribution over several wide values (some differing only in
/// their high bits) builds and queries correctly.
#[test]
fn test_vfunc2_uniform_u128() -> Result<()> {
    let distinct: [u128; 6] = [
        0,
        1,
        1u128 << 64,
        (1u128 << 64) | 1,
        1u128 << 100,
        (1u128 << 100) | 7,
    ];
    let n = 5000usize;
    let keys: Vec<u64> = (0..n)
        .map(u64::try_from)
        .collect::<std::result::Result<_, _>>()?;
    let values: Vec<u128> = (0..n).map(|i| distinct[i % distinct.len()]).collect();

    let func: VFunc2<u64, BitFieldVec<Box<[u128]>>> =
        VFunc2::try_par_new(&keys, &values, no_logging![])?;

    for (&key, &expected) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), expected, "key {key}");
    }
    Ok(())
}

/// Build and query VFunc2 over the narrow value backends (`u32` and `u64`),
/// covering the value widths below `u128`.
#[test]
fn test_vfunc2_narrow_value_widths() -> Result<()> {
    let n = 3000usize;
    let keys: Vec<u64> = (0..n)
        .map(u64::try_from)
        .collect::<std::result::Result<_, _>>()?;

    // u32 backend, skewed values.
    let v32: Vec<u32> = keys
        .iter()
        .map(|&k| {
            if k % 4 == 0 {
                1_000_000
            } else {
                u32::try_from(k % 7).unwrap()
            }
        })
        .collect();
    let f32: VFunc2<u64, BitFieldVec<Box<[u32]>>> =
        VFunc2::try_par_new(&keys, &v32, no_logging![])?;
    for (&key, &expected) in keys.iter().zip(&v32) {
        assert_eq!(f32.get(key), expected, "u32 key {key}");
    }

    // u64 backend, roughly uniform values.
    let v64: Vec<u64> = keys.iter().map(|&k| (k % 5) * 1_000_000_000).collect();
    let f64: VFunc2<u64, BitFieldVec<Box<[u64]>>> =
        VFunc2::try_par_new(&keys, &v64, no_logging![])?;
    for (&key, &expected) in keys.iter().zip(&v64) {
        assert_eq!(f64.get(key), expected, "u64 key {key}");
    }
    Ok(())
}
