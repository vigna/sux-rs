/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]
use anyhow::Result;
use dsi_progress_logger::no_logging;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::io::Cursor;
use sux::func::{Lcp2MmphfInt, Lcp2MmphfStr};
use sux::utils::{FromSlice, LineLender};

#[test]
fn test_small_u64() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: Lcp2MmphfInt<u64> =
        Lcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_monotone_1000_u64() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();
    let func: Lcp2MmphfInt<u64> = Lcp2MmphfInt::try_new(FromSlice::new(&keys), n, no_logging![])?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_signed_i64() -> Result<()> {
    let keys: Vec<i64> = vec![-100, -10, -1, 0, 1, 10, 100];
    let func: Lcp2MmphfInt<i64> =
        Lcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_empty_u64() -> Result<()> {
    let keys: Vec<u64> = vec![];
    let func: Lcp2MmphfInt<u64> = Lcp2MmphfInt::try_new(FromSlice::new(&keys), 0, no_logging![])?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    Ok(())
}

fn keys_lender(keys: &[&str]) -> LineLender<Cursor<Vec<u8>>> {
    let data = keys.join("\n");
    LineLender::new(Cursor::new(data.into_bytes()))
}

#[test]
fn test_small_str() -> Result<()> {
    let mut keys = vec!["alpha", "beta", "delta", "gamma"];
    keys.sort();
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_new(keys_lender(&keys), keys.len(), no_logging![])?;
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_str_1000() -> Result<()> {
    let mut keys: Vec<String> = (0..1000).map(|i| format!("key_{:06}", i)).collect();
    keys.sort();
    let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_new(keys_lender(&refs), refs.len(), no_logging![])?;
    for (i, key) in refs.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_empty_str() -> Result<()> {
    let keys: Vec<&str> = vec![];
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_new(keys_lender(&keys), 0, no_logging![])?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    Ok(())
}

#[test]
fn test_slice_u8() -> Result<()> {
    use sux::func::Lcp2MmphfSliceU8;
    let keys: Vec<Vec<u8>> = vec![
        b"alpha".to_vec(),
        b"beta".to_vec(),
        b"delta".to_vec(),
        b"gamma".to_vec(),
    ];
    let func: Lcp2MmphfSliceU8 =
        Lcp2MmphfSliceU8::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_slice()), i);
    }
    Ok(())
}
