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
use sux::bits::BitFieldVec;
use sux::func::{BitSignedFunc, LcpMmphf, LcpMmphfInt, SignedFunc};
use sux::traits::TryIntoUnaligned;
use sux::utils::FromSlice;

type SignedLcpMmphfStr = SignedFunc<LcpMmphf<str>, Box<[u64]>>;
type SignedLcpMmphfSliceU8 = SignedFunc<LcpMmphf<[u8]>, Box<[u64]>>;
type SignedLcpMmphfInt<T> = SignedFunc<LcpMmphfInt<T>, Box<[u64]>>;
type BitSignedLcpMmphfInt<T> = BitSignedFunc<LcpMmphfInt<T>, BitFieldVec<Box<[usize]>>>;
type BitSignedLcpMmphfStr = BitSignedFunc<LcpMmphf<str>, BitFieldVec<Box<[usize]>>>;

// ── String tests ────────────────────────────────────────────────────

#[test]
fn test_str_small() -> Result<()> {
    let keys = vec![
        "alpha".to_owned(),
        "beta".to_owned(),
        "delta".to_owned(),
        "gamma".to_owned(),
    ];
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            func.get(key.as_str()),
            Some(i),
            "key {key:?} at position {i}"
        );
    }
    assert_eq!(func.get("not_a_key"), None);
    assert_eq!(func.get(""), None);
    Ok(())
}

#[test]
fn test_str_single() -> Result<()> {
    let keys = vec!["hello".to_owned()];
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), 1, no_logging![])?;
    let func = func.try_into_unaligned()?;
    assert_eq!(func.get("hello"), Some(0));
    assert_eq!(func.get("world"), None);
    assert_eq!(func.len(), 1);
    assert!(!func.is_empty());
    Ok(())
}

#[test]
fn test_str_empty() -> Result<()> {
    let keys: Vec<String> = vec![];
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), 0, no_logging![])?;
    let func = func.try_into_unaligned()?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    assert_eq!(func.get("anything"), None);
    Ok(())
}

#[test]
fn test_str_1000() -> Result<()> {
    let mut keys: Vec<String> = (0..1000).map(|i| format!("key_{:06}", i)).collect();
    keys.sort();
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            func.get(key.as_str()),
            Some(i),
            "key {key:?} at position {i}"
        );
    }
    assert_eq!(func.get("zzz_not_present"), None);
    assert_eq!(func.get("key_999999"), None);
    Ok(())
}

// ── [u8] tests ──────────────────────────────────────────────────────

#[test]
fn test_slice_u8_small() -> Result<()> {
    let keys: Vec<Vec<u8>> = vec![
        b"alpha".to_vec(),
        b"beta".to_vec(),
        b"delta".to_vec(),
        b"gamma".to_vec(),
    ];
    let func: SignedLcpMmphfSliceU8 =
        SignedLcpMmphfSliceU8::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_slice()), Some(i));
    }
    assert_eq!(func.get(b"not_a_key".as_slice()), None);
    Ok(())
}

// ── Integer tests ───────────────────────────────────────────────────

#[test]
fn test_u64_small() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: SignedLcpMmphfInt<u64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(999), None);
    assert_eq!(func.get(0), None);
    Ok(())
}

#[test]
fn test_u64_empty() -> Result<()> {
    let keys: Vec<u64> = vec![];
    let func: SignedLcpMmphfInt<u64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), 0, no_logging![])?;
    let func = func.try_into_unaligned()?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    assert_eq!(func.get(42), None);
    Ok(())
}

#[test]
fn test_u64_1000() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();

    let func: SignedLcpMmphfInt<u64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), n, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(0), None);
    Ok(())
}

#[test]
fn test_signed_i64() -> Result<()> {
    let keys: Vec<i64> = vec![-100, -10, -1, 0, 1, 10, 100];
    let func: SignedLcpMmphfInt<i64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(999), None);
    assert_eq!(func.get(-999), None);
    Ok(())
}

#[test]
fn test_u32() -> Result<()> {
    let keys: Vec<u32> = vec![1, 100, 1000, 10000, 100000];
    let func: SignedLcpMmphfInt<u32> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(42), None);
    Ok(())
}

// ── BitSigned tests (sub-word-width hashes) ─────────────────────────

#[test]
fn test_bit_signed_u64_8bit() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: BitSignedLcpMmphfInt<u64> =
        BitSignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(999), None);
    Ok(())
}

#[test]
fn test_bit_signed_str_12bit() -> Result<()> {
    let keys = vec![
        "alpha".to_owned(),
        "beta".to_owned(),
        "delta".to_owned(),
        "gamma".to_owned(),
    ];
    let func: BitSignedLcpMmphfStr =
        BitSignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), 12, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            func.get(key.as_str()),
            Some(i),
            "key {key:?} at position {i}"
        );
    }
    assert_eq!(func.get("not_a_key"), None);
    Ok(())
}

#[test]
fn test_bit_signed_u64_1000() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();

    let func: BitSignedLcpMmphfInt<u64> =
        BitSignedLcpMmphfInt::try_new(FromSlice::new(&keys), n, 16, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(0), None);
    Ok(())
}
