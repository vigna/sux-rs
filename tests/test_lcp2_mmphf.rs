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
use sux::traits::TryIntoUnaligned;
use sux::utils::{FromSlice, LineLender};

#[test]
fn test_small_u64() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func = Lcp2MmphfInt::<u64>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?
        .try_into_unaligned()
        .unwrap();
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
    let func = Lcp2MmphfInt::<u64>::try_new(FromSlice::new(&keys), n, no_logging![])?
        .try_into_unaligned()
        .unwrap();
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_signed_i64() -> Result<()> {
    let keys: Vec<i64> = vec![-100, -10, -1, 0, 1, 10, 100];
    let func = Lcp2MmphfInt::<i64>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?
        .try_into_unaligned()
        .unwrap();
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_empty_u64() -> Result<()> {
    let keys: Vec<u64> = vec![];
    let func = Lcp2MmphfInt::<u64>::try_new(FromSlice::new(&keys), 0, no_logging![])?
        .try_into_unaligned()
        .unwrap();
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
    let func = func.try_into_unaligned().unwrap();
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
    let func = func.try_into_unaligned().unwrap();
    for (i, key) in refs.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_empty_str() -> Result<()> {
    let keys: Vec<&str> = vec![];
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_new(keys_lender(&keys), 0, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
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
    let func = func.try_into_unaligned().unwrap();
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_slice()), i);
    }
    Ok(())
}

// ── Parallel constructors ───────────────────────────────────────────

#[test]
fn test_par_small_u64() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: Lcp2MmphfInt<u64> = Lcp2MmphfInt::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_par_monotone_1000_u64() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let func: Lcp2MmphfInt<u64> = Lcp2MmphfInt::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_par_signed_i64() -> Result<()> {
    let keys: Vec<i64> = vec![-100, -10, -1, 0, 1, 10, 100];
    let func: Lcp2MmphfInt<i64> = Lcp2MmphfInt::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_par_empty_u64() -> Result<()> {
    let keys: Vec<u64> = vec![];
    let func: Lcp2MmphfInt<u64> = Lcp2MmphfInt::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    Ok(())
}

#[test]
fn test_par_small_str() -> Result<()> {
    let mut keys = vec!["alpha", "beta", "delta", "gamma"];
    keys.sort();
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_par_str_1000() -> Result<()> {
    let mut keys: Vec<String> = (0..1000).map(|i| format!("key_{:06}", i)).collect();
    keys.sort();
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_str()), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_par_empty_str() -> Result<()> {
    let keys: Vec<&str> = vec![];
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    Ok(())
}

#[test]
fn test_par_slice_u8() -> Result<()> {
    use sux::func::Lcp2MmphfSliceU8;
    let keys: Vec<Vec<u8>> = vec![
        b"alpha".to_vec(),
        b"beta".to_vec(),
        b"delta".to_vec(),
        b"gamma".to_vec(),
    ];
    let func: Lcp2MmphfSliceU8 = Lcp2MmphfSliceU8::try_par_new(&keys, no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_slice()), i);
    }
    Ok(())
}

/// Querying absent keys must never index the `remap` table out of bounds. The
/// first VFunc can return any frequent-LCP index in `[0, escape)`, but
/// construction previously sized `remap` to only the frequent lengths, so an
/// absent key landing in `[num_freq, escape)` panicked in safe `get`.
#[test]
fn test_absent_keys_do_not_panic() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..2000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();
    let func = Lcp2MmphfInt::<u64>::try_new(FromSlice::new(&keys), n, no_logging![])?
        .try_into_unaligned()
        .unwrap();
    let present: std::collections::HashSet<u64> = keys.iter().copied().collect();
    let mut probe = SmallRng::seed_from_u64(12345);
    let mut checked = 0u64;
    for _ in 0..100_000 {
        let k = probe.random::<u64>();
        if present.contains(&k) {
            continue;
        }
        // The result is arbitrary for an absent key, but it must not panic.
        std::hint::black_box(func.get(k));
        checked += 1;
    }
    assert!(checked > 0);
    Ok(())
}

/// A common prefix longer than 65535 bits must not wrap the stored LCP length.
/// On 64-bit `LcpLen` is already u32, so this is a smoke test here; on 32-bit
/// (i686 CI) it fails before the u16 -> u32 widening.
#[test]
fn test_long_common_prefix_str() -> Result<()> {
    // 8200 bytes = 65600 bits of shared prefix, then a differing byte.
    let prefix = "a".repeat(8200);
    let a = format!("{prefix}0");
    let b = format!("{prefix}1");
    let mut keys = vec![a.as_str(), b.as_str()];
    keys.sort();
    let func: Lcp2MmphfStr = Lcp2MmphfStr::try_new(keys_lender(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned().unwrap();
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "long-prefix key at position {i}");
    }
    Ok(())
}

#[test]
fn test_key_count_mismatch() {
    // Sorted keys but n larger than the actual count: the constructor must
    // return Err (the count assert is now a recoverable error), not panic.
    let keys: Vec<u64> = vec![100, 200];
    let result: Result<Lcp2MmphfInt<u64>> =
        Lcp2MmphfInt::try_new(FromSlice::new(&keys), 3, no_logging![]);
    assert!(result.is_err());
}
