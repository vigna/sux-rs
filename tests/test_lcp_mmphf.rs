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
use sux::func::shard_edge::FuseLge3Shards;
use sux::func::{LcpMmphfInt, LcpMmphfStr};
use sux::utils::{FromSlice, LineLender};

/// Helper: build a `LineLender` from a slice of sorted string keys.
fn keys_lender(keys: &[&str]) -> LineLender<Cursor<Vec<u8>>> {
    let data = keys.join("\n");
    LineLender::new(Cursor::new(data.into_bytes()))
}

#[test]
fn test_small() -> Result<()> {
    let mut keys = vec!["alpha", "beta", "delta", "gamma"];
    keys.sort();
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(
        keys_lender(&keys),
        keys.len(),
        no_logging![],
    )?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_single() -> Result<()> {
    let keys = vec!["hello"];
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(keys_lender(&keys), 1, no_logging![])?;
    assert_eq!(func.get("hello"), 0);
    assert_eq!(func.len(), 1);
    assert!(!func.is_empty());
    Ok(())
}

#[test]
fn test_two_keys() -> Result<()> {
    let keys = vec!["aaa", "bbb"];
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(keys_lender(&keys), 2, no_logging![])?;
    assert_eq!(func.get("aaa"), 0);
    assert_eq!(func.get("bbb"), 1);
    Ok(())
}

#[test]
fn test_monotone_1000() -> Result<()> {
    let mut keys: Vec<String> = (0..1000).map(|i| format!("key_{:06}", i)).collect();
    keys.sort();
    let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(
        keys_lender(&refs),
        refs.len(),
        no_logging![],
    )?;

    for (i, key) in refs.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_common_prefixes() -> Result<()> {
    let keys: Vec<String> = (0..200)
        .map(|i| format!("very_long_common_prefix_{:04}", i))
        .collect();
    let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(
        keys_lender(&refs),
        refs.len(),
        no_logging![],
    )?;

    for (i, key) in refs.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_diverse_keys() -> Result<()> {
    let mut keys = vec![
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli",
        "vanilla",
        "watermelon",
        "ximenia",
        "yuzu",
        "zucchini",
    ];
    keys.sort();
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(
        keys_lender(&keys),
        keys.len(),
        no_logging![],
    )?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_unsorted_error() {
    let keys = vec!["beta", "alpha"];
    let result = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(keys_lender(&keys), 2, no_logging![]);
    assert!(result.is_err());
}

#[test]
fn test_small_u64() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func = LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(
        FromSlice::new(&keys),
        keys.len(),
        no_logging![],
    )?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_single_u64() -> Result<()> {
    let keys: Vec<u64> = vec![42];
    let func =
        LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(FromSlice::new(&keys), 1, no_logging![])?;
    assert_eq!(func.get(42), 0);
    assert_eq!(func.len(), 1);
    Ok(())
}

#[test]
fn test_two_u64() -> Result<()> {
    let keys: Vec<u64> = vec![100, 200];
    let func =
        LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(FromSlice::new(&keys), 2, no_logging![])?;
    assert_eq!(func.get(100), 0);
    assert_eq!(func.get(200), 1);
    Ok(())
}

#[test]
fn test_monotone_1000_u64() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();

    let func =
        LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(FromSlice::new(&keys), n, no_logging![])?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_dense_u64() -> Result<()> {
    // Consecutive integers — minimal bit-level LCP differences.
    let keys: Vec<u64> = (1000..1200).collect();
    let func = LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(
        FromSlice::new(&keys),
        keys.len(),
        no_logging![],
    )?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_sparse_u64() -> Result<()> {
    // Widely spaced values.
    let keys: Vec<u64> = vec![1, 1 << 20, 1 << 40, 1 << 60, u64::MAX - 1];
    let func = LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(
        FromSlice::new(&keys),
        keys.len(),
        no_logging![],
    )?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_u32() -> Result<()> {
    let keys: Vec<u32> = vec![1, 100, 1000, 10000, 100000];
    let func = LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(
        FromSlice::new(&keys),
        keys.len(),
        no_logging![],
    )?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_unsorted_error_u64() {
    let keys: Vec<u64> = vec![50, 30, 10];
    let result =
        LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(FromSlice::new(&keys), 3, no_logging![]);
    assert!(result.is_err());
}

#[test]
fn test_empty_str() -> Result<()> {
    let keys: Vec<&str> = vec![];
    let func = LcpMmphfStr::<[u64; 2], FuseLge3Shards>::try_new(keys_lender(&keys), 0, no_logging![])?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    Ok(())
}

#[test]
fn test_empty_u64() -> Result<()> {
    let keys: Vec<u64> = vec![];
    let func =
        LcpMmphfInt::<_, [u64; 2], FuseLge3Shards>::try_new(FromSlice::new(&keys), 0, no_logging![])?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    Ok(())
}

#[test]
fn test_signed_i64_crossing_zero() -> Result<()> {
    // Keys spanning the sign boundary — the critical case.
    let keys: Vec<i64> = vec![-100, -10, -1, 0, 1, 10, 100];
    let func: LcpMmphfInt<i64> =
        LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_signed_i64_extremes() -> Result<()> {
    let keys: Vec<i64> = vec![i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX];
    let func: LcpMmphfInt<i64> =
        LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_signed_i32() -> Result<()> {
    let keys: Vec<i32> = vec![-1000, -100, -10, -1, 0, 1, 10, 100, 1000];
    let func: LcpMmphfInt<i32> =
        LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_signed_i64_all_negative() -> Result<()> {
    let keys: Vec<i64> = vec![-500, -400, -300, -200, -100];
    let func: LcpMmphfInt<i64> =
        LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}

#[test]
fn test_signed_i64_random_1000() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut keys: Vec<i64> = (0..1000).map(|_| rng.random::<i64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();

    let func: LcpMmphfInt<i64> = LcpMmphfInt::try_new(FromSlice::new(&keys), n, no_logging![])?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key} at position {i}");
    }
    Ok(())
}
