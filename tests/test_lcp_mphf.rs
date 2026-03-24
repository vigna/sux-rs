/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]
use anyhow::Result;
use dsi_progress_logger::no_logging;
use std::io::{BufReader, Cursor};
use sux::func::LcpMinPerfHashFunc;
use sux::utils::LineLender;

/// Helper: build a `LineLender` from a slice of sorted string keys.
fn keys_lender(keys: &[&str]) -> LineLender<BufReader<Cursor<Vec<u8>>>> {
    let data = keys.join("\n");
    LineLender::new(BufReader::new(Cursor::new(data.into_bytes())))
}

#[test]
fn test_small() -> Result<()> {
    let mut keys = vec!["alpha", "beta", "delta", "gamma"];
    keys.sort();
    let func = LcpMinPerfHashFunc::new(keys_lender(&keys), keys.len(), no_logging![])?;

    // Must be a permutation of 0..n.
    let mut ranks: Vec<usize> = keys.iter().map(|k| func.get(k)).collect();
    ranks.sort();
    assert_eq!(ranks, vec![0, 1, 2, 3]);

    // Must be monotone: sorted keys -> get(key_i) == i.
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }

    Ok(())
}

#[test]
fn test_single() -> Result<()> {
    let keys = vec!["hello"];
    let func = LcpMinPerfHashFunc::new(keys_lender(&keys), 1, no_logging![])?;
    assert_eq!(func.get("hello"), 0);
    assert_eq!(func.len(), 1);
    assert!(!func.is_empty());
    Ok(())
}

#[test]
fn test_two_keys() -> Result<()> {
    let keys = vec!["aaa", "bbb"];
    let func = LcpMinPerfHashFunc::new(keys_lender(&keys), 2, no_logging![])?;
    assert_eq!(func.get("aaa"), 0);
    assert_eq!(func.get("bbb"), 1);
    Ok(())
}

/// Nine keys carefully chosen so that each bucket boundary produces a
/// distinct LCP byte prefix:
///   bucket 0 (keys 0..4): LCP with empty prev = 0, prefix = b""
///   bucket 1 (keys 4..8): LCP("daa","d") = 1, prefix = b"d"
///   bucket 2 (key 8):     LCP("dadaaa","dad") = 3, prefix = b"dad"
#[test]
fn test_nine_keys_distinct_lcps() -> Result<()> {
    let keys: Vec<&str> = vec!["a", "b", "c", "d", "daa", "dab", "dac", "dad", "dadaaa"];
    // Already in sorted order.
    let func = LcpMinPerfHashFunc::new(keys_lender(&keys), keys.len(), no_logging![])?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i, "key {key:?} at position {i}");
    }
    Ok(())
}

#[test]
fn test_unsorted_error() {
    let keys = vec!["beta", "alpha"];
    let result = LcpMinPerfHashFunc::new(keys_lender(&keys), 2, no_logging![]);
    assert!(result.is_err());
}
