/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Build a [`VFunc`] mapping words to their index, then wrap it in a
//! [`SignedFunc`] to reject unknown keys.
//!
//! Run with:
//! ```text
//! cargo run --release --example example_vfunc
//! ```

use anyhow::Result;
use dsi_progress_logger::ProgressLogger;
use sux::bits::BitFieldVec;
use sux::func::{LcpMmphfStr, SignedFunc, VFunc};
use sux::traits::TryIntoUnaligned;
use sux::utils::{FromCloneableIntoIterator, FromSlice};

fn main() -> Result<()> {
    // ── Keys ───────────────────────────────────────────────────────
    let keys = vec![
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    ];
    let n = keys.len();

    // ── VFunc: map each key to its index ───────────────────────────
    let mut pl = ProgressLogger::default();

    let func = <VFunc<str, BitFieldVec<Box<[usize]>>>>::try_new(
        FromSlice::new(&keys),
        FromCloneableIntoIterator::from(0..n),
        n,
        &mut pl,
    )?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), i);
    }
    // Querying an unknown key returns an arbitrary value — no error.
    let _ = func.get("unknown");

    println!(
        "VFunc built: {} keys, get(\"gamma\") = {}",
        n,
        func.get("gamma")
    );

    // ── SignedFunc: wrap VFunc with verification hashes ─────────────
    // Hash width is u16 → false-positive rate 2⁻¹⁶ ≈ 0.0015%.
    let signed = <SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>>, Box<[u16]>>>::try_new(
        FromSlice::new(&keys),
        n,
        &mut pl,
    )?;

    // Convert to unaligned access for faster queries.
    let signed = signed.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(signed.get(key), Some(i));
    }
    // Unknown keys are (almost certainly) rejected.
    assert_eq!(signed.get("unknown"), None);

    println!(
        "SignedFunc built: get(\"gamma\") = {:?}, get(\"unknown\") = {:?}",
        signed.get("gamma"),
        signed.get("unknown"),
    );

    // ── Signed LcpMmphf: monotone MMPHF for sorted keys ───────────
    // Keys must be in strictly increasing lexicographic order.
    let mut sorted_keys: Vec<String> = keys.iter().map(|s| s.to_string()).collect();
    sorted_keys.sort();

    let mmphf =
        <SignedFunc<LcpMmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&sorted_keys), n, &mut pl)?;
    let mmphf = mmphf.try_into_unaligned()?;

    for (rank, key) in sorted_keys.iter().enumerate() {
        assert_eq!(mmphf.get(key.as_str()), Some(rank));
    }
    assert_eq!(mmphf.get("not_in_set"), None);

    println!(
        "SignedFunc<LcpMmphfStr> built: get(\"alpha\") = {:?}",
        mmphf.get("alpha"),
    );

    Ok(())
}
