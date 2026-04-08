/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Build a [`VFilter`] from a set of keys and check membership.
//!
//! Run with:
//! ```text
//! cargo run --release --example example_vfilter
//! ```

use anyhow::Result;
use dsi_progress_logger::ProgressLogger;
use sux::bits::BitFieldVec;
use sux::dict::VFilter;
use sux::func::VFunc;
use sux::utils::FromCloneableIntoIterator;

fn main() -> Result<()> {
    let n = 1_000_000;

    // ── Full-width filter (Box<[u8]> → 8 hash bits → 2⁻⁸ FP rate) ─
    let mut pl = ProgressLogger::default();

    let filter = <VFilter<VFunc<usize, Box<[u8]>>>>::try_new(
        FromCloneableIntoIterator::from(0..n),
        n,
        &mut pl,
    )?;

    // All keys in the original set are found.
    for i in 0..n {
        assert!(filter.contains(i));
    }
    // Index syntax also works.
    assert!(filter[42]);

    println!(
        "Box<[u8]> filter: {} keys, {} hash bits",
        filter.len(),
        filter.hash_bits(),
    );

    // ── Compact filter (BitFieldVec, 5 hash bits → 2⁻⁵ FP rate) ───
    let filter = <VFilter<VFunc<usize, BitFieldVec<Box<[usize]>>>>>::try_new(
        FromCloneableIntoIterator::from(0..n),
        n,
        5, // filter_bits: fewer bits → less space, more false positives
        &mut pl,
    )?;

    for i in 0..n {
        assert!(filter.contains(i));
    }

    println!(
        "BitFieldVec filter: {} keys, {} hash bits",
        filter.len(),
        filter.hash_bits(),
    );

    Ok(())
}
