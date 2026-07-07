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
