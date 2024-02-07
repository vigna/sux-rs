/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use dsi_progress_logger::*;
use epserde::prelude::*;
use sux::{func::VFunc, prelude::VFuncBuilder, utils::FromIntoIterator};

#[test]
fn test_func() -> anyhow::Result<()> {
    let mut pl = ProgressLogger::default();

    for offline in [false, true] {
        for n in [10_usize, 100, 1000, 100000] {
            let func = VFuncBuilder::default()
                .log2_buckets(4)
                .offline(offline)
                .build(
                    FromIntoIterator::from(0..n),
                    FromIntoIterator::from(0_usize..),
                    &mut pl,
                )?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            func.serialize(&mut cursor).unwrap();
            cursor.set_position(0);
            let func = VFunc::<usize>::deserialize_eps(cursor.as_bytes()).unwrap();
            pl.start("Querying...");
            for i in 0..n {
                assert_eq!(i, func.get(&i));
            }
            pl.done_with_count(n);
        }
    }

    Ok(())
}

#[test]
fn test_dup_key() {
    assert!(VFuncBuilder::<usize, usize>::default()
        .build(
            FromIntoIterator::from(std::iter::repeat(0).take(10)),
            FromIntoIterator::from(0..),
            &mut Option::<ProgressLogger>::None
        )
        .is_err());
}
