/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use dsi_progress_logger::ProgressLogger;
use epserde::prelude::*;
use sux::{func::VFunc, prelude::VFuncBuilder};

#[test]
fn test_func() -> anyhow::Result<()> {
    let mut pl = ProgressLogger::default();

    for offline in [false, true] {
        for n in [10, 100, 1000, 100000] {
            let func =
                VFuncBuilder::<_>::default()
                    .offline(offline)
                    .build(0..n, &(0..), Some(&mut pl))?;
            let mut cursor = epserde::new_aligned_cursor();
            func.serialize(&mut cursor).unwrap();
            cursor.set_position(0);
            let buf = cursor.into_inner();
            let func = VFunc::<u64>::deserialize_eps(&buf).unwrap();
            pl.start("Querying...");
            for i in 0..n {
                assert_eq!(i, func.get(&i) as u64);
            }
            pl.done_with_count(n as usize);
        }
    }

    Ok(())
}

#[test]
fn test_dup_key() {
    assert!(VFuncBuilder::<_>::default()
        .build(std::iter::repeat(0).take(10), &(0..), None)
        .is_err());
}
