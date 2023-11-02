/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use dsi_progress_logger::*;
use epserde::prelude::*;
use sux::{func::VFunc, prelude::VFuncBuilder, utils::IntoOkLender};

#[test]
fn test_func() -> anyhow::Result<()> {
    let mut pl = ProgressLogger::default();

    for offline in [false, true] {
        for n in [10, 100, 1000, 100000] {
            let func = VFuncBuilder::<usize, usize>::default()
                .offline(offline)
                .build(IntoOkLender::from((0..n)), &(0..).into(), &mut pl)?;
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
    assert!(VFuncBuilder::<usize, usize>::default()
        .build(
            std::iter::repeat(0).take(10).into(),
            &(0..).into(),
            &mut Option::<ProgressLogger>::None
        )
        .is_err());
}
