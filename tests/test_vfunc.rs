/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use dsi_progress_logger::ProgressLogger;
use epserde::prelude::*;
use sux::func::vfunc::VFunc;

#[test]
fn test_func() {
    let mut pl = ProgressLogger::default();

    for n in [10, 100, 1000, 100000] {
        let func = VFunc::<_>::new(0..n as u64, &(0..), &mut Some(&mut pl));
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
