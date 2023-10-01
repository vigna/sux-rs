/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use dsi_progress_logger::ProgressLogger;
use epserde::prelude::*;
use sux::prelude::func::Function;

#[test]
fn test_func() {
    let mut pl = ProgressLogger::default();

    for n in [10, 100, 1000, 100000, 10000000] {
        let func = Function::<_>::new(0..n as u64, &mut (0..), &mut Some(&mut pl));
        let mut cursor = epserde::new_aligned_cursor();
        func.serialize(&mut cursor).unwrap();
        cursor.set_position(0);
        let buf = cursor.into_inner();
        let func = Function::<_>::deserialize_eps(&buf).unwrap();
        pl.start("Querying...");
        for i in 0..n {
            assert_eq!(i, func.get(&i) as usize);
        }
        pl.done_with_count(n);
    }
}
