/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use epserde::prelude::*;
use sux::prelude::HyperLogLogVec;

#[test]
fn test_hyperloglog_vec() -> Result<()> {
    //let path = "hyperloglogvec";
    for end in [
        5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
    ] {
        let mut hll = HyperLogLogVec::new(5, 5, end).unwrap();
        for i in 0..end {
            hll.insert(0, &i);
        }
        let estimate = hll.estimate_cardinality(0);
        let approx_ratio = (1.0 - end as f64 / estimate as f64).abs();
        println!("{:.4} {}: {}", approx_ratio, end, estimate);
        assert!(approx_ratio < 0.3);

        /* DECOMMENT WHEN EPSERDE IS UPDATED
        hll.store(path)?;

        let hll2 = <HyperLogLogVec>::load_mem(path)?;

        for i in 0..end {
            assert_eq!(
                hll.iter_regs(i).collect::<Vec<_>>(),
                hll2.iter_regs(i).collect::<Vec<_>>(),
            );
        }
        std::fs::remove_file(path)?; 
        */
    }

    Ok(())
}
