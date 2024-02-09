/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use core::sync::atomic::Ordering;
use epserde::prelude::*;
use sux::prelude::*;

#[test]
fn test_hyperloglog_vec() -> Result<()> {
    let path = "hyperloglogvec";
    let path2 = "hyperloglogvecatomic";
    for end in [
        5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
    ] {
        let ahll = AtomicHyperLogLogVec::new(5, 5, end).unwrap();
        let mut hll = HyperLogLogVec::new(5, 5, end).unwrap();
        for i in 0..end {
            ahll.insert(0, &i, Ordering::SeqCst);
            hll.insert(0, &i);
        }
        let estimate = hll.estimate_cardinality(0);
        let estimate2 = ahll.estimate_cardinality(0, Ordering::SeqCst);
        let approx_ratio = (1.0 - end as f64 / estimate as f64).abs();
        println!("{:.4} {}: {} {}", approx_ratio, end, estimate, estimate2);
        assert!(approx_ratio < 0.3);
        assert!(((estimate - estimate2) / estimate).abs() < 0.05);

        let nahll = ahll.convert_to()?;
        hll.store(path)?;
        nahll.store(path2)?;

        let hll2 = <HyperLogLogVec>::load_mem(path)?;
        let nahll2 = <HyperLogLogVec>::load_mem(path)?;

        for i in 0..end {
            assert_eq!(
                hll.iter_regs(i).collect::<Vec<_>>(),
                hll2.iter_regs(i).collect::<Vec<_>>(),
            );
            assert_eq!(
                hll.iter_regs(i).collect::<Vec<_>>(),
                nahll2.iter_regs(i).collect::<Vec<_>>(),
            )
        }
        std::fs::remove_file(path)?;
        std::fs::remove_file(path2)?;
    }

    Ok(())
}
