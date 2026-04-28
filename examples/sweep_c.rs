/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Sweep peelability across n values using the current c() and
//! log2_seg_size() from `Fuse3NoShards`.
//!
//! For each n, tries building a VFunc with 1-bit values. Reports
//! any n where construction fails.
//!
//! Usage:
//!   cargo run --release --example sweep_c [start] [end] [trials]
//!
//! Defaults: start=1, end=1_000_000, trials=5

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use sux::func::shard_edge::Fuse3NoShards;
use sux::func::{VBuilder, VFunc};

fn try_build(n: usize, seed: u64) -> bool {
    if n == 0 {
        return true;
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<usize> = (0..n).map(|_| rng.random_range(0..2)).collect();

    let builder = VBuilder::<Box<[usize]>, [u64; 2], Fuse3NoShards>::default();
    let pl = dsi_progress_logger::no_logging![];

    VFunc::<u64, Box<[usize]>, _, _>::try_par_new_with_builder(
        &keys,
        &values,
        builder,
        &mut pl.clone(),
    )
    .is_ok()
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let start: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let end: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let trials: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    eprintln!("Sweep n={start}..={end}, {trials} trials per n");

    let failed_count = AtomicUsize::new(0);
    let tested = AtomicUsize::new(0);

    let results: Vec<(usize, usize)> = (start..=end)
        .into_par_iter()
        .map(|n| {
            let failures: usize = (0..trials).filter(|&t| !try_build(n, t as u64)).count();

            let t = tested.fetch_add(1, Ordering::Relaxed) + 1;
            if failures > 0 {
                failed_count.fetch_add(1, Ordering::Relaxed);
                eprintln!("n={n} FAILED {failures}/{trials}");
            }
            if t % 10000 == 0 {
                eprintln!(
                    "progress: {t}/{} ({} failures so far)",
                    end - start + 1,
                    failed_count.load(Ordering::Relaxed)
                );
            }
            (n, failures)
        })
        .collect();

    let total_failures: usize = results.iter().filter(|(_, f)| *f > 0).count();
    eprintln!(
        "\nDone. {total_failures}/{} values of n had at least one failure.",
        end - start + 1
    );

    for (n, failures) in &results {
        if *failures > 0 {
            println!("{n}\t{failures}");
        }
    }
}
