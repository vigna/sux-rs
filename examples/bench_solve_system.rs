/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Per-phase timing of CompVFunc's force_index_peeler path
//! (materialized) vs streaming PackedEdge path.
//!
//! Builds the same shard data and runs both approaches, printing
//! per-phase durations.

use dsi_progress_logger::no_logging;
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::Geometric;
use std::time::Instant;
use sux::func::CompVFunc;

fn main() {
    for &n in &[100_000usize] {
        let keys: Vec<u64> = (0..n as u64).collect();
        let mut rng = SmallRng::seed_from_u64(0);
        let g = Geometric::new(0.5).unwrap();
        let values: Vec<usize> = (0..n).map(|_| g.sample(&mut rng) as usize + 1).collect();

        // Warm up
        let _ = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]);

        let trials = 10;
        let mut times = Vec::with_capacity(trials);
        for _ in 0..trials {
            let t0 = Instant::now();
            let _ = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[trials / 2];
        eprintln!(
            "n={n}: median={median:.4}s ({:.1} µs/key)",
            median * 1e6 / n as f64
        );
    }
}
