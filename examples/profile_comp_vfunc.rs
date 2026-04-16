/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! One-shot CompVFunc build at a given size, intended for profiling
//! with `samply` or `cargo flamegraph`. Prints wall-clock build time
//! to stderr and exits.
//!
//! Run with: `cargo run --release --example profile_comp_vfunc -- 100000000`

use dsi_progress_logger::no_logging;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Geometric};
use sux::func::CompVFunc;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .expect("usage: profile_comp_vfunc <n>")
        .parse()
        .expect("n must be a positive integer");

    let mut rng = SmallRng::seed_from_u64(42);
    let geo = Geometric::new(0.5).unwrap();
    let values: Vec<usize> = (0..n).map(|_| geo.sample(&mut rng) as usize).collect();
    let keys: Vec<usize> = (0..n).collect();

    let start = std::time::Instant::now();
    let func =
        CompVFunc::<usize>::try_par_new(&keys, &values, no_logging![]).expect("build failed");
    let elapsed = start.elapsed();
    eprintln!(
        "built CompVFunc n={n} in {:.3}s ({} ns/key)",
        elapsed.as_secs_f64(),
        elapsed.as_nanos() / n as u128
    );
    std::hint::black_box(func);
}
