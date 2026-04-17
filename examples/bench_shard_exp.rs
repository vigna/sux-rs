/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Experiment: build CompVFunc across sizes with uniform(0..256),
//! logging shard/entropy/overhead details.

use dsi_progress_logger::{ProgressLog, ProgressLogger};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;
use sux::func::codec::Huffman;
use sux::func::{CompVFunc, VBuilder};

fn run(n: usize) {
    let keys: Vec<u64> = (0..n as u64).collect();
    let mut rng = SmallRng::seed_from_u64(0);
    let values: Vec<usize> = (0..n).map(|_| rng.random_range(0..256)).collect();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);

    let t0 = Instant::now();
    let func = CompVFunc::<u64>::try_par_new_with_builder(
        &keys,
        &values,
        Huffman::new(),
        VBuilder::default(),
        &mut pl,
    );
    let dt = t0.elapsed();

    match func {
        Ok(f) => {
            let mut wrong = 0;
            for i in 0..n.min(1000) {
                if f.get(keys[i]) != values[i] {
                    wrong += 1;
                }
            }
            eprintln!(
                "n={n} built in {:.3}s ({:.1} µs/key) w={} wrong={wrong}",
                dt.as_secs_f64(),
                dt.as_secs_f64() * 1e6 / n as f64,
                f.global_max_codeword_length(),
            );
        }
        Err(e) => eprintln!("n={n} FAILED: {e}"),
    }
    eprintln!();
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    for &n in &[
        10, 100, 1000, 5000, 10000, 50000, 80000, 90000, 99000, 100000, 101000, 110000, 150000,
        200000, 400000, 800000, 1000000, 10000000,
    ] {
        run(n);
    }
}
