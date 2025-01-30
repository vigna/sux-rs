/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::{Ok, Result};
use clap::Parser;
use dsi_progress_logger::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks Rank9", long_about = None)]
struct Args {
    /// The number of elements
    n: usize,

    /// The number of values to test
    t: usize,

    /// The number of test repetitions
    #[arg(short, long, default_value = "0.5")]
    density: f64,

    /// The number of test repetitions
    #[arg(short, long, default_value = "10")]
    repeats: usize,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();
    let n = args.n;
    let mut bit_vec = BitVec::new(n);
    let mut rng = SmallRng::seed_from_u64(0);
    for i in 0..n {
        if rng.random_bool(args.density) {
            bit_vec.set(i, true);
        }
    }

    let rank9: Rank9 = Rank9::new(bit_vec);

    println!();

    let mut pos = Vec::with_capacity(args.t);
    for _ in 0..args.t {
        pos.push(rng.random_range(0..n));
    }

    for _ in 0..args.repeats {
        let mut pl = ProgressLogger::default();

        pl.start("Benchmarking rank...");
        for &p in &pos {
            black_box(rank9.rank(p));
        }
        pl.done_with_count(args.t);
    }

    Ok(())
}
