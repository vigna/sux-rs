/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks Elias-Fano", long_about = None)]
struct Args {
    /// The number of elements
    n: usize,

    /// The size of the universe
    u: usize,

    /// The number of values to test
    t: usize,

    /// The number of test repetitions
    #[arg(short, long, default_value = "0.5")]
    density: f64,

    /// The number of test repetitions
    #[arg(short, long, default_value = "10")]
    repeats: usize,
}

fn main() {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let args = Args::parse();
    let mut values = Vec::with_capacity(args.n);
    let mut rng = SmallRng::seed_from_u64(0);
    for _ in 0..args.n {
        values.push(rng.gen_range(0..args.u));
    }
    values.sort();
    // Build Elias-Fano
    let mut elias_fano_builder = EliasFanoBuilder::new(args.n, args.u);
    for value in &values {
        elias_fano_builder.push(*value).unwrap();
    }
    // Add an index on ones
    let elias_fano: EliasFano<QuantumIndex> = elias_fano_builder.build().convert_to().unwrap();
    // Add an index on zeros
    let elias_fano: EliasFano<QuantumZeroIndex<QuantumIndex>> = elias_fano.convert_to().unwrap();

    let mut ranks = Vec::with_capacity(args.t);
    for _ in 0..args.t {
        ranks.push(rng.gen_range(0..args.n));
    }

    let mut u = 0;

    for _ in 0..args.repeats {
        let mut pl = ProgressLogger::default();
        pl.start("Benchmarking get()...");
        for &rank in &ranks {
            u ^= elias_fano.get(rank);
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking get_unchecked()...");
        for &rank in &ranks {
            unsafe {
                u ^= elias_fano.get_unchecked(rank);
            }
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking succ()...");
        for _ in 0..args.t {
            u ^= elias_fano
                .succ(&rng.gen_range(0..args.u))
                .unwrap_or((0, 0))
                .0;
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking succ_unchecked()...");
        let upper_bound = *values.last().unwrap();
        for _ in 0..args.t {
            u ^= unsafe { elias_fano.succ_unchecked(&rng.gen_range(0..upper_bound)).0 };
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking iter()...");
        for i in &elias_fano {
            u ^= i;
        }
        pl.done_with_count(args.n);
    }

    black_box(u);
}
