/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use mem_dbg::DbgFlags;
use mem_dbg::MemDbg;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks Elias-Fano", long_about = None)]
struct Args {
    /// The number of elements in the list.
    n: usize,

    /// The size of the universe.
    u: usize,

    /// The number of values to test.
    t: usize,

    /// The number of test repetitions.
    #[arg(short, long, default_value = "10")]
    repeats: usize,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();
    let mut values = Vec::with_capacity(args.n);
    let mut rng = SmallRng::seed_from_u64(0);
    for _ in 0..args.n {
        values.push(rng.random_range(0..args.u));
    }
    values.sort();
    // Build Elias-Fano
    let mut elias_fano_builder = EliasFanoBuilder::new(args.n, args.u);
    for value in &values {
        elias_fano_builder.push(*value);
    }

    let mut elias_fano_builder = EliasFanoBuilder::new(args.n, args.u);
    for value in &values {
        elias_fano_builder.push(*value);
    }
    // Same as defaults
    const LOG2_ONES_PER_INVENTORY: usize = 12;
    const LOG2_U64_PER_INVENTORY: usize = 3;
    // Add an index on zeros
    let elias_fano_s: EliasFano<
        SelectZeroAdaptConst<
            SelectAdaptConst<_, _, LOG2_ONES_PER_INVENTORY, LOG2_U64_PER_INVENTORY>,
            _,
            LOG2_ONES_PER_INVENTORY,
            LOG2_U64_PER_INVENTORY,
        >,
    > = unsafe {
        elias_fano_builder
            .build()
            .map_high_bits(|high_bits| SelectZeroAdaptConst::new(SelectAdaptConst::new(high_bits)))
    };

    println!();
    elias_fano_s
        .mem_dbg(DbgFlags::default() | DbgFlags::PERCENTAGE)
        .unwrap();

    let mut ranks = Vec::with_capacity(args.t);
    for _ in 0..args.t {
        ranks.push(rng.random_range(0..args.n));
    }

    for _ in 0..args.repeats {
        let mut pl = ProgressLogger::default();

        pl.start("Benchmarking s.get()...");
        for &rank in &ranks {
            black_box(elias_fano_s.get(rank));
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.get_unchecked()...");
        for &rank in &ranks {
            unsafe {
                black_box(elias_fano_s.get_unchecked(rank));
            }
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.succ()...");
        for _ in 0..args.t {
            black_box(
                elias_fano_s
                    .succ(rng.random_range(0..args.u))
                    .unwrap_or((0, 0))
                    .0,
            );
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.succ_unchecked::<false>()...");
        let upper_bound = *values.last().unwrap();
        for _ in 0..args.t {
            black_box(unsafe {
                elias_fano_s
                    .succ_unchecked::<false>(rng.random_range(0..upper_bound))
                    .0
            });
        }
        pl.done_with_count(args.t);

        let first = *values.first().unwrap();

        pl.start("Benchmarking s.pred()...");
        for _ in 0..args.t {
            black_box(
                elias_fano_s
                    .pred(rng.random_range(first..args.u))
                    .unwrap_or((0, 0))
                    .0,
            );
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.pred_unchecked::<false>()...");
        for _ in 0..args.t {
            black_box(unsafe {
                elias_fano_s
                    .pred_unchecked::<false>(rng.random_range(first..args.u))
                    .0
            });
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.iter()...");
        for i in &elias_fano_s {
            black_box(i);
        }
        pl.done_with_count(args.n);
    }

    Ok(())
}
