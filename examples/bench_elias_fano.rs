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

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

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
        elias_fano_builder.push(*value)?;
    }

    // Frequency of ones in the inventory for one-level index
    const FIXED1_LOG2_ONES_PER_INVENTORY: usize = 8;
    // Add an index on ones
    let elias_fano_q: EliasFano<SelectFixed1<_, _, FIXED1_LOG2_ONES_PER_INVENTORY>> =
        elias_fano_builder.build().convert_to()?;
    // Add an index on zeros
    let elias_fano_q: EliasFano<
        SelectZeroFixed1<
            SelectFixed1<_, _, FIXED1_LOG2_ONES_PER_INVENTORY>,
            _,
            FIXED1_LOG2_ONES_PER_INVENTORY,
        >,
    > = elias_fano_q.convert_to()?;

    elias_fano_q.mem_dbg(DbgFlags::default() | DbgFlags::PERCENTAGE)?;

    let mut elias_fano_builder = EliasFanoBuilder::new(args.n, args.u);
    for value in &values {
        elias_fano_builder.push(*value)?;
    }
    const FIXED2_LOG2_ONES_PER_INVENTORY: usize = 10;
    const FIXED2_LOG2_U64_PER_INVENTORY: usize = 2;
    // Add an index on ones
    let elias_fano_s: EliasFano<
        SelectFixed2<_, _, FIXED2_LOG2_ONES_PER_INVENTORY, FIXED2_LOG2_U64_PER_INVENTORY>,
    > = elias_fano_builder.build().convert_to()?;
    // Add an index on zeros
    let elias_fano_s: EliasFano<
        SelectZeroFixed2<
            SelectFixed2<_, _, FIXED2_LOG2_ONES_PER_INVENTORY, FIXED2_LOG2_U64_PER_INVENTORY>,
            _,
            FIXED2_LOG2_ONES_PER_INVENTORY,
            FIXED2_LOG2_U64_PER_INVENTORY,
        >,
    > = elias_fano_s.convert_to()?;

    println!();
    elias_fano_s.mem_dbg(DbgFlags::default() | DbgFlags::PERCENTAGE)?;

    let mut ranks = Vec::with_capacity(args.t);
    for _ in 0..args.t {
        ranks.push(rng.gen_range(0..args.n));
    }

    for _ in 0..args.repeats {
        let mut pl = ProgressLogger::default();

        pl.start("Benchmarking q.get()...");
        for &rank in &ranks {
            black_box(elias_fano_q.get(rank));
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking q.get_unchecked()...");
        for &rank in &ranks {
            unsafe {
                black_box(elias_fano_q.get_unchecked(rank));
            }
        }
        pl.done_with_count(args.t);

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

        pl.start("Benchmarking q.succ()...");
        for _ in 0..args.t {
            black_box(
                elias_fano_q
                    .succ(&rng.gen_range(0..args.u))
                    .unwrap_or((0, 0))
                    .0,
            );
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking q.succ_unchecked::<false>()...");
        let upper_bound = *values.last().unwrap();
        for _ in 0..args.t {
            black_box(unsafe {
                elias_fano_q
                    .succ_unchecked::<false>(&rng.gen_range(0..upper_bound))
                    .0
            });
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.succ()...");
        for _ in 0..args.t {
            black_box(
                elias_fano_s
                    .succ(&rng.gen_range(0..args.u))
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
                    .succ_unchecked::<false>(&rng.gen_range(0..upper_bound))
                    .0
            });
        }
        pl.done_with_count(args.t);

        let first = *values.first().unwrap();

        pl.start("Benchmarking q.pred()...");
        for _ in 0..args.t {
            black_box(
                elias_fano_q
                    .pred(&(rng.gen_range(first..args.u)))
                    .unwrap_or((0, 0))
                    .0,
            );
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking q.pred_unchecked::<false>()...");
        for _ in 0..args.t {
            black_box(unsafe {
                elias_fano_q
                    .pred_unchecked::<false>(&rng.gen_range(first..args.u))
                    .0
            });
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarkins s.pred()...");
        for _ in 0..args.t {
            black_box(
                elias_fano_s
                    .pred(&(rng.gen_range(first..args.u)))
                    .unwrap_or((0, 0))
                    .0,
            );
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking s.pred_unchecked::<false>()...");
        for _ in 0..args.t {
            black_box(unsafe {
                elias_fano_s
                    .pred_unchecked::<false>(&rng.gen_range(first..args.u))
                    .0
            });
        }
        pl.done_with_count(args.t);

        pl.start("Benchmarking q.iter()...");
        for i in &elias_fano_q {
            black_box(i);
        }
        pl.done_with_count(args.n);

        pl.start("Benchmarking s.iter()...");
        for i in &elias_fano_s {
            black_box(i);
        }
        pl.done_with_count(args.n);
    }

    Ok(())
}
