/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::hint::black_box;

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use xorf::{BinaryFuse16, Filter, Fuse16};

#[derive(Parser, Debug)]
#[command(about = "Benchmark Fuse with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The number of integers in the function
    n: usize,
    #[arg(short)]
    /// The number of samples
    s: usize,
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let mut pl = ProgressLogger::default();

    let args = Args::parse();

    let keys: Vec<u64> = (0..args.n as u64).collect();
    pl.start("Building...");
    let filter = BinaryFuse16::try_from(&keys).unwrap();
    pl.done_with_count(args.n);

    pl.start("Querying...");
    for i in 0..args.s as u64 {
        black_box(filter.contains(&i));
    }
    pl.done_with_count(args.s);

    Ok(())
}