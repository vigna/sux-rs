/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::hint::black_box;
use sux::prelude::*;
use value_traits::slices::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks bit-field vectors", long_about = None)]
struct Args {
    /// The width of the elements of the vector.
    width: usize,
    /// The base-2 logarithm of the length of the vector.
    log2_size: usize,

    /// The number of test repetitions.
    #[arg(short, long, default_value = "10")]
    repeats: usize,

    /// The number of elements to get and set.
    #[arg(short, long, default_value = "10000000")]
    n: usize,
}

pub fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let mut a = BitFieldVec::<usize>::new(args.width, 1 << args.log2_size);
    let mask = (1 << args.log2_size) - 1;

    let mut pl = ProgressLogger::default();

    for _ in 0..args.repeats {
        let mut rand = SmallRng::seed_from_u64(0);
        pl.item_name("write");
        pl.start("Writing...");
        for _ in 0..args.n {
            let x = rand.random::<u64>() as usize & mask;
            unsafe { a.set_value_unchecked(x, 1) };
            black_box(());
        }
        pl.done_with_count(args.n);

        pl.item_name("read");
        pl.start("Reading (random)...");
        for _ in 0..args.n {
            unsafe {
                black_box(a.get_value_unchecked(rand.random::<u64>() as usize & mask));
            }
        }
        pl.done_with_count(args.n);

        pl.start("Reading (sequential)...");
        for i in 0..args.n {
            black_box(unsafe { a.get_value_unchecked(i) });
        }
        pl.done_with_count(args.n);

        let mut iter = a.into_unchecked_iter();
        pl.item_name("item");
        pl.start("Scanning (unchecked) ...");
        for _ in 0..args.n {
            black_box(unsafe { iter.next_unchecked() });
        }
        pl.done_with_count(args.n);

        let mut iter = a.into_rev_unchecked_iter();
        pl.item_name("item");
        pl.start("Scanning (reverse unchecked) ...");
        for _ in 0..args.n {
            black_box(unsafe { iter.next_unchecked() });
        }
        pl.done_with_count(args.n);
    }

    Ok(())
}
