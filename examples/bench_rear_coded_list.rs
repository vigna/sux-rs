/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use lender::prelude::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks construction and access fo read-coded lists", long_about = None)]
struct Args {
    /// The file to read, every line will be inserted in the RCA.
    file_path: String,

    #[arg(short, long, default_value = "4")]
    /// Fully write every string with index multiple of k.
    k: usize,

    #[arg(short, long, default_value = "10000")]
    /// How many iterations of random access speed test
    accesses: usize,
}

pub fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let mut rcab = RearCodedListBuilder::new(args.k);
    let mut pl = ProgressLogger::default();
    pl.display_memory(true).item_name("line");

    let lines = std::io::BufReader::new(std::fs::File::open(&args.file_path)?);

    pl.start("Inserting...");

    for_!(result in  LineLender::new(lines) {
        match result {
            Ok(line) => {
                rcab.push(line);
                pl.light_update();
            }
            Err(e) => {
                panic!("Error reading input: {}", e);
            }
        }
    });

    pl.done();

    rcab.print_stats();
    let rca = rcab.build();

    let mut rand = SmallRng::seed_from_u64(0);

    let start = std::time::Instant::now();
    for _ in 0..args.accesses {
        let i = rand.gen::<usize>() % rca.len();
        let _ = black_box(rca.get(i));
    }
    let elapsed = start.elapsed();
    println!(
        "avg_rnd_access_speed: {} ns/access",
        elapsed.as_nanos() as f64 / args.accesses as f64
    );

    Ok(())
}
