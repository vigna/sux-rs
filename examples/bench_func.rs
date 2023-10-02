/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::ProgressLogger;
use epserde::prelude::*;
use sux::func::vfunc::VFunc;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks functions", long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    // A file containing UTF-8 keys, one per line.
    filename: Option<String>,
    #[arg(short)]
    // Use the 64-bit keys [0..n).
    n: usize,
    // A name for the ε-serde serialized function with u64 keys.
    func: String,
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let mut pl = ProgressLogger::default();

    let args = Args::parse();
    let func = VFunc::<_>::load_mem(&args.func)?;

    if let Some(filename) = args.filename {
        let keys = sux::utils::file::FilenameIntoIterator(&filename)
            .into_iter()
            .take(args.n)
            .collect::<Vec<_>>();
        pl.start("Querying...");
        for i in 0..keys.len() {
            std::hint::black_box(func.get(&i));
        }
        pl.done_with_count(keys.len());
    } else {
        pl.start("Querying...");
        for i in 0..args.n {
            assert_eq!(i, func.get(&i));
        }
        pl.done_with_count(args.n);
    }

    Ok(())
}
