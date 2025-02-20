/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::prelude::*;
use lender::*;
use sux::{
    bits::BitFieldVec,
    func::VFunc,
    utils::{LineLender, ZstdLineLender},
};

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    #[arg(short)]
    /// The maximum number strings to use from the file, or the number of 64-bit keys.
    n: usize,
    /// A name for the ε-serde serialized function with u64 keys.
    func: String,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let mut pl = ProgressLogger::default();

    let args = Args::parse();

    if let Some(filename) = args.filename {
        let func = VFunc::<_, _, BitFieldVec<usize>>::load_mem(&args.func)?;
        let keys: Vec<_> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| x.unwrap().to_owned())
                .take(args.n)
                .collect()
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| x.unwrap().to_owned())
                .take(args.n)
                .collect()
        };

        pl.start("Querying...");
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(i, func.get(key));
        }
        pl.done_with_count(keys.len());
    } else {
        let func = VFunc::<_, _, BitFieldVec<usize>, [u64; 2], true>::load_mem(&args.func)?;

        pl.start("Querying (independent)...");
        for i in 0..args.n {
            std::hint::black_box(func.get(&i));
        }
        pl.done_with_count(args.n);

        pl.start("Querying (dependent)...");
        let mut x = 0;
        for i in 0..args.n {
            x = std::hint::black_box(func.get(&(i ^ (x & 1))));
        }
        pl.done_with_count(args.n);
    }

    Ok(())
}
