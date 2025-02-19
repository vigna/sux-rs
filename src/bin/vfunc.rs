/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use sux::bits::BitFieldVec;
use sux::prelude::VFuncBuilder;
use sux::utils::{FromIntoIterator, LineLender, ZstdLineLender};

#[derive(Parser, Debug)]
#[command(about = "Generate a VFunc mapping each input to its rank and serialize it with ε-serde", long_about = None)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .args(&["filename", "n"]),
))]
struct Args {
    #[arg(short, long)]
    /// A file containing UTF-8 keys, one per line.
    filename: Option<String>,
    #[arg(short)]
    /// Use the 64-bit keys [0..n). Mainly useful for testing and debugging.
    n: Option<usize>,
    /// Use this number of threads.
    #[arg(short, long)]
    threads: Option<usize>,
    /// A name for the ε-serde serialized function.
    func: String,
    /// The filename containing the keys is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// Use disk-based buckets to reduce memory usage at construction time.
    #[arg(short, long)]
    offline: bool,
    /// The number of high bits defining the number of buckets. Very large key
    /// sets may benefit from a larger number of buckets.
    #[arg(short = 'H', long, default_value_t = 8)]
    high_bits: u32,
    /// Create an approximate dictionary with this number of bits
    #[arg(short, long)]
    dict: Option<usize>,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);

    if let Some(filename) = args.filename {
        let mut builder = VFuncBuilder::<_, _, BitFieldVec<usize>, [u64; 2]>::default()
            .offline(args.offline)
            .log2_buckets(args.high_bits);

        if let Some(threads) = args.threads {
            builder = builder.max_num_threads(threads);
        }

        let func = if args.zstd {
            builder.try_build(
                ZstdLineLender::from_path(&filename)?,
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        } else {
            builder.try_build(
                LineLender::from_path(&filename)?,
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        };
        func.store(&args.func)?;
    }

    if let Some(n) = args.n {
        let mut builder = VFuncBuilder::<_, _, BitFieldVec<usize>, [u64; 2], true>::default()
            .offline(args.offline)
            .expected_num_keys(n)
            .log2_buckets(args.high_bits);
        if let Some(threads) = args.threads {
            builder = builder.max_num_threads(threads);
        }

        let func = if let Some(dict_bits) = args.dict {
            builder.try_build(
                FromIntoIterator::from(0_usize..n),
                FromIntoIterator::from(itertools::repeat_n((1 << dict_bits) - 1, n)),
                &mut pl,
            )?
        } else {
            builder.try_build(
                FromIntoIterator::from(0_usize..n),
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        };

        func.store(&args.func)?;
    }
    Ok(())
}
