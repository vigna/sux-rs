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
use sux::prelude::VBuilder;
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
    /// Whether the file is compressed with zstd.
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
    /// A 64-bit seed for the pseudorandom number generator.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);

    if let Some(filename) = args.filename {
        let mut builder = VBuilder::<_, _, BitFieldVec<usize>, [u64; 2]>::default()
            .offline(args.offline)
            .log2_buckets(args.high_bits);

        if let Some(seed) = args.seed {
            builder = builder.seed(seed);
        }
        if let Some(threads) = args.threads {
            builder = builder.max_num_threads(threads);
        }

        let func = if args.zstd {
            builder.try_build_func(
                ZstdLineLender::from_path(&filename)?,
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        } else {
            builder.try_build_func(
                LineLender::from_path(&filename)?,
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        };
        func.store(&args.func)?;
        return Ok(());
    }

    if let Some(_dict_bits) = args.dict {
        if let Some(n) = args.n {
            let mut builder = VBuilder::<_, _, Vec<u8>, [u64; 2], true, _>::default()
                .offline(args.offline)
                .expected_num_keys(n)
                .log2_buckets(args.high_bits);
            if let Some(seed) = args.seed {
                builder = builder.seed(seed);
            }
            if let Some(threads) = args.threads {
                builder = builder.max_num_threads(threads);
            }
            let filter = builder.try_build_filter(FromIntoIterator::from(0_usize..n), &mut pl)?;
            filter.store(&args.func)?;
        }
    } else if let Some(n) = args.n {
        let mut builder = VBuilder::<_, _, BitFieldVec<usize>, [u64; 2], true>::default()
            .offline(args.offline)
            .expected_num_keys(n)
            .log2_buckets(args.high_bits);
        if let Some(seed) = args.seed {
            builder = builder.seed(seed);
        }
        if let Some(threads) = args.threads {
            builder = builder.max_num_threads(threads);
        }

        let func = builder.try_build_func(
            FromIntoIterator::from(0_usize..n),
            FromIntoIterator::from(0_usize..),
            &mut pl,
        )?;
        func.store(&args.func)?;
    }
    Ok(())
}
