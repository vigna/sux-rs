/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
#![cfg(feature = "cli")]

use anyhow::{bail, Result};
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::{Lender, Lending};
use std::io::BufReader;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Compress a file containing one string per line using a Rear Coded List", long_about = None)]

struct Args {
    /// A file containing UTF-8 strings, one per line.
    filename: String,
    #[arg(short, long)]
    /// A name for the ε-serde serialized RCL. Defaults to the "{filename}.rcl" if not specified.
    dst: Option<String>,
    /// The filename containing the keys is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// The number of high bits defining the number of buckets. Very large key sets may benefit from a larger number of buckets.
    #[arg(short = 'k', long, default_value_t = 8)]
    k: usize,
}

fn compress<
    S: AsRef<str> + ?Sized,
    L: Lender + for<'lend> Lending<'lend, Lend = Result<&'lend S, std::io::Error>>,
>(
    mut lines: L,
    args: Args,
) -> anyhow::Result<()> {
    let dst_file_path = args.dst.unwrap_or(format!("{}.rcl", args.filename));
    let dst_file = std::fs::File::create(dst_file_path).unwrap();
    let mut dst_file = std::io::BufWriter::new(dst_file);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Reading the input file...");

    let mut rclb = RearCodedListBuilder::new(args.k);

    while let Some(result) = lines.next() {
        match result {
            Ok(line) => {
                pl.light_update();
                rclb.push(line);
            }
            Err(e) => {
                bail!("Error reading input: {}", e);
            }
        }
    }

    pl.done();

    rclb.print_stats();

    let rcl = rclb.build();

    rcl.serialize(&mut dst_file)?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let file = std::fs::File::open(&args.filename).unwrap();
    if args.zstd {
        compress(ZstdLineLender::new(file)?, args)?;
    } else {
        compress(LineLender::new(BufReader::new(file)), args)?;
    }

    Ok(())
}
