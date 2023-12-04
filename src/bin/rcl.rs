/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::io::BufRead;

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Compress a file containing one string per line using a Rear Coded List", long_about = None)]

struct Args {
    /// A file containing UTF-8 strings, one per line.
    filename: String,
    #[arg(short, long)]
    /// A name for the ε-serde serialized RCL. Defaults to the "{filename}.rcl" if not specified.
    dst: Option<String>,
    /// The number of high bits defining the number of buckets. Very large key sets may benefit from a larger number of buckets.
    #[arg(short = 'k', long, default_value_t = 8)]
    k: usize,
}

fn compress<R: BufRead>(file: R, args: Args) {
    let dst_file_path = args.dst.unwrap_or(format!("{}.rcl", args.filename));
    let dst_file = std::fs::File::create(dst_file_path).unwrap();
    let mut dst_file = std::io::BufWriter::new(dst_file);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Reading the input file...");

    let mut rclb = RearCodedListBuilder::new(args.k);

    for line in file.lines() {
        let line = line.unwrap();
        rclb.push(line.as_str());
        pl.light_update();
    }
    pl.done();

    rclb.print_stats();

    let rcl = rclb.build();

    rcl.serialize(&mut dst_file).unwrap();
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let args = Args::parse();

    let file = std::fs::File::open(&args.filename).unwrap();
    let file = std::io::BufReader::new(file);
    compress(file, args);

    Ok(())
}
