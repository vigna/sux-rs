/*
 *
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use std::{borrow::Borrow, io::BufRead};

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::for_;
use sux::{prelude::*, utils::LineLender};

#[derive(Parser, Debug)]
#[command(about = "Builds a rear-coded list starting from a list of UTF-8 encoded strings.", long_about = None)]
struct Args {
    /// A file containing UTF-8 strings, one per line, or - for standard input.
    source: String,
    /// A name for the ε-serde serialized rear-coded list.
    dest: String,
    /// The number of strings in a block. Higher values provide more compression
    /// at the expense of slower access.
    #[arg(short = 'k', long, default_value_t = 8)]
    k: usize,
}

fn compress<BR: BufRead>(buf_read: BR, dest: impl Borrow<str>, k: usize) -> Result<()> {
    let mut rclb = RearCodedListBuilder::new(k);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Reading the input file...");

    for_![result in LineLender::new(buf_read) {
        match result {
            Ok(line) => {
                rclb.push(line);
            }
            Err(e) => {
                pl.info(format_args!("Error reading line: {}", e));
                return Result::Err(e.into());
            }
        }
        pl.light_update();
    }];

    pl.done();

    rclb.print_stats();

    let rcl = rclb.build();
    let dst_file = std::fs::File::create(dest.borrow()).expect("Cannot create destination file");
    let mut dst_file = std::io::BufWriter::new(dst_file);
    rcl.serialize(&mut dst_file)
        .expect("Cannot serialize rear-coded list");
    Ok(())
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    if args.source == "-" {
        let stdin = std::io::stdin();
        let stdin = stdin.lock();
        compress(stdin, args.dest, args.k)?;
    } else {
        let file = std::fs::File::open(&args.source).expect("Cannot open source file");
        let buf_ref = std::io::BufReader::new(file);
        compress(buf_ref, args.dest, args.k)?;
    }

    Ok(())
}
