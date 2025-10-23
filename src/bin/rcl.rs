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
use sux::{init_env_logger, prelude::*, utils::LineLender};

#[derive(Parser, Debug)]
#[command(about = "Builds a rear-coded list starting from a list of UTF-8 encoded strings.", long_about = None)]
struct Args {
    /// A file containing UTF-8 strings, one per line, or - for standard input.
    source: String,
    /// A name for the ε-serde serialized rear-coded list.
    dest: String,
    /// The list of strings is not sorted (no checks, no indexing).
    #[arg(long, default_value_t = false)]
    unsorted: bool,
    /// The number of strings in a block. Higher values provide more compression
    /// at the expense of slower access.
    #[arg(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Use the slower direct-to-disk construction algorithm, which uses very little memory.
    #[arg(long, default_value_t = false)]
    low_mem: bool,
}

fn compress<BR: BufRead, const SORTED: bool>(
    buf_read: BR,
    dest: impl Borrow<str>,
    k: usize,
) -> Result<()> {
    let mut rclb = RearCodedListBuilder::<SORTED>::new(k);

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
    unsafe {
        rcl.serialize(&mut dst_file)
            .expect("Cannot serialize rear-coded list")
    };
    Ok(())
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    if args.low_mem {
        if args.source == "-" {
            panic!("Low-memory mode cannot read from standard input");
        }
        let lender = LineLender::from_path(&args.source)?;
        if args.unsorted {
            rear_coded_list::store::<_, _, false>(args.k, lender, args.dest)?;
        } else {
            rear_coded_list::store::<_, _, true>(args.k, lender, args.dest)?;
        }
    } else {
        if args.source == "-" {
            let stdin = std::io::stdin();
            let stdin = stdin.lock();
            if args.unsorted {
                compress::<_, false>(stdin, args.dest, args.k)?;
            } else {
                compress::<_, true>(stdin, args.dest, args.k)?;
            }
        } else {
            let file = std::fs::File::open(&args.source).expect("Cannot open source file");
            let buf_ref = std::io::BufReader::new(file);
            if args.unsorted {
                compress::<_, false>(buf_ref, args.dest, args.k)?;
            } else {
                compress::<_, true>(buf_ref, args.dest, args.k)?;
            }
        }
    }

    Ok(())
}
