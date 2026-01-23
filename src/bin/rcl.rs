/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use std::{borrow::Borrow, io::BufRead};

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use sux::{init_env_logger, prelude::*, utils::DekoBufLineLender};

/// Macro to handle the repeated pattern of checking args.unsorted
/// and calling a function with the appropriate const generic parameter.
/// When args.unsorted is true, we pass false for SORTED (meaning not sorted).
/// When args.unsorted is false, we pass true for SORTED.
macro_rules! call_with_sorted {
    // For compress function with single type parameter
    ($unsorted:expr, compress, $($args:expr),*) => {
        if $unsorted {
            compress::<_, false>($($args),*)
        } else {
            compress::<_, true>($($args),*)
        }
    };
    // For rear_coded_list::store with one type parameter and one const generic
    ($unsorted:expr, rear_coded_list::store_str, $($args:expr),*) => {
        if $unsorted {
            rear_coded_list::store_str::<_,_, false>($($args),*)
        } else {
            rear_coded_list::store_str::<_,_,  true>($($args),*)
        }
    };
}

#[derive(Parser, Debug)]
#[command(about = "Builds a rear-coded list starting from a list of UTF-8 encoded strings.", long_about = None)]
struct Args {
    /// A file containing UTF-8 strings, one per line, or - for standard input; it can be compressed with any format supported by the deko crate.
    source: String,
    /// A name for the ε-serde serialized rear-coded list.
    dest: String,
    /// The list of strings is not sorted (no checks, no indexing).
    #[arg(long, default_value_t = false)]
    unsorted: bool,
    /// The number of strings in a block: higher values provide more compression
    /// at the expense of slower access.
    #[arg(short = 'r', long, default_value_t = 8)]
    ratio: usize,
    /// Use the slower direct-to-disk construction algorithm, which uses very little memory (cannot be used with stdin input).
    #[arg(long, default_value_t = false)]
    low_mem: bool,
}

fn compress<R: BufRead, const SORTED: bool>(
    mut lender: DekoBufLineLender<R>,
    dest: impl Borrow<str>,
    ratio: usize,
) -> Result<()> {
    let mut rclb = RearCodedListBuilder::<str, Vec<usize>, SORTED>::new(ratio);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Reading the input file...");

    loop {
        match lender.next() {
            Ok(None) => break,
            Ok(Some(line)) => {
                rclb.push(line);
            }
            Err(e) => {
                pl.info(format_args!("Error reading line: {}", e));
                return Result::Err(e.into());
            }
        }
        pl.light_update();
    }

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
        let lender = DekoBufLineLender::from_path(&args.source)?;
        call_with_sorted!(
            args.unsorted,
            rear_coded_list::store_str,
            args.ratio,
            lender,
            args.dest
        )?;
    } else if args.source == "-" {
        let stdin = DekoBufLineLender::new(std::io::BufReader::new(std::io::stdin().lock()))?;
        call_with_sorted!(args.unsorted, compress, stdin, args.dest, args.ratio)?;
    } else {
        let lender = DekoBufLineLender::from_path(&args.source)?;
        call_with_sorted!(args.unsorted, compress, lender, args.dest, args.ratio)?;
    }

    Ok(())
}
