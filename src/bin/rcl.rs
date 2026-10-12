/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */
use std::{borrow::Borrow, io::Read, num::NonZeroUsize};

use anyhow::{Context, Result, bail};
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use sux::{
    cli::{AnyLineLender, Decompression, init_env_logger},
    prelude::*,
};

#[derive(Parser, Debug)]
#[command(about = "Builds a rear-coded list starting from a list of UTF-8 encoded strings.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// A file containing UTF-8 strings, one per line, or - for standard input.​
    source: String,
    /// A name for the ε-serde serialized rear-coded list.​
    dest: String,
    /// Assume the input is not sorted (disables checks and indexing); the
    /// output must then be loaded as a `RearCodedListStr<false>`.​
    #[arg(long)]
    unsorted: bool,
    /// The number of strings in a block: higher values provide more compression
    /// at the expense of slower access.​
    #[arg(short = 'r', long, default_value_t = NonZeroUsize::new(8).expect("nonzero default"))]
    ratio: NonZeroUsize,
    /// Use the slower direct-to-disk construction algorithm, which uses very little memory (cannot be used with stdin input).​
    #[arg(long)]
    low_mem: bool,
    /// Decompress the input using this format (by default, it is read as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
}

fn compress<R: Read, const SORTED: bool>(
    mut lender: AnyLineLender<R>,
    dest: impl Borrow<str>,
    ratio: usize,
) -> Result<()> {
    let mut rclb = RearCodedListBuilder::<str, SORTED>::new(ratio);

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
    let dst_file = std::fs::File::create(dest.borrow())
        .with_context(|| format!("cannot create file '{}'", dest.borrow()))?;
    let mut dst_file = std::io::BufWriter::new(dst_file);
    // SAFETY: rcl was built by RearCodedListBuilder, so all serialized
    // representation invariants are established by safe construction.
    unsafe {
        rcl.serialize(&mut dst_file)
            .context("cannot serialize rear-coded list")?
    };
    std::io::Write::flush(&mut dst_file).context("cannot flush serialized rear-coded list")?;
    Ok(())
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    if args.low_mem {
        if args.source == "-" {
            bail!("low-memory mode cannot read from standard input");
        }
        // Refuse to clobber the input: low-memory mode truncates the
        // destination (store_str calls File::create) before the two
        // streaming passes read the source, so rcl f f --low-mem would
        // destroy f and emit an empty list.
        if let Ok(dst_canon) = std::fs::canonicalize(&args.dest) {
            let src_canon = std::fs::canonicalize(&args.source)
                .with_context(|| format!("cannot resolve source path '{}'", args.source))?;
            if src_canon == dst_canon {
                bail!(
                    "low-memory mode cannot write the output '{}' over its input '{}'",
                    args.dest,
                    args.source
                );
            }
        }
        let lender = AnyLineLender::from_path(&args.source, args.decompress)?;
        if args.unsorted {
            rear_coded_list::store_str::<_, _, false>(args.ratio.get(), lender, args.dest)?;
        } else {
            rear_coded_list::store_str::<_, _, true>(args.ratio.get(), lender, args.dest)?;
        }
    } else if args.source == "-" {
        let stdin = AnyLineLender::new(std::io::stdin().lock(), args.decompress)?;
        if args.unsorted {
            compress::<_, false>(stdin, args.dest, args.ratio.get())?;
        } else {
            compress::<_, true>(stdin, args.dest, args.ratio.get())?;
        }
    } else {
        let lender = AnyLineLender::from_path(&args.source, args.decompress)?;
        if args.unsorted {
            compress::<_, false>(lender, args.dest, args.ratio.get())?;
        } else {
            compress::<_, true>(lender, args.dest, args.ratio.get())?;
        }
    }

    Ok(())
}
