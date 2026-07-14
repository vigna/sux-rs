/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use std::{borrow::Borrow, io::BufRead, num::NonZeroUsize};

use anyhow::{Context, Result, bail};
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
#[command(about = "Builds a rear-coded list starting from a list of UTF-8 encoded strings.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// A file containing UTF-8 strings, one per line, or - for standard input; it can be compressed with any format supported by the deko crate.​
    source: String,
    /// A name for the ε-serde serialized rear-coded list.​
    dest: String,
    /// Assume the input is not sorted (disables checks and indexing).​
    #[arg(long)]
    unsorted: bool,
    /// The number of strings in a block: higher values provide more compression
    /// at the expense of slower access.​
    #[arg(short = 'r', long, default_value_t = NonZeroUsize::new(8).expect("nonzero default"))]
    ratio: NonZeroUsize,
    /// Use the slower direct-to-disk construction algorithm, which uses very little memory (cannot be used with stdin input).​
    #[arg(long)]
    low_mem: bool,
}

fn compress<R: BufRead, const SORTED: bool>(
    mut lender: DekoBufLineLender<R>,
    dest: impl Borrow<str>,
    ratio: usize,
) -> Result<()> {
    let mut rclb = RearCodedListBuilder::<str, SORTED>::new(ratio);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Reading the input file...");

    // Track the previous line so unsorted input is reported as a clean error
    // instead of panicking inside RearCodedListBuilder::push (SORTED only).
    let mut prev = String::new();
    let mut have_prev = false;

    loop {
        match lender.next() {
            Ok(None) => break,
            Ok(Some(line)) => {
                if SORTED {
                    if have_prev && line < prev.as_str() {
                        bail!(
                            "input is not sorted ({line:?} follows {prev:?}); pass --unsorted to build an unsorted list"
                        );
                    }
                    prev.clear();
                    prev.push_str(line);
                    have_prev = true;
                }
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
    // SAFETY: `rcl` was built by `RearCodedListBuilder`, so all serialized
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
        // destination (`store_str` calls `File::create`) before the two
        // streaming passes read the source, so `rcl f f --low-mem` would
        // destroy `f` and emit an empty list.
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
        let lender = DekoBufLineLender::from_path(&args.source)?;
        call_with_sorted!(
            args.unsorted,
            rear_coded_list::store_str,
            args.ratio.get(),
            lender,
            args.dest
        )?;
    } else if args.source == "-" {
        let stdin = DekoBufLineLender::new(std::io::BufReader::new(std::io::stdin().lock()))?;
        call_with_sorted!(args.unsorted, compress, stdin, args.dest, args.ratio.get())?;
    } else {
        let lender = DekoBufLineLender::from_path(&args.source)?;
        call_with_sorted!(args.unsorted, compress, lender, args.dest, args.ratio.get())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_with_lines(lines: &[&str]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        f
    }

    #[test]
    fn compress_sorted_rejects_unsorted_input() {
        let src = temp_with_lines(&["banana", "apple"]);
        let lender = DekoBufLineLender::from_path(src.path().to_str().unwrap()).unwrap();
        let dest = tempfile::NamedTempFile::new().unwrap();
        let err = compress::<_, true>(lender, dest.path().to_str().unwrap(), 8).unwrap_err();
        assert!(err.to_string().contains("not sorted"), "got: {err}");
    }

    #[test]
    fn compress_sorted_accepts_duplicates() {
        // Equal adjacent keys are valid in a sorted rear-coded list.
        let src = temp_with_lines(&["apple", "apple", "banana"]);
        let lender = DekoBufLineLender::from_path(src.path().to_str().unwrap()).unwrap();
        let dest = tempfile::NamedTempFile::new().unwrap();
        compress::<_, true>(lender, dest.path().to_str().unwrap(), 8).unwrap();
    }

    #[test]
    fn store_str_sorted_rejects_unsorted_input() {
        // The low-memory path streams through rear_coded_list::store_str, which
        // must also reject unsorted input gracefully rather than panicking.
        let src = temp_with_lines(&["banana", "apple"]);
        let lender = DekoBufLineLender::from_path(src.path().to_str().unwrap()).unwrap();
        let dest = tempfile::NamedTempFile::new().unwrap();
        let err = rear_coded_list::store_str::<str, _, true>(8, lender, dest.path()).unwrap_err();
        assert!(err.to_string().contains("not sorted"), "got: {err}");
    }

    #[test]
    fn store_str_sorted_accepts_duplicates() {
        // Equal adjacent keys are valid on the low-memory path too.
        let src = temp_with_lines(&["apple", "apple", "banana"]);
        let lender = DekoBufLineLender::from_path(src.path().to_str().unwrap()).unwrap();
        let dest = tempfile::NamedTempFile::new().unwrap();
        rear_coded_list::store_str::<str, _, true>(8, lender, dest.path()).unwrap();
    }
}
