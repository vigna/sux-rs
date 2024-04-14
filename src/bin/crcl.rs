/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
#![cfg(feature = "cli")]

use anyhow::{bail, Result};
use clap::Parser;
use dsi_bitstream::codes::huffman::HuffmanTree;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::{Lender, Lending};
use mem_dbg::{DbgFlags, MemDbg};
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
    mut lender_func: impl FnMut() -> L,
    args: Args,
) -> anyhow::Result<()> {
    let dst_file_path = args.dst.unwrap_or(format!("{}.crcl", args.filename));
    let dst_file = std::fs::File::create(dst_file_path).unwrap();
    let mut dst_file = std::io::BufWriter::new(dst_file);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Reading the input file...");

    let mut lines = lender_func();
    let mut counts = [0; 256];
    while let Some(result) = lines.next() {
        match result {
            Ok(line) => {
                pl.light_update();
                for c in line.as_ref().bytes() {
                    counts[c as usize] += 1;
                }
            }
            Err(e) => {
                bail!("Error reading input: {}", e);
            }
        }
    }

    pl.done();

    pl.start("Reading the input file...");

    let huffman = HuffmanTree::new(&counts)?;
    let mut rcab = <CompressedRearCodedListBuilder>::new(
        args.k,
        compressed_rcl::huffman::Encoder::new(
            huffman,
            compressed_rcl::ef::OffsetsBuilder::default(),
        ),
    );

    let mut lines = lender_func();
    while let Some(result) = lines.next() {
        match result {
            Ok(line) => {
                pl.light_update();
                rcab.push(line);
            }
            Err(e) => {
                bail!("Error reading input: {}", e);
            }
        }
    }

    pl.done();

    dbg!(rcab.stats());

    let rcl = rcab.build().unwrap();

    rcl.mem_dbg(DbgFlags::default()).unwrap();

    rcl.serialize(&mut dst_file)?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let filename = args.filename.clone();
    if args.zstd {
        compress(
            move || ZstdLineLender::new(std::fs::File::open(&filename).unwrap()).unwrap(),
            args,
        )?;
    } else {
        compress(
            || LineLender::new(BufReader::new(std::fs::File::open(&filename).unwrap())),
            args,
        )?;
    }

    Ok(())
}
