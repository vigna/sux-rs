/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::ProgressLogger;
use epserde::prelude::*;
use sux::func::vigna::Function;
use sux::prelude::CompactArray;
use sux::utils::file::FilenameIntoIterator;
use sux::utils::FilenameZstdIntoIterator;

#[derive(Parser, Debug)]
#[command(about = "Generate a function and serialize it with ε-serde", long_about = None)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .args(&["filename", "n"]),
))]
struct Args {
    #[arg(short = 'f', long)]
    // A file containing UTF-8 keys, one per line.
    filename: Option<String>,
    #[arg(short)]
    // Use the 64-bit keys [0..n).
    n: Option<usize>,
    // A name for the ε-serde serialized function.
    func: String,
    // The filename containing the keys is compressed with zstd.
    #[arg(short)]
    zstd: bool,
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let args = Args::parse();

    let mut pl = ProgressLogger::default();

    if let Some(filename) = args.filename {
        let func = if args.zstd {
            Function::<_>::new(
                FilenameZstdIntoIterator(&filename),
                &(0..),
                &mut Some(&mut pl),
            )
        } else {
            Function::<_>::new_offline(FilenameIntoIterator(&filename), &(0..), &mut Some(&mut pl))?
        };
        func.store(&args.func)?;
    }

    if let Some(n) = args.n {
        let func = Function::<_, CompactArray<Vec<usize>>>::new_offline(
            0..n as u64,
            &(0..),
            &mut Some(&mut pl),
        )?;

        func.store(&args.func)?;
    }
    Ok(())
}
