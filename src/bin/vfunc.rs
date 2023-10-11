/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::ProgressLogger;
use epserde::ser::Serialize;
use sux::func::vfunc::VFunc;
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
    #[arg(short, long)]
    /// A file containing UTF-8 keys, one per line.
    filename: Option<String>,
    #[arg(short)]
    /// Use the 64-bit keys [0..n).
    n: Option<usize>,
    /// A name for the ε-serde serialized function.
    func: String,
    /// The filename containing the keys is compressed with zstd.
    #[arg(short, long)]
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
            VFunc::<_>::new_offline(
                FilenameZstdIntoIterator(&filename),
                &(0..),
                &mut Some(&mut pl),
            )?
        } else {
            VFunc::<_>::new_offline(FilenameIntoIterator(&filename), &(0..), &mut Some(&mut pl))?
        };
        func.store(&args.func)?;
    }

    if let Some(n) = args.n {
        let func = VFunc::<_>::new_offline(0..n, &(0..), &mut Some(&mut pl))?;

        func.store(&args.func)?;
    }
    Ok(())
}
