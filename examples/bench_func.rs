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

#[derive(Parser, Debug)]
#[command(about = "Functions", long_about = None)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .args(&["filename", "n"]),
))]
struct Args {
    // A name for the ε-serde serialized function with u64 keys.
    func: String,
    #[arg(short)]
    // Use the 64-bit keys [0..n).
    n: usize,
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let mut pl = ProgressLogger::default();

    let args = Args::parse();
    let func = Function::<_>::load_mem(&args.func)?;
    pl.start("Querying...");
    for i in 0..args.n {
        assert_eq!(i, func.get(&i));
    }
    pl.done_with_count(args.n);

    Ok(())
}
