/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::hint::black_box;

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use sux::{func::VBuilder, utils::FromIntoIterator};

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The number of integers in the function
    n: usize,
    #[arg(short)]
    /// The number of samples
    s: usize,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let mut pl = ProgressLogger::default();

    let args = Args::parse();

    pl.start("Building...");
    let builder = VBuilder::<u64, u16, Vec<_>, [u64; 2]>::default();

    let vfunc = builder.try_build_func(
        FromIntoIterator::from(0..args.n as u64),
        FromIntoIterator::from(0_u16..),
        &mut pl,
    )?;

    pl.done_with_count(args.n);

    pl.item_name("nodes");
    pl.start("Querying...");
    for i in 0..args.s as u64 {
        black_box(vfunc.get(&i));
    }
    pl.done_with_count(args.s);

    Ok(())
}
