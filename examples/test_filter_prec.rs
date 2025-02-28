/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use average::{Estimate, MeanWithError};
use clap::Parser;
use dsi_progress_logger::{no_logging, progress_logger, ProgressLog};
use epserde::prelude::*;
use rdst::RadixKey;
use sux::{
    bits::BitFieldVec,
    func::{Fuse3Shards, ShardEdge, VBuilder, VFilter, VFunc},
    utils::{FromIntoIterator, Sig, SigVal, ToSig},
};
#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    /// Number of keys
    n: usize,
    /// Precision
    dict: u32,
    /// Sample size
    s: usize,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
}

fn _main<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    SigVal<S, ()>: RadixKey,
    usize: ToSig<S>,
    VFilter<usize, VFunc<usize, BitFieldVec, S, E>>: TypeHash,
{
    let mut m = MeanWithError::new();

    let mut pl = progress_logger![item_name = "sample"];

    pl.start("Sampling...");

    for seed in 0..args.s {
        let filter = VBuilder::<_, usize, BitFieldVec<usize>, S, E, ()>::default()
            .log2_buckets(4)
            .offline(false)
            .seed(seed as u64)
            .try_build_filter(FromIntoIterator::from(0..args.n), args.dict, no_logging![])?;

        let mut c = 0;
        for i in 0..args.n {
            c += filter.contains(&(i + args.n)) as usize;
        }

        let error_rate = c as f64 / args.n as f64;
        m.add(error_rate);
        println!(
            "Error rate: {} (1/{}); Stats: {} ± {} (1/{})",
            error_rate,
            1. / error_rate,
            m.mean(),
            m.error(),
            1. / m.mean()
        );
        pl.update();
    }

    pl.done();

    println!("{} ± {}", m.mean(), m.error());

    Ok(())
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    if args.sig64 {
        _main::<[u64; 1], Fuse3Shards>(args)?;
    } else {
        _main::<[u64; 2], Fuse3Shards>(args)?;
    }

    Ok(())
}
