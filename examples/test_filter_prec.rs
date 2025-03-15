/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use std::ops::{BitXor, BitXorAssign};

use anyhow::Result;
use average::{Estimate, MeanWithError};
use clap::Parser;
use common_traits::{UnsignedInt, UpcastableFrom};
use dsi_progress_logger::{no_logging, progress_logger, ProgressLog};
use epserde::prelude::*;
use rdst::RadixKey;
use sux::{
    bits::BitFieldVec,
    dict::VFilter,
    func::{
        shard_edge::{FuseLge3NoShards, FuseLge3Shards, ShardEdge},
        VBuilder, VFunc,
    },
    utils::{EmptyVal, FromIntoIterator, Sig, SigVal, ToSig},
};
#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    /// Number of keys
    n: usize,
    /// Precision
    dict: usize,
    /// Number of seeds
    s: usize,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
}

fn _main<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    usize: ToSig<S>,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    u128: UpcastableFrom<usize>,
    VFilter<usize, VFunc<usize, usize, BitFieldVec, S, E>>: TypeHash,
{
    let mut max = MeanWithError::new();
    let mut min = MeanWithError::new();
    let mut avg = MeanWithError::new();

    let mut pl = progress_logger![item_name = "sample"];

    pl.start("Sampling...");

    for seed in 0..args.s {
        let filter = VBuilder::<_, BitFieldVec<usize>, S, E>::default()
            .log2_buckets(4)
            .offline(false)
            .seed(seed as u64)
            .try_build_filter(FromIntoIterator::from(0..args.n), args.dict, no_logging![])?;

        let mut counts = vec![0; 1_usize << args.dict];
        let samples = args.n * args.n;
        for i in 0..samples {
            counts[filter.get(i + args.n)] += filter.contains(i + args.n) as usize;
        }

        let max_error_rate = counts
            .iter()
            .map(|&c| c as f64 / (samples as f64 / (1 << args.dict) as f64))
            .fold(-f64::INFINITY, |a, b| a.max(b));
        let min_error_rate = counts
            .iter()
            .map(|&c| c as f64 / (samples as f64 / (1 << args.dict) as f64))
            .fold(f64::INFINITY, |a, b| a.min(b));
        let avg_error_rate = counts.iter().sum::<usize>() as f64 / samples as f64;
        max.add(max_error_rate);
        min.add(min_error_rate);
        avg.add(avg_error_rate);
        println!(
            "Max Error rate: {} (1/{}); Stats: {} ± {} (1/{})",
            max_error_rate,
            1. / max_error_rate,
            max.mean(),
            max.error(),
            1. / max.mean()
        );
        println!(
            "Min Error rate: {} (1/{}); Stats: {} ± {} (1/{})",
            min_error_rate,
            1. / min_error_rate,
            min.mean(),
            min.error(),
            1. / min.mean()
        );
        println!(
            "Avg Error rate: {} (1/{}); Stats: {} ± {} (1/{})",
            avg_error_rate,
            1. / avg_error_rate,
            avg.mean(),
            avg.error(),
            1. / avg.mean()
        );

        pl.update();
    }

    pl.done();

    Ok(())
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    if args.sig64 {
        _main::<[u64; 1], FuseLge3NoShards>(args)?;
    } else {
        _main::<[u64; 2], FuseLge3Shards>(args)?;
    }

    Ok(())
}
