/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use lender::*;
use sux::{
    bits::BitFieldVec,
    func::{shard_edge::*, *},
    utils::{LineLender, Sig, ToSig, ZstdLineLender},
};

fn bench(n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
        eprintln!("{} ns/key", timings.last().unwrap());
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "Min: {} Median: {} Max: {} Average: {}",
        timings[0],
        timings[timings.len() / 2],
        timings.last().unwrap(),
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The maximum number of strings to read from the file, or the number of 64-bit keys.
    n: usize,
    /// A name for the ε-serde serialized function.
    func: String,
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    /// The number of repetitions.
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
    /// Do not use sharding.
    #[arg(long)]
    no_shards: bool,
    /// Use slower edge logic reducing the probability of duplicate arcs for big
    /// shards.
    #[arg(long, conflicts_with = "sig64")]
    big_shards: bool,
    /// Use 3-hypergraphs.
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with_all = ["sig64", "big_shards"])]
    mwhc: bool,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    #[cfg(feature = "mwhc")]
    if args.mwhc {
        return if args.no_shards {
            main_with_types::<[u64; 2], Mwhc3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], Mwhc3Shards>(args)
        };
    }
    if args.no_shards {
        if args.sig64 {
            // TODO
            main_with_types::<[u64; 1], FuseLge3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], FuseLge3NoShards>(args)
        }
    } else {
        if args.big_shards {
            main_with_types::<[u64; 2], FuseLge3BigShards>(args)
        } else {
            main_with_types::<[u64; 2], FuseLge3Shards>(args)
        }
    }
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    VFunc<usize, usize, BitFieldVec, S, E>: Deserialize,
    VFunc<str, usize, BitFieldVec, S, E>: Deserialize,
{
    if let Some(filename) = args.filename {
        let keys: Vec<_> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| x.unwrap().to_owned())
                .take(args.n)
                .collect()
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| x.unwrap().to_owned())
                .take(args.n)
                .collect()
        };

        let func = VFunc::<str, usize, BitFieldVec, S, E>::load_full(&args.func)?;
        bench(args.n, args.repeats, || {
            for key in &keys {
                std::hint::black_box(func.get(key.as_str()));
            }
        });
    } else {
        // No filename
        let func = VFunc::<usize, usize, BitFieldVec<usize>, S, E>::load_full(&args.func)?;
        bench(args.n, args.repeats, || {
            let mut key: usize = 0;
            for _ in 0..args.n {
                key = key.wrapping_add(0x9e3779b97f4a7c15);
                std::hint::black_box(func.get(key));
            }
        });
    }
    Ok(())
}
