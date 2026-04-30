/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use fallible_iterator::FallibleIterator;
use lender::*;
use sux::{
    bits::{BitFieldVec, BitFieldVecU},
    func::{shard_edge::*, *},
    utils::{LineLender, Sig, ToSig, ZstdLineLender},
};

#[cfg(target_pointer_width = "64")]
const INCR: usize = 0x9e3779b97f4a7c15;
#[cfg(not(target_pointer_width = "64"))]
const INCR: usize = 0x9e3779b9;

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
#[command(about = "Benchmarks VFunc with strings or 64-bit integers.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// The maximum number of strings to read from the file, or the number of 64-bit keys.​
    n: usize,
    /// A name for the ε-serde serialized function.​
    func: String,
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.​
    filename: Option<String>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// The input file is compressed with zstd.​
    #[arg(short, long)]
    zstd: bool,
    /// Shard/edge type.​
    #[arg(long, value_enum, default_value_t)]
    edge: sux::cli::EdgeType,
    /// Use unaligned reads.​
    #[arg(long)]
    unaligned: bool,
}

fn main() -> Result<()> {
    use sux::cli::EdgeType;
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    match args.edge {
        EdgeType::Fuse3NoShards64 => main_with_types::<[u64; 1], Fuse3NoShards>(args),
        EdgeType::Fuse3NoShards128 => main_with_types::<[u64; 2], Fuse3NoShards>(args),
        EdgeType::Fuse3 => main_with_types::<[u64; 2], Fuse3Shards>(args),
        EdgeType::FuseLge3 => main_with_types::<[u64; 2], FuseLge3Shards>(args),
        EdgeType::FuseLge3FullSigs => main_with_types::<[u64; 2], FuseLge3FullSigs>(args),
        #[cfg(feature = "mwhc")]
        EdgeType::Mwhc3 => main_with_types::<[u64; 2], Mwhc3Shards>(args),
        #[cfg(feature = "mwhc")]
        EdgeType::Mwhc3NoShards => main_with_types::<[u64; 2], Mwhc3NoShards>(args),
    }
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>: Deserialize,
    VFunc<str, BitFieldVec<Box<[usize]>>, S, E>: Deserialize,
    VFunc<usize, BitFieldVecU<Box<[usize]>>, S, E>: Deserialize,
    VFunc<str, BitFieldVecU<Box<[usize]>>, S, E>: Deserialize,
{
    if let Some(filename) = args.filename {
        let keys: Vec<_> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        };

        if args.unaligned {
            let func =
                unsafe { VFunc::<str, BitFieldVecU<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                for key in &keys {
                    std::hint::black_box(func.get(key.as_str()));
                }
            });
        } else {
            let func =
                unsafe { VFunc::<str, BitFieldVec<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                for key in &keys {
                    std::hint::black_box(func.get(key.as_str()));
                }
            });
        }
    } else {
        if args.unaligned {
            let func =
                unsafe { VFunc::<usize, BitFieldVecU<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut key: usize = 0;
                for _ in 0..args.n {
                    key = key.wrapping_add(INCR);
                    std::hint::black_box(func.get(key));
                }
            });
        } else {
            let func =
                unsafe { VFunc::<usize, BitFieldVec<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut key: usize = 0;
                for _ in 0..args.n {
                    key = key.wrapping_add(INCR);
                    std::hint::black_box(func.get(key));
                }
            });
        }
    }
    Ok(())
}
