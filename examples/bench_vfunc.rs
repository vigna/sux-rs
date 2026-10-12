/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use sux::{
    bits::{BitFieldVec, BitFieldVecU},
    cli::{Decompression, INCR, bench, init_env_logger, pack_strings, reservoir_sample},
    func::{shard_edge::*, *},
    utils::{Sig, ToSig},
};

#[derive(Parser, Debug)]
#[command(about = "Benchmarks VFunc with strings or 64-bit integers.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// The number of queries to perform.​
    n: usize,
    /// A name for the ε-serde serialized function.​
    func: String,
    /// A file containing UTF-8 keys, one per line, from which queries are
    /// sampled. If not specified, queries are multiples of a large odd
    /// constant, and thus mostly not keys of a structure built with -n (but
    /// queries take the same time on keys and non-keys).​
    #[arg(short = 'f', long)]
    filename: Option<String>,
    /// Decompress the file of keys using this format (by default, it is read
    /// as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Shard/edge type.​
    #[arg(long = "shard-edge", short = 'E', value_enum, default_value_t)]
    shard_edge: sux::cli::ShardEdgeType,
    /// Use unaligned reads.​
    #[arg(long, short)]
    unaligned: bool,
}

fn main() -> Result<()> {
    use sux::cli::ShardEdgeType;
    init_env_logger()?;

    let args = Args::parse();

    match args.shard_edge {
        ShardEdgeType::Fuse3NoShards64 => main_with_types::<[u64; 1], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3NoShards128 => main_with_types::<[u64; 2], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3Shards => main_with_types::<[u64; 2], Fuse3Shards>(args),
        ShardEdgeType::FuseLge3Shards => main_with_types::<[u64; 2], FuseLge3Shards>(args),
        ShardEdgeType::FuseLge3FullSigs => main_with_types::<[u64; 2], FuseLge3FullSigs>(args),
        #[cfg(feature = "mwhc")]
        ShardEdgeType::Mwhc3 => main_with_types::<[u64; 2], Mwhc3Shards>(args),
        #[cfg(feature = "mwhc")]
        ShardEdgeType::Mwhc3NoShards => main_with_types::<[u64; 2], Mwhc3NoShards>(args),
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
    if let Some(ref filename) = args.filename {
        let (packed, packed_offsets) = {
            let queries = reservoir_sample(filename, args.n, 42, args.decompress)?;
            pack_strings(&queries, args.n)
        };

        if args.unaligned {
            let func =
                unsafe { VFunc::<str, BitFieldVecU<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for i in 0..args.n {
                    let s = packed_offsets[i];
                    let e = packed_offsets[i + 1];
                    let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                    u ^= func.get(q);
                }
                std::hint::black_box(u);
            });
        } else {
            let func =
                unsafe { VFunc::<str, BitFieldVec<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for i in 0..args.n {
                    let s = packed_offsets[i];
                    let e = packed_offsets[i + 1];
                    let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                    u ^= func.get(q);
                }
                std::hint::black_box(u);
            });
        }
    } else {
        if args.unaligned {
            let func =
                unsafe { VFunc::<usize, BitFieldVecU<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                let mut key: usize = 0;
                for _ in 0..args.n {
                    key = key.wrapping_add(INCR);
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        } else {
            let func =
                unsafe { VFunc::<usize, BitFieldVec<Box<[usize]>>, S, E>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                let mut key: usize = 0;
                for _ in 0..args.n {
                    key = key.wrapping_add(INCR);
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        }
    }
    Ok(())
}
