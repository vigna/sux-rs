/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
#![allow(clippy::collapsible_else_if)]
use std::ops::{BitXor, BitXorAssign};

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::{Serialize, SerializeInner};
use epserde::traits::{TypeHash, ZeroCopy};
use lender::Lender;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::func::{shard_edge::*, *};
use sux::prelude::VBuilder;
use sux::traits::{BitFieldSlice, Word};
use sux::utils::{EmptyVal, FromIntoIterator, LineLender, Sig, SigVal, ToSig, ZstdLineLender};

#[derive(Parser, Debug)]
#[command(about = "Creates a VFunc mapping each input to its rank and serializes it with ε-serde", long_about = None)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .args(&["filename", "n"]),
))]
struct Args {
    /// The number of keys. If no filename is provided, use the 64-bit keys
    /// [0..n).
    n: usize,
    /// A name for the ε-serde serialized function.
    func: Option<String>,
    #[arg(short, long)]
    /// A file containing UTF-8 keys, one per line. At most N keys will be read.
    filename: Option<String>,
    /// Use this number of threads.
    #[arg(short, long)]
    threads: Option<usize>,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// Use disk-based buckets to reduce memory usage at construction time.
    #[arg(short, long)]
    offline: bool,
    /// Sort shards and check for duplicate signatures.
    #[arg(short, long)]
    check_dups: bool,
    /// A 64-bit seed for the pseudorandom number generator.
    #[arg(long)]
    seed: Option<u64>,
    /// Use 64-bit signatures.
    #[arg(long, requires = "no_shards")]
    sig64: bool,
    /// Always use the low-mem peel-by-signature algorithm (slightly slower).
    #[arg(long)]
    low_mem: bool,
    /// Always use the high-mem peel-by-signature algorithm (slightly faster).
    #[arg(long, conflicts_with = "low_mem")]
    high_mem: bool,
    /// Do not use sharding.
    #[arg(long)]
    no_shards: bool,
    /// Use random 3-hypergraph.
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with = "sig64")]
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
            main_with_types::<[u64; 1], FuseLge3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], FuseLge3NoShards>(args)
        }
    } else {
        main_with_types::<[u64; 2], FuseLge3Shards>(args)
    }
}

fn set_builder<W: ZeroCopy + Word, D: BitFieldSlice<W> + Send + Sync, S, E: ShardEdge<S, 3>>(
    builder: VBuilder<W, D, S, E>,
    args: &Args,
) -> VBuilder<W, D, S, E> {
    let mut builder = builder
        .offline(args.offline)
        .check_dups(args.check_dups)
        .expected_num_keys(args.n);
    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }
    if let Some(threads) = args.threads {
        builder = builder.max_num_threads(threads);
    }
    if args.low_mem {
        builder = builder.low_mem(true);
    }
    if args.high_mem {
        builder = builder.low_mem(false);
    }
    builder
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    <E as SerializeInner>::SerType: ShardEdge<S, 3>, // Weird
    VFunc<usize, usize, BitFieldVec, S, E>: Serialize,
    VFunc<str, usize, BitFieldVec, S, E>: Serialize,
    VFunc<usize, u8, Box<[u8]>, S, E>: Serialize,
    VFunc<str, u8, Box<[u8]>, S, E>: Serialize + TypeHash, // TODO: this is weird
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    let n = args.n;

    if let Some(ref filename) = &args.filename {
        let builder = set_builder(VBuilder::<_, BitFieldVec<usize>, S, E>::default(), &args);
        let func = if args.zstd {
            builder.try_build_func(
                ZstdLineLender::from_path(filename)?.take(n),
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        } else {
            builder.try_build_func(
                LineLender::from_path(filename)?.take(n),
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?
        };
        if let Some(filename) = args.func {
            func.store(filename)?;
        }
    } else {
        let builder = set_builder(VBuilder::<_, BitFieldVec<usize>, S, E>::default(), &args);
        let func = builder.try_build_func(
            FromIntoIterator::from(0_usize..n),
            FromIntoIterator::from(0_usize..),
            &mut pl,
        )?;
        if let Some(filename) = args.func {
            func.store(filename)?;
        }
    }

    Ok(())
}
