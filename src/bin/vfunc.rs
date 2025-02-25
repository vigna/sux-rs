/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use epserde::traits::{TypeHash, ZeroCopy};
use lender::Lender;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::func::{ShardEdge, VFilter, VFunc};
use sux::prelude::VBuilder;
use sux::traits::{BitFieldSlice, Word};
use sux::utils::{FromIntoIterator, LineLender, Sig, SigVal, ToSig, ZstdLineLender};

#[derive(Parser, Debug)]
#[command(about = "Generate a VFunc mapping each input to its rank and serialize it with ε-serde", long_about = None)]
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
    func: String,
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
    /// The number of high bits defining the number of buckets. Very large key
    /// sets may benefit from a larger number of buckets.
    #[arg(short = 'H', long, default_value_t = 8)]
    high_bits: u32,
    /// Create an approximate 8-bit dictionary
    #[arg(short, long)]
    dict: bool,
    /// A 64-bit seed for the pseudorandom number generator.
    #[arg(long)]
    seed: Option<u64>,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
    /// Do not use sharding.
    #[arg(long)]
    no_shards: bool,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    if args.no_shards {
        if args.sig64 {
            main_with_types::<[u64; 1], false>(args)
        } else {
            main_with_types::<[u64; 2], false>(args)
        }
    } else {if args.sig64 {
        main_with_types::<[u64;1], true>(args)
    } else {
        main_with_types::<[u64; 2], true>(args)
    }
}
}

fn set_up_builder<
    T: ?Sized + Send + Sync + ToSig<S>,
    W: ZeroCopy + Word,
    D: BitFieldSlice<W> + Send + Sync,
    S,
    const SHARDED: bool,
    V,
>(
    mut builder: VBuilder<T, W, D, S, SHARDED, V>,
    args: &Args,
) -> VBuilder<T, W, D, S, SHARDED, V> {
    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }
    if let Some(threads) = args.threads {
        builder = builder.max_num_threads(threads);
    }
    builder
}

fn main_with_types<S: Sig + ShardEdge, const SHARDED: bool>(args: Args) -> Result<()>
where
    SigVal<S, usize>: RadixKey,
    SigVal<S, ()>: RadixKey,
    str: ToSig<S>,
    usize: ToSig<S>,
    VFunc<str, usize, BitFieldVec, S, SHARDED>: Serialize,
    VFunc<usize, usize, BitFieldVec, S, SHARDED>: Serialize,
    VFunc<usize, u8, Vec<u8>, S, SHARDED>: Serialize,
    VFunc<str, u8, Vec<u8>, S, SHARDED>: Serialize + TypeHash, // TODO: this is weird
    VFilter<u8, VFunc<usize, u8, Vec<u8>, S, SHARDED>>: Serialize,
{
    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    let n = args.n;

    if let Some(ref filename) = &args.filename {
        if args.dict {
            let mut builder = VBuilder::<str, u8, Vec<u8>, S, SHARDED, ()>::default()
                .offline(args.offline)
                .log2_buckets(args.high_bits);
            builder = set_up_builder(builder, &args);
            let filter = if args.zstd {
                builder.try_build_filter(ZstdLineLender::from_path(filename)?.take(n), &mut pl)?
            } else {
                let t = LineLender::from_path(filename)?.take(n);
                builder.try_build_filter(t, &mut pl)?
            };
            filter.store(&args.func)?;
        } else {
            let mut builder = VBuilder::<_, _, BitFieldVec<usize>, S, SHARDED>::default()
                .offline(args.offline)
                .log2_buckets(args.high_bits);
            builder = set_up_builder(builder, &args);
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
            func.store(&args.func)?;
        }
    } else if args.dict {
        let mut builder = VBuilder::<_, u8, Vec<u8>, S, SHARDED, ()>::default()
            .offline(args.offline)
            .expected_num_keys(n)
            .log2_buckets(args.high_bits);
        builder = set_up_builder(builder, &args);
        let filter = builder.try_build_filter(FromIntoIterator::from(0_usize..n), &mut pl)?;
        filter.store(&args.func)?;
    } else {
        let mut builder = VBuilder::<_, _, BitFieldVec<usize>, S, SHARDED>::default()
            .offline(args.offline)
            .expected_num_keys(n)
            .log2_buckets(args.high_bits);
        builder = set_up_builder(builder, &args);
        let func = builder.try_build_func(
            FromIntoIterator::from(0_usize..n),
            FromIntoIterator::from(0_usize..),
            &mut pl,
        )?;
        func.store(&args.func)?;
    }

    Ok(())
}
