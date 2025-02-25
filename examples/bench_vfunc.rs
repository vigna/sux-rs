/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

 #![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::prelude::*;
use lender::*;
use rdst::RadixKey;
use sux::{
    bits::BitFieldVec,
    func::{ShardEdge, VFilter, VFunc},
    utils::{LineLender, Sig, SigVal, ToSig, ZstdLineLender},
};

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    #[arg(short)]
    /// The maximum number strings to use from the file, or the number of 64-bit keys.
    n: usize,
    /// A name for the ε-serde serialized function with u64 keys.
    func: String,
    /// The function is an approximate 8-bit dictionary
    #[arg(short, long)]
    dict: bool,
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
            main_with_types::<[u64;1], false>(args)
        } else {
            main_with_types::<[u64; 2], false>(args)
        }
    } else {
        if args.sig64 {
            main_with_types::<[u64;1], true>(args)
        } else {
            main_with_types::<[u64; 2], true>(args)
        }
    }
}

fn main_with_types<S: Sig + ShardEdge, const SHARDED: bool>(args: Args) -> Result<()>
where
    SigVal<S, usize>: RadixKey,
    SigVal<S, ()>: RadixKey,
    str: ToSig<S>,
    usize: ToSig<S>,
    VFunc<str, usize, BitFieldVec, S, SHARDED>: Deserialize,
    VFunc<usize, usize, BitFieldVec, S, SHARDED>: Deserialize,
    VFunc<usize, u8, Vec<u8>, S, SHARDED>: Deserialize,
    VFunc<str, u8, Vec<u8>, S, SHARDED>: Deserialize + TypeHash, // TODO: this is weird
    VFilter<u8, VFunc<usize, u8, Vec<u8>, S, SHARDED>>: Deserialize,
{
    let mut pl = ProgressLogger::default();

    if let Some(filename) = args.filename {
        let mut keys: Vec<_> = if args.zstd {
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

        if args.dict {
            let filter = VFilter::<u8, VFunc<str, u8, Vec<u8>, S, SHARDED>>::load_full(&args.func)?;

            pl.start("Querying (independent)...");
            for key in &keys {
                std::hint::black_box(filter.get(key));
            }
            pl.done_with_count(args.n);

            pl.start("Querying (dependent)...");
            let mut x = 0;
            for key in &mut keys {
                debug_assert!(!key.is_empty());
                unsafe {
                    // This as horrible as it can be, and will probably
                    // do harm if a key is the empty string, but we avoid
                    // testing
                    *key.as_bytes_mut().get_unchecked_mut(0) ^= x & 1;
                }
                x = std::hint::black_box(filter.get(key));
            }
            pl.done_with_count(args.n);
        } else {
            let func = VFunc::<str, usize, BitFieldVec, S, SHARDED>::load_full(&args.func)?;

            pl.start("Querying (independent)...");
            for key in &keys {
                std::hint::black_box(func.get(key));
            }
            pl.done_with_count(args.n);

            pl.start("Querying (dependent)...");
            let mut x = 0;
            for key in &mut keys {
                debug_assert!(!key.is_empty());
                unsafe {
                    // This as horrible as it can be, and will probably
                    // do harm if a key is the empty string, but we avoid
                    // testing
                    *key.as_bytes_mut().get_unchecked_mut(0) ^= (x & 1) as u8;
                }
                x = func.get(key);
                std::hint::black_box(());
            }
            pl.done_with_count(args.n);
        }
    } else {
        // No filename
        if args.dict {
            let filter =
                VFilter::<u8, VFunc<usize, u8, Vec<u8>, S, SHARDED>>::load_full(&args.func)?;

            pl.start("Querying (independent)...");
            for i in 0..args.n {
                std::hint::black_box(filter.get(&i));
            }
            pl.done_with_count(args.n);

            pl.start("Querying (dependent)...");
            let mut x = 0;
            for i in 0..args.n {
                x = filter.contains(&(i ^ (x & 1))) as usize;
                std::hint::black_box(());
            }
            pl.done_with_count(args.n);
        } else {
            let func =
                VFunc::<usize, usize, BitFieldVec<usize>, S, SHARDED>::load_full(&args.func)?;

            pl.start("Querying (independent)...");
            for i in 0..args.n {
                std::hint::black_box(func.get(&i));
            }
            pl.done_with_count(args.n);

            pl.start("Querying (dependent)...");
            let mut x = 0;
            for i in 0..args.n {
                x = std::hint::black_box(func.get(&(i ^ (x & 1))));
            }
            pl.done_with_count(args.n);
        }
    }
    Ok(())
}
