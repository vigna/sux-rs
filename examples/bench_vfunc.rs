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
    func::*,
    utils::{LineLender, Sig, SigVal, ToSig, ZstdLineLender},
};

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The maximum number strings to use from the file, or the number of 64-bit keys.
    n: usize,
    /// A name for the ε-serde serialized function with u64 keys.
    func: String,
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
    /// Do not use sharding.
    #[arg(long)]
    no_shards: bool,
    /// Use 3-hypergraphs.
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
            // TODO
            main_with_types::<[u64; 1], Fuse3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], Fuse3NoShards>(args)
        }
    } else {
        main_with_types::<[u64; 2], Fuse3Shards>(args)
    }
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    SigVal<S, usize>: RadixKey,
    SigVal<S, ()>: RadixKey,
    str: ToSig<S>,
    String: ToSig<S>,
    usize: ToSig<S>,
    VFunc<usize, usize, BitFieldVec, S, E>: Deserialize,
    VFunc<str, usize, BitFieldVec, S, E>: Deserialize,
    VFunc<usize, u8, Box<[u8]>, S, E>: Deserialize,
    VFunc<str, u8, Box<[u8]>, S, E>: Deserialize + TypeHash, // TODO: this is weird
    VFilter<u8, VFunc<usize, u8, Box<[u8]>, S, E>>: Deserialize,
{
    let mut pl = progress_logger![item_name = "key"];

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

        let func = VFunc::<str, usize, BitFieldVec, S, E>::load_full(&args.func)?;

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
    } else {
        // No filename
        let func = VFunc::<usize, usize, BitFieldVec<usize>, S, E>::load_full(&args.func)?;

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
    Ok(())
}
