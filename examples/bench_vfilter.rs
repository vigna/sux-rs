/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use common_traits::{CastableFrom, DowncastableInto};
use dsi_progress_logger::*;
use epserde::prelude::*;
use lender::*;
use rdst::RadixKey;
use sux::{
    bits::BitFieldVec,
    dict::VFilter,
    func::{shard_edge::*, *},
    traits::{BitFieldSlice, Word},
    utils::{LineLender, Sig, SigVal, ToSig, ZstdLineLender},
};

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The maximum number strings to use from the file, or the number of 64-bit keys.
    n: usize,
    /// A name for the ε-serde serialized function with u64 keys.
    func: String,
    /// The number of bits of the hashes used by the filter.
    #[arg(short, long, default_value_t = 8)]
    bits: u32,
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
    /// Use slower edge logic reducing the probability of duplicate arcs for big
    /// shards.
    #[arg(long, conflicts_with_all = ["sig64", "mwhc"])]
    big_shards: bool,
    /// Use 3-hypergraphs.
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with = "sig64")]
    mwhc: bool,
}

macro_rules! fuse {
    ($args: expr, $main: ident, $ty: ty) => {
        if $args.no_shards {
            if $args.sig64 {
                $main::<$ty, [u64; 1], FuseLge3NoShards>($args)
            } else {
                $main::<$ty, [u64; 2], FuseLge3NoShards>($args)
            }
        } else {
            if $args.big_shards {
                $main::<$ty, [u64; 2], FuseLge3BigShards>($args)
            } else {
                $main::<$ty, [u64; 2], FuseLge3Shards>($args)
            }
        }
    };
}

#[cfg(feature = "mwhc")]
macro_rules! mwhc {
    ($args: expr, $main: ident, $ty: ty) => {
        if $args.no_shards {
            $main::<$ty, [u64; 2], Mwhc3NoShards>($args)
        } else {
            $main::<$ty, [u64; 2], Mwhc3Shards>($args)
        }
    };
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    #[cfg(feature = "mwhc")]
    if args.mwhc {
        return match args.bits {
            8 => mwhc!(args, main_with_types_boxed_slice, u8),
            16 => mwhc!(args, main_with_types_boxed_slice, u16),
            32 => mwhc!(args, main_with_types_boxed_slice, u32),
            64 => mwhc!(args, main_with_types_boxed_slice, u64),
            _ => mwhc!(args, main_with_types_bit_field_vec, u64),
        };
    }

    match args.bits {
        8 => fuse!(args, main_with_types_boxed_slice, u8),
        16 => fuse!(args, main_with_types_boxed_slice, u16),
        32 => fuse!(args, main_with_types_boxed_slice, u32),
        64 => fuse!(args, main_with_types_boxed_slice, u64),
        _ => fuse!(args, main_with_types_bit_field_vec, u64),
    }
}

fn main_with_types_boxed_slice<
    W: ZeroCopy + Word + CastableFrom<u64> + DowncastableInto<u8> + TypeHash + AlignHash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    args: Args,
) -> Result<()>
where
    SigVal<S, usize>: RadixKey,
    SigVal<S, ()>: RadixKey,
    str: ToSig<S>,
    String: ToSig<S>,
    usize: ToSig<S>,
    Box<[W]>: BitFieldSlice<W>,
    VFunc<usize, usize, BitFieldVec, S, E>: DeserializeInner + TypeHash + AlignHash,
    VFunc<str, usize, BitFieldVec, S, E>: DeserializeInner + TypeHash + AlignHash,
    VFunc<usize, W, Box<[W]>, S, E>: DeserializeInner + TypeHash + AlignHash,
    VFunc<str, W, Box<[W]>, S, E>: DeserializeInner + TypeHash + AlignHash, // TODO: this is weird
    VFilter<W, VFunc<usize, W, Box<[W]>, S, E>>: DeserializeInner + TypeHash + AlignHash,
    VFilter<W, VFunc<str, W, Box<[W]>, S, E>>: DeserializeInner + TypeHash + AlignHash,
{
    let mut pl = progress_logger![item_name = "key"];

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

        let filter = VFilter::<W, VFunc<str, W, Box<[W]>, S, E>>::load_full(&args.func)?;

        pl.start("Querying...");
        for key in &keys {
            std::hint::black_box(filter.get(key.as_str()));
        }
        pl.done_with_count(args.n);
    } else {
        // No filename
        let filter = VFilter::<W, VFunc<usize, W, Box<[W]>, S, E>>::load_full(&args.func)?;
        let mut key: usize = 0;

        pl.start("Querying...");
        for i in 0..args.n {
            key = key.wrapping_add(0x9e3779b97f4a7c15);
            std::hint::black_box(filter.contains(i));
        }
        pl.done_with_count(args.n);
    }
    Ok(())
}

fn main_with_types_bit_field_vec<
    W: ZeroCopy + Word + CastableFrom<u64> + DowncastableInto<u8> + TypeHash + AlignHash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    args: Args,
) -> Result<()>
where
    SigVal<S, usize>: RadixKey,
    SigVal<S, ()>: RadixKey,
    str: ToSig<S>,
    String: ToSig<S>,
    usize: ToSig<S>,
    VFunc<usize, usize, BitFieldVec, S, E>: DeserializeInner + TypeHash + AlignHash,
    VFunc<str, usize, BitFieldVec, S, E>: DeserializeInner + TypeHash + AlignHash,
    VFunc<usize, W, BitFieldVec<W>, S, E>: DeserializeInner + TypeHash + AlignHash,
    VFunc<str, W, BitFieldVec<W>, S, E>: DeserializeInner + TypeHash + AlignHash, // TODO: this is weird
    VFilter<W, VFunc<usize, W, BitFieldVec<W>, S, E>>: DeserializeInner + TypeHash + AlignHash,
    VFilter<W, VFunc<str, W, BitFieldVec<W>, S, E>>: DeserializeInner + TypeHash + AlignHash,
{
    let mut pl = progress_logger![item_name = "key"];

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

        let filter = VFilter::<W, VFunc<str, W, BitFieldVec<W>, S, E>>::load_full(&args.func)?;

        pl.start("Querying...");
        for key in &keys {
            std::hint::black_box(filter.get(key.as_str()));
        }
        pl.done_with_count(args.n);
    } else {
        // No filename
        let filter = VFilter::<W, VFunc<usize, W, BitFieldVec<W>, S, E>>::load_full(&args.func)?;

        pl.start("Querying...");
        for i in 0..args.n {
            std::hint::black_box(filter.contains(i));
        }
        pl.done_with_count(args.n);
    }
    Ok(())
}
