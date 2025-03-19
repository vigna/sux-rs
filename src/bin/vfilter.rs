/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
#![allow(clippy::collapsible_else_if)]
use std::ops::{BitXor, BitXorAssign};

use anyhow::Result;
use clap::{ArgGroup, Parser};
use common_traits::{CastableFrom, UpcastableFrom};
use dsi_progress_logger::*;
use epserde::ser::{Serialize, SerializeInner};
use epserde::traits::{AlignHash, TypeHash, ZeroCopy};
use lender::Lender;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::dict::VFilter;
use sux::func::{shard_edge::*, *};
use sux::prelude::VBuilder;
use sux::traits::{BitFieldSlice, BitFieldSliceMut, Word};
use sux::utils::{EmptyVal, FromIntoIterator, LineLender, Sig, SigVal, ToSig, ZstdLineLender};

#[derive(Parser, Debug)]
#[command(about = "Creates a VFilter and serializes it with ε-serde", long_about = None)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .args(&["filename", "n"]),
))]
struct Args {
    /// The number of keys. If no filename is provided, use the 64-bit keys
    /// [0..n).
    n: usize,
    /// An optional name for the ε-serde serialized function.
    filter: Option<String>,
    /// The number of bits of the hashes used by the filter.
    #[arg(short, long, default_value_t = 8)]
    bits: usize,
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
    /// The target relative space overhead due to sharding.
    #[arg(long, default_value_t = 0.001)]
    eps: f64,
    /// Always use the low-mem peel-by-signature algorithm (slightly slower).
    #[arg(long)]
    low_mem: bool,
    /// Always use the high-mem peel-by-signature algorithm (slightly faster).
    #[arg(long, conflicts_with = "low_mem")]
    high_mem: bool,
    /// Do not use sharding.
    #[arg(long)]
    no_shards: bool,
    /// Use slower edge logic reducing the probability of duplicate arcs for big
    /// shards.
    #[arg(long, conflicts_with = "sig64")]
    full_sigs: bool,
    /// Use 3-hypergraphs.
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with_all = ["sig64", "full_sigs"])]
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
            if $args.full_sigs {
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

fn set_builder<W: ZeroCopy + Word, D: BitFieldSlice<W> + Send + Sync, S, E: ShardEdge<S, 3>>(
    builder: VBuilder<W, D, S, E>,
    args: &Args,
) -> VBuilder<W, D, S, E> {
    let mut builder = builder
        .offline(args.offline)
        .check_dups(args.check_dups)
        .expected_num_keys(args.n)
        .eps(args.eps);
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

fn main_with_types_boxed_slice<
    W: Word + ZeroCopy + Send + Sync + CastableFrom<u64> + SerializeInner + TypeHash + AlignHash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    args: Args,
) -> Result<()>
where
    <W as SerializeInner>::SerType: Word + ZeroCopy,
    str: ToSig<S>,
    usize: ToSig<S>,
    u128: UpcastableFrom<W>,
    SigVal<S, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    Box<[W]>: BitFieldSlice<W> + BitFieldSliceMut<W>,
    for<'a> <Box<[W]> as BitFieldSliceMut<W>>::ChunksMut<'a>: Send,
    for<'a> <<Box<[W]> as BitFieldSliceMut<W>>::ChunksMut<'a> as Iterator>::Item: Send,
    <E as SerializeInner>::SerType: ShardEdge<S, 3>, // Weird
    VFunc<usize, usize, BitFieldVec, S, E>: Serialize,
    VFunc<str, usize, BitFieldVec, S, E>: Serialize,
    VFunc<usize, W, Box<[W]>, S, E>: Serialize,
    VFunc<str, W, Box<[W]>, S, E>: Serialize + TypeHash, // TODO: this is weird
    VFilter<W, VFunc<usize, W, Box<[W]>, S, E>>: Serialize,
    VFilter<W, VFunc<usize, W, Box<[W]>, S, <E as SerializeInner>::SerType>>: TypeHash,
    VFilter<
        <W as SerializeInner>::SerType,
        VFunc<str, W, Box<[W]>, S, <E as SerializeInner>::SerType>,
    >: TypeHash + AlignHash, // Weird
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    let n = args.n;

    if let Some(ref filename) = &args.filename {
        let builder = set_builder(VBuilder::<W, Box<[W]>, S, E>::default(), &args);
        let filter = if args.zstd {
            builder.try_build_filter(ZstdLineLender::from_path(filename)?.take(n), &mut pl)?
        } else {
            let t = LineLender::from_path(filename)?.take(n);
            builder.try_build_filter(t, &mut pl)?
        };
        if let Some(filename) = args.filter {
            filter.store(filename)?;
        }
    } else {
        let builder = set_builder(VBuilder::<W, Box<[W]>, S, E>::default(), &args);
        let filter = builder.try_build_filter(FromIntoIterator::from(0_usize..n), &mut pl)?;
        if let Some(filename) = args.filter {
            filter.store(filename)?;
        }
    }

    Ok(())
}

fn main_with_types_bit_field_vec<
    W: Word + ZeroCopy + Send + Sync + CastableFrom<u64> + SerializeInner + TypeHash + AlignHash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    args: Args,
) -> Result<()>
where
    <W as SerializeInner>::SerType: Word + ZeroCopy,
    str: ToSig<S>,
    usize: ToSig<S>,
    u128: UpcastableFrom<W>,
    SigVal<S, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    BitFieldVec<W>: BitFieldSlice<W> + BitFieldSliceMut<W>,
    for<'a> <BitFieldVec<W> as BitFieldSliceMut<W>>::ChunksMut<'a>: Send,
    for<'a> <<BitFieldVec<W> as BitFieldSliceMut<W>>::ChunksMut<'a> as Iterator>::Item: Send,
    <E as SerializeInner>::SerType: ShardEdge<S, 3>, // Weird
    VFunc<usize, usize, BitFieldVec, S, E>: Serialize,
    VFunc<str, usize, BitFieldVec, S, E>: Serialize,
    VFunc<usize, W, BitFieldVec<W>, S, E>: Serialize,
    VFunc<str, W, BitFieldVec<W>, S, E>: Serialize + TypeHash, // TODO: this is weird
    VFilter<W, VFunc<usize, W, BitFieldVec<W>, S, E>>: Serialize,
    VFilter<W, VFunc<usize, W, BitFieldVec<W>, S, <E as SerializeInner>::SerType>>: TypeHash,
    VFilter<
        <W as SerializeInner>::SerType,
        VFunc<str, W, BitFieldVec<W>, S, <E as SerializeInner>::SerType>,
    >: TypeHash + AlignHash, // Weird
    VFilter<W, VFunc<str, W, BitFieldVec<W>, S, E>>: SerializeInner,
    <VFilter<W, VFunc<str, W, BitFieldVec<W>, S, E>> as SerializeInner>::SerType:
        TypeHash + AlignHash, // Weird
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    let n = args.n;

    if let Some(ref filename) = &args.filename {
        let builder = set_builder(VBuilder::<W, BitFieldVec<W>, S, E>::default(), &args);
        let filter = if args.zstd {
            builder.try_build_filter(
                ZstdLineLender::from_path(filename)?.take(n),
                args.bits,
                &mut pl,
            )?
        } else {
            let t = LineLender::from_path(filename)?.take(n);
            builder.try_build_filter(t, args.bits, &mut pl)?
        };
        if let Some(filename) = args.filter {
            filter.store(filename)?;
        }
    } else {
        let builder = set_builder(VBuilder::<W, BitFieldVec<W>, S, E>::default(), &args);
        let filter =
            builder.try_build_filter(FromIntoIterator::from(0_usize..n), args.bits, &mut pl)?;
        if let Some(filename) = args.filter {
            filter.store(filename)?;
        }
    }

    Ok(())
}
