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
use epserde::ser::Serialize;
use lender::FallibleLender;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::cli::{BuilderArgs, ShardingArgs};
use sux::dict::VFilter;
use sux::func::{shard_edge::*, *};
use sux::init_env_logger;
use sux::prelude::VBuilder;
use sux::traits::{BitFieldSliceMut, Word};
use sux::utils::{
    BinSafe, DekoBufLineLender, EmptyVal, FromCloneableIntoIterator, Sig, SigVal, ToSig,
};
use value_traits::slices::SliceByValueMut;

#[derive(Parser, Debug)]
#[command(about = "Creates a VFilter and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .multiple(true)
                .args(&["filename", "n"]),
))]
struct Args {
    /// The number of keys; if no filename is provided, use the 64-bit keys
    /// [0 . . n).​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line (at most N keys will be read); it can be compressed with any format supported by the deko crate.​
    #[arg(short, long)]
    filename: Option<String>,
    /// An optional name for the ε-serde serialized filter.​
    filter: Option<String>,
    /// The number of bits of the hashes used by the filter.​
    #[arg(short, long, default_value_t = 8)]
    bits: usize,
    #[clap(flatten)]
    builder: BuilderArgs,
    #[clap(flatten)]
    sharding: ShardingArgs,
    /// Use slower edge logic reducing the probability of duplicate arcs for big
    /// shards.​
    #[arg(long, conflicts_with_all = ["sig64", "no_shards"])]
    full_sigs: bool,
    /// Use 3-hypergraphs.​
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with_all = ["sig64", "full_sigs"])]
    mwhc: bool,
}

macro_rules! fuse {
    ($args: expr, $main: ident, $ty: ty) => {
        if $args.sharding.no_shards {
            if $args.sharding.sig64 {
                $main::<$ty, [u64; 1], FuseLge3NoShards>($args)
            } else {
                $main::<$ty, [u64; 2], FuseLge3NoShards>($args)
            }
        } else {
            if $args.full_sigs {
                $main::<$ty, [u64; 2], FuseLge3FullSigs>($args)
            } else {
                $main::<$ty, [u64; 2], FuseLge3Shards>($args)
            }
        }
    };
}

#[cfg(feature = "mwhc")]
macro_rules! mwhc {
    ($args: expr, $main: ident, $ty: ty) => {
        if $args.sharding.no_shards {
            $main::<$ty, [u64; 2], Mwhc3NoShards>($args)
        } else {
            $main::<$ty, [u64; 2], Mwhc3Shards>($args)
        }
    };
}

fn main() -> Result<()> {
    init_env_logger()?;

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

fn set_builder<D: Send + Sync, S, E: ShardEdge<S, 3>>(
    builder: VBuilder<D, S, E>,
    args: &Args,
) -> VBuilder<D, S, E> {
    let mut builder = args.builder.configure(builder);
    if let Some(n) = args.n {
        builder = builder.expected_num_keys(n);
    }
    builder
}

fn main_with_types_boxed_slice<W: Word + BinSafe, S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    Box<[W]>: BitFieldSliceMut<Value = W>,
    for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
    for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
        Send + BitFieldSliceMut<Value = W>,
    VFunc<usize, BitFieldVec, S, E>: Serialize,
    VFunc<str, BitFieldVec, S, E>: Serialize,
    VFunc<usize, Box<[W]>, S, E>: Serialize,
    VFunc<str, Box<[W]>, S, E>: Serialize,
    VFilter<VFunc<usize, Box<[W]>, S, E>>: Serialize,
    VFilter<VFunc<str, Box<[W]>, S, E>>: Serialize,
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    if let Some(filename) = &args.filename {
        let n = args.n.unwrap_or(usize::MAX);
        let builder = set_builder(VBuilder::<Box<[W]>, S, E>::default(), &args);
        let filter = <VFilter<VFunc<str, Box<[W]>, S, E>>>::try_new_with_builder(
            DekoBufLineLender::from_path(filename)?.take(n),
            args.n.unwrap_or(0),
            builder,
            &mut pl,
        )?;
        if let Some(filename) = args.filter {
            unsafe { filter.store(filename) }?;
        }
    } else {
        let n = args.n.unwrap();
        let builder = set_builder(VBuilder::<Box<[W]>, S, E>::default(), &args);
        let filter = <VFilter<VFunc<usize, Box<[W]>, S, E>>>::try_new_with_builder(
            FromCloneableIntoIterator::from(0_usize..n),
            n,
            builder,
            &mut pl,
        )?;
        if let Some(filename) = args.filter {
            unsafe { filter.store(filename) }?;
        }
    }

    Ok(())
}

fn main_with_types_bit_field_vec<W: Word + BinSafe, S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    VFilter<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>: Serialize,
    VFilter<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>: Serialize,
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    if let Some(filename) = &args.filename {
        let n = args.n.unwrap_or(usize::MAX);
        let builder = set_builder(VBuilder::<BitFieldVec<Box<[W]>>, S, E>::default(), &args);
        let filter = <VFilter<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>>::try_new_with_builder(
            DekoBufLineLender::from_path(filename)?.take(n),
            args.n.unwrap_or(0),
            args.bits,
            builder,
            &mut pl,
        )?;
        if let Some(filename) = args.filter {
            unsafe { filter.store(filename)? };
        }
    } else {
        let n = args.n.unwrap();
        let mut builder = set_builder(VBuilder::<BitFieldVec<Box<[W]>>, S, E>::default(), &args);
        builder = builder.expected_num_keys(n);
        let filter = <VFilter<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>>::try_new_with_builder(
            FromCloneableIntoIterator::from(0_usize..n),
            n,
            args.bits,
            builder,
            &mut pl,
        )?;
        if let Some(filename) = args.filter {
            unsafe { filter.store(filename)? };
        }
    }

    Ok(())
}
