/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use std::ops::{BitXor, BitXorAssign};

use anyhow::{Result, bail};
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use mem_dbg::{FlatType, MemSize};
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::cli::{BuilderArgs, ShardingArgs, read_lines_concatenated, str_slice_from_offsets};
use sux::dict::VFilter;
use sux::func::{shard_edge::*, *};
use sux::init_env_logger;
use sux::prelude::VBuilder;
use sux::traits::{BitFieldSliceMut, TryIntoUnaligned, Word};
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
                .args(["filename", "n"]),
))]
struct Args {
    /// The number of keys; if no filename is provided, use the 64-bit keys
    /// [0 . . n).​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line (at most N keys will be read); it can be compressed with any format supported by the deko crate.​
    #[arg(short, long)]
    filename: Option<String>,
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long)]
    unaligned: bool,
    /// An optional name for the ε-serde serialized filter.​
    filter: Option<String>,
    /// The number of bits of the hashes used by the filter.​
    #[arg(short, long, default_value_t = 8)]
    bits: usize,
    /// Use the single-threaded *sequential* construction path that
    /// streams keys through a lender instead of materialising them
    /// in memory. Slower for large builds (sig-store population is
    /// the bottleneck) but useful when keys don't fit in RAM. The
    /// default is the parallel, in-memory path.​
    #[arg(short, long)]
    sequential: bool,
    #[clap(flatten)]
    builder: BuilderArgs,
    #[clap(flatten)]
    sharding: ShardingArgs,
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

macro_rules! dispatch_edge {
    ($args: expr, $main: ident, $ty: ty) => {{
        use sux::cli::ShardEdgeType;
        match $args.sharding.shard_edge {
            ShardEdgeType::Fuse3NoShards64 => $main::<$ty, [u64; 1], Fuse3NoShards>($args),
            ShardEdgeType::Fuse3NoShards128 => $main::<$ty, [u64; 2], Fuse3NoShards>($args),
            ShardEdgeType::Fuse3Shards => $main::<$ty, [u64; 2], Fuse3Shards>($args),
            ShardEdgeType::FuseLge3Shards => $main::<$ty, [u64; 2], FuseLge3Shards>($args),
            ShardEdgeType::FuseLge3FullSigs => $main::<$ty, [u64; 2], FuseLge3FullSigs>($args),
            #[cfg(feature = "mwhc")]
            ShardEdgeType::Mwhc3 => $main::<$ty, [u64; 2], Mwhc3Shards>($args),
            #[cfg(feature = "mwhc")]
            ShardEdgeType::Mwhc3NoShards => $main::<$ty, [u64; 2], Mwhc3NoShards>($args),
        }
    }};
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    match args.bits {
        8 => dispatch_edge!(args, main_with_types_boxed_slice, u8),
        16 => dispatch_edge!(args, main_with_types_boxed_slice, u16),
        32 => dispatch_edge!(args, main_with_types_boxed_slice, u32),
        64 => dispatch_edge!(args, main_with_types_boxed_slice, u64),
        _ => dispatch_edge!(args, main_with_types_bit_field_vec, u64),
    }
}

fn main_with_types_boxed_slice<
    W: Word + BinSafe + MemSize + FlatType,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
>(
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
    Box<[W]>: MemSize + FlatType,
{
    if args.unaligned {
        bail!("--unaligned is not supported for backend Box<[W]>; use a custom bit width");
    }

    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    pl.log_interval(args.log.log_interval);

    if let Some(filename) = &args.filename {
        let n = args.n.unwrap_or(usize::MAX);
        let mut builder = args
            .builder
            .configure(VBuilder::<Box<[W]>, S, E>::default());
        if let Some(n_hint) = args.n {
            builder = builder.expected_num_keys(n_hint);
        }
        if args.sequential {
            let filter = <VFilter<VFunc<str, Box<[W]>, S, E>>>::try_new_with_builder(
                DekoBufLineLender::from_path(filename)?.take(n),
                builder,
                &mut pl,
            )?;
            if let Some(filename) = args.filter {
                unsafe { filter.store(filename) }?;
            }
        } else {
            let (buffer, offsets) = read_lines_concatenated(filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            if let Some(n_hint) = args.n {
                if keys.len() != n_hint {
                    bail!(
                        "key count mismatch: read {} keys, expected {n_hint}",
                        keys.len()
                    );
                }
            }
            let filter = <VFilter<VFunc<str, Box<[W]>, S, E>>>::try_par_new_with_builder(
                &keys, builder, &mut pl,
            )?;
            if let Some(filename) = args.filter {
                unsafe { filter.store(filename) }?;
            }
        }
    } else {
        let n = args.n.unwrap();
        let builder = args
            .builder
            .configure(VBuilder::<Box<[W]>, S, E>::default())
            .expected_num_keys(n);
        if args.sequential {
            let filter = <VFilter<VFunc<usize, Box<[W]>, S, E>>>::try_new_with_builder(
                FromCloneableIntoIterator::from(0_usize..n),
                builder,
                &mut pl,
            )?;
            if let Some(filename) = args.filter {
                unsafe { filter.store(filename) }?;
            }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let filter = <VFilter<VFunc<usize, Box<[W]>, S, E>>>::try_par_new_with_builder(
                &keys, builder, &mut pl,
            )?;
            if let Some(filename) = args.filter {
                unsafe { filter.store(filename) }?;
            }
        }
    }

    Ok(())
}

fn main_with_types_bit_field_vec<
    W: Word + BinSafe + MemSize + FlatType,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, usize>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    VFilter<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    VFilter<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    BitFieldVec<Box<[W]>>: MemSize + FlatType,
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    pl.log_interval(args.log.log_interval);

    if let Some(filename) = &args.filename {
        let n = args.n.unwrap_or(usize::MAX);
        let mut builder = args
            .builder
            .configure(VBuilder::<BitFieldVec<Box<[W]>>, S, E>::default());
        if let Some(n_hint) = args.n {
            builder = builder.expected_num_keys(n_hint);
        }
        if args.sequential {
            let filter = <VFilter<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>>::try_new_with_builder(
                DekoBufLineLender::from_path(filename)?.take(n),
                args.bits,
                builder,
                &mut pl,
            )?;
            if let Some(filename) = args.filter {
                if args.unaligned {
                    unsafe { filter.try_into_unaligned().unwrap().store(filename)? };
                } else {
                    unsafe { filter.store(filename)? };
                }
            }
        } else {
            let (buffer, offsets) = read_lines_concatenated(filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            if let Some(n_hint) = args.n {
                if keys.len() != n_hint {
                    bail!(
                        "key count mismatch: read {} keys, expected {n_hint}",
                        keys.len()
                    );
                }
            }
            let filter =
                <VFilter<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>>::try_par_new_with_builder(
                    &keys, args.bits, builder, &mut pl,
                )?;
            if let Some(filename) = args.filter {
                if args.unaligned {
                    unsafe { filter.try_into_unaligned().unwrap().store(filename)? };
                } else {
                    unsafe { filter.store(filename)? };
                }
            }
        }
    } else {
        let n = args.n.unwrap();
        let builder = args
            .builder
            .configure(VBuilder::<BitFieldVec<Box<[W]>>, S, E>::default())
            .expected_num_keys(n);
        if args.sequential {
            let filter =
                <VFilter<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>>::try_new_with_builder(
                    FromCloneableIntoIterator::from(0_usize..n),
                    args.bits,
                    builder,
                    &mut pl,
                )?;
            if let Some(filename) = args.filter {
                if args.unaligned {
                    unsafe { filter.try_into_unaligned().unwrap().store(filename)? };
                } else {
                    unsafe { filter.store(filename)? };
                }
            }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let filter =
                <VFilter<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>>::try_par_new_with_builder(
                    &keys, args.bits, builder, &mut pl,
                )?;
            if let Some(filename) = args.filter {
                if args.unaligned {
                    unsafe { filter.try_into_unaligned().unwrap().store(filename)? };
                } else {
                    unsafe { filter.store(filename)? };
                }
            }
        }
    }

    Ok(())
}
