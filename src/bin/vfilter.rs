/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use std::ops::{BitXor, BitXorAssign};

use anyhow::Result;
use clap::{ArgGroup, Parser};
use epserde::ser::Serialize;
use lender::FallibleLender;
use mem_dbg::{FlatType, MemSize};
use num_primitive::PrimitiveNumberAs;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::cli::{
    AnyLineLender, BuilderArgs, Decompression, KeyBuffer, ShardingArgs, check_num_keys,
    init_env_logger, store, store_as_is,
};
use sux::dict::VFilter;
use sux::func::{shard_edge::*, *};
use sux::traits::{BitFieldSliceMut, TryIntoUnaligned, Word};
use sux::utils::{
    BinSafe, EmptyVal, FromCloneableIntoIterator, FromIntoFallibleLenderFactory, Sig, SigVal, ToSig,
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
    /// The number of keys: if a file of keys is specified, its first N keys
    /// are used, and it is an error if it contains fewer keys; otherwise, the
    /// 64-bit keys [0 . . N) are used.​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line.​
    #[arg(short, long)]
    filename: Option<String>,
    /// Decompress the file of keys using this format (by default, it is read
    /// as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
    /// Save the structure in unaligned form (faster); this option has no
    /// effect if the hashes have 8, 16, 32, or 64 bits, as they are then
    /// stored in a boxed slice.​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized filter (if not specified, the filter
    /// is built but not saved).​
    filter: Option<String>,
    /// The number of bits of the hashes used by the filter.​
    #[arg(short, long, default_value_t = 8)]
    bits: usize,
    /// Hashes keys sequentially without loading them in RAM.​
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
            ShardEdgeType::FuseAug3Shards => $main::<$ty, [u64; 2], FuseAug3Shards>($args),
            ShardEdgeType::FuseAug3FullSigs => $main::<$ty, [u64; 2], FuseAug3FullSigs>($args),
            #[allow(deprecated)]
            ShardEdgeType::FuseLge3Shards => $main::<$ty, [u64; 2], FuseLge3Shards>($args),
            #[allow(deprecated)]
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
    u64: PrimitiveNumberAs<W>,
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    Box<[W]>: BitFieldSliceMut<Value = W>,
    for<'a> <Box<[W]> as SliceByValueMut>::ChunksMut<'a>: Send,
    for<'a> <<Box<[W]> as SliceByValueMut>::ChunksMut<'a> as Iterator>::Item:
        Send + BitFieldSliceMut<Value = W>,
    VFilter<usize, Box<[W]>, S, E>: Serialize,
    VFilter<str, Box<[W]>, S, E>: Serialize,
{
    if args.unaligned {
        log::warn!(
            "--unaligned has no effect with {}-bit hashes, which are stored in a boxed slice",
            args.bits
        );
    }

    let mut pl = args.log.progress_logger();
    let mut builder = args
        .builder
        .configure(VBuilder::<Box<[W]>, S, E>::default());
    if let Some(n) = args.n {
        builder = builder.expected_num_keys(n);
    }

    if let Some(filename) = &args.filename {
        if args.sequential {
            let n = args.n.unwrap_or(usize::MAX);
            let filter = <VFilter<str, Box<[W]>, S, E>>::try_new_with_builder(
                FromIntoFallibleLenderFactory::new(|| {
                    AnyLineLender::from_path(filename, args.decompress).map(|keys| keys.take(n))
                })?,
                builder,
                &mut pl,
            )?;
            check_num_keys(filter.len(), args.n)?;
            unsafe { store_as_is(&filter, args.filter.as_ref()) }
        } else {
            let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
            let filter = <VFilter<str, Box<[W]>, S, E>>::try_par_new_with_builder(
                &buffer.keys(),
                builder,
                &mut pl,
            )?;
            unsafe { store_as_is(&filter, args.filter.as_ref()) }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let filter = <VFilter<usize, Box<[W]>, S, E>>::try_new_with_builder(
                FromCloneableIntoIterator::from(0_usize..n),
                builder,
                &mut pl,
            )?;
            unsafe { store_as_is(&filter, args.filter.as_ref()) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let filter = <VFilter<usize, Box<[W]>, S, E>>::try_par_new_with_builder(
                &keys, builder, &mut pl,
            )?;
            unsafe { store_as_is(&filter, args.filter.as_ref()) }
        }
    }
}

fn main_with_types_bit_field_vec<
    W: Word + BinSafe + MemSize + FlatType,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
>(
    args: Args,
) -> Result<()>
where
    u64: PrimitiveNumberAs<W>,
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    VFilter<usize, BitFieldVec<Box<[W]>>, S, E>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    VFilter<str, BitFieldVec<Box<[W]>>, S, E>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = args.log.progress_logger();
    let mut builder = args
        .builder
        .configure(VBuilder::<BitFieldVec<Box<[W]>>, S, E>::default());
    if let Some(n) = args.n {
        builder = builder.expected_num_keys(n);
    }

    if let Some(filename) = &args.filename {
        if args.sequential {
            let n = args.n.unwrap_or(usize::MAX);
            let filter = <VFilter<str, BitFieldVec<Box<[W]>>, S, E>>::try_new_with_builder(
                FromIntoFallibleLenderFactory::new(|| {
                    AnyLineLender::from_path(filename, args.decompress).map(|keys| keys.take(n))
                })?,
                args.bits,
                builder,
                &mut pl,
            )?;
            check_num_keys(filter.len(), args.n)?;
            unsafe { store(filter, args.filter.as_ref(), args.unaligned) }
        } else {
            let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
            let filter = <VFilter<str, BitFieldVec<Box<[W]>>, S, E>>::try_par_new_with_builder(
                &buffer.keys(),
                args.bits,
                builder,
                &mut pl,
            )?;
            unsafe { store(filter, args.filter.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let filter = <VFilter<usize, BitFieldVec<Box<[W]>>, S, E>>::try_new_with_builder(
                FromCloneableIntoIterator::from(0_usize..n),
                args.bits,
                builder,
                &mut pl,
            )?;
            unsafe { store(filter, args.filter.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let filter = <VFilter<usize, BitFieldVec<Box<[W]>>, S, E>>::try_par_new_with_builder(
                &keys, args.bits, builder, &mut pl,
            )?;
            unsafe { store(filter, args.filter.as_ref(), args.unaligned) }
        }
    }
}
