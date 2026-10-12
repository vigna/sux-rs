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
    AnyLineLender, BuilderArgs, Decompression, HashTypes, KeyBuffer, ShardingArgs, check_num_keys,
    init_env_logger, store,
};
use sux::func::signed::SignedFunc;
use sux::func::{shard_edge::*, *};
use sux::traits::{TryIntoUnaligned, Word};
use sux::utils::{FromCloneableIntoIterator, FromIntoFallibleLenderFactory, Sig, SigVal, ToSig};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) VFunc mapping each input to its rank and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
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
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function (if not specified, the
    /// function is built but not saved).​
    func: Option<String>,
    /// Sign the function using hashes of this type, obtaining a SignedFunc:
    /// queries with keys outside the set return None, except for a fraction
    /// 2^-b of false positives, where b is the number of bits of the type.​
    #[arg(long)]
    hash_type: Option<HashTypes>,
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

/// The functions built by this binary.
type Func<K, S, E> = VFunc<K, BitFieldVec<Box<[usize]>>, S, E>;

/// The signed functions built by this binary.
type Signed<K, S, E, W> = SignedFunc<Func<K, S, E>, Box<[W]>>;

fn main() -> Result<()> {
    use sux::cli::ShardEdgeType;
    init_env_logger()?;

    let args = Args::parse();

    macro_rules! build_with_types {
        ($s:ty, $e:ty) => {
            match args.hash_type {
                None => build::<$s, $e>(&args),
                Some(HashTypes::U8) => build_signed::<$s, $e, u8>(&args),
                Some(HashTypes::U16) => build_signed::<$s, $e, u16>(&args),
                Some(HashTypes::U32) => build_signed::<$s, $e, u32>(&args),
                Some(HashTypes::U64) => build_signed::<$s, $e, u64>(&args),
            }
        };
    }
    match args.sharding.shard_edge {
        ShardEdgeType::Fuse3NoShards64 => build_with_types!([u64; 1], Fuse3NoShards),
        ShardEdgeType::Fuse3NoShards128 => build_with_types!([u64; 2], Fuse3NoShards),
        ShardEdgeType::Fuse3Shards => build_with_types!([u64; 2], Fuse3Shards),
        ShardEdgeType::FuseLge3Shards => build_with_types!([u64; 2], FuseLge3Shards),
        ShardEdgeType::FuseLge3FullSigs => build_with_types!([u64; 2], FuseLge3FullSigs),
        #[cfg(feature = "mwhc")]
        ShardEdgeType::Mwhc3 => build_with_types!([u64; 2], Mwhc3Shards),
        #[cfg(feature = "mwhc")]
        ShardEdgeType::Mwhc3NoShards => build_with_types!([u64; 2], Mwhc3NoShards),
    }
}

/// Returns a builder configured by the command-line options.
fn new_builder<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
    args: &Args,
) -> VBuilder<BitFieldVec<Box<[usize]>>, S, E> {
    let builder = args.builder.configure(VBuilder::default());
    match args.n {
        Some(n) => builder.expected_num_keys(n),
        None => builder,
    }
}

/// Builds a function.
fn build<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>(args: &Args) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    Func<str, S, E>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    Func<usize, S, E>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = args.log.progress_logger();
    let builder = new_builder(args);

    if let Some(filename) = &args.filename {
        if args.sequential {
            let n = args.n.unwrap_or(usize::MAX);
            let func = <Func<str, S, E>>::try_new_with_builder(
                FromIntoFallibleLenderFactory::new(|| {
                    AnyLineLender::from_path(filename, args.decompress).map(|keys| keys.take(n))
                })?,
                FromCloneableIntoIterator::from(0_usize..),
                builder,
                &mut pl,
            )?;
            check_num_keys(func.len(), args.n)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            // Parallel: read all keys into a single buffer for cache-friendly
            // access during signature hashing
            let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
            let keys = buffer.keys();
            let values: Vec<usize> = (0..keys.len()).collect();
            let func =
                <Func<str, S, E>>::try_par_new_with_builder(&keys, &values, builder, &mut pl)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let func = <Func<usize, S, E>>::try_new_with_builder(
                FromCloneableIntoIterator::from(0_usize..n),
                FromCloneableIntoIterator::from(0_usize..),
                builder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let values: Vec<usize> = (0..n).collect();
            let func =
                <Func<usize, S, E>>::try_par_new_with_builder(&keys, &values, builder, &mut pl)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    }
}

/// Builds a function signed with hashes of type `W`.
fn build_signed<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType, W: Word>(
    args: &Args,
) -> Result<()>
where
    u64: PrimitiveNumberAs<W>,
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    Signed<str, S, E, W>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    Signed<usize, S, E, W>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = args.log.progress_logger();
    let builder = new_builder(args);

    if let Some(filename) = &args.filename {
        if args.sequential {
            let n = args.n.unwrap_or(usize::MAX);
            let func = <Signed<str, S, E, W>>::try_new_with_builder(
                FromIntoFallibleLenderFactory::new(|| {
                    AnyLineLender::from_path(filename, args.decompress).map(|keys| keys.take(n))
                })?,
                builder,
                &mut pl,
            )?;
            check_num_keys(func.len(), args.n)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
            let func =
                <Signed<str, S, E, W>>::try_par_new_with_builder(&buffer.keys(), builder, &mut pl)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let func = <Signed<usize, S, E, W>>::try_new_with_builder(
                FromCloneableIntoIterator::new(0_usize..n),
                builder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <Signed<usize, S, E, W>>::try_par_new_with_builder(&keys, builder, &mut pl)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    }
}
