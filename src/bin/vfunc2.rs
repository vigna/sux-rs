/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use anyhow::Result;
use clap::{ArgGroup, Parser};
use epserde::ser::Serialize;
use lender::FallibleLender;
use mem_dbg::{FlatType, MemSize};
use rdst::RadixKey;
use std::ops::{BitXor, BitXorAssign};
use sux::bits::BitFieldVec;
use sux::cli::{
    AnyLineLender, BuilderArgs, Decompression, KeyBuffer, ShardingArgs, ValueArgs, check_num_keys,
    count_lines, init_env_logger, store,
};
use sux::func::VBuilder;
use sux::func::VFunc2;
use sux::func::shard_edge::*;
use sux::traits::TryIntoUnaligned;
use sux::utils::{
    FromCloneableIntoIterator, FromIntoFallibleLenderFactory, FromSlice, Sig, SigVal, ToSig,
};

#[derive(Parser, Debug)]
#[command(
    about = "Creates a VFunc2 (two-step static function) mapping each key to an integer value and serializes it with ε-serde.",
    long_about = None,
    next_line_help = true,
    max_term_width = 100,
)]
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
    #[clap(flatten)]
    values: ValueArgs,
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function (if not specified, the
    /// function is built but not saved).​
    func: Option<String>,
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

fn main() -> Result<()> {
    use sux::cli::ShardEdgeType;
    init_env_logger()?;

    let args = Args::parse();
    match args.sharding.shard_edge {
        ShardEdgeType::Fuse3NoShards64 => main_with_types::<[u64; 1], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3NoShards128 => main_with_types::<[u64; 2], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3Shards => main_with_types::<[u64; 2], Fuse3Shards>(args),
        ShardEdgeType::FuseLge3Shards => main_with_types::<[u64; 2], FuseLge3Shards>(args),
        ShardEdgeType::FuseLge3FullSigs => main_with_types::<[u64; 2], FuseLge3FullSigs>(args),
        #[cfg(feature = "mwhc")]
        ShardEdgeType::Mwhc3 => main_with_types::<[u64; 2], Mwhc3Shards>(args),
        #[cfg(feature = "mwhc")]
        ShardEdgeType::Mwhc3NoShards => main_with_types::<[u64; 2], Mwhc3NoShards>(args),
    }
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    VFunc2<str, BitFieldVec<Box<[usize]>>, S, E>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    VFunc2<usize, BitFieldVec<Box<[usize]>>, S, E>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = args.log.progress_logger();
    let mut builder = args
        .builder
        .configure(VBuilder::<BitFieldVec<Box<[usize]>>, S, E>::default());
    if let Some(n) = args.n {
        builder = builder.expected_num_keys(n);
    }

    if let Some(filename) = &args.filename {
        let n = match args.n {
            // Generated values need the number of keys
            None if args.values.file.is_none() => Some(count_lines(filename, args.decompress)?),
            n => n,
        };
        let values = args.values.values(n)?;
        let n = values.len();
        args.values.check_keys(filename, args.decompress, n)?;
        if args.sequential {
            let keys = FromIntoFallibleLenderFactory::new(|| {
                AnyLineLender::from_path(filename, args.decompress).map(|keys| keys.take(n))
            })?;
            let func = <VFunc2<str, BitFieldVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                builder,
                &mut pl,
            )?;
            check_num_keys(func.len(), Some(n))?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let buffer = KeyBuffer::from_path(filename, Some(n), args.decompress)?;
            let func = <VFunc2<str, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &buffer.keys(),
                &values,
                builder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        let values = args.values.values(Some(n))?;
        if args.sequential {
            let keys = FromCloneableIntoIterator::from(0_usize..n);
            let func = <VFunc2<usize, BitFieldVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                builder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <VFunc2<usize, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, builder, &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    }
}
