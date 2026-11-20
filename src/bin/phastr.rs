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
use num_primitive::PrimitiveNumberAs;
use sux::bits::BitFieldVec;
use sux::cli::{
    AnyLineLender, Decompression, HashTypes, KeyBuffer, PHastRArgs, check_num_keys,
    init_env_logger, store,
};
use sux::func::phastr::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder, SignedFunc};
use sux::traits::{TryIntoUnaligned, Word};
use sux::utils::{FromCloneableIntoIterator, FromIntoFallibleLenderFactory};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) PHastR minimal perfect hash function on a set of keys and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
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
    /// Save the structure in unaligned form (faster).​
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
    phastr: PHastRArgs,
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    let builder = args.phastr.builder(args.n);

    macro_rules! build_with_seeds {
        ($d:ty) => {
            match args.hash_type {
                None => build::<$d>(&args, builder.clone()),
                Some(HashTypes::U8) => build_signed::<$d, u8>(&args, builder.clone()),
                Some(HashTypes::U16) => build_signed::<$d, u16>(&args, builder.clone()),
                Some(HashTypes::U32) => build_signed::<$d, u32>(&args, builder.clone()),
                Some(HashTypes::U64) => build_signed::<$d, u64>(&args, builder.clone()),
            }
        };
    }
    // The default number of bits of a seed is at most 8, so seeds are
    // stored in bytes unless more bits are requested
    let build = || {
        if args.phastr.wide_seeds() {
            build_with_seeds!(BitFieldVec<Box<[usize]>>)
        } else {
            build_with_seeds!(Box<[u8]>)
        }
    };
    args.phastr.install(build)
}

/// Builds a function with seeds stored in `D`.
fn build<D: SeedStoreBuild + MemSize + FlatType + TryIntoUnaligned + Send + Sync>(
    args: &Args,
    builder: PHastRBuilder,
) -> Result<()>
where
    PHastR<str, D>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    PHastR<usize, D>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = args.log.progress_logger();

    if let Some(filename) = &args.filename {
        if args.sequential {
            let func = <PHastR<str, D>>::try_new_with_builder(
                AnyLineLender::from_path(filename, args.decompress)?
                    .take(args.n.unwrap_or(usize::MAX)),
                builder,
                &mut pl,
            )?;
            check_num_keys(func.len(), args.n)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
            let keys = buffer.keys();
            let func = <PHastR<str, D>>::try_par_new_with_builder(&keys, builder, &mut pl)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let func = <PHastR<usize, D>>::try_new_with_builder(
                FromCloneableIntoIterator::new(0..n),
                builder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <PHastR<usize, D>>::try_par_new_with_builder(&keys, builder, &mut pl)?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    }
}

/// Builds a function with seeds stored in `D`, signed with hashes of type `W`.
fn build_signed<
    D: SeedStoreBuild + MemSize + FlatType + TryIntoUnaligned + Send + Sync,
    W: Word + Send + Sync,
>(
    args: &Args,
    builder: PHastRBuilder,
) -> Result<()>
where
    u64: PrimitiveNumberAs<W>,
    SignedFunc<PHastR<str, D>, Box<[W]>>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<PHastR<usize, D>, Box<[W]>>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = args.log.progress_logger();

    if let Some(filename) = &args.filename {
        if args.sequential {
            // The keys are read twice: to build the function, and to sign it
            let n = args.n.unwrap_or(usize::MAX);
            let func = <SignedFunc<PHastR<str, D>, Box<[W]>>>::try_new_with_builder(
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
            let keys = buffer.keys();
            let func = <SignedFunc<PHastR<str, D>, Box<[W]>>>::try_par_new_with_builder(
                &keys, builder, &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let func = <SignedFunc<PHastR<usize, D>, Box<[W]>>>::try_new_with_builder(
                FromCloneableIntoIterator::new(0..n),
                builder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <SignedFunc<PHastR<usize, D>, Box<[W]>>>::try_par_new_with_builder(
                &keys, builder, &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    }
}
