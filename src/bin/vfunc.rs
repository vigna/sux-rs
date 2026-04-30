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
use sux::cli::{
    BuilderArgs, HashTypes, ShardingArgs, read_lines_concatenated, str_slice_from_offsets,
};
use sux::func::VFunc2;
use sux::func::signed::SignedFunc;
use sux::func::{shard_edge::*, *};
use sux::init_env_logger;
use sux::prelude::VBuilder;
use sux::traits::TryIntoUnaligned;
use sux::utils::{
    DekoBufLineLender, EmptyVal, FromCloneableIntoIterator, FromSlice, Sig, SigVal, ToSig,
};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) VFunc mapping each input to its rank and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
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
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Use the two-step variant (less space for skewed distributions, slightly slower queries). In this case, values are the number of trailing zeros of the keys.​
    #[arg(long, conflicts_with = "hash_type")]
    two_step: bool,
    /// Sign the function using hashes of this type.​
    #[arg(long)]
    hash_type: Option<HashTypes>,
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

fn main() -> Result<()> {
    use sux::cli::ShardEdgeType;
    init_env_logger()?;

    let args = Args::parse();

    if args.two_step {
        return main_two_step(args);
    }

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

// ── Sequential (lender-based) build macros ─────────────────────────

macro_rules! filename_save_sign_seq(
    ($h: ty, $builder:expr, $filename: expr, $func: expr, $unaligned: expr, $n: expr, $pl: expr) => {{
        let func =
            <SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[$h]>>>::try_new_with_builder(
                DekoBufLineLender::from_path($filename)?.take($n),
                $builder,
                &mut $pl,
            )?;
        if let Some(filename) = $func {
            if $unaligned {
                unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
            } else {
                unsafe { func.store(filename) }?;
            }
        }
    }}
);

macro_rules! n_save_sign_seq(
    ($h: ty, $builder:expr, $n: expr, $func: expr, $unaligned: expr, $pl: expr) => {{
        let func =
            <SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[$h]>>>::try_new_with_builder(
                FromCloneableIntoIterator::new(0_usize..$n),
                $builder,
                &mut $pl,
            )?;
        if let Some(filename) = $func {
            if $unaligned {
                unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
            } else {
                unsafe { func.store(filename) }?;
            }
        }
    }}
);

// ── Parallel (slice-based) build macros ────────────────────────────

macro_rules! filename_save_sign_par(
    ($h: ty, $builder:expr, $keys:expr, $func: expr, $unaligned: expr, $pl: expr) => {{
        let func =
            <SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[$h]>>>::try_par_new_with_builder(
                $keys,
                $builder,
                &mut $pl,
            )?;
        if let Some(filename) = $func {
            if $unaligned {
                unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
            } else {
                unsafe { func.store(filename) }?;
            }
        }
    }}
);

macro_rules! n_save_sign_par(
    ($h: ty, $builder:expr, $keys:expr, $func: expr, $unaligned: expr, $pl: expr) => {{
        let func =
            <SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[$h]>>>::try_par_new_with_builder(
                $keys,
                $builder,
                &mut $pl,
            )?;
        if let Some(filename) = $func {
            if $unaligned {
                unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
            } else {
                unsafe { func.store(filename) }?;
            }
        }
    }}
);

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, EmptyVal>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: BitXor + BitXorAssign,
    VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    VFunc<str, BitFieldVec<Box<[usize]>>, S, E>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    VFunc<usize, Box<[u8]>, S, E>: Serialize,
    VFunc<str, Box<[u8]>, S, E>: Serialize,
    SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u8]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u16]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u32]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u64]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u8]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u16]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u32]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u64]>>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
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
            .configure(VBuilder::<BitFieldVec<Box<[usize]>>, S, E>::default());
        if let Some(n_hint) = args.n {
            builder = builder.expected_num_keys(n_hint);
        }
        if args.sequential {
            match args.hash_type {
                None => {
                    let func = <VFunc<str, BitFieldVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                        DekoBufLineLender::from_path(filename)?.take(n),
                        FromCloneableIntoIterator::from(0_usize..),
                        builder,
                        &mut pl,
                    )?;
                    if let Some(filename) = args.func {
                        if args.unaligned {
                            unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                        } else {
                            unsafe { func.store(filename) }?;
                        }
                    }
                }
                Some(HashTypes::U8) => {
                    filename_save_sign_seq!(u8, builder, filename, args.func, args.unaligned, n, pl)
                }
                Some(HashTypes::U16) => {
                    filename_save_sign_seq!(
                        u16,
                        builder,
                        filename,
                        args.func,
                        args.unaligned,
                        n,
                        pl
                    )
                }
                Some(HashTypes::U32) => {
                    filename_save_sign_seq!(
                        u32,
                        builder,
                        filename,
                        args.func,
                        args.unaligned,
                        n,
                        pl
                    )
                }
                Some(HashTypes::U64) => {
                    filename_save_sign_seq!(
                        u64,
                        builder,
                        filename,
                        args.func,
                        args.unaligned,
                        n,
                        pl
                    )
                }
            }
        } else {
            // Parallel: read all keys into a single concatenated
            // buffer, then build a `Vec<&str>` of slices into it for
            // cache-friendly access during sig hashing.
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
            match args.hash_type {
                None => {
                    let values: Vec<usize> = (0..keys.len()).collect();
                    let func =
                        <VFunc<str, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                            &keys, &values, builder, &mut pl,
                        )?;
                    if let Some(filename) = args.func {
                        if args.unaligned {
                            unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                        } else {
                            unsafe { func.store(filename) }?;
                        }
                    }
                }
                Some(HashTypes::U8) => {
                    filename_save_sign_par!(u8, builder, &keys, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U16) => {
                    filename_save_sign_par!(u16, builder, &keys, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U32) => {
                    filename_save_sign_par!(u32, builder, &keys, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U64) => {
                    filename_save_sign_par!(u64, builder, &keys, args.func, args.unaligned, pl)
                }
            }
        }
    } else {
        let n = args.n.unwrap();
        let builder = args
            .builder
            .configure(VBuilder::<BitFieldVec<Box<[usize]>>, S, E>::default());
        if args.sequential {
            match args.hash_type {
                None => {
                    let func =
                        <VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                            FromCloneableIntoIterator::from(0_usize..n),
                            FromCloneableIntoIterator::from(0_usize..),
                            builder,
                            &mut pl,
                        )?;
                    if let Some(filename) = args.func {
                        if args.unaligned {
                            unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                        } else {
                            unsafe { func.store(filename) }?;
                        }
                    }
                }
                Some(HashTypes::U8) => {
                    n_save_sign_seq!(u8, builder, n, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U16) => {
                    n_save_sign_seq!(u16, builder, n, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U32) => {
                    n_save_sign_seq!(u32, builder, n, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U64) => {
                    n_save_sign_seq!(u64, builder, n, args.func, args.unaligned, pl)
                }
            }
        } else {
            // Parallel: materialise keys as `Vec<usize>`. Costs `n * 8`
            // bytes of memory but lets the sig-hashing phase run on all
            // cores.
            let keys: Vec<usize> = (0..n).collect();
            match args.hash_type {
                None => {
                    let values: Vec<usize> = (0..n).collect();
                    let func =
                        <VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                            &keys, &values, builder, &mut pl,
                        )?;
                    if let Some(filename) = args.func {
                        if args.unaligned {
                            unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                        } else {
                            unsafe { func.store(filename) }?;
                        }
                    }
                }
                Some(HashTypes::U8) => {
                    n_save_sign_par!(u8, builder, &keys, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U16) => {
                    n_save_sign_par!(u16, builder, &keys, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U32) => {
                    n_save_sign_par!(u32, builder, &keys, args.func, args.unaligned, pl)
                }
                Some(HashTypes::U64) => {
                    n_save_sign_par!(u64, builder, &keys, args.func, args.unaligned, pl)
                }
            }
        }
    }

    Ok(())
}

fn main_two_step(args: Args) -> Result<()> {
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    pl.log_interval(args.log.log_interval);

    let builder = args.builder.to_builder();

    if let Some(filename) = &args.filename {
        let n = if let Some(n) = args.n {
            n
        } else {
            pl.info(format_args!("Counting keys..."));
            let mut lender = DekoBufLineLender::from_path(filename)?;
            let mut count = 0usize;
            while FallibleLender::next(&mut lender)?.is_some() {
                count += 1;
            }
            pl.info(format_args!("Found {count} keys"));
            count
        };
        if args.sequential {
            let func: VFunc2<str, BitFieldVec<Box<[usize]>>> = VFunc2::try_new_with_builder(
                DekoBufLineLender::from_path(filename)?.take(n),
                FromCloneableIntoIterator::from(0_usize..),
                builder,
                &mut pl,
            )?;
            if let Some(filename) = args.func {
                if args.unaligned {
                    unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                } else {
                    unsafe { func.store(filename) }?;
                }
            }
        } else {
            let (buffer, offsets) = read_lines_concatenated(filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            if keys.len() != n {
                bail!("key count mismatch: read {} keys, expected {n}", keys.len());
            }
            let values: Vec<usize> = (0..n).collect();
            let func: VFunc2<str, BitFieldVec<Box<[usize]>>> =
                VFunc2::try_par_new_with_builder(&keys, &values, builder, &mut pl)?;
            if let Some(filename) = args.func {
                if args.unaligned {
                    unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                } else {
                    unsafe { func.store(filename) }?;
                }
            }
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let keys: Vec<usize> = (0..n).collect();
            let vals: Vec<usize> = (0..n).map(|k| (k + 1).trailing_zeros() as usize).collect();
            let func: VFunc2<usize, BitFieldVec<Box<[usize]>>> = VFunc2::try_new_with_builder(
                FromSlice::new(&keys),
                FromSlice::new(&vals),
                builder,
                &mut pl,
            )?;
            if let Some(filename) = args.func {
                if args.unaligned {
                    unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                } else {
                    unsafe { func.store(filename) }?;
                }
            }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let values: Vec<usize> = (0..n).map(|k| (k + 1).trailing_zeros() as usize).collect();
            let func: VFunc2<usize, BitFieldVec<Box<[usize]>>> =
                VFunc2::try_par_new_with_builder(&keys, &values, builder, &mut pl)?;
            if let Some(filename) = args.func {
                if args.unaligned {
                    unsafe { func.try_into_unaligned().unwrap().store(filename) }?;
                } else {
                    unsafe { func.store(filename) }?;
                }
            }
        }
    }

    Ok(())
}
