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
use sux::cli::{BuilderArgs, HashTypes, ShardingArgs, read_concat_lines, str_slice_from_offsets};
use sux::func::signed::SignedFunc;
use sux::func::{shard_edge::*, *};
use sux::init_env_logger;
use sux::prelude::VBuilder;
use sux::traits::TryIntoUnaligned;
use sux::utils::{DekoBufLineLender, EmptyVal, FromCloneableIntoIterator, Sig, SigVal, ToSig};

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
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Sign the function using hashes of this type.​
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
    let mut pl = ProgressLogger::default();
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
            let (buffer, offsets) = read_concat_lines(filename, n)?;
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
            .configure(VBuilder::<BitFieldVec<Box<[usize]>>, S, E>::default())
            .expected_num_keys(n);
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
            let keys: Vec<usize> = (0..n).collect();
            let values: Vec<usize> = (0..n).collect();
            match args.hash_type {
                None => {
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
