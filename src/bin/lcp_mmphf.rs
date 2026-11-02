/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use num_primitive::PrimitiveNumberAs;
use sux::bits::BitFieldVec;
use sux::cli::{
    AnyLineLender, BuilderArgs, Decompression, HashTypes, KeyBuffer, check_func_num_keys,
    count_lines, gen_sorted_keys, init_env_logger, store,
};
use sux::func::VBuilder;
use sux::func::lcp_mmphf::*;
use sux::func::lcp2_mmphf::*;
use sux::func::shard_edge::{Fuse3FullSigs, Fuse3Shards, FuseAug3FullSigs, FuseAug3Shards};
use sux::func::signed::*;
use sux::traits::{TryIntoUnaligned, Word};
use sux::utils::{FromIntoFallibleLenderFactory, FromSlice};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) LCP-based monotone minimal perfect hash function \
    mapping each sorted key to its rank and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .multiple(true)
                .args(["filename", "n"]),
))]
struct Args {
    /// The number of keys: if a file of keys is specified, its first N keys
    /// are used, and it is an error if it contains fewer keys; otherwise, N
    /// distinct pseudorandom 64-bit keys, always the same for a given N, are
    /// sorted and used.​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing sorted UTF-8 keys, one per line.​
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
    /// Use the two-step variant (less space, slightly slower queries).​
    #[arg(long, short = '2')]
    two_step: bool,
    /// Use full signatures in the inner functions, which is necessary with
    /// more than ten billion keys (by default, inner functions use 64-bit
    /// local signatures).​
    #[arg(long)]
    full_sigs: bool,
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
    log: sux::cli::LogIntervalArg,
}

fn main() -> Result<()> {
    init_env_logger()?;
    let args = Args::parse();

    if args.full_sigs {
        full_sigs::run(&args)
    } else {
        // The function on keys has n keys
        check_func_num_keys::<[u64; 2], FuseAug3Shards>(args.n, "--full-sigs")?;
        local_sigs::run(&args)
    }
}

/// Returns the number of keys to read sequentially from a file: N, if
/// specified, or the number of lines of the file.
fn num_keys(args: &Args, filename: &str, pl: &mut impl ProgressLog) -> Result<usize> {
    if let Some(n) = args.n {
        return Ok(n);
    }
    pl.info(format_args!("Counting keys..."));
    let count = count_lines(filename, args.decompress)?;
    pl.info(format_args!("Found {count} keys"));
    Ok(count)
}

/// Defines a module `$module` containing the functions that build and store
/// MMPHFs whose inner functions use the given shard/edge logics: `$e0` for the
/// function on keys, `$f0` for the function on keys with infrequent LCP
/// lengths of the two-step variant, and `$e1` for the function on prefixes.
macro_rules! impl_build {
    ($module:ident, $e0:ty, $f0:ty, $e1:ty) => {
        mod $module {
            use super::*;

            type D = BitFieldVec<Box<[usize]>>;
            type Lcp = LcpMmphfStr<D, [u64; 2], $e0, [u64; 2], $e1>;
            type Lcp2 = Lcp2MmphfStr<D, [u64; 2], $e0, $f0, [u64; 2], $e1>;
            type LcpInt = LcpMmphfInt<u64, D, [u64; 2], $e0, [u64; 2], $e1>;
            type Lcp2Int = Lcp2MmphfInt<u64, D, [u64; 2], $e0, $f0, [u64; 2], $e1>;

            pub fn run(args: &Args) -> Result<()> {
                match args.hash_type {
                    None => build(args),
                    Some(HashTypes::U8) => build_signed::<u8>(args),
                    Some(HashTypes::U16) => build_signed::<u16>(args),
                    Some(HashTypes::U32) => build_signed::<u32>(args),
                    Some(HashTypes::U64) => build_signed::<u64>(args),
                }
            }

            /// Builds a function.
            fn build(args: &Args) -> Result<()> {
                let mut pl = args.log.progress_logger();
                let builder = args.builder.configure(VBuilder::default());

                if let Some(filename) = &args.filename {
                    if args.sequential {
                        let n = num_keys(args, filename, &mut pl)?;
                        let keys = FromIntoFallibleLenderFactory::new(|| {
                            AnyLineLender::from_path(filename, args.decompress)
                                .map(|keys| keys.take(n))
                        })?;
                        if args.two_step {
                            let mmphf = <Lcp2>::try_new_with_builder(keys, n, builder, &mut pl)?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        } else {
                            let mmphf = <Lcp>::try_new_with_builder(keys, n, builder, &mut pl)?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        }
                    } else {
                        let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
                        let keys = buffer.keys();
                        if args.two_step {
                            let mmphf = <Lcp2>::try_par_new_with_builder(&keys, builder, &mut pl)?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        } else {
                            let mmphf = <Lcp>::try_par_new_with_builder(&keys, builder, &mut pl)?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        }
                    }
                } else {
                    let n = args.n.unwrap();
                    let keys = gen_sorted_keys(n, 0);
                    if args.sequential {
                        let keys = FromSlice::new(&keys);
                        if args.two_step {
                            let mmphf = <Lcp2Int>::try_new_with_builder(keys, n, builder, &mut pl)?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        } else {
                            let mmphf = <LcpInt>::try_new_with_builder(keys, n, builder, &mut pl)?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        }
                    } else if args.two_step {
                        let mmphf = <Lcp2Int>::try_par_new_with_builder(&keys, builder, &mut pl)?;
                        unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                    } else {
                        let mmphf = <LcpInt>::try_par_new_with_builder(&keys, builder, &mut pl)?;
                        unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                    }
                }
            }

            /// Builds a function signed with hashes of type `W`.
            fn build_signed<W: Word>(args: &Args) -> Result<()>
            where
                u64: PrimitiveNumberAs<W>,
                SignedFunc<Lcp, Box<[W]>>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
                SignedFunc<Lcp2, Box<[W]>>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
                SignedFunc<LcpInt, Box<[W]>>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
                SignedFunc<Lcp2Int, Box<[W]>>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
            {
                let mut pl = args.log.progress_logger();
                let builder = args.builder.configure(VBuilder::default());

                if let Some(filename) = &args.filename {
                    if args.sequential {
                        let n = num_keys(args, filename, &mut pl)?;
                        let keys = FromIntoFallibleLenderFactory::new(|| {
                            AnyLineLender::from_path(filename, args.decompress)
                                .map(|keys| keys.take(n))
                        })?;
                        if args.two_step {
                            let mmphf = <SignedFunc<Lcp2, Box<[W]>>>::try_new_with_builder(
                                keys, n, builder, &mut pl,
                            )?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        } else {
                            let mmphf = <SignedFunc<Lcp, Box<[W]>>>::try_new_with_builder(
                                keys, n, builder, &mut pl,
                            )?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        }
                    } else {
                        let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
                        let keys = buffer.keys();
                        if args.two_step {
                            let mmphf = <SignedFunc<Lcp2, Box<[W]>>>::try_par_new_with_builder(
                                &keys, builder, &mut pl,
                            )?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        } else {
                            let mmphf = <SignedFunc<Lcp, Box<[W]>>>::try_par_new_with_builder(
                                &keys, builder, &mut pl,
                            )?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        }
                    }
                } else {
                    let n = args.n.unwrap();
                    let keys = gen_sorted_keys(n, 0);
                    if args.sequential {
                        let keys = FromSlice::new(&keys);
                        if args.two_step {
                            let mmphf = <SignedFunc<Lcp2Int, Box<[W]>>>::try_new_with_builder(
                                keys, n, builder, &mut pl,
                            )?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        } else {
                            let mmphf = <SignedFunc<LcpInt, Box<[W]>>>::try_new_with_builder(
                                keys, n, builder, &mut pl,
                            )?;
                            unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                        }
                    } else if args.two_step {
                        let mmphf = <SignedFunc<Lcp2Int, Box<[W]>>>::try_par_new_with_builder(
                            &keys, builder, &mut pl,
                        )?;
                        unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                    } else {
                        let mmphf = <SignedFunc<LcpInt, Box<[W]>>>::try_par_new_with_builder(
                            &keys, builder, &mut pl,
                        )?;
                        unsafe { store(mmphf, args.func.as_ref(), args.unaligned) }
                    }
                }
            }
        }
    };
}

impl_build!(local_sigs, FuseAug3Shards, Fuse3Shards, FuseAug3Shards);
impl_build!(full_sigs, FuseAug3FullSigs, Fuse3FullSigs, FuseAug3FullSigs);
