/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use sux::cli::{BuilderArgs, HashTypes, read_lines_concatenated, str_slice_from_offsets};
use sux::func::lcp_mmphf::*;
use sux::func::lcp2_mmphf::*;
use sux::func::signed::*;
use sux::init_env_logger;
use sux::traits::TryIntoUnaligned;
use sux::utils::{DekoBufLineLender, FromSlice};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) LCP-based monotone minimal perfect hash function \
    mapping each sorted string to its rank and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// A file containing sorted UTF-8 keys, one per line; it can be
    /// compressed with any format supported by the deko crate.​
    filename: String,
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long)]
    unaligned: bool,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// The number of keys (if not given, a counting pass is performed).​
    #[arg(short, long)]
    n: Option<usize>,
    /// Use the two-step variant (less space, slightly slower queries).​
    #[arg(long)]
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
}

fn main() -> Result<()> {
    init_env_logger()?;
    let args = Args::parse();

    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    let n = if let Some(n) = args.n {
        n
    } else {
        pl.info(format_args!("Counting keys..."));
        let mut lender = DekoBufLineLender::from_path(&args.filename)?;
        let mut count = 0usize;
        while FallibleLender::next(&mut lender)?.is_some() {
            count += 1;
        }
        pl.info(format_args!("Found {count} keys"));
        count
    };

    let builder = args.builder.to_builder();

    if args.two_step {
        build_two_step(&args, n, builder, &mut pl)
    } else {
        build_single(&args, n, builder, &mut pl)
    }
}

fn build_single(
    args: &Args,
    n: usize,
    builder: sux::func::VBuilder<sux::bits::BitFieldVec<Box<[usize]>>>,
    pl: &mut (impl ProgressLog + Clone + Send + Sync),
) -> Result<()> {
    match args.hash_type {
        None => {
            if args.sequential {
                let lender = DekoBufLineLender::from_path(&args.filename)?;
                let mmphf: LcpMmphfStr = LcpMmphfStr::try_new_with_builder(lender, n, builder, pl)?;
                if let Some(ref f) = args.func {
                    if args.unaligned {
                        unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                    } else {
                        unsafe { mmphf.store(f) }?;
                    }
                }
            } else {
                // Parallel: read all keys into a single concatenated
                // buffer, then build a `Vec<&str>` of slices into it.
                let (buffer, offsets) = read_lines_concatenated(&args.filename, n)?;
                let keys = str_slice_from_offsets(&buffer, &offsets);
                let mmphf: LcpMmphfStr = LcpMmphfStr::try_par_new_with_builder(&keys, builder, pl)?;
                if let Some(ref f) = args.func {
                    if args.unaligned {
                        unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                    } else {
                        unsafe { mmphf.store(f) }?;
                    }
                }
            }
        }
        Some(ref ht) => {
            // Signed variants need the keys addressable a second time
            // (for signing, after the main build), so we materialise
            // them as a `Vec<&str>` over a single concatenated buffer
            // — one big allocation, cache-friendly access — usable
            // by both paths (sequential rewinds via `FromSlice`,
            // parallel consumes the slice directly).
            let (buffer, offsets) = read_lines_concatenated(&args.filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            let n = keys.len();
            if args.sequential {
                macro_rules! build_seq {
                    ($h:ty) => {{
                        let mmphf: SignedFunc<LcpMmphfStr, Box<[$h]>> =
                            <SignedFunc<LcpMmphfStr, Box<[$h]>>>::try_new_with_builder(
                                FromSlice::new(&keys),
                                n,
                                builder,
                                pl,
                            )?;
                        if let Some(ref f) = args.func {
                            if args.unaligned {
                                unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                            } else {
                                unsafe { mmphf.store(f) }?;
                            }
                        }
                    }};
                }
                match ht {
                    HashTypes::U8 => build_seq!(u8),
                    HashTypes::U16 => build_seq!(u16),
                    HashTypes::U32 => build_seq!(u32),
                    HashTypes::U64 => build_seq!(u64),
                }
            } else {
                macro_rules! build_par {
                    ($h:ty) => {{
                        let mmphf: SignedFunc<LcpMmphfStr, Box<[$h]>> =
                            <SignedFunc<LcpMmphfStr, Box<[$h]>>>::try_par_new_with_builder(
                                &keys, builder, pl,
                            )?;
                        if let Some(ref f) = args.func {
                            if args.unaligned {
                                unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                            } else {
                                unsafe { mmphf.store(f) }?;
                            }
                        }
                    }};
                }
                match ht {
                    HashTypes::U8 => build_par!(u8),
                    HashTypes::U16 => build_par!(u16),
                    HashTypes::U32 => build_par!(u32),
                    HashTypes::U64 => build_par!(u64),
                }
            }
        }
    }
    Ok(())
}

fn build_two_step(
    args: &Args,
    n: usize,
    builder: sux::func::VBuilder<sux::bits::BitFieldVec<Box<[usize]>>>,
    pl: &mut (impl ProgressLog + Clone + Send + Sync),
) -> Result<()> {
    match args.hash_type {
        None => {
            if args.sequential {
                let lender = DekoBufLineLender::from_path(&args.filename)?;
                let mmphf: Lcp2MmphfStr =
                    Lcp2MmphfStr::try_new_with_builder(lender, n, builder, pl)?;
                if let Some(ref f) = args.func {
                    if args.unaligned {
                        unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                    } else {
                        unsafe { mmphf.store(f) }?;
                    }
                }
            } else {
                let (buffer, offsets) = read_lines_concatenated(&args.filename, n)?;
                let keys = str_slice_from_offsets(&buffer, &offsets);
                let mmphf: Lcp2MmphfStr =
                    Lcp2MmphfStr::try_par_new_with_builder(&keys, builder, pl)?;
                if let Some(ref f) = args.func {
                    if args.unaligned {
                        unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                    } else {
                        unsafe { mmphf.store(f) }?;
                    }
                }
            }
        }
        Some(ref ht) => {
            // See the comment in `build_single`: signed paths
            // materialise keys through a single concatenated buffer
            // for both sequential and parallel variants.
            let (buffer, offsets) = read_lines_concatenated(&args.filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            let n = keys.len();
            if args.sequential {
                macro_rules! build_seq {
                    ($h:ty) => {{
                        let mmphf: SignedFunc<Lcp2MmphfStr, Box<[$h]>> =
                            <SignedFunc<Lcp2MmphfStr, Box<[$h]>>>::try_new(
                                FromSlice::new(&keys),
                                n,
                                pl,
                            )?;
                        if let Some(ref f) = args.func {
                            if args.unaligned {
                                unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                            } else {
                                unsafe { mmphf.store(f) }?;
                            }
                        }
                    }};
                }
                match ht {
                    HashTypes::U8 => build_seq!(u8),
                    HashTypes::U16 => build_seq!(u16),
                    HashTypes::U32 => build_seq!(u32),
                    HashTypes::U64 => build_seq!(u64),
                }
            } else {
                macro_rules! build_par {
                    ($h:ty) => {{
                        let mmphf: SignedFunc<Lcp2MmphfStr, Box<[$h]>> =
                            <SignedFunc<Lcp2MmphfStr, Box<[$h]>>>::try_par_new(&keys, pl)?;
                        if let Some(ref f) = args.func {
                            if args.unaligned {
                                unsafe { mmphf.try_into_unaligned().unwrap().store(f) }?;
                            } else {
                                unsafe { mmphf.store(f) }?;
                            }
                        }
                    }};
                }
                match ht {
                    HashTypes::U8 => build_par!(u8),
                    HashTypes::U16 => build_par!(u16),
                    HashTypes::U32 => build_par!(u32),
                    HashTypes::U64 => build_par!(u64),
                }
            }
        }
    }
    Ok(())
}
