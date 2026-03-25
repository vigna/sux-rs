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
use sux::cli::{BuilderArgs, HashTypes};
use sux::func::lcp_mmphf::*;
use sux::func::lcp2_mmphf::*;
use sux::func::signed_lcp_mmphf::*;
use sux::init_env_logger;
use sux::utils::{DekoBufLineLender, FromSlice};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) LCP-based monotone minimal perfect hash function \
    mapping each sorted string to its rank and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// A file containing sorted UTF-8 keys, one per line; it can be
    /// compressed with any format supported by the deko crate.​
    filename: String,
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
    builder: sux::func::VBuilder<usize, sux::bits::BitFieldVec<Box<[usize]>>>,
    pl: &mut (impl ProgressLog + Clone + Send + Sync),
) -> Result<()> {
    match args.hash_type {
        None => {
            let lender = DekoBufLineLender::from_path(&args.filename)?;
            let mmphf: LcpMmphfStr = LcpMmphfStr::try_new_with_builder(lender, n, builder, pl)?;
            if let Some(ref f) = args.func {
                unsafe { mmphf.store(f) }?;
            }
        }
        Some(ref ht) => {
            let keys = read_keys(&args.filename, n)?;
            let n = keys.len();
            macro_rules! build {
                ($h:ty) => {{
                    let mmphf: SignedLcpMmphfStr<Box<[$h]>> =
                        SignedLcpMmphfStr::try_new_with_builder(
                            FromSlice::new(&keys),
                            n,
                            builder,
                            pl,
                        )?;
                    if let Some(ref f) = args.func {
                        unsafe { mmphf.store(f) }?;
                    }
                }};
            }
            match ht {
                HashTypes::U8 => build!(u8),
                HashTypes::U16 => build!(u16),
                HashTypes::U32 => build!(u32),
                HashTypes::U64 => build!(u64),
            }
        }
    }
    Ok(())
}

fn build_two_step(
    args: &Args,
    n: usize,
    builder: sux::func::VBuilder<usize, sux::bits::BitFieldVec<Box<[usize]>>>,
    pl: &mut (impl ProgressLog + Clone + Send + Sync),
) -> Result<()> {
    match args.hash_type {
        None => {
            let lender = DekoBufLineLender::from_path(&args.filename)?;
            let mmphf: Lcp2MmphfStr = Lcp2MmphfStr::try_new_with_builder(lender, n, builder, pl)?;
            if let Some(ref f) = args.func {
                unsafe { mmphf.store(f) }?;
            }
        }
        Some(ref ht) => {
            let keys = read_keys(&args.filename, n)?;
            let n = keys.len();
            macro_rules! build {
                ($h:ty) => {{
                    let mmphf: SignedLcp2MmphfStr<Box<[$h]>> =
                        SignedLcp2MmphfStr::try_new(FromSlice::new(&keys), n, pl)?;
                    if let Some(ref f) = args.func {
                        unsafe { mmphf.store(f) }?;
                    }
                }};
            }
            match ht {
                HashTypes::U8 => build!(u8),
                HashTypes::U16 => build!(u16),
                HashTypes::U32 => build!(u32),
                HashTypes::U64 => build!(u64),
            }
        }
    }
    Ok(())
}

/// Reads keys into memory (needed for Clone lender in signed variants).
fn read_keys(filename: &str, max_n: usize) -> Result<Vec<String>> {
    let mut keys = Vec::new();
    let mut lender = DekoBufLineLender::from_path(filename)?;
    let mut count = 0usize;
    while let Some(key) = FallibleLender::next(&mut lender)? {
        keys.push(key.to_owned());
        count += 1;
        if count >= max_n {
            break;
        }
    }
    Ok(keys)
}
