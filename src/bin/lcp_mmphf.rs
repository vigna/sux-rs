/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fmt::Display;

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use sux::bits::BitFieldVec;
use sux::func::lcp_mmphf::*;
use sux::func::signed_lcp_mmphf::*;
use sux::init_env_logger;
use sux::prelude::VBuilder;
use sux::utils::{DekoBufLineLender, FromSlice};

#[derive(clap::ValueEnum, Clone, Debug)]
enum HashTypes {
    U8,
    U16,
    U32,
    U64,
}

impl Display for HashTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashTypes::U8 => write!(f, "u8"),
            HashTypes::U16 => write!(f, "u16"),
            HashTypes::U32 => write!(f, "u32"),
            HashTypes::U64 => write!(f, "u64"),
        }
    }
}

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
    /// Use this number of threads.​
    #[arg(short, long)]
    threads: Option<usize>,
    /// Use disk-based buckets to reduce memory usage at construction time.​
    #[arg(short, long)]
    offline: bool,
    /// Sign the function using hashes of this type.​
    #[arg(long)]
    hash_type: Option<HashTypes>,
}

fn main() -> Result<()> {
    init_env_logger()?;
    let args = Args::parse();

    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    // Count keys if -n was not given.
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

    // Configure the VBuilder for the main (offset_lcp_length) VFunc.
    let mut builder = VBuilder::<_, BitFieldVec<Box<[usize]>>>::default()
        .offline(args.offline);
    if let Some(t) = args.threads {
        builder = builder.max_num_threads(t);
    }

    match args.hash_type {
        None => {
            let lender = DekoBufLineLender::from_path(&args.filename)?;
            let mmphf: LcpMmphfStr =
                LcpMmphfStr::new_with_builder(lender, n, builder, &mut pl)?;
            if let Some(filename) = args.func {
                unsafe { mmphf.store(filename) }?;
            }
        }
        Some(ref ht) => {
            // Read keys into memory (Clone lender needed for hash pass).
            let mut keys: Vec<String> = Vec::with_capacity(n);
            let mut lender = DekoBufLineLender::from_path(&args.filename)?;
            let mut count = 0usize;
            while let Some(key) = FallibleLender::next(&mut lender)? {
                keys.push(key.to_owned());
                count += 1;
                if count >= n {
                    break;
                }
            }
            let n = keys.len();

            macro_rules! build_signed {
                ($h:ty) => {{
                    let mmphf: SignedLcpMmphfStr<Box<[$h]>> =
                        SignedLcpMmphfStr::new_with_builder(
                            FromSlice::new(&keys),
                            n,
                            builder,
                            &mut pl,
                        )?;
                    if let Some(filename) = &args.func {
                        unsafe { mmphf.store(filename) }?;
                    }
                }};
            }
            match ht {
                HashTypes::U8 => build_signed!(u8),
                HashTypes::U16 => build_signed!(u16),
                HashTypes::U32 => build_signed!(u32),
                HashTypes::U64 => build_signed!(u64),
            }
        }
    }

    Ok(())
}
