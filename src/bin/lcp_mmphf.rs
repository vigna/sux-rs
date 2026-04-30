/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::{Result, bail};
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::cli::{BuilderArgs, HashTypes, read_lines_concatenated, str_slice_from_offsets};
use sux::func::lcp_mmphf::*;
use sux::func::lcp2_mmphf::*;
use sux::func::signed::*;
use sux::init_env_logger;
use sux::traits::TryIntoUnaligned;
use sux::utils::{DekoBufLineLender, FromCloneableIntoIterator};

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
    /// The number of keys; if no filename is provided, use the 64-bit keys
    /// [0 . . n).​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing sorted UTF-8 keys, one per line; it can be
    /// compressed with any format supported by the deko crate.​
    #[arg(short, long)]
    filename: Option<String>,
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long)]
    unaligned: bool,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
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
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

fn main() -> Result<()> {
    init_env_logger()?;
    let args = Args::parse();

    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;
    pl.log_interval(args.log.log_interval);

    let n = if let Some(ref filename) = args.filename {
        if let Some(n) = args.n {
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
        }
    } else {
        args.n.unwrap()
    };

    let builder = args.builder.to_builder();

    if args.two_step {
        build_two_step(&args, n, builder, &mut pl)
    } else {
        build_single(&args, n, builder, &mut pl)
    }
}

#[derive(Clone)]
struct RandomSortedKeys {
    n: usize,
    k: u64,
    seed: u64,
}

impl RandomSortedKeys {
    fn new(n: usize) -> Self {
        Self {
            n,
            k: ((1.9 * u64::MAX as f64) / n as f64) as u64,
            seed: rand::rng().random(),
        }
    }
}

impl IntoIterator for RandomSortedKeys {
    type Item = u64;
    type IntoIter = RandomSortedKeysIter;
    fn into_iter(self) -> Self::IntoIter {
        RandomSortedKeysIter {
            remaining: self.n,
            k: self.k,
            acc: 0,
            rng: SmallRng::seed_from_u64(self.seed),
        }
    }
}

struct RandomSortedKeysIter {
    remaining: usize,
    k: u64,
    acc: u64,
    rng: SmallRng,
}

impl Iterator for RandomSortedKeysIter {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.acc += self.rng.random_range(1..self.k);
        Some(self.acc)
    }
}

macro_rules! maybe_store {
    ($func:expr, $out:expr, $unaligned:expr) => {
        if let Some(ref f) = $out {
            if $unaligned {
                unsafe { $func.try_into_unaligned().unwrap().store(f) }?;
            } else {
                unsafe { $func.store(f) }?;
            }
        }
    };
}

fn build_single(
    args: &Args,
    n: usize,
    builder: sux::func::VBuilder<sux::bits::BitFieldVec<Box<[usize]>>>,
    pl: &mut (impl ProgressLog + Clone + Send + Sync),
) -> Result<()> {
    if let Some(ref filename) = args.filename {
        match args.hash_type {
            None => {
                if args.sequential {
                    let lender = DekoBufLineLender::from_path(filename)?;
                    let mmphf: LcpMmphfStr =
                        LcpMmphfStr::try_new_with_builder(lender, n, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                } else {
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
                    let mmphf: LcpMmphfStr =
                        LcpMmphfStr::try_par_new_with_builder(&keys, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                }
            }
            Some(ref ht) => {
                if args.sequential {
                    macro_rules! build_seq {
                        ($h:ty) => {{
                            let lender = DekoBufLineLender::from_path(filename)?;
                            let mmphf: SignedFunc<LcpMmphfStr, Box<[$h]>> =
                                <SignedFunc<LcpMmphfStr, Box<[$h]>>>::try_new_with_builder(
                                    lender, n, builder, pl,
                                )?;
                            maybe_store!(mmphf, args.func, args.unaligned);
                        }};
                    }
                    match ht {
                        HashTypes::U8 => build_seq!(u8),
                        HashTypes::U16 => build_seq!(u16),
                        HashTypes::U32 => build_seq!(u32),
                        HashTypes::U64 => build_seq!(u64),
                    }
                } else {
                    let (buffer, offsets) = read_lines_concatenated(filename, n)?;
                    let keys = str_slice_from_offsets(&buffer, &offsets);
                    macro_rules! build_par {
                        ($h:ty) => {{
                            let mmphf: SignedFunc<LcpMmphfStr, Box<[$h]>> =
                                <SignedFunc<LcpMmphfStr, Box<[$h]>>>::try_par_new_with_builder(
                                    &keys, builder, pl,
                                )?;
                            maybe_store!(mmphf, args.func, args.unaligned);
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
    } else {
        let random_keys = RandomSortedKeys::new(n);
        if args.sequential {
            let lender = FromCloneableIntoIterator::new(random_keys);
            match args.hash_type {
                None => {
                    let mmphf: LcpMmphfInt<u64> =
                        LcpMmphfInt::try_new_with_builder(lender, n, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                }
                Some(ref ht) => {
                    macro_rules! build_seq {
                        ($h:ty) => {{
                            let mmphf: SignedFunc<LcpMmphfInt<u64>, Box<[$h]>> =
                                <SignedFunc<LcpMmphfInt<u64>, Box<[$h]>>>::try_new_with_builder(
                                    lender, n, builder, pl,
                                )?;
                            maybe_store!(mmphf, args.func, args.unaligned);
                        }};
                    }
                    match ht {
                        HashTypes::U8 => build_seq!(u8),
                        HashTypes::U16 => build_seq!(u16),
                        HashTypes::U32 => build_seq!(u32),
                        HashTypes::U64 => build_seq!(u64),
                    }
                }
            }
        } else {
            let keys: Vec<u64> = random_keys.into_iter().collect();
            match args.hash_type {
                None => {
                    let mmphf: LcpMmphfInt<u64> =
                        LcpMmphfInt::try_par_new_with_builder(&keys, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                }
                Some(ref ht) => {
                    macro_rules! build_par {
                        ($h:ty) => {{
                            let mmphf: SignedFunc<LcpMmphfInt<u64>, Box<[$h]>> =
                                <SignedFunc<LcpMmphfInt<u64>, Box<[$h]>>>::try_par_new_with_builder(
                                    &keys, builder, pl,
                                )?;
                            maybe_store!(mmphf, args.func, args.unaligned);
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
    }
    Ok(())
}

fn build_two_step(
    args: &Args,
    n: usize,
    builder: sux::func::VBuilder<sux::bits::BitFieldVec<Box<[usize]>>>,
    pl: &mut (impl ProgressLog + Clone + Send + Sync),
) -> Result<()> {
    if let Some(ref filename) = args.filename {
        match args.hash_type {
            None => {
                if args.sequential {
                    let lender = DekoBufLineLender::from_path(filename)?;
                    let mmphf: Lcp2MmphfStr =
                        Lcp2MmphfStr::try_new_with_builder(lender, n, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                } else {
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
                    let mmphf: Lcp2MmphfStr =
                        Lcp2MmphfStr::try_par_new_with_builder(&keys, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                }
            }
            Some(ref ht) => {
                if args.sequential {
                    macro_rules! build_seq {
                        ($h:ty) => {{
                            let lender = DekoBufLineLender::from_path(filename)?;
                            let mmphf: SignedFunc<Lcp2MmphfStr, Box<[$h]>> =
                                <SignedFunc<Lcp2MmphfStr, Box<[$h]>>>::try_new(lender, n, pl)?;
                            maybe_store!(mmphf, args.func, args.unaligned);
                        }};
                    }
                    match ht {
                        HashTypes::U8 => build_seq!(u8),
                        HashTypes::U16 => build_seq!(u16),
                        HashTypes::U32 => build_seq!(u32),
                        HashTypes::U64 => build_seq!(u64),
                    }
                } else {
                    let (buffer, offsets) = read_lines_concatenated(filename, n)?;
                    let keys = str_slice_from_offsets(&buffer, &offsets);
                    macro_rules! build_par {
                        ($h:ty) => {{
                            let mmphf: SignedFunc<Lcp2MmphfStr, Box<[$h]>> =
                                <SignedFunc<Lcp2MmphfStr, Box<[$h]>>>::try_par_new(&keys, pl)?;
                            maybe_store!(mmphf, args.func, args.unaligned);
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
    } else {
        let random_keys = RandomSortedKeys::new(n);
        if args.sequential {
            let lender = FromCloneableIntoIterator::new(random_keys);
            match args.hash_type {
                None => {
                    let mmphf: Lcp2MmphfInt<u64> =
                        Lcp2MmphfInt::try_new_with_builder(lender, n, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                }
                Some(ref ht) => {
                    macro_rules! build_seq {
                        ($h:ty) => {{
                            let mmphf: SignedFunc<Lcp2MmphfInt<u64>, Box<[$h]>> =
                                <SignedFunc<Lcp2MmphfInt<u64>, Box<[$h]>>>::try_new(lender, n, pl)?;
                            maybe_store!(mmphf, args.func, args.unaligned);
                        }};
                    }
                    match ht {
                        HashTypes::U8 => build_seq!(u8),
                        HashTypes::U16 => build_seq!(u16),
                        HashTypes::U32 => build_seq!(u32),
                        HashTypes::U64 => build_seq!(u64),
                    }
                }
            }
        } else {
            let keys: Vec<u64> = random_keys.into_iter().collect();
            match args.hash_type {
                None => {
                    let mmphf: Lcp2MmphfInt<u64> =
                        Lcp2MmphfInt::try_par_new_with_builder(&keys, builder, pl)?;
                    maybe_store!(mmphf, args.func, args.unaligned);
                }
                Some(ref ht) => {
                    macro_rules! build_par {
                        ($h:ty) => {{
                            let mmphf: SignedFunc<Lcp2MmphfInt<u64>, Box<[$h]>> =
                                <SignedFunc<Lcp2MmphfInt<u64>, Box<[$h]>>>::try_par_new(&keys, pl)?;
                            maybe_store!(mmphf, args.func, args.unaligned);
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
    }
    Ok(())
}
