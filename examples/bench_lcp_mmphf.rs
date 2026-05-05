/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use fallible_iterator::FallibleIterator;
use lender::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::{
    bits::BitFieldVecU,
    func::{lcp2_mmphf::*, lcp_mmphf::*},
    utils::{LineLender, ZstdLineLender},
};

fn bench(n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
        eprintln!("{} ns/key", timings.last().unwrap());
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "Min: {} Median: {} Max: {} Average: {}",
        timings[0],
        timings[timings.len() / 2],
        timings.last().unwrap(),
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}

/// Generate k sorted distinct random u64 keys.
fn gen_sorted_keys(k: usize, seed: u64) -> Vec<u64> {
    let extra = (((k as u128) * (k as u128)) >> 65) as usize;
    let draw = k + 2 * extra;
    let mut rng = SmallRng::seed_from_u64(seed);
    loop {
        let mut keys: Vec<u64> = (0..draw).map(|_| rng.random()).collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() >= k {
            keys.truncate(k);
            return keys;
        }
    }
}

/// Shuffle keys into a random query order (Fisher–Yates).
fn shuffle_keys(keys: &[u64], seed: u64) -> Vec<u64> {
    let mut shuffled = keys.to_vec();
    let mut rng = SmallRng::seed_from_u64(seed);
    for i in (1..shuffled.len()).rev() {
        let j = (rng.random::<u64>() % (i as u64 + 1)) as usize;
        shuffled.swap(i, j);
    }
    shuffled
}

#[derive(Parser, Debug)]
#[command(
    about = "Benchmarks LcpMmphf / Lcp2Mmphf with strings or 64-bit integers.",
    long_about = None,
    next_line_help = true,
    max_term_width = 100
)]
struct Args {
    /// The maximum number of strings to read from the file, or the number of 64-bit keys.​
    n: usize,
    /// A name for the ε-serde serialized function.​
    func: String,
    /// A file containing sorted UTF-8 keys, one per line. If not specified, random sorted u64 keys are used.​
    #[arg(short = 'f', long)]
    filename: Option<String>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// The input file is compressed with zstd.​
    #[arg(short, long)]
    zstd: bool,
    /// Use the two-step variant (Lcp2Mmphf).​
    #[arg(long, short = '2')]
    two_step: bool,
    /// Use unaligned reads.​
    #[arg(long, short)]
    unaligned: bool,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    if args.two_step {
        main_two_step(args)
    } else {
        main_single(args)
    }
}

fn main_single(args: Args) -> Result<()> {
    if let Some(ref filename) = args.filename {
        let keys: Vec<String> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        };
        let n = keys.len();

        if args.unaligned {
            let func = unsafe {
                <LcpMmphfStr<BitFieldVecU<Box<[usize]>>>>::load_full(&args.func)
            }?;
            bench(n, args.repeats, || {
                let mut u = 0usize;
                for key in &keys {
                    u ^= func.get(key.as_str());
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe {
                <LcpMmphfStr>::load_full(&args.func)
            }?;
            bench(n, args.repeats, || {
                let mut u = 0usize;
                for key in &keys {
                    u ^= func.get(key.as_str());
                }
                std::hint::black_box(u);
            });
        }
    } else {
        let keys = gen_sorted_keys(args.n, 0);
        let queries = shuffle_keys(&keys, 42);

        if args.unaligned {
            let func =
                unsafe { LcpMmphfInt::<u64, BitFieldVecU<Box<[usize]>>>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for &key in &queries {
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe { LcpMmphfInt::<u64>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for &key in &queries {
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        }
    }
    Ok(())
}

fn main_two_step(args: Args) -> Result<()> {
    if let Some(ref filename) = args.filename {
        let keys: Vec<String> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        };
        let n = keys.len();

        if args.unaligned {
            let func = unsafe {
                <Lcp2MmphfStr<BitFieldVecU<Box<[usize]>>>>::load_full(&args.func)
            }?;
            bench(n, args.repeats, || {
                let mut u = 0usize;
                for key in &keys {
                    u ^= func.get(key.as_str());
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe {
                <Lcp2MmphfStr>::load_full(&args.func)
            }?;
            bench(n, args.repeats, || {
                let mut u = 0usize;
                for key in &keys {
                    u ^= func.get(key.as_str());
                }
                std::hint::black_box(u);
            });
        }
    } else {
        let keys = gen_sorted_keys(args.n, 0);
        let queries = shuffle_keys(&keys, 42);

        if args.unaligned {
            let func =
                unsafe { Lcp2MmphfInt::<u64, BitFieldVecU<Box<[usize]>>>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for &key in &queries {
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe { Lcp2MmphfInt::<u64>::load_full(&args.func) }?;
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for &key in &queries {
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        }
    }
    Ok(())
}
