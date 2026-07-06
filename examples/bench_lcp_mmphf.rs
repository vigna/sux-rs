/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::{
    bits::BitFieldVecU,
    cli::{pack_strings, reservoir_sample},
    func::{lcp_mmphf::*, lcp2_mmphf::*},
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

/// Sample n random keys from the key set (with replacement).
fn sample_queries(keys: &[u64], n: usize, seed: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| keys[(rng.random::<u64>() % keys.len() as u64) as usize])
        .collect()
}

#[derive(Parser, Debug)]
#[command(
    about = "Benchmarks LcpMmphf / Lcp2Mmphf with strings or 64-bit integers.",
    long_about = None,
    next_line_help = true,
    max_term_width = 100
)]
struct Args {
    /// The number of queries to perform.​
    n: usize,
    /// A name for the ε-serde serialized function.​
    func: String,
    /// A file containing sorted UTF-8 keys, one per line; it can be compressed with any format supported by the deko crate. If not specified, random sorted u64 keys are used.​
    #[arg(short = 'f', long)]
    filename: Option<String>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Use the two-step variant (Lcp2Mmphf).​
    #[arg(long, short = '2')]
    two_step: bool,
    /// Use unaligned reads.​
    #[arg(long, short)]
    unaligned: bool,
}

fn main() -> Result<()> {
    sux::init_env_logger()?;

    let args = Args::parse();

    if args.two_step {
        main_two_step(args)
    } else {
        main_single(args)
    }
}

fn main_single(args: Args) -> Result<()> {
    if let Some(ref filename) = args.filename {
        if args.unaligned {
            let func = unsafe { <LcpMmphfStr<BitFieldVecU<Box<[usize]>>>>::load_full(&args.func) }?;
            let (packed, packed_offsets) = {
                let queries = reservoir_sample(filename, args.n, 42)?;
                pack_strings(&queries, args.n)
            };
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for i in 0..args.n {
                    let s = packed_offsets[i];
                    let e = packed_offsets[i + 1];
                    let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                    u ^= func.get(q);
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe { <LcpMmphfStr>::load_full(&args.func) }?;
            let (packed, packed_offsets) = {
                let queries = reservoir_sample(filename, args.n, 42)?;
                pack_strings(&queries, args.n)
            };
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for i in 0..args.n {
                    let s = packed_offsets[i];
                    let e = packed_offsets[i + 1];
                    let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                    u ^= func.get(q);
                }
                std::hint::black_box(u);
            });
        }
    } else {
        if args.unaligned {
            let func =
                unsafe { LcpMmphfInt::<u64, BitFieldVecU<Box<[usize]>>>::load_full(&args.func) }?;
            let keys = gen_sorted_keys(func.len(), 0);
            let queries = sample_queries(&keys, args.n, 42);
            drop(keys);
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for &key in &queries {
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe { LcpMmphfInt::<u64>::load_full(&args.func) }?;
            let keys = gen_sorted_keys(func.len(), 0);
            let queries = sample_queries(&keys, args.n, 42);
            drop(keys);
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
        if args.unaligned {
            let func =
                unsafe { <Lcp2MmphfStr<BitFieldVecU<Box<[usize]>>>>::load_full(&args.func) }?;
            let (packed, packed_offsets) = {
                let queries = reservoir_sample(filename, args.n, 42)?;
                pack_strings(&queries, args.n)
            };
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for i in 0..args.n {
                    let s = packed_offsets[i];
                    let e = packed_offsets[i + 1];
                    let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                    u ^= func.get(q);
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe { <Lcp2MmphfStr>::load_full(&args.func) }?;
            let (packed, packed_offsets) = {
                let queries = reservoir_sample(filename, args.n, 42)?;
                pack_strings(&queries, args.n)
            };
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for i in 0..args.n {
                    let s = packed_offsets[i];
                    let e = packed_offsets[i + 1];
                    let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                    u ^= func.get(q);
                }
                std::hint::black_box(u);
            });
        }
    } else {
        if args.unaligned {
            let func =
                unsafe { Lcp2MmphfInt::<u64, BitFieldVecU<Box<[usize]>>>::load_full(&args.func) }?;
            let keys = gen_sorted_keys(func.len(), 0);
            let queries = sample_queries(&keys, args.n, 42);
            drop(keys);
            bench(args.n, args.repeats, || {
                let mut u = 0usize;
                for &key in &queries {
                    u ^= func.get(key);
                }
                std::hint::black_box(u);
            });
        } else {
            let func = unsafe { Lcp2MmphfInt::<u64>::load_full(&args.func) }?;
            let keys = gen_sorted_keys(func.len(), 0);
            let queries = sample_queries(&keys, args.n, 42);
            drop(keys);
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
