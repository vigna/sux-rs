/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::{
    bits::{BitFieldVec, BitFieldVecU},
    cli::{Decompression, bench, gen_sorted_keys, init_env_logger, pack_strings, reservoir_sample},
    func::{
        lcp_mmphf::*,
        lcp2_mmphf::*,
        shard_edge::{Fuse3FullSigs, Fuse3Shards, FuseAug3FullSigs, FuseAug3Shards},
    },
};

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
    /// A file containing sorted UTF-8 keys, one per line. If not specified, random sorted u64 keys are used.​
    #[arg(short = 'f', long)]
    filename: Option<String>,
    /// Decompress the file of keys using this format (by default, it is read
    /// as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Use the two-step variant (Lcp2Mmphf).​
    #[arg(long, short = '2')]
    two_step: bool,
    /// Use unaligned reads.​
    #[arg(long, short)]
    unaligned: bool,
    /// The function was built using full signatures (option --full-sigs of
    /// lcp_mmphf).​
    #[arg(long)]
    full_sigs: bool,
}

/// Benchmarks a structure of type `$t` on strings sampled from `$filename`.
macro_rules! bench_str {
    ($t:ty, $args:expr, $filename:expr) => {{
        let func = unsafe { <$t>::load_full(&$args.func) }?;
        let (packed, packed_offsets) = {
            let queries = reservoir_sample($filename, $args.n, 42, $args.decompress)?;
            pack_strings(&queries, $args.n)
        };
        bench($args.n, $args.repeats, || {
            let mut u = 0usize;
            for i in 0..$args.n {
                let s = packed_offsets[i];
                let e = packed_offsets[i + 1];
                let q = unsafe { std::str::from_utf8_unchecked(&packed[s..e]) };
                u ^= func.get(q);
            }
            std::hint::black_box(u);
        });
    }};
}

/// Benchmarks a structure of type `$t` on random sorted u64 keys.
macro_rules! bench_int {
    ($t:ty, $args:expr) => {{
        let func = unsafe { <$t>::load_full(&$args.func) }?;
        let keys = gen_sorted_keys(func.len(), 0);
        let queries = sample_queries(&keys, $args.n, 42);
        drop(keys);
        bench($args.n, $args.repeats, || {
            let mut u = 0usize;
            for &key in &queries {
                u ^= func.get(key);
            }
            std::hint::black_box(u);
        });
    }};
}

/// Benchmarks a structure whose inner functions use the shard/edge types
/// `$e0` (key map), `$f0` (infrequent LCP lengths of the two-step variant),
/// and `$e1` (LCP-to-bucket map), as the `lcp_mmphf` binary does.
macro_rules! bench_with_types {
    ($args:expr, $e0:ty, $f0:ty, $e1:ty) => {{
        type A = BitFieldVec<Box<[usize]>>;
        type U = BitFieldVecU<Box<[usize]>>;
        match (&$args.filename, $args.two_step, $args.unaligned) {
            (Some(f), false, false) => {
                bench_str!(LcpMmphfStr<A, [u64; 2], $e0, [u64; 2], $e1>, $args, f)
            }
            (Some(f), false, true) => {
                bench_str!(LcpMmphfStr<U, [u64; 2], $e0, [u64; 2], $e1>, $args, f)
            }
            (Some(f), true, false) => {
                bench_str!(Lcp2MmphfStr<A, [u64; 2], $e0, $f0, [u64; 2], $e1>, $args, f)
            }
            (Some(f), true, true) => {
                bench_str!(Lcp2MmphfStr<U, [u64; 2], $e0, $f0, [u64; 2], $e1>, $args, f)
            }
            (None, false, false) => {
                bench_int!(LcpMmphfInt<u64, A, [u64; 2], $e0, [u64; 2], $e1>, $args)
            }
            (None, false, true) => {
                bench_int!(LcpMmphfInt<u64, U, [u64; 2], $e0, [u64; 2], $e1>, $args)
            }
            (None, true, false) => {
                bench_int!(Lcp2MmphfInt<u64, A, [u64; 2], $e0, $f0, [u64; 2], $e1>, $args)
            }
            (None, true, true) => {
                bench_int!(Lcp2MmphfInt<u64, U, [u64; 2], $e0, $f0, [u64; 2], $e1>, $args)
            }
        }
    }};
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    if args.full_sigs {
        bench_with_types!(args, FuseAug3FullSigs, Fuse3FullSigs, FuseAug3FullSigs);
    } else {
        bench_with_types!(args, FuseAug3Shards, Fuse3Shards, FuseAug3Shards);
    }
    Ok(())
}
