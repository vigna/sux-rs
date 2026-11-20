/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use sux::{
    bits::BitFieldVec,
    cli::{Decompression, INCR, bench, init_env_logger, pack_strings, reservoir_sample},
    func::{PHastR, PHunc},
    traits::Unaligned,
};

#[derive(Parser, Debug)]
#[command(about = "Benchmarks PHunc with strings or 64-bit integers.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// The number of queries to perform.​
    n: usize,
    /// A name for the ε-serde serialized function.​
    func: String,
    /// A file containing UTF-8 keys, one per line, from which queries are
    /// sampled. If not specified, queries are multiples of a large odd
    /// constant, and thus mostly not keys of a structure built with -n (but
    /// queries take the same time on keys and non-keys).​
    #[arg(short = 'f', long)]
    filename: Option<String>,
    /// Decompress the file of keys using this format (by default, it is read
    /// as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// The number of bits of a seed used to build the function [default: 8];
    /// seeds of more than 8 bits are stored in a bit-field vector.​
    #[arg(long)]
    seed_bits: Option<u32>,
    /// Use unaligned reads.​
    #[arg(long, short)]
    unaligned: bool,
}

/// The functions built by the `phunc` binary, with seeds stored in `D`.
type Func<K, D> = PHunc<PHastR<K, D>, BitFieldVec<Box<[usize]>>>;

/// Benchmarks a function on strings, given the queries.
macro_rules! bench_strings {
    ($args:expr, $t:ty, $packed:expr, $packed_offsets:expr) => {{
        let func = unsafe { <$t>::load_full(&$args.func) }?;
        bench($args.n, $args.repeats, || {
            let mut u = 0usize;
            for i in 0..$args.n {
                let s = $packed_offsets[i];
                let e = $packed_offsets[i + 1];
                let q = unsafe { std::str::from_utf8_unchecked(&$packed[s..e]) };
                u ^= func.get(q);
            }
            std::hint::black_box(u);
        });
    }};
}

/// Benchmarks a function on 64-bit integers.
macro_rules! bench_integers {
    ($args:expr, $t:ty) => {{
        let func = unsafe { <$t>::load_full(&$args.func) }?;
        bench($args.n, $args.repeats, || {
            let mut u = 0usize;
            let mut key: usize = 0;
            for _ in 0..$args.n {
                key = key.wrapping_add(INCR);
                u ^= func.get(key);
            }
            std::hint::black_box(u);
        });
    }};
}

/// Benchmarks a function with seeds stored in the given type.
macro_rules! bench_with_seeds {
    ($args:expr, $d:ty) => {{
        if let Some(ref filename) = $args.filename {
            let (packed, packed_offsets) = {
                let queries = reservoir_sample(filename, $args.n, 42, $args.decompress)?;
                pack_strings(&queries, $args.n)
            };
            if $args.unaligned {
                bench_strings!($args, Unaligned<Func<str, $d>>, packed, packed_offsets)
            } else {
                bench_strings!($args, Func<str, $d>, packed, packed_offsets)
            }
        } else {
            if $args.unaligned {
                bench_integers!($args, Unaligned<Func<usize, $d>>)
            } else {
                bench_integers!($args, Func<usize, $d>)
            }
        }
    }};
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    if args.seed_bits.is_some_and(|b| b > 8) {
        bench_with_seeds!(args, BitFieldVec<Box<[usize]>>)
    } else {
        bench_with_seeds!(args, Box<[u8]>)
    }
    Ok(())
}
