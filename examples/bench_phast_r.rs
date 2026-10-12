/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Benchmarks construction time, space, and query time of PHastR on 64-bit
//! integer keys.

use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder};
use sux::traits::TryIntoUnaligned;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks PHastR on 64-bit integer keys", long_about = None)]
struct Args {
    /// The number of keys.​
    n: usize,
    /// The number of queries.​
    #[arg(short, long, default_value_t = 10_000_000)]
    queries: usize,
    /// The number of bits of a seed; seeds of more than 8 bits are stored in
    /// a bit-field vector.​
    #[arg(long, default_value_t = 8)]
    seed_bits: u32,
    /// The base-2 logarithm of the number of layouts.​
    #[arg(long, default_value_t = 2)]
    log2_layouts: u32,
    /// The base-2 logarithm of the slice length.​
    #[arg(long, default_value_t = 10)]
    log2_slice_len: u32,
    /// The expected number of keys per bucket.​
    #[arg(long, default_value_t = 4.25)]
    bucket_size: f64,
}

/// Builds a function, reporting construction time and space.
fn build<D: SeedStoreBuild + MemSize + mem_dbg::FlatType>(
    args: &Args,
    keys: &[u64],
) -> PHastR<u64, D> {
    let builder = PHastRBuilder::default()
        .seed_bits(args.seed_bits)
        .log2_layouts(args.log2_layouts)
        .log2_slice_len(args.log2_slice_len)
        .bucket_size(args.bucket_size);
    let start = Instant::now();
    let phf: PHastR<u64, D> =
        PHastR::try_par_new_with_builder(keys, builder, no_logging![]).unwrap();
    let build = start.elapsed().as_nanos() as f64 / keys.len() as f64;
    let bits = phf.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64;
    eprintln!("Construction: {build:.1} ns/key");
    eprintln!("Space: {bits:.4} bits/key ({} levels)", phf.num_levels());
    phf
}

/// Checks that `get` maps the keys to distinct values, and
/// reports the time of queries running through the keys in sequence (each
/// query hashes its key, so accesses to the function are random anyway).
fn query(get: impl Fn(&u64) -> usize, keys: &[u64], queries: usize) {
    let mut seen = vec![false; keys.len()];
    for key in keys {
        let v = get(key);
        assert!(!seen[v], "Duplicate output {v}");
        seen[v] = true;
    }

    let mut acc = 0usize;
    let start = Instant::now();
    for key in keys.iter().cycle().take(queries) {
        acc = acc.wrapping_add(get(key));
    }
    std::hint::black_box(acc);
    eprintln!(
        "Queries: {:.1} ns/query",
        start.elapsed().as_nanos() as f64 / queries as f64
    );
}

fn main() {
    let args = Args::parse();
    let keys: Vec<u64> = (0..args.n as u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x1234_5678)
        .collect();
    if args.seed_bits <= 8 {
        let phf = build::<Box<[u8]>>(&args, &keys);
        query(|key| phf.get(key), &keys, args.queries);
    } else {
        // Seeds stored in a bit-field vector are faster to read unaligned
        let phf = build::<BitFieldVec<Box<[usize]>>>(&args, &keys)
            .try_into_unaligned()
            .unwrap();
        query(|key| phf.get(key), &keys, args.queries);
    }
}
