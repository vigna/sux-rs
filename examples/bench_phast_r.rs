/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Benchmarks construction time, space, and query time of PHast-R on 64-bit
//! integer keys.

use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::{PHastR, PHastRBuilder, SeedStore, SeedStoreBuild};

#[derive(Parser, Debug)]
#[command(about = "Benchmarks PHast-R on 64-bit integer keys", long_about = None)]
struct Args {
    /// The number of keys.
    n: usize,
    /// The number of random queries.
    #[arg(short, long, default_value_t = 10_000_000)]
    queries: usize,
    /// The number of bits per seed (byte seeds are used for 8 bits or less).
    #[arg(short, long, default_value_t = 8)]
    seed_bits: u32,
    /// The base-2 logarithm of the number of patterns.
    #[arg(short = 'R', long, default_value_t = 2)]
    log2_patterns: u32,
    /// The base-2 logarithm of the slice length.
    #[arg(short = 'L', long, default_value_t = 10)]
    log2_slice_len: u32,
    /// The expected number of keys per bucket.
    #[arg(short = 'l', long, default_value_t = 4.5)]
    bucket_size: f64,
}

fn run<D: SeedStoreBuild + SeedStore + MemSize + mem_dbg::FlatType>(args: &Args, keys: &[u64]) {
    let builder = PHastRBuilder::default()
        .seed_bits(args.seed_bits)
        .log2_patterns(args.log2_patterns)
        .log2_slice_len(args.log2_slice_len)
        .bucket_size(args.bucket_size);
    let start = Instant::now();
    let phf: PHastR<u64, D> = builder.try_build(keys, no_logging![]).unwrap();
    let build = start.elapsed().as_nanos() as f64 / keys.len() as f64;
    let bits = phf.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64;
    eprintln!("Construction: {build:.1} ns/key");
    eprintln!("Space: {bits:.4} bits/key ({} levels)", phf.num_levels());

    let mut seen = vec![false; keys.len()];
    for key in keys {
        let v = phf.get(key);
        assert!(!seen[v], "Duplicate output {v}");
        seen[v] = true;
    }

    let n = keys.len() as u64;
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let mut acc = 0usize;
    let start = Instant::now();
    for _ in 0..args.queries {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        acc = acc.wrapping_add(phf.get(keys[(((x >> 32) * n) >> 32) as usize]));
    }
    std::hint::black_box(acc);
    eprintln!(
        "Queries: {:.1} ns/query",
        start.elapsed().as_nanos() as f64 / args.queries as f64
    );
}

fn main() {
    let args = Args::parse();
    let keys: Vec<u64> = (0..args.n as u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x1234_5678)
        .collect();
    if args.seed_bits <= 8 {
        run::<Box<[u8]>>(&args, &keys);
    } else {
        run::<BitFieldVec<Box<[usize]>>>(&args, &keys);
    }
}
