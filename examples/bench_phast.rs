/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]

use anyhow::Result;
use clap::Parser;
use ph::{
    BuildDefaultSeededHasher,
    fmph::Bits8,
    phast::{self, DefaultCompressedArray, Params, SeedOnly, bits_per_seed_to_100_bucket_size},
};
use sux::bit_field_vec;
use value_traits::slices::*;

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

#[derive(Parser, Debug)]
#[command(about = "Benchmark Phast+ + array with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The maximum number of strings to read from the file, or the number of 64-bit keys.
    n: usize,
    /// The maximum number of samples to take when testing query speed.
    samples: usize,
    /// Use this number of threads.
    threads: usize,
    /// The number of repetitions.
    #[arg(short, long, default_value = "5")]
    repeats: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let keys = (0..args.n as u64).collect::<Vec<_>>();

    let start = std::time::Instant::now();

    let func: phast::Function2<Bits8, SeedOnly, DefaultCompressedArray> =
        phast::Function2::with_slice_p_threads_hash_sc(
            &keys,
            &Params::new(Bits8, bits_per_seed_to_100_bucket_size(8)),
            args.threads,
            BuildDefaultSeededHasher::default(),
            SeedOnly,
        );

    let mut output = bit_field_vec![(args.n - 1).ilog2() as usize + 1 => 0; args.n];
    for i in 0..args.n {
        output.set_value(func.get(&(i as u64)) as usize, i);
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let ns_per_key = elapsed.as_nanos() as f64 / args.n as f64;
    eprintln!(
        "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
        secs, args.n, ns_per_key
    );

    bench(args.samples, args.repeats, || {
        let mut key: u64 = 0;
        let mut acc: usize = 0;
        for _ in 0..args.samples {
            key = key.wrapping_add(0x9e3779b97f4a7c15);
            acc ^= unsafe { output.get_unaligned_unchecked(func.get(&key) as usize) };
        }
        std::hint::black_box(acc);
    });

    Ok(())
}
