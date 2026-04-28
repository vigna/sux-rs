/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Experiment: build CompVFunc across sizes with uniform(0..256)
//! and Zipf(1, 1_000_000), logging shard/entropy/overhead details.

use dsi_progress_logger::{ProgressLog, ProgressLogger};
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;
use sux::func::codec::Huffman;
use sux::func::shard_edge::Fuse3Shards;
use sux::func::{CompVFunc, VBuilder};

fn zipf_cdf(s: f64, n: usize) -> Vec<f64> {
    let h: f64 = (1..=n).map(|i| 1.0 / (i as f64).powf(s)).sum();
    let mut cdf = vec![0.0; n];
    let mut acc = 0.0;
    for i in 1..=n {
        acc += 1.0 / (i as f64).powf(s) / h;
        cdf[i - 1] = acc;
    }
    cdf
}

fn sample_zipf(cdf: &[f64], rng: &mut SmallRng) -> usize {
    let u: f64 = (rng.random::<u64>() >> 11) as f64 / ((1u64 << 53) as f64);
    cdf.partition_point(|&p| p < u) + 1
}

fn run(n: usize, values: &[usize]) {
    let keys: Vec<u64> = (0..n as u64).collect();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);

    let t0 = Instant::now();
    let func = CompVFunc::<u64>::try_par_new_with_builder(
        &keys,
        &values[..n],
        Huffman::new(),
        VBuilder::<_, _, Fuse3Shards>::default(),
        &mut pl,
    );
    let dt = t0.elapsed();

    match func {
        Ok(f) => {
            let mut wrong = 0;
            for i in 0..n.min(1000) {
                if f.get(keys[i]) != values[i] {
                    wrong += 1;
                }
            }
            let bits = f.mem_size(SizeFlags::default()) * 8;
            let bits_per_key = bits as f64 / n as f64;
            let entropy: f64 = values[..n]
                .iter()
                .fold(
                    std::collections::HashMap::<usize, usize>::new(),
                    |mut m, &v| {
                        *m.entry(v).or_default() += 1;
                        m
                    },
                )
                .values()
                .map(|&c| {
                    let p = c as f64 / n as f64;
                    -p * p.log2()
                })
                .sum();
            let overhead = if entropy > 0.0 {
                100.0 * (bits_per_key / entropy - 1.0)
            } else {
                0.0
            };
            eprintln!(
                "n={n} {:.3}s {:.1}µs/key {:.3}bits/key H={:.3} oh={:+.1}% w={} wrong={wrong}",
                dt.as_secs_f64(),
                dt.as_secs_f64() * 1e6 / n as f64,
                bits_per_key,
                entropy,
                overhead,
                f.global_max_codeword_length(),
            );
        }
        Err(e) => eprintln!("n={n} FAILED: {e}"),
    }
    eprintln!();
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    let max_n = 10_000_000_000_usize;
    let sizes: Vec<usize> = vec![
        100_000_000,
        200_000_000,
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        10_000_000_000_usize,
    ];

    // ── Uniform(0..256) ──
    eprintln!("=== Uniform(0..256) ===");
    let mut rng = SmallRng::seed_from_u64(0);
    let uniform_vals: Vec<usize> = (0..max_n).map(|_| rng.random_range(0..256)).collect();
    for &n in &sizes {
        run(n, &uniform_vals);
    }

    // ── Zipf(1, 1_000_000) ──
    eprintln!("=== Zipf(1, 1_000_000) ===");
    let cdf = zipf_cdf(1.0, 1_000_000);
    let mut rng = SmallRng::seed_from_u64(0);
    let zipf_vals: Vec<usize> = (0..max_n).map(|_| sample_zipf(&cdf, &mut rng)).collect();
    for &n in &sizes {
        run(n, &zipf_vals);
    }
}
