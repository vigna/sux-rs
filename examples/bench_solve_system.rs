/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Benchmark CompVFunc construction at n=100M with forced
//! high-mem vs low-mem peeling. Zipf(1, 1000) distribution
//! produces wider codewords (w ≈ 10) than geometric.

use dsi_progress_logger::no_logging;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;
use sux::bits::BitVec;
use sux::func::codec::{Decoder, HuffmanConf};
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

fn make_builder(low_mem: Option<bool>) -> VBuilder<BitVec<Box<[usize]>>, [u64; 2], Fuse3Shards> {
    let b = VBuilder::default();
    match low_mem {
        Some(lm) => b.low_mem(lm),
        None => b,
    }
}

fn bench(n: usize, keys: &[u64], values: &[usize], low_mem: Option<bool>, trials: usize) {
    let label = match low_mem {
        Some(true) => "low-mem",
        Some(false) => "high-mem",
        None => "auto",
    };

    // Warm up
    let _ = CompVFunc::<u64>::try_par_new_with_builder(
        keys,
        values,
        HuffmanConf::new(),
        make_builder(low_mem),
        no_logging![],
    );

    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let func = CompVFunc::<u64>::try_par_new_with_builder(
            keys,
            values,
            HuffmanConf::new(),
            make_builder(low_mem),
            no_logging![],
        )
        .expect("build");
        let dt = t0.elapsed().as_secs_f64();
        times.push(dt);
        if times.len() == 1 {
            eprintln!(
                "  max_codeword_len = {} esym_len  ={}",
                func.decoder().max_codeword_len(),
                func.decoder().escaped_symbols_len()
            );
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[trials / 2];
    let min = times[0];
    let max = times[trials - 1];
    eprintln!(
        "[{label:>8}] n={n}: min={min:.3}s median={median:.3}s max={max:.3}s ({:.1} µs/key)",
        median * 1e6 / n as f64
    );
}

fn main() {
    let n = 100_000_000usize;
    let trials = 5;

    eprintln!("Generating {n} keys + Zipf(1, 1000000) values...");
    let keys: Vec<u64> = (0..n as u64).collect();
    let cdf = zipf_cdf(1.0, 1_000_000);
    let mut rng = SmallRng::seed_from_u64(0);
    let values: Vec<usize> = (0..n).map(|_| sample_zipf(&cdf, &mut rng)).collect();
    eprintln!("Done generating. Starting benchmarks...\n");

    bench(n, &keys, &values, Some(false), trials);
    bench(n, &keys, &values, Some(true), trials);
    bench(n, &keys, &values, None, trials);
}
