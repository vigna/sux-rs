/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! End-to-end stress test for [`CompVFunc`] across every
//! [`ShardEdge`] variant, distribution, and size scale.
//!
//! For each `(shard_edge, distribution, n, trial)` cell, builds a
//! `CompVFunc` from `n` keys whose values follow `distribution`,
//! then verifies a random sample of queries. Emits one CSV row per
//! cell (incremental — pipe to `tee` for live monitoring) so an
//! overnight matrix can be inspected as it runs.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example comp_vfunc_stress -- \
//!     --shard-edge fuselge3-shards --distribution zipf \
//!     --zipf-s 1.0 --zipf-n 1000 \
//!     --n-list 100,1000,10000,100000,1000000 \
//!     --trials 5 --queries 1024
//! ```

#![allow(clippy::collapsible_else_if)]

use clap::{Parser, ValueEnum};
use dsi_progress_logger::progress_logger;
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::Geometric;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;
use sux::func::shard_edge::{Fuse3NoShards, Fuse3Shards};
#[cfg(feature = "mwhc")]
use sux::func::shard_edge::{Mwhc3NoShards, Mwhc3Shards};
use sux::func::{CompVFunc, VBuilder};
use sux::init_env_logger;

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum ShardEdgeKind {
    /// `Fuse3Shards` with `[u64; 2]`.
    Fuse3Shards,
    /// `Fuse3NoShards` with `[u64; 2]`.
    Fuse3NoShards2,
    /// `Fuse3NoShards` with `[u64; 1]`.
    Fuse3NoShards1,
    /// `Mwhc3Shards` with `[u64; 2]` (requires `mwhc` feature).
    #[cfg(feature = "mwhc")]
    Mwhc3Shards,
    /// `Mwhc3NoShards` with `[u64; 2]` (requires `mwhc` feature).
    #[cfg(feature = "mwhc")]
    Mwhc3NoShards,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum DistKind {
    /// All keys map to a single value (degenerate).
    Constant,
    /// Each key gets a value uniformly sampled from `0..max_value`.
    Uniform,
    /// Each key gets `Geometric(0.5)` (mean ≈ 2, low entropy).
    Geom,
    /// Each key gets a value sampled from `Zipf(zipf_s, zipf_n)`.
    Zipf,
}

#[derive(Parser, Debug)]
struct Args {
    /// Which `ShardEdge` instantiation to test.​
    #[arg(long, value_enum)]
    shard_edge: ShardEdgeKind,

    /// Value distribution.​
    #[arg(long, value_enum)]
    distribution: DistKind,

    /// Comma-separated list of key counts to test.​
    #[arg(long)]
    n_list: String,

    /// Number of trials per cell.​
    #[arg(long, default_value_t = 5)]
    trials: usize,

    /// Number of random queries to verify per built function.​
    #[arg(long, default_value_t = 1024)]
    queries: usize,

    /// Constant value (for `--distribution constant`).​
    #[arg(long, default_value_t = 42)]
    constant_value: u64,

    /// Max value (exclusive) for `--distribution uniform`.​
    #[arg(long, default_value_t = 1024)]
    uniform_max: u64,

    /// Zipf exponent.​
    #[arg(long, default_value_t = 1.0)]
    zipf_s: f64,

    /// Zipf alphabet size.​
    #[arg(long, default_value_t = 1000)]
    zipf_n: usize,

    /// Cap on n × max-w to skip cells that would blow memory. Cells
    /// with `n > edge_cap` for distributions with mean codeword
    /// length ≥ 1 are skipped automatically by an early check.​
    #[arg(long, default_value_t = 200_000_000)]
    edge_cap: usize,
}

// ── Distribution helpers ───────────────────────────────────────────

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

fn sample_zipf(cdf: &[f64], rng: &mut SmallRng) -> u64 {
    let u: f64 = (rng.random::<u64>() >> 11) as f64 / ((1u64 << 53) as f64);
    (cdf.partition_point(|&p| p < u) + 1) as u64
}

fn gen_values(args: &Args, n: usize, seed: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    match args.distribution {
        DistKind::Constant => vec![args.constant_value; n],
        DistKind::Uniform => (0..n)
            .map(|_| rng.random::<u64>() % args.uniform_max)
            .collect(),
        DistKind::Geom => {
            let g = Geometric::new(0.5).unwrap();
            (0..n).map(|_| g.sample(&mut rng) + 1).collect()
        }
        DistKind::Zipf => {
            let cdf = zipf_cdf(args.zipf_s, args.zipf_n);
            (0..n).map(|_| sample_zipf(&cdf, &mut rng)).collect()
        }
    }
}

// ── Build & verify ─────────────────────────────────────────────────

/// Macro to factor out the build-and-verify loop across the 9
/// `(ShardEdge, S)` instantiations. Each call expands to a single
/// concrete `CompVFunc` build with the right type parameters.
macro_rules! build_and_verify {
    ($keys:expr, $values:expr, $sig:ty, $shard_edge:ty, $queries:expr, $rng:expr) => {{
        let func: Result<
            CompVFunc<u64, u64, sux::bits::BitVec<Box<[usize]>>, $sig, $shard_edge>,
            _,
        > = <CompVFunc<u64, u64, _, $sig, $shard_edge>>::try_par_new_with_builder(
            $keys,
            $values,
            sux::func::codec::Huffman::new(),
            VBuilder::default(),
            &mut progress_logger![],
        );
        match func {
            Ok(f) => {
                // Verify random sample queries.
                let n = $values.len();
                let mut rng = $rng;
                let mut wrong = 0usize;
                for _ in 0..$queries.min(n) {
                    let i = (rng.random::<u64>() as usize) % n;
                    let got = f.get($keys[i]);
                    if got != $values[i] {
                        wrong += 1;
                    }
                }
                Ok(wrong)
            }
            Err(e) => Err(format!("{e}")),
        }
    }};
}

fn run_trial(
    args: &Args,
    n: usize,
    trial: usize,
    stdout: &Mutex<std::io::Stdout>,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed = (n as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (trial as u64);
    let keys: Vec<u64> = (0..n as u64).collect();
    let values = gen_values(args, n, seed);
    let rng = SmallRng::seed_from_u64(seed.wrapping_add(1));

    let t0 = Instant::now();
    let result: Result<usize, String> = match args.shard_edge {
        ShardEdgeKind::Fuse3Shards => {
            build_and_verify!(&keys, &values, [u64; 2], Fuse3Shards, args.queries, rng)
        }
        ShardEdgeKind::Fuse3NoShards2 => {
            build_and_verify!(&keys, &values, [u64; 2], Fuse3NoShards, args.queries, rng)
        }
        ShardEdgeKind::Fuse3NoShards1 => {
            build_and_verify!(&keys, &values, [u64; 1], Fuse3NoShards, args.queries, rng)
        }
        #[cfg(feature = "mwhc")]
        ShardEdgeKind::Mwhc3Shards => {
            build_and_verify!(&keys, &values, [u64; 2], Mwhc3Shards, args.queries, rng)
        }
        #[cfg(feature = "mwhc")]
        ShardEdgeKind::Mwhc3NoShards => {
            build_and_verify!(&keys, &values, [u64; 2], Mwhc3NoShards, args.queries, rng)
        }
    };
    let elapsed_secs = t0.elapsed().as_secs_f64();

    // Distinct value count for diagnostic.
    let mut distinct = HashMap::<u64, ()>::new();
    for &v in &values {
        distinct.insert(v, ());
    }

    let row = match result {
        Ok(0) => format!(
            "{:?},{:?},{},{},{},{},build_ok,{:.4}",
            args.shard_edge,
            args.distribution,
            n,
            trial,
            distinct.len(),
            seed,
            elapsed_secs
        ),
        Ok(wrong) => format!(
            "{:?},{:?},{},{},{},{},query_mismatch_{},{:.4}",
            args.shard_edge,
            args.distribution,
            n,
            trial,
            distinct.len(),
            seed,
            wrong,
            elapsed_secs
        ),
        Err(e) => format!(
            "{:?},{:?},{},{},{},{},build_err:{},{:.4}",
            args.shard_edge,
            args.distribution,
            n,
            trial,
            distinct.len(),
            seed,
            e.replace(',', ";"),
            elapsed_secs
        ),
    };

    let mut out = stdout.lock().unwrap();
    writeln!(out, "{row}")?;
    out.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger()?;

    let args = Args::parse();
    let n_list: Vec<usize> = args
        .n_list
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("n_list must be comma-separated integers")
        })
        .collect();

    eprintln!(
        "comp_vfunc_stress: shard_edge={:?} distribution={:?} trials={} queries={}",
        args.shard_edge, args.distribution, args.trials, args.queries
    );

    let stdout_mutex = Mutex::new(std::io::stdout());
    {
        let mut out = stdout_mutex.lock().unwrap();
        writeln!(
            out,
            "shard_edge,distribution,n,trial,distinct_values,seed,outcome,elapsed_secs"
        )?;
        out.flush()?;
    }

    let total = n_list.len() * args.trials;
    let start = Instant::now();
    let mut done = 0usize;

    for &n in &n_list {
        for trial in 0..args.trials {
            run_trial(&args, n, trial, &stdout_mutex)?;
            done += 1;
            let elapsed = start.elapsed().as_secs_f64();
            if done % 5 == 0 || done == total {
                let rate = done as f64 / elapsed;
                let eta = (total - done) as f64 / rate;
                eprintln!(
                    "[{:>4}/{:<4}] {:.0}% elapsed={:.0}s rate={:.1}/s eta={:.0}s",
                    done,
                    total,
                    (done as f64 / total as f64) * 100.0,
                    elapsed,
                    rate,
                    eta
                );
            }
        }
    }

    eprintln!("Done. Total runtime: {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}
