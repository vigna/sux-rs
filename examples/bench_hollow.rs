/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Benchmark for [`HtDistMmphf`] construction and query.
//!
//! Usage:
//! ```text
//! cargo run --release --features epserde --example bench_hollow -- -f keys.txt 1000000
//! ```
//!
//! Reads up to N sorted keys from a file (one per line) and builds an
//! `HtDistMmphf`. Then queries all keys and reports timings.

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{DbgFlags, MemDbg, MemSize};
use std::io::BufRead;
use sux::bits::BitFieldVec;
use sux::func::hollow_trie::HtDistMmphfStr;
use sux::func::lcp_mmphf::LcpMmphf;
use sux::func::shard_edge::FuseLge3Shards;
use sux::utils::FromSlice;

type DefLcpStr = LcpMmphf<str, BitFieldVec<Box<[usize]>>, [u64; 2], FuseLge3Shards>;

fn bench(label: &str, n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
        eprintln!("{label}: {} ns/key", timings.last().unwrap());
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "{label} -- Min: {:.2} Median: {:.2} Max: {:.2} Avg: {:.2} ns/key",
        timings[0],
        timings[timings.len() / 2],
        timings.last().unwrap(),
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}

#[derive(Parser, Debug)]
#[command(about = "Benchmarks HtDistMmphf.", long_about = None)]
struct Args {
    /// Maximum number of keys to read.
    n: usize,
    /// File containing sorted UTF-8 keys, one per line.
    #[arg(short = 'f', long)]
    filename: String,
    /// Number of query repetitions.
    #[arg(short, long, default_value = "5")]
    repeats: usize,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    let args = Args::parse();

    // Read keys
    eprintln!("Reading up to {} keys from {}...", args.n, args.filename);
    let file = std::fs::File::open(&args.filename)?;
    let reader = std::io::BufReader::new(file);
    let mut keys: Vec<String> = Vec::with_capacity(args.n);
    for line in reader.lines() {
        if keys.len() >= args.n {
            break;
        }
        keys.push(line?);
    }
    let n = keys.len();
    eprintln!("Read {n} keys");

    // Build
    eprintln!("Building HtDistMmphf...");
    let start = std::time::Instant::now();
    let func = HtDistMmphfStr::try_new(FromSlice::new(&keys), n, no_logging![])?;
    func.mem_dbg(DbgFlags::default())?;
    let build_time = start.elapsed();
    eprintln!(
        "Build time: {:.3} s ({:.0} ns/key)",
        build_time.as_secs_f64(),
        build_time.as_nanos() as f64 / n as f64,
    );

    // Full verification
    for i in 0..n {
        let result = func.get(keys[i].as_str());
        assert_eq!(result, i, "HtDistMmphf verification failed at key {i}");
    }
    eprintln!("Verified all {n} keys");

    // Query benchmark
    bench("HtDist query", n, args.repeats, || {
        let mut sum = 0usize;
        for key in &keys {
            sum = sum.wrapping_add(func.get(key.as_str()));
        }
        std::hint::black_box(sum);
    });

    let ht_bits = func.mem_size(mem_dbg::SizeFlags::default()) * 8;
    eprintln!(
        "HtDistMmphf: {:.2} bits/key ({ht_bits} bits)\n",
        ht_bits as f64 / n as f64,
    );

    // ── LcpMmphfStr for comparison ─────────────────────────────────
    eprintln!("Building LcpMmphfStr...");
    let start = std::time::Instant::now();
    let lcp = DefLcpStr::try_new(FromSlice::new(&keys), n, no_logging![])?;
    let lcp_build = start.elapsed();
    eprintln!(
        "Build time: {:.3} s ({:.0} ns/key)",
        lcp_build.as_secs_f64(),
        lcp_build.as_nanos() as f64 / n as f64,
    );

    for i in 0..n {
        assert_eq!(
            lcp.get(keys[i].as_str()),
            i,
            "LcpMmphf verification failed at {i}"
        );
    }
    let lcp_bits = lcp.mem_size(mem_dbg::SizeFlags::default()) * 8;
    eprintln!(
        "LcpMmphfStr: {:.2} bits/key ({lcp_bits} bits)",
        lcp_bits as f64 / n as f64,
    );

    bench("LcpMmphf query", n, args.repeats, || {
        let mut sum = 0usize;
        for key in &keys {
            sum = sum.wrapping_add(lcp.get(key.as_str()));
        }
        std::hint::black_box(sum);
    });

    eprintln!(
        "\nBuild ratio: {:.2}x (LcpMmphf/HtDist)",
        lcp_build.as_secs_f64() / build_time.as_secs_f64()
    );

    Ok(())
}
