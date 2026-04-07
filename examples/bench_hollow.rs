/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Benchmark for [`HtDistMmphf`] and [`HtDistMmphfInt`] construction and query,
//! compared with [`LcpMmphf`] and [`LcpMmphfInt`].
//!
//! Usage:
//! ```text
//! # String keys from file
//! cargo run --release --features "epserde,clap" --example bench_hollow -- -f trec.terms 1000000
//!
//! # Integer keys
//! cargo run --release --features clap --example bench_hollow -- --int 1000000
//! cargo run --release --features clap --example bench_hollow -- --int 100000000
//! ```

use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use std::io::BufRead;
use sux::bits::BitFieldVec;
use sux::func::hollow_trie::{HtDistMmphfInt, HtDistMmphfStr};
use sux::func::lcp_mmphf::{LcpMmphf, LcpMmphfInt};
use sux::func::shard_edge::FuseLge3Shards;
use sux::traits::TryIntoUnaligned;
use sux::utils::FromSlice;
use std::time::Instant;

type DefLcpStr = LcpMmphf<str, BitFieldVec<Box<[usize]>>, [u64; 2], FuseLge3Shards>;

fn bench(label: &str, n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "  {label}: min {:.1} med {:.1} ns/key",
        timings[0],
        timings[timings.len() / 2],
    );
}

fn timed<R>(label: &str, n: usize, f: impl FnOnce() -> R) -> R {
    let start = Instant::now();
    let r = f();
    let d = start.elapsed();
    eprintln!(
        "  {label}: {:.3}s ({:.0} ns/key)",
        d.as_secs_f64(),
        d.as_nanos() as f64 / n as f64,
    );
    r
}

fn bits_per_key(obj: &impl MemSize, n: usize) -> f64 {
    obj.mem_size(SizeFlags::default()) as f64 * 8.0 / n as f64
}

// ── String benchmarks ───────────────────────────────────────────────

fn bench_strings(path: &str, max_n: usize, repeats: usize, ht_only: bool) -> Result<()> {
    eprintln!("Reading keys from {path}...");
    let file = std::fs::File::open(path)?;
    let mut keys: Vec<String> = std::io::BufReader::new(file)
        .lines()
        .take(max_n)
        .collect::<std::io::Result<_>>()?;
    let sorted = keys.windows(2).all(|w| w[0] < w[1]);
    if !sorted {
        keys.sort();
        keys.dedup();
    }
    let n = keys.len();
    eprintln!("{n} keys{}\n", if sorted { " (already sorted)" } else { " (sorted)" });

    // ── HtDistMmphf ──
    eprintln!("=== HtDistMmphf<str> ===");
    let ht = timed("Build", n, || {
        HtDistMmphfStr::try_new(FromSlice::new(&keys), n, no_logging![])
            .unwrap()
            .try_into_unaligned()
            .unwrap()
    });

    // Verify
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(ht.get(key.as_str()), i, "HtDist mismatch at {i}");
    }
    eprintln!("  Space: {:.2} bits/key", bits_per_key(&ht, n));

    bench("Query", n, repeats, || {
        let mut sum = 0usize;
        for key in &keys {
            sum = sum.wrapping_add(ht.get(key.as_str()));
        }
        std::hint::black_box(sum);
    });

    if !ht_only {
        eprintln!();

        // ── LcpMmphf ──
        eprintln!("=== LcpMmphf<str> ===");
        let lcp = timed("Build", n, || {
            DefLcpStr::try_new(FromSlice::new(&keys), n, no_logging![])
                .unwrap()
                .try_into_unaligned()
                .unwrap()
        });

        for (i, key) in keys.iter().enumerate() {
            assert_eq!(lcp.get(key.as_str()), i, "LcpMmphf mismatch at {i}");
        }
        eprintln!("  Space: {:.2} bits/key", bits_per_key(&lcp, n));

        bench("Query", n, repeats, || {
            let mut sum = 0usize;
            for key in &keys {
                sum = sum.wrapping_add(lcp.get(key.as_str()));
            }
            std::hint::black_box(sum);
        });
    }

    Ok(())
}

// ── Integer benchmarks ──────────────────────────────────────────────

fn bench_integers(n: usize, repeats: usize) -> Result<()> {
    eprintln!("Generating {n} random sorted usize keys...");
    // Use splitmix64 to generate random keys (no rand dependency issues)
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut keys: Vec<usize> = (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z ^ (z >> 31)) as usize
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let n = keys.len();
    eprintln!("{n} unique keys\n");

    // ── HtDistMmphfInt ──
    eprintln!("=== HtDistMmphfInt<usize> ===");
    let ht = timed("Build", n, || {
        HtDistMmphfInt::<usize>::try_new(FromSlice::new(&keys), n, no_logging![])
            .unwrap()
            .try_into_unaligned()
            .unwrap()
    });

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(ht.get(key), i, "HtDistInt mismatch at {i}");
    }
    eprintln!("  Space: {:.2} bits/key", bits_per_key(&ht, n));

    bench("Query", n, repeats, || {
        let mut sum = 0usize;
        for &key in &keys {
            sum = sum.wrapping_add(ht.get(key));
        }
        std::hint::black_box(sum);
    });

    eprintln!();

    // ── LcpMmphfInt ──
    eprintln!("=== LcpMmphfInt<usize> ===");
    let lcp = timed("Build", n, || {
        LcpMmphfInt::<usize>::try_new(FromSlice::new(&keys), n, no_logging![])
            .unwrap()
            .try_into_unaligned()
            .unwrap()
    });

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(lcp.get(key), i, "LcpMmphfInt mismatch at {i}");
    }
    eprintln!("  Space: {:.2} bits/key", bits_per_key(&lcp, n));

    bench("Query", n, repeats, || {
        let mut sum = 0usize;
        for &key in &keys {
            sum = sum.wrapping_add(lcp.get(key));
        }
        std::hint::black_box(sum);
    });

    Ok(())
}

// ── Main ────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Benchmarks HtDistMmphf vs LcpMmphf.", long_about = None)]
struct Args {
    /// File containing sorted UTF-8 keys, one per line.
    #[arg(short = 'f', long)]
    filename: Option<String>,
    /// Number of integer keys to generate.
    #[arg(long)]
    int: Option<usize>,
    /// Maximum number of keys (for file mode).
    #[arg(default_value = "18446744073709551615")]
    n: usize,
    /// Number of query repetitions.
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Skip LcpMmphf (only bench HtDist).
    #[arg(long)]
    ht_only: bool,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .try_init()
        .ok();

    let args = Args::parse();

    if let Some(path) = &args.filename {
        bench_strings(path, args.n, args.repeats, args.ht_only)?;
    }
    if let Some(n) = args.int {
        bench_integers(n, args.repeats)?;
    }
    if args.filename.is_none() && args.int.is_none() {
        eprintln!("Usage: --int <N> or -f <file> [N]");
        std::process::exit(1);
    }

    Ok(())
}
