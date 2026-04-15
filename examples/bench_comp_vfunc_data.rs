/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Builds a [`CompVFunc`] from `keys.txt` and `values.txt` (one decimal
//! integer per line, parallel) and times random queries through the
//! unaligned variant. Mirrors the C `test_csf3_byte_array` setup so the
//! Rust and C numbers can be compared on the *same* dataset.
//!
//! Run from the repo root:
//!
//! ```sh
//! cargo run --release --example bench_comp_vfunc_data -- keys.txt values.txt
//! ```

use anyhow::{Context, Result};
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Geometric};
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;
use sux::bits::{BitFieldVec, BitVecU};
use sux::func::{CompVFunc, VFunc};
use sux::traits::TryIntoUnaligned;
use sux::utils::FromSlice;

trait GetU {
    fn get_u(&self, key: usize) -> u64;
}

impl GetU for CompVFunc<usize, BitVecU<Box<[usize]>>> {
    #[inline(always)]
    fn get_u(&self, key: usize) -> u64 {
        self.get(key)
    }
}

fn read_decimals(path: &str) -> Result<Vec<u64>> {
    let s = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let mut out = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(
            line.parse::<u64>()
                .with_context(|| format!("parse {line}"))?,
        );
    }
    Ok(out)
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    let mut args = env::args().skip(1);
    let first = args.next();
    let (keys, values): (Vec<usize>, Vec<u64>) = match first.as_deref() {
        Some("--synth-geometric") => {
            // Synthetic geometric(0.5), default 1M keys or as requested
            // via the next CLI arg. Using the same harness as the file
            // mode lets us separate methodology (criterion vs
            // hand-coded loop) from data shape and sweep over n cleanly.
            let n: usize = args
                .next()
                .as_deref()
                .map(|s| s.parse().expect("n must be an integer"))
                .unwrap_or(1_000_000);
            eprintln!("synthetic geometric(0.5), n={n}");
            let keys: Vec<usize> = (0..n).collect();
            let mut rng = SmallRng::seed_from_u64(42);
            let geo = Geometric::new(0.5).unwrap();
            let values: Vec<u64> = (0..n).map(|_| geo.sample(&mut rng)).collect();
            (keys, values)
        }
        Some(p) => {
            let keys_path = p.to_string();
            let values_path = args.next().unwrap_or_else(|| "values.txt".to_string());
            eprintln!("Reading keys from {keys_path}");
            let keys: Vec<usize> = read_decimals(&keys_path)?
                .into_iter()
                .map(|x| x as usize)
                .collect();
            eprintln!("Reading values from {values_path}");
            let values = read_decimals(&values_path)?;
            (keys, values)
        }
        None => {
            eprintln!(
                "usage: bench_comp_vfunc_data [keys.txt values.txt | --synth-geometric] [repeats]"
            );
            return Ok(());
        }
    };
    assert_eq!(keys.len(), values.len(), "keys and values must be parallel");
    let n = keys.len();
    let max_value = values.iter().copied().max().unwrap_or(0);
    let distinct: std::collections::HashSet<_> = values.iter().copied().collect();
    eprintln!(
        "n = {n}, distinct values = {}, max value = {max_value}",
        distinct.len()
    );

    // ── Diagnostic: what does our Huffman codec produce? ──
    {
        use std::collections::HashMap;
        use sux::func::codec::{Codec, Coder, Huffman};
        let mut freqs: HashMap<u64, u64> = HashMap::new();
        for &v in &values {
            *freqs.entry(v).or_insert(0) += 1;
        }
        let coder = Huffman::new().build_coder(&freqs);
        let mut total_bits: u64 = 0;
        let mut by_len: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for (&v, &f) in &freqs {
            let l = coder.codeword_length(v);
            total_bits += f * l as u64;
            *by_len.entry(l).or_insert(0) += 1;
        }
        eprintln!(
            "Rust Huffman: total = {} bits, avg = {:.4} b/codeword",
            total_bits,
            total_bits as f64 / n as f64
        );
        for (l, c) in &by_len {
            eprintln!("  length {l}: {c} symbols");
        }
    }

    eprintln!("Building CompVFunc...");
    let t0 = Instant::now();
    let func = CompVFunc::<usize>::try_par_new(&keys, &values, no_logging![]).expect("build");
    let comp_build_secs = t0.elapsed().as_secs_f64();
    let bytes = func.mem_size(SizeFlags::default());
    let bpk = bytes as f64 * 8.0 / n as f64;
    eprintln!(
        "Built in {comp_build_secs:.2}s. Size: {bytes} B = {bpk:.2} bits/key (w = {}, esc_len = {}, esym_len = {})",
        func.global_max_codeword_length(),
        func.escape_length(),
        func.escaped_symbol_length(),
    );

    // ── Comparison build: flat VFunc on the same keys/values ──
    // Fits each value in `ceil(log2(max + 1))` bits. Build time
    // should be proportional to the peeling work, which is ~ one
    // edge per key — so CompVFunc/VFunc ≈ average codeword length
    // (the empirical entropy of the value distribution, rounded up
    // by the codec) if the peelers are equally efficient.
    let values_usize: Vec<usize> = values.iter().map(|&v| v as usize).collect();
    let flat_t0 = Instant::now();
    let flat = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
        FromSlice::new(&keys),
        FromSlice::new(&values_usize),
        n,
        Default::default(),
        no_logging![],
    )
    .expect("VFunc build");
    let flat_build_secs = flat_t0.elapsed().as_secs_f64();
    let flat_bytes = flat.mem_size(SizeFlags::default());
    let flat_bpk = flat_bytes as f64 * 8.0 / n as f64;
    eprintln!(
        "Flat VFunc built in {flat_build_secs:.2}s. Size: {flat_bytes} B = {flat_bpk:.2} bits/key"
    );
    eprintln!(
        "CompVFunc / VFunc build ratio: {:.2}×  (target ≈ entropy of the value distribution)",
        comp_build_secs / flat_build_secs.max(1e-9)
    );

    // Verify a sample of keys round-trip correctly.
    let mut rng = SmallRng::seed_from_u64(0);
    for _ in 0..1024 {
        let i = rng.random::<u64>() as usize % n;
        assert_eq!(
            func.get(keys[i]),
            values[i],
            "round-trip failure at key {}",
            keys[i]
        );
    }
    eprintln!("1024-sample round-trip OK");

    let mut func_u = func
        .try_into_unaligned()
        .expect("CompVFunc TryIntoUnaligned failed");
    eprintln!(
        "default decoder strategy: {}",
        if func_u.is_decoder_branchless() {
            "branchless"
        } else {
            "branchy"
        }
    );

    // Two query streams. The "sequential" stream walks the first
    // NUM_QUERIES keys in file order, matching what the C
    // `test_csf3_byte_array` harness does. The "random" stream draws
    // NUM_QUERIES key positions uniformly at random — same number of
    // queries, but each load goes to an unpredictable element of the
    // input keys array, defeating hardware prefetch on the input.
    const NUM_QUERIES: usize = 1_000_000;
    let seq_queries: Vec<usize> = keys.iter().copied().take(NUM_QUERIES).collect();
    let mut rng = SmallRng::seed_from_u64(7);
    let rnd_queries: Vec<usize> = (0..NUM_QUERIES)
        .map(|_| keys[rng.random::<u64>() as usize % n])
        .collect();

    let repeats = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(11);

    fn bench_stream(name: &str, queries: &[usize], func_u: &impl GetU, repeats: usize) {
        let mut timings: Vec<f64> = Vec::with_capacity(repeats);
        for r in 0..repeats {
            let t0 = Instant::now();
            let mut acc: u64 = 0;
            for &q in queries {
                acc ^= func_u.get_u(q);
            }
            black_box(acc);
            let elapsed = t0.elapsed().as_nanos() as f64 / queries.len() as f64;
            timings.push(elapsed);
            eprintln!("[{name}] repeat {r}: {elapsed:.2} ns/query");
        }
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "[{name}] min {:.2}, median {:.2}, max {:.2}, avg {:.2} ns/query",
            timings[0],
            timings[timings.len() / 2],
            timings.last().copied().unwrap_or(0.0),
            timings.iter().sum::<f64>() / timings.len() as f64,
        );
    }

    func_u.set_decoder_branchless(false);
    eprintln!("=== branchy decoder (forced) ===");
    bench_stream("seq  branchy   ", &seq_queries, &func_u, repeats);
    bench_stream("rnd  branchy   ", &rnd_queries, &func_u, repeats);

    func_u.set_decoder_branchless(true);
    eprintln!("=== branchless decoder (forced) ===");
    bench_stream("seq  branchless", &seq_queries, &func_u, repeats);
    bench_stream("rnd  branchless", &rnd_queries, &func_u, repeats);

    Ok(())
}
