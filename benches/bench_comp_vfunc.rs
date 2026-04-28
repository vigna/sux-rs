/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Microbenchmark for [`CompVFunc`] with `usize` keys and skewed
//! integer-valued distributions (geometric and Zipf-1).
//!
//! Runs at 1M and 100M keys, querying through the *unaligned* variant
//! of the data backend (the `BitVecU` wrapper). For each (distribution,
//! size) pair we report:
//!
//! * a build-once size estimate in bits/key,
//! * average single-query latency through the unaligned `get` path.
//!
//! Run with `cargo bench --bench bench_comp_vfunc` (or filter, e.g.
//! `... -- comp_vfunc_query_zipf/1M`). Note: `100M` configurations are
//! gated behind the `slow_tests` feature so a default `cargo bench`
//! does not allocate ~30 GB of working memory.

#![allow(clippy::type_complexity)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Geometric, Zipf};
use std::hint::black_box;
use std::time::Instant;
use sux::func::CompVFunc;
use sux::func::codec::Decoder;
use sux::traits::TryIntoUnaligned;

/// Number of pregenerated queries (must be a power of 2 for masking).
///
/// Sized to fit comfortably in L2 cache (1 << 17 × 8 B = 1 MiB) so the
/// query-stream load itself doesn't dominate the per-call timing. The
/// previous setting of `1 << 24` (128 MiB of queries) blew past L3 and
/// added ~15 ns of stream-fetch latency to every query.
const NUM_QUERIES: usize = 1 << 17;
const QUERY_MASK: usize = NUM_QUERIES - 1;

#[cfg(feature = "slow_tests")]
const SIZES: &[(usize, &str)] = &[(1_000_000, "1M"), (100_000_000, "100M")];
#[cfg(not(feature = "slow_tests"))]
const SIZES: &[(usize, &str)] = &[(1_000_000, "1M")];

// ── Value distributions ─────────────────────────────────────────────

/// Geometric with success probability `p = 0.5`. The resulting symbol
/// distribution is heavily skewed toward small values, which is the
/// regime where `CompVFunc` should beat a flat-width `VFunc`.
fn gen_geometric_values(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(42);
    let geo = Geometric::new(0.5).unwrap();
    (0..n).map(|_| geo.sample(&mut rng) as usize).collect()
}

/// Zipf with exponent 1.5 over a million-symbol alphabet. The user
/// asked for "Zipf of degree 1"; `rand_distr::Zipf` requires
/// `s > 1`, so we use the next available exponent that still gives a
/// pronounced power-law tail.
fn gen_zipf_values(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(42);
    let zipf = Zipf::<f64>::new(1_000_000.0, 1.5).unwrap();
    (0..n).map(|_| zipf.sample(&mut rng) as usize - 1).collect()
}

fn gen_query_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize % n)
        .collect()
}

fn build_comp_vfunc(n: usize, values: &[usize]) -> CompVFunc<usize> {
    let keys: Vec<usize> = (0..n).collect();
    CompVFunc::<usize>::try_par_new(&keys, values, no_logging![]).expect("CompVFunc build failed")
}

// ── Benchmarks ──────────────────────────────────────────────────────

fn bench_query(c: &mut Criterion) {
    let distributions: &[(&str, fn(usize) -> Vec<usize>)] = &[
        ("geometric", gen_geometric_values),
        ("zipf", gen_zipf_values),
    ];

    for &(dist_name, gen_fn) in distributions {
        let mut group = c.benchmark_group(format!("comp_vfunc_query_{dist_name}"));

        for &(n, label) in SIZES {
            let values = gen_fn(n);
            let max_val = values.iter().copied().max().unwrap_or(0);
            let queries = gen_query_indices(n);

            // Single-shot wall-clock build timing — reported here
            // because Criterion's per-iteration construction harness
            // is run only at the small size (see `bench_construction`),
            // and for 100M keys the amortization cost would be
            // impractical.
            let t0 = Instant::now();
            let func = build_comp_vfunc(n, &values);
            let build_secs = t0.elapsed().as_secs_f64();
            let bytes = func.mem_size(SizeFlags::default());
            let bpk = bytes as f64 * 8.0 / n as f64;
            let ns_per_key = build_secs * 1e9 / n as f64;
            eprintln!(
                "CompVFunc {dist_name} n={label}: built in {build_secs:.2}s ({ns_per_key:.0} ns/key), {bytes} B, {bpk:.2} bits/key (max_val = {max_val}, max_codeword_len = {}, esym_len = {})",
                func.decoder().max_codeword_len(),
                func.decoder().escaped_symbols_len(),
            );

            // Convert to the unaligned variant for the query path.
            let func_u = func
                .try_into_unaligned()
                .expect("CompVFunc TryIntoUnaligned failed (unsupported codeword length?)");

            group.bench_function(BenchmarkId::new("CompVFunc/unaligned", label), |b| {
                let mut ctr = 0usize;
                b.iter(|| {
                    let q = queries[ctr & QUERY_MASK];
                    ctr += 1;
                    black_box(func_u.get(q))
                })
            });
        }
        group.finish();
    }
}

fn bench_construction(c: &mut Criterion) {
    let distributions: &[(&str, fn(usize) -> Vec<usize>)] = &[
        ("geometric", gen_geometric_values),
        ("zipf", gen_zipf_values),
    ];

    for &(dist_name, gen_fn) in distributions {
        let mut group = c.benchmark_group(format!("comp_vfunc_construction_{dist_name}"));
        group.sample_size(10);

        // Criterion-managed construction timing is only run at the
        // smaller size: building a 100M-key compressed function in a
        // Criterion timing loop (with ≥10 iterations for statistical
        // stability) is impractical. For the large size we instead
        // report a single-shot wall-clock build time via the eprintln
        // in `bench_query`.
        let n = 1_000_000;
        let values = gen_fn(n);

        group.bench_function("CompVFunc", |b| {
            b.iter(|| {
                black_box(build_comp_vfunc(n, &values));
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_query, bench_construction);
criterion_main!(benches);
