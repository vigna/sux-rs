/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::bits::BitFieldVec;
use sux::dict::VFilter;
use sux::func::VFunc;
use sux::traits::TryIntoUnaligned;

/// Number of pregenerated queries (must be a power of 2 for masking).
const NUM_QUERIES: usize = 1 << 17;
const QUERY_MASK: usize = NUM_QUERIES - 1;

#[cfg(feature = "slow_tests")]
const SIZES: &[(usize, &str)] = &[(1_000_000, "1M"), (100_000_000, "100M")];
#[cfg(not(feature = "slow_tests"))]
const SIZES: &[(usize, &str)] = &[(1_000_000, "1M")];

fn gen_query_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize % n)
        .collect()
}

fn build_filter_box_u8(n: usize) -> VFilter<VFunc<usize, Box<[u8]>>> {
    let keys: Vec<usize> = (0..n).collect();
    <VFilter<VFunc<usize, Box<[u8]>>>>::try_par_new(&keys, no_logging![]).unwrap()
}

fn build_filter_bfv(
    n: usize,
    filter_bits: usize,
) -> VFilter<VFunc<usize, BitFieldVec<Box<[usize]>>>> {
    let keys: Vec<usize> = (0..n).collect();
    <VFilter<VFunc<usize, BitFieldVec<Box<[usize]>>>>>::try_par_new(
        &keys,
        filter_bits,
        no_logging![],
    )
    .unwrap()
}

// ── Benchmarks ──────────────────────────────────────────────────────

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("vfilter_query");

    for &(n, label) in SIZES {
        let queries = gen_query_indices(n);

        // Box<[u8]> — 8-bit hashes, FPR = 1/256
        let filter_u8 = build_filter_box_u8(n);
        let bytes = filter_u8.mem_size(SizeFlags::default());
        let bpk = bytes as f64 * 8.0 / n as f64;
        eprintln!("VFilter Box<[u8]> n={label}: {bytes} B, {bpk:.2} bits/key");

        group.bench_function(BenchmarkId::new("Box_u8", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(filter_u8.contains(q))
            })
        });

        // BitFieldVec 9-bit — non-power-of-2 width, FPR = 1/512
        let filter_bfv9 = build_filter_bfv(n, 9);
        let bytes = filter_bfv9.mem_size(SizeFlags::default());
        let bpk = bytes as f64 * 8.0 / n as f64;
        eprintln!("VFilter BitFieldVec 9b n={label}: {bytes} B, {bpk:.2} bits/key");

        group.bench_function(BenchmarkId::new("BFV_9b", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(filter_bfv9.contains(q))
            })
        });

        // BitFieldVec 9-bit unaligned
        let filter_bfv9_u = filter_bfv9.try_into_unaligned().unwrap();

        group.bench_function(BenchmarkId::new("BFV_9b/unaligned", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(filter_bfv9_u.contains(q))
            })
        });
    }
    group.finish();
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("vfilter_construction");
    group.sample_size(10);

    let n = 1_000_000;

    group.bench_function("Box_u8", |b| {
        b.iter(|| {
            black_box(build_filter_box_u8(n));
        })
    });

    group.bench_function("BFV_9b", |b| {
        b.iter(|| {
            black_box(build_filter_bfv(n, 9));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_query, bench_construction);
criterion_main!(benches);
