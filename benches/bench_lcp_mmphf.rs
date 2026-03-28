use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::func::{Lcp2MmphfInt, Lcp2MmphfStr, LcpMmphfInt, LcpMmphfStr};
use sux::traits::TryIntoUnaligned;
use sux::utils::FromSlice;

/// Number of pregenerated queries (must be a power of 2 for masking).
const NUM_QUERIES: usize = 1 << 24;
const QUERY_MASK: usize = NUM_QUERIES - 1;

const SIZES: &[(usize, &str)] = &[(1_000_000, "1M"), (100_000_000, "100M")];

// ── Helpers ──────────────────────────────────────────────────────────

fn gen_sorted_u64(n: usize) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = Vec::with_capacity(n + n / 10);
    for _ in 0..n + n / 10 {
        keys.push(rng.random::<u64>());
    }
    keys.sort_unstable();
    keys.dedup();
    keys.truncate(n);
    assert!(
        keys.len() == n,
        "Not enough unique u64 values: got {} need {n}",
        keys.len()
    );
    keys
}

/// Sorted strings derived from sorted u64s: convert to string, then
/// re-sort lexicographically.
fn gen_sorted_strings(n: usize) -> Vec<String> {
    let ints = gen_sorted_u64(n);
    let mut strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
    strings.sort();
    strings.dedup();
    strings.truncate(n);
    assert_eq!(strings.len(), n);
    strings
}

fn gen_query_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize % n)
        .collect()
}

// ── Integer query ───────────────────────────────────────────────────

fn bench_int_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_int_query");

    for &(n, label) in SIZES {
        let keys = gen_sorted_u64(n);
        let query_indices = gen_query_indices(n);
        let queries: Vec<u64> = query_indices.iter().map(|&i| keys[i]).collect();

        // LcpMmphf (single-packed)
        let func: LcpMmphfInt<u64> =
            LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
        let bytes = func.mem_size(SizeFlags::default());
        eprintln!(
            "LcpMmphfInt n={label}: {bytes} bytes, {:.2} bits/key",
            bytes as f64 * 8.0 / n as f64
        );

        group.bench_function(BenchmarkId::new("LcpMmphf", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(func.get(q))
            })
        });

        // LcpMmphf (unaligned)
        let func_u = func.try_into_unaligned().unwrap();

        group.bench_function(BenchmarkId::new("LcpMmphf/unaligned", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(func_u.get(q))
            })
        });

        // Lcp2Mmphf (two-step)
        let func2: Lcp2MmphfInt<u64> =
            Lcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
        let bytes2 = func2.mem_size(SizeFlags::default());
        eprintln!(
            "Lcp2MmphfInt n={label}: {bytes2} bytes, {:.2} bits/key",
            bytes2 as f64 * 8.0 / n as f64
        );

        group.bench_function(BenchmarkId::new("Lcp2Mmphf", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(func2.get(q))
            })
        });

        // Lcp2Mmphf (unaligned)
        let func2_u = func2.try_into_unaligned().unwrap();

        group.bench_function(BenchmarkId::new("Lcp2Mmphf/unaligned", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(func2_u.get(q))
            })
        });
    }
    group.finish();
}

// ── String query ────────────────────────────────────────────────────

fn bench_str_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_str_query");

    for &(n, label) in SIZES {
        let keys = gen_sorted_strings(n);
        let query_indices = gen_query_indices(n);

        // Pack query strings contiguously.
        let mut packed_data = Vec::new();
        let mut packed_offsets = Vec::with_capacity(NUM_QUERIES + 1);
        packed_offsets.push(0u32);
        for &i in &query_indices {
            packed_data.extend_from_slice(keys[i].as_bytes());
            packed_offsets.push(packed_data.len() as u32);
        }

        // LcpMmphf (single-packed)
        let func: LcpMmphfStr =
            LcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
        let bytes = func.mem_size(SizeFlags::default());
        eprintln!(
            "LcpMmphfStr n={label}: {bytes} bytes, {:.2} bits/key",
            bytes as f64 * 8.0 / n as f64
        );

        group.bench_function(BenchmarkId::new("LcpMmphf", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let idx = ctr & QUERY_MASK;
                ctr += 1;
                let start = packed_offsets[idx] as usize;
                let end = packed_offsets[idx + 1] as usize;
                let q = unsafe { std::str::from_utf8_unchecked(&packed_data[start..end]) };
                black_box(func.get(q))
            })
        });

        // LcpMmphf (unaligned)
        let func_u = func.try_into_unaligned().unwrap();

        group.bench_function(BenchmarkId::new("LcpMmphf/unaligned", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let idx = ctr & QUERY_MASK;
                ctr += 1;
                let start = packed_offsets[idx] as usize;
                let end = packed_offsets[idx + 1] as usize;
                let q = unsafe { std::str::from_utf8_unchecked(&packed_data[start..end]) };
                black_box(func_u.get(q))
            })
        });

        // Lcp2Mmphf (two-step)
        let func2: Lcp2MmphfStr =
            Lcp2MmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
        let bytes2 = func2.mem_size(SizeFlags::default());
        eprintln!(
            "Lcp2MmphfStr n={label}: {bytes2} bytes, {:.2} bits/key",
            bytes2 as f64 * 8.0 / n as f64
        );

        group.bench_function(BenchmarkId::new("Lcp2Mmphf", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let idx = ctr & QUERY_MASK;
                ctr += 1;
                let start = packed_offsets[idx] as usize;
                let end = packed_offsets[idx + 1] as usize;
                let q = unsafe { std::str::from_utf8_unchecked(&packed_data[start..end]) };
                black_box(func2.get(q))
            })
        });

        // Lcp2Mmphf (unaligned)
        let func2_u = func2.try_into_unaligned().unwrap();

        group.bench_function(BenchmarkId::new("Lcp2Mmphf/unaligned", label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let idx = ctr & QUERY_MASK;
                ctr += 1;
                let start = packed_offsets[idx] as usize;
                let end = packed_offsets[idx + 1] as usize;
                let q = unsafe { std::str::from_utf8_unchecked(&packed_data[start..end]) };
                black_box(func2_u.get(q))
            })
        });
    }
    group.finish();
}

// ── Construction ────────────────────────────────────────────────────

fn bench_int_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_int_construction");
    group.sample_size(10);

    for &(n, label) in SIZES {
        let keys = gen_sorted_u64(n);

        group.bench_function(BenchmarkId::new("LcpMmphf", label), |b| {
            b.iter(|| {
                let func: LcpMmphfInt<u64> =
                    LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
                black_box(&func);
            })
        });

        group.bench_function(BenchmarkId::new("Lcp2Mmphf", label), |b| {
            b.iter(|| {
                let func: Lcp2MmphfInt<u64> =
                    Lcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])
                        .unwrap();
                black_box(&func);
            })
        });
    }
    group.finish();
}

fn bench_str_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_str_construction");
    group.sample_size(10);

    for &(n, label) in SIZES {
        let keys = gen_sorted_strings(n);

        group.bench_function(BenchmarkId::new("LcpMmphf", label), |b| {
            b.iter(|| {
                let func: LcpMmphfStr =
                    LcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
                black_box(&func);
            })
        });

        group.bench_function(BenchmarkId::new("Lcp2Mmphf", label), |b| {
            b.iter(|| {
                let func: Lcp2MmphfStr =
                    Lcp2MmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])
                        .unwrap();
                black_box(&func);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_int_query,
    bench_str_query,
    bench_int_construction,
    bench_str_construction,
);
criterion_main!(benches);
