use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dsi_progress_logger::no_logging;
use lender::*;
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::func::{LcpMmphfInt, LcpMmphfStr};
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
    assert!(strings.len() == n);
    strings
}

fn gen_query_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize % n)
        .collect()
}

// ── String lender from a Vec<String> ─────────────────────────────────
//
// Yields `&str` from a `Vec<String>`, rewindable, no extra allocation.

struct StringVecLender<'a> {
    strings: &'a [String],
    pos: usize,
}

impl<'a> StringVecLender<'a> {
    fn new(strings: &'a [String]) -> Self {
        Self { strings, pos: 0 }
    }
}

impl<'lend, 'a> FallibleLending<'lend> for StringVecLender<'a> {
    type Lend = &'lend str;
}

impl<'a> FallibleLender for StringVecLender<'a> {
    check_covariance_fallible!();
    type Error = core::convert::Infallible;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        if self.pos < self.strings.len() {
            let s = self.strings[self.pos].as_str();
            self.pos += 1;
            Ok(Some(s))
        } else {
            Ok(None)
        }
    }
}

impl<'a> sux::utils::FallibleRewindableLender for StringVecLender<'a> {
    type RewindError = core::convert::Infallible;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        Ok(StringVecLender {
            strings: self.strings,
            pos: 0,
        })
    }
}

// ── Integer benchmarks ───────────────────────────────────────────────

fn bench_int_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_int_construction");
    group.sample_size(10);

    for &(n, label) in SIZES {
        let keys = gen_sorted_u64(n);
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let func: LcpMmphfInt<u64> =
                    LcpMmphfInt::new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
                black_box(&func);
            })
        });
    }
    group.finish();
}

fn bench_int_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_int_query");

    for &(n, label) in SIZES {
        let keys = gen_sorted_u64(n);
        let func: LcpMmphfInt<u64> =
            LcpMmphfInt::new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();

        let query_indices = gen_query_indices(n);
        let queries: Vec<u64> = query_indices.iter().map(|&i| keys[i]).collect();

        let total_bytes = func.mem_size(SizeFlags::default());
        let bits_per_key = total_bytes as f64 * 8.0 / n as f64;
        eprintln!(
            "LcpMinPerfHashFuncInt n={label}: {total_bytes} bytes, {bits_per_key:.2} bits/key"
        );

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let q = queries[ctr & QUERY_MASK];
                ctr += 1;
                black_box(func.get(q))
            })
        });
    }
    group.finish();
}

// ── String benchmarks ────────────────────────────────────────────────

fn bench_str_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_str_construction");
    group.sample_size(10);

    for &(n, label) in SIZES {
        let keys = gen_sorted_strings(n);
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let func: LcpMmphfStr =
                    LcpMmphfStr::new(StringVecLender::new(&keys), keys.len(), no_logging![])
                        .unwrap();
                black_box(&func);
            })
        });
    }
    group.finish();
}

fn bench_str_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcp_mphf_str_query");

    for &(n, label) in SIZES {
        let keys = gen_sorted_strings(n);
        let func: LcpMmphfStr =
            LcpMmphfStr::new(StringVecLender::new(&keys), keys.len(), no_logging![]).unwrap();

        // Pack query strings contiguously to avoid pointer-chasing
        // cache misses from heap-allocated String data.
        let query_indices = gen_query_indices(n);
        let mut packed_data = Vec::new();
        let mut packed_offsets = Vec::with_capacity(NUM_QUERIES + 1);
        packed_offsets.push(0u32);
        for &i in &query_indices {
            packed_data.extend_from_slice(keys[i].as_bytes());
            packed_offsets.push(packed_data.len() as u32);
        }

        let total_bytes = func.mem_size(SizeFlags::default());
        let bits_per_key = total_bytes as f64 * 8.0 / n as f64;
        eprintln!(
            "LcpMinPerfHashFuncStr n={label}: {total_bytes} bytes, {bits_per_key:.2} bits/key"
        );

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let idx = ctr & QUERY_MASK;
                ctr += 1;
                let start = packed_offsets[idx] as usize;
                let end = packed_offsets[idx + 1] as usize;
                // SAFETY: packed_data was built from valid UTF-8 String data.
                let q = unsafe { std::str::from_utf8_unchecked(&packed_data[start..end]) };
                black_box(func.get(q))
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
