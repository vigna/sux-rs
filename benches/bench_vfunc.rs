use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Geometric, Zipf};
use std::hint::black_box;
use sux::bits::BitFieldVec;
use sux::func::vfunc2::VFunc2;
use sux::func::{VBuilder, VFunc};
use sux::utils::FromSlice;

/// Number of pregenerated queries (must be a power of 2 for masking).
const NUM_QUERIES: usize = 1 << 24;
const QUERY_MASK: usize = NUM_QUERIES - 1;

const SIZES: &[(usize, &str)] = &[(1_000_000, "1M"), (100_000_000, "100M")];

// ── Value distributions ─────────────────────────────────────────────

fn gen_geometric_values(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(42);
    let geo = Geometric::new(0.5).unwrap();
    (0..n).map(|_| geo.sample(&mut rng) as usize).collect()
}

fn gen_zipf_values(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(42);
    let zipf = Zipf::<f64>::new(1_000_000.0, 1.5).unwrap();
    (0..n).map(|_| zipf.sample(&mut rng) as usize - 1).collect()
}

fn gen_identity_values(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn gen_query_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize % n)
        .collect()
}

// ── Build helpers ───────────────────────────────────────────────────

fn build_vfunc(n: usize, values: &[usize]) -> VFunc<usize, usize, BitFieldVec<Box<[usize]>>> {
    let keys: Vec<usize> = (0..n).collect();
    VBuilder::<_, BitFieldVec<Box<[usize]>>>::default()
        .expected_num_keys(n)
        .try_build_func::<usize, usize>(
            FromSlice::new(&keys),
            FromSlice::new(values),
            no_logging![],
        )
        .unwrap()
}

fn build_vfunc2(n: usize, values: &[usize]) -> VFunc2<usize> {
    let keys: Vec<usize> = (0..n).collect();
    VFunc2::try_new(
        FromSlice::new(&keys),
        FromSlice::new(values),
        n,
        no_logging![],
    )
    .unwrap()
}

// ── Benchmarks ──────────────────────────────────────────────────────

fn bench_query(c: &mut Criterion) {
    let distributions: &[(&str, fn(usize) -> Vec<usize>)] = &[
        ("geometric", gen_geometric_values),
        ("zipf", gen_zipf_values),
        ("identity", gen_identity_values),
    ];

    for &(dist_name, gen_fn) in distributions {
        let mut group = c.benchmark_group(format!("vfunc_query_{dist_name}"));

        for &(n, label) in SIZES {
            let values = gen_fn(n);
            let max_val = values.iter().copied().max().unwrap_or(0);
            let query_indices = gen_query_indices(n);
            let queries: Vec<usize> = query_indices;

            // VFunc
            let vfunc = build_vfunc(n, &values);
            let vf_bytes = vfunc.mem_size(SizeFlags::default());
            let vf_bpk = vf_bytes as f64 * 8.0 / n as f64;
            eprintln!(
                "VFunc  {dist_name} n={label}: {vf_bytes} B, {vf_bpk:.2} bits/key (max_val={max_val})"
            );

            group.bench_function(BenchmarkId::new("VFunc", label), |b| {
                let mut ctr = 0usize;
                b.iter(|| {
                    let q = queries[ctr & QUERY_MASK];
                    ctr += 1;
                    black_box(vfunc.get(q))
                })
            });

            // VFunc2
            let vfunc2 = build_vfunc2(n, &values);
            let vf2_bytes = vfunc2.mem_size(SizeFlags::default());
            let vf2_bpk = vf2_bytes as f64 * 8.0 / n as f64;
            eprintln!("VFunc2 {dist_name} n={label}: {vf2_bytes} B, {vf2_bpk:.2} bits/key");

            group.bench_function(BenchmarkId::new("VFunc2", label), |b| {
                let mut ctr = 0usize;
                b.iter(|| {
                    let q = queries[ctr & QUERY_MASK];
                    ctr += 1;
                    black_box(vfunc2.get(q))
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
        ("identity", gen_identity_values),
    ];

    for &(dist_name, gen_fn) in distributions {
        let mut group = c.benchmark_group(format!("vfunc_construction_{dist_name}"));
        group.sample_size(10);

        let n = 1_000_000;
        let values = gen_fn(n);

        group.bench_function("VFunc", |b| {
            b.iter(|| {
                black_box(build_vfunc(n, &values));
            })
        });

        group.bench_function("VFunc2", |b| {
            b.iter(|| {
                black_box(build_vfunc2(n, &values));
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_query, bench_construction,);
criterion_main!(benches);
