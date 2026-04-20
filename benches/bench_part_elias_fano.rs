use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::dict::{EfSeqDict, EliasFanoBuilder, PartEliasFanoBuilder};
use sux::traits::{IndexedSeq, Pred, PredUnchecked, Succ, SuccUnchecked};

const NUM_QUERIES: usize = 1 << 20;
const QUERY_MASK: usize = NUM_QUERIES - 1;

/// (n, l) pairs: element count and number of lower bits.
/// u = 2^l * n determines the universe density.
const CONFIGS: &[(usize, usize)] = &[
    (1 << 16, 4),
    (1 << 16, 8),
    (1 << 16, 16),
    (1 << 20, 4),
    (1 << 20, 8),
    (1 << 20, 16),
];

fn n_label(n: usize) -> &'static str {
    match n {
        65536 => "64K",
        1048576 => "1M",
        _ => "?",
    }
}

fn gen_sorted_values(n: usize, l: usize) -> Vec<usize> {
    let u = (1usize << l) * n;
    let mut rng = SmallRng::seed_from_u64(0);
    let mut values: Vec<usize> = (0..n).map(|_| rng.random_range(0..u)).collect();
    values.sort_unstable();
    values
}

fn build_ef(values: &[usize]) -> EfSeqDict<usize> {
    let n = values.len();
    let u = *values.last().unwrap();
    let mut builder = EliasFanoBuilder::new(n, u);
    for &v in values {
        builder.push(v);
    }
    builder.build_with_seq_and_dict()
}

fn build_pef(values: &[usize]) -> sux::dict::PartEliasFano {
    let n = values.len();
    let u = *values.last().unwrap();
    let mut builder = PartEliasFanoBuilder::new(n, u);
    for &v in values {
        builder.push(v);
    }
    builder.build()
}

fn gen_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random_range(0..n))
        .collect()
}

fn gen_values(values: &[usize]) -> Vec<usize> {
    let first = values[0];
    let last = *values.last().unwrap();
    let mut rng = SmallRng::seed_from_u64(2);
    (0..NUM_QUERIES)
        .map(|_| rng.random_range(first..=last))
        .collect()
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("pef_get");
    for &(n, l) in CONFIGS {
        let values = gen_sorted_values(n, l);
        let ef = build_ef(&values);
        let pef = build_pef(&values);
        let queries = gen_indices(n);
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("EF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let i = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(ef.get(i))
            })
        });

        group.bench_function(BenchmarkId::new("PEF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let i = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(pef.get(i))
            })
        });
    }
    group.finish();
}

fn bench_succ(c: &mut Criterion) {
    let mut group = c.benchmark_group("pef_succ");
    for &(n, l) in CONFIGS {
        let values = gen_sorted_values(n, l);
        let ef = build_ef(&values);
        let pef = build_pef(&values);
        let queries = gen_values(&values);
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("EF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(Succ::succ(&ef, v))
            })
        });

        group.bench_function(BenchmarkId::new("PEF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(Succ::succ(&pef, v))
            })
        });
    }
    group.finish();
}

fn bench_pred(c: &mut Criterion) {
    let mut group = c.benchmark_group("pef_pred");
    for &(n, l) in CONFIGS {
        let values = gen_sorted_values(n, l);
        let ef = build_ef(&values);
        let pef = build_pef(&values);
        let queries = gen_values(&values);
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("EF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(Pred::pred(&ef, v))
            })
        });

        group.bench_function(BenchmarkId::new("PEF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(Pred::pred(&pef, v))
            })
        });
    }
    group.finish();
}

fn bench_succ_unchecked(c: &mut Criterion) {
    let mut group = c.benchmark_group("pef_succ_unchecked");
    for &(n, l) in CONFIGS {
        let values = gen_sorted_values(n, l);
        let ef = build_ef(&values);
        let pef = build_pef(&values);
        let queries = gen_values(&values);
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("EF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(unsafe { SuccUnchecked::succ_unchecked::<false>(&ef, v) })
            })
        });

        group.bench_function(BenchmarkId::new("PEF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(unsafe { SuccUnchecked::succ_unchecked::<false>(&pef, v) })
            })
        });
    }
    group.finish();
}

fn bench_pred_unchecked(c: &mut Criterion) {
    let mut group = c.benchmark_group("pef_pred_unchecked");
    for &(n, l) in CONFIGS {
        let values = gen_sorted_values(n, l);
        let ef = build_ef(&values);
        let pef = build_pef(&values);
        let queries = gen_values(&values);
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("EF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(unsafe { PredUnchecked::pred_unchecked::<false>(&ef, v) })
            })
        });

        group.bench_function(BenchmarkId::new("PEF", &param), |b| {
            let mut ctr = 0usize;
            b.iter(|| {
                let v = queries[ctr & QUERY_MASK];
                ctr = ctr.wrapping_add(1);
                black_box(unsafe { PredUnchecked::pred_unchecked::<false>(&pef, v) })
            })
        });
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("pef_build");
    for &(n, l) in CONFIGS {
        let values = gen_sorted_values(n, l);
        let u = *values.last().unwrap();
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("EF", &param), |b| {
            b.iter(|| {
                let mut builder = EliasFanoBuilder::new(n, u);
                for &v in &values {
                    builder.push(v);
                }
                black_box(builder.build_with_seq_and_dict());
            })
        });

        group.bench_function(BenchmarkId::new("PEF", &param), |b| {
            b.iter(|| {
                let mut builder = PartEliasFanoBuilder::new(n, u);
                for &v in &values {
                    builder.push(v);
                }
                black_box(builder.build());
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get,
    bench_succ,
    bench_pred,
    bench_succ_unchecked,
    bench_pred_unchecked,
    bench_build,
);
criterion_main!(benches);
