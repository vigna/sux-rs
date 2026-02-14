use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::prelude::*;
use sux::traits::indexed_dict::IndexedSeq;

const N: usize = 10_000_000;

/// (label, upper_bound) pairs chosen so that l = floor(log2(u/n)) gives
/// the desired number of lower bits.
const CONFIGS: &[(usize, usize)] = &[
    (2, 4 * N),      // l = 2
    (4, 16 * N),     // l = 4
    (8, 256 * N),    // l = 8
    (16, 65536 * N), // l = 16
];

type EfBench = EliasFano<
    SelectZeroAdaptConst<
        SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>,
        Box<[usize]>,
        12,
        3,
    >,
>;

fn build_ef(u: usize) -> EfBench {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut values = Vec::with_capacity(N);
    for _ in 0..N {
        values.push(rng.random_range(0..u));
    }
    values.sort();

    let mut builder = EliasFanoBuilder::new(N, u);
    for &v in &values {
        builder.push(v);
    }

    unsafe {
        builder
            .build()
            .map_high_bits(|h| SelectZeroAdaptConst::new(SelectAdaptConst::new(h)))
    }
}

fn bench_forward_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_forward_iter");
    for &(l, u) in CONFIGS {
        let ef = build_ef(u);
        group.bench_function(BenchmarkId::from_parameter(format!("l={l}")), |b| {
            b.iter(|| {
                for v in &ef {
                    black_box(v);
                }
            })
        });
    }
    group.finish();
}

fn bench_back_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_back_iter");
    for &(l, u) in CONFIGS {
        let ef = build_ef(u);
        group.bench_function(BenchmarkId::from_parameter(format!("l={l}")), |b| {
            b.iter(|| {
                for v in ef.iter_back() {
                    black_box(v);
                }
            })
        });
    }
    group.finish();
}

fn bench_iter_bidi_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_iter_bidi_forwards");
    for &(l, u) in CONFIGS {
        let ef = build_ef(u);
        group.bench_function(BenchmarkId::from_parameter(format!("l={l}")), |b| {
            b.iter(|| {
                let (_, iter) = unsafe { ef.iter_bidi_from_succ_unchecked::<false>(0) };
                for v in iter {
                    black_box(v);
                }
            })
        });
    }
    group.finish();
}

fn bench_iter_bidi_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_iter_bidi_backwards");
    for &(l, u) in CONFIGS {
        let ef = build_ef(u);
        group.bench_function(BenchmarkId::from_parameter(format!("l={l}")), |b| {
            b.iter(|| {
                let (_, iter) =
                    unsafe { ef.iter_bidi_from_succ_unchecked::<false>(ef.get_unchecked(N - 1)) };
                for v in iter {
                    black_box(v);
                }
            })
        });
    }
    group.finish();
}

fn bench_sequential_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_sequential_select");
    for &(l, u) in CONFIGS {
        let ef = build_ef(u);
        group.bench_function(BenchmarkId::from_parameter(format!("l={l}")), |b| {
            b.iter(|| {
                for i in 0..N {
                    black_box(unsafe { IndexedSeq::get_unchecked(&ef, i) });
                }
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_iter,
    bench_back_iter,
    bench_iter_bidi,
    bench_sequential_select,
);
criterion_main!(benches);
