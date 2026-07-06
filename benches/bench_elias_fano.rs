use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::dict::EliasFanoConcurrentBuilder;
use sux::prelude::*;
use sux::traits::{IndexedSeq, Pred, PredUnchecked, Succ, SuccUnchecked, TryIntoUnaligned};

/// Number of pregenerated queries (must be a power of 2 for masking).
const NUM_QUERIES: usize = 1 << 20;
const QUERY_MASK: usize = NUM_QUERIES - 1;

/// (n, l) pairs: element count (power of 2) and desired number of lower bits.
/// The upper bound u = 2^l * n is chosen so that l = floor(log₂(u/n)).
const CONFIGS: &[(usize, usize)] = &[
    (1 << 20, 2),
    (1 << 20, 4),
    (1 << 20, 8),
    (1 << 20, 16),
    (1 << 30, 2),
    (1 << 30, 4),
    (1 << 30, 8),
    (1 << 30, 16),
];

fn n_label(n: usize) -> &'static str {
    if n == 1 << 20 { "1M" } else { "1G" }
}

type EfAligned = EliasFano<
    u64,
    SelectZeroAdaptConst<
        SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>,
        Box<[usize]>,
        12,
        3,
    >,
>;

/// Build an Elias–Fano structure with `n` elements and `l` lower bits.
/// Returns the structure and the first/last values in the monotone sequence.
fn build_ef(n: usize, l: usize) -> (EfAligned, u64, u64) {
    let u = (1u64 << l) * n as u64;
    let mut rng = SmallRng::seed_from_u64(0);
    let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
    values.sort_unstable();

    let first = values[0];
    let last = values[n - 1];

    let mut builder = EliasFanoBuilder::new(n, u);
    for &v in &values {
        builder.push(v);
    }
    drop(values);

    let ef = unsafe {
        builder
            .build()
            .map_high_bits(|h| SelectZeroAdaptConst::new(SelectAdaptConst::new(h)))
    };
    (ef, first, last)
}

/// Generate `NUM_QUERIES` random indices in [0, n) using bit masking.
fn gen_indices(n: usize) -> Vec<usize> {
    let mask = n - 1; // n is a power of 2
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize & mask)
        .collect()
}

/// Generate `NUM_QUERIES` random values suitable for succ/pred queries.
/// All values are in [first, u/2), well within the valid range.
fn gen_values(n: usize, l: usize, first: u64) -> Vec<u64> {
    let u = (1u64 << l) * n as u64;
    let mask = (u >> 1) - 1;
    let mut rng = SmallRng::seed_from_u64(2);
    (0..NUM_QUERIES)
        .map(|_| (rng.random::<u64>() & mask).max(first))
        .collect()
}

/// Each benchmark iteration performs a single operation, cycling through
/// the pregenerated query array using a counter and masking.
macro_rules! bench_ef {
    (index, $fn_name:ident, $group_name:expr, |$ef:ident, $q:ident| $op:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            for &(n, l) in CONFIGS {
                let (ef, _, _) = build_ef(n, l);
                let queries = gen_indices(n);
                let param = format!("{}/l={}", n_label(n), l);

                group.bench_function(BenchmarkId::new("aligned", &param), |b| {
                    let mut ctr = 0usize;
                    b.iter(|| {
                        let $q = queries[ctr & QUERY_MASK];
                        ctr = ctr.wrapping_add(1);
                        let $ef = &ef;
                        black_box($op)
                    })
                });

                let ef = ef.try_into_unaligned().unwrap();

                group.bench_function(BenchmarkId::new("unaligned", &param), |b| {
                    let mut ctr = 0usize;
                    b.iter(|| {
                        let $q = queries[ctr & QUERY_MASK];
                        ctr = ctr.wrapping_add(1);
                        let $ef = &ef;
                        black_box($op)
                    })
                });
            }
            group.finish();
        }
    };
    (value, $fn_name:ident, $group_name:expr, |$ef:ident, $q:ident| $op:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            for &(n, l) in CONFIGS {
                let (ef, first, _) = build_ef(n, l);
                let queries = gen_values(n, l, first);
                let param = format!("{}/l={}", n_label(n), l);

                group.bench_function(BenchmarkId::new("aligned", &param), |b| {
                    let mut ctr = 0usize;
                    b.iter(|| {
                        let $q = queries[ctr & QUERY_MASK];
                        ctr = ctr.wrapping_add(1);
                        let $ef = &ef;
                        black_box($op)
                    })
                });

                let ef = ef.try_into_unaligned().unwrap();

                group.bench_function(BenchmarkId::new("unaligned", &param), |b| {
                    let mut ctr = 0usize;
                    b.iter(|| {
                        let $q = queries[ctr & QUERY_MASK];
                        ctr = ctr.wrapping_add(1);
                        let $ef = &ef;
                        black_box($op)
                    })
                });
            }
            group.finish();
        }
    };
}

bench_ef!(
    index,
    bench_get_unchecked,
    "ef_get_unchecked",
    |ef, i| unsafe { IndexedSeq::get_unchecked(ef, i) }
);

bench_ef!(index, bench_get, "ef_get", |ef, i| IndexedSeq::get(ef, i));

bench_ef!(
    value,
    bench_succ_unchecked,
    "ef_succ_unchecked",
    |ef, v| unsafe { SuccUnchecked::succ_unchecked::<false>(ef, v) }
);

bench_ef!(value, bench_succ, "ef_succ", |ef, v| Succ::succ(ef, v));

bench_ef!(
    value,
    bench_pred_unchecked,
    "ef_pred_unchecked",
    |ef, v| unsafe { PredUnchecked::pred_unchecked::<false>(ef, v) }
);

bench_ef!(value, bench_pred, "ef_pred", |ef, v| Pred::pred(ef, v));

bench_ef!(
    value,
    bench_rank_unchecked,
    "ef_rank_unchecked",
    |ef, v| unsafe { PredUnchecked::rank_unchecked(ef, v) }
);

bench_ef!(value, bench_rank, "ef_rank", |ef, v| Pred::rank(ef, v));

fn bench_build_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_build_seq");
    for &(n, l) in CONFIGS {
        let u = (1u64 << l) * n as u64;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
        values.sort_unstable();
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("push", &param), |b| {
            b.iter(|| {
                let mut builder = EliasFanoBuilder::new(n, u);
                for &v in &values {
                    builder.push(v);
                }
                black_box(builder.build());
            })
        });
    }
    group.finish();
}

fn bench_build_concurrent(c: &mut Criterion) {
    let thread_counts = [4, 8, 16];
    let mut group = c.benchmark_group("ef_build_conc");
    for &(n, l) in CONFIGS {
        let u = (1u64 << l) * n as u64;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
        values.sort_unstable();

        for num_threads in thread_counts {
            let param = format!("{}/l={}/t={}", n_label(n), l, num_threads);
            let chunk_size = n.div_ceil(num_threads);
            let chunks: Vec<(usize, &[u64])> = values
                .chunks(chunk_size)
                .enumerate()
                .map(|(i, chunk)| (i * chunk_size, chunk))
                .collect();

            group.bench_function(BenchmarkId::new("set", &param), |b| {
                b.iter(|| {
                    let efcb = EliasFanoConcurrentBuilder::new(n, u);
                    std::thread::scope(|s| {
                        for &(start, chunk) in &chunks {
                            let efcb = &efcb;
                            s.spawn(move || {
                                for (j, &v) in chunk.iter().enumerate() {
                                    unsafe { efcb.set(start + j, v) };
                                }
                            });
                        }
                    });
                    black_box(efcb.build());
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get_unchecked,
    bench_get,
    bench_succ_unchecked,
    bench_succ,
    bench_pred_unchecked,
    bench_pred,
    bench_rank_unchecked,
    bench_rank,
    bench_build_sequential,
    bench_build_concurrent,
);
criterion_main!(benches);
