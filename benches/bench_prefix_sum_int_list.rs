use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::prelude::*;
use value_traits::slices::SliceByValue;

const SMALL: usize = 1 << 20;
const LARGE: usize = 1 << 30;

/// Builds a `PrefixSumIntList` of `n` elements drawn from a geometric
/// distribution: each value is `trailing_zeros(random_u64())`,
/// giving values in [0 . . 63] with *P*(*k*) = 2⁻*ᴷ*.
fn build_geometric(n: usize) -> PrefixSumIntList {
    let values: Vec<usize> = {
        let mut rng = SmallRng::seed_from_u64(0);
        (0..n)
            .map(|_| {
                let r: u64 = rng.random();
                r.trailing_zeros() as usize
            })
            .collect()
    };
    PrefixSumIntList::new(&values)
}

/// Builds a `PrefixSumIntList` of `n` elements drawn from a Zipf distribution
/// on the first billion integers with exponent 1 (≈1/*x*).
fn build_zipf(n: usize) -> PrefixSumIntList {
    let values: Vec<usize> = {
        let mut rng = SmallRng::seed_from_u64(42);
        let distr = rand_distr::Zipf::new(1E9_f64, 1.0).unwrap();
        (0..n)
            .map(|_| {
                let x: f64 = distr.sample(&mut rng);
                x as usize - 1
            })
            .collect()
    };
    PrefixSumIntList::new(&values)
}

fn bench_geometric(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_sum_int_list_ef_geom");
    for &(label, n) in &[("2^20", SMALL), ("2^30", LARGE)] {
        let list = build_geometric(n);
        let mask = (n - 1) as u64;
        let mut rng = SmallRng::seed_from_u64(1);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let index = (rng.random::<u64>() & mask) as usize;
                black_box(list.index_value(index))
            })
        });
    }

    group.finish();
}

fn bench_zipf(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_sum_int_list_ef_zipf");
    for &(label, n) in &[("2^20", SMALL), ("2^30", LARGE)] {
        let list = build_zipf(n);
        let mask = (n - 1) as u64;
        let mut rng = SmallRng::seed_from_u64(1);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let index = (rng.random::<u64>() & mask) as usize;
                black_box(list.index_value(index))
            })
        });
    }

    group.finish();
}

fn to_vec_prefix_sums(list: PrefixSumIntList) -> PrefixSumIntList<usize, Vec<usize>> {
    unsafe {
        list.map_prefix_sums(|d| {
            let n = d.len();
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(d.index_value(i));
            }
            v
        })
    }
}

fn bench_geometric_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_sum_int_list_vec_geom");
    for &(label, n) in &[("2^20", SMALL), ("2^30", LARGE)] {
        let list = to_vec_prefix_sums(build_geometric(n));
        let mask = (n - 1) as u64;
        let mut rng = SmallRng::seed_from_u64(1);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let index = (rng.random::<u64>() & mask) as usize;
                black_box(list.index_value(index))
            })
        });
    }

    group.finish();
}

fn bench_zipf_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_sum_int_list_vec_zipf");
    for &(label, n) in &[("2^20", SMALL), ("2^30", LARGE)] {
        let list = to_vec_prefix_sums(build_zipf(n));
        let mask = (n - 1) as u64;
        let mut rng = SmallRng::seed_from_u64(1);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let index = (rng.random::<u64>() & mask) as usize;
                black_box(list.index_value(index))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_geometric,
    bench_zipf,
    bench_geometric_vec,
    bench_zipf_vec,
);
criterion_main!(benches);
