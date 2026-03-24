use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::prelude::*;
use sux::traits::Word;
use value_traits::slices::SliceByValue;

#[cfg(not(target_pointer_width = "64"))]
const SIZES: [(&str, usize); 1] = [("2^20", 1 << 20)];
#[cfg(target_pointer_width = "64")]
const SIZES: [(&str, usize); 2] = [("2^20", 1 << 20), ("2^30", 1 << 30)];

/// Builds a `CompIntList` of `n` elements drawn from a geometric
/// distribution: each value is `trailing_zeros(random_u64())`,
/// giving values in [0 . . 63] with *P*(*k*) = 2⁻*ᴷ*.
fn build_geometric(n: usize) -> CompIntList<u64> {
    let mut rng = SmallRng::seed_from_u64(0);
    let values: Vec<u64> = (0..n)
        .map(|_| {
            let r: u64 = rng.random();
            r.trailing_zeros() as u64
        })
        .collect();
    CompIntList::new(0, &values)
}

/// Builds a `CompIntList` of `n` elements drawn from a Zipf distribution
/// on the first billion integers with exponent 1 (≈1/*x*).
fn build_zipf(n: usize) -> CompIntList<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    let distr = rand_distr::Zipf::new(1E9_f64, 1.0).unwrap();
    let values: Vec<u64> = (0..n)
        .map(|_| {
            let x: f64 = distr.sample(&mut rng);
            x as u64 - 1
        })
        .collect();
    CompIntList::new(0, &values)
}

fn bench_geometric(c: &mut Criterion) {
    let mut group = c.benchmark_group("comp_int_list_ef_geom");
    for &(label, n) in &SIZES {
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
    let mut group = c.benchmark_group("comp_int_list_ef_zipf");
    for &(label, n) in &SIZES {
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

fn to_vec_delimiters<V: Word>(list: CompIntList<V>) -> CompIntList<V, Vec<u64>> {
    unsafe {
        list.map_delimiters(|d| {
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
    let mut group = c.benchmark_group("comp_int_list_vec_geom");
    for &(label, n) in &SIZES {
        let list = to_vec_delimiters(build_geometric(n));
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
    let mut group = c.benchmark_group("comp_int_list_vec_zipf");
    for &(label, n) in &SIZES {
        let list = to_vec_delimiters(build_zipf(n));
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
