use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::prelude::*;
use value_traits::slices::SliceByValue;

const SMALL: usize = 1 << 20;
const LARGE: usize = 1 << 30;

/// Builds a `CompIntList` of `n` elements drawn from a geometric
/// distribution: each value is `trailing_zeros(random_u64())`,
/// giving values in [0 . . 63] with *P*(*k*) = 2⁻*ᴷ*.
fn build_geometric(n: usize) -> CompIntList {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut builder = CompIntListBuilder::new(0);
    for _ in 0..n {
        let r: u64 = rng.random();
        let value = r.trailing_zeros() as u64;
        builder.push(value);
    }
    builder.build()
}

/// Builds a `CompIntList` of `n` elements drawn from a Zipf distribution
/// on the first billion integers with exponent 1 (≈1/*x*).
fn build_zipf(n: usize) -> CompIntList {
    let mut rng = SmallRng::seed_from_u64(42);
    let distr = rand_distr::Zipf::new(1E9_f64, 1.0).unwrap();
    let mut builder = CompIntListBuilder::new(0);
    for _ in 0..n {
        let x: f64 = distr.sample(&mut rng);
        builder.push(x as u64 - 1);
    }
    builder.build()
}

fn bench_geometric(c: &mut Criterion) {
    let mut group = c.benchmark_group("comp_int_list_geometric");
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
    let mut group = c.benchmark_group("comp_int_list_zipf");
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

criterion_group!(benches, bench_geometric, bench_zipf);
criterion_main!(benches);
