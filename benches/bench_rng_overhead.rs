use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn bench_rng(c: &mut Criterion) {
    let len = 1u64 << 30;
    let mut rng = SmallRng::seed_from_u64(0);
    c.bench_function("rng_overhead", |b| {
        b.iter(|| black_box(((rng.gen::<u64>() as u128).wrapping_mul(len as u128) >> 64) as usize))
    });
}

fn bench_rng_non_uniform(c: &mut Criterion) {
    let len = 1u64 << 30;
    let num_ones_first_half = (len as usize / 2) / 4096;
    let num_ones_second_half = (len as usize / 2) / 512;
    let mut rng = SmallRng::seed_from_u64(0);
    c.bench_function("rng_non_uniform_overhead", |b| {
        b.iter(|| {
            black_box(if rng.gen_bool(0.5) {
                ((rng.gen::<u64>() as u128).wrapping_mul(num_ones_first_half as u128) >> 64)
                    as usize
            } else {
                num_ones_first_half
                    + ((rng.gen::<u64>() as u128).wrapping_mul(num_ones_second_half as u128) >> 64)
                        as usize
            })
        })
    });
}

criterion_group!(benches, bench_rng, bench_rng_non_uniform);
criterion_main!(benches);
