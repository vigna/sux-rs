use criterion::{black_box, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::{Rank11, Rank9};
use sux::traits::Rank;

const LENS: [u64; 11] = [
    1u64 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
    1 << 24,
    1 << 25,
    1 << 26,
    1 << 27,
    1 << 28,
    1 << 29,
    1 << 30,
];

const DENSITIES: [f64; 3] = [0.25, 0.5, 0.75];

const REPS: usize = 5;

pub fn bench_rank9(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank9");

    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in DENSITIES {
            // possible repetitions
            for i in 0..REPS {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let rank9: Rank9 = Rank9::new(bits);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let p = ((rng.gen::<u64>() as u128).wrapping_mul(len as u128) >> 64)
                                as usize;
                            black_box(unsafe { rank9.rank_unchecked(p) });
                        })
                    },
                );
            }
        }
    }

    bench_group.finish();
}

pub fn bench_rank11(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank11");

    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in DENSITIES {
            // possible repetitions
            for i in 0..REPS {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let rank11: Rank11 = Rank11::new(bits);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let p = ((rng.gen::<u64>() as u128).wrapping_mul(len as u128) >> 64)
                                as usize;
                            black_box(unsafe { rank11.rank_unchecked(p) });
                        })
                    },
                );
            }
        }
    }

    bench_group.finish();
}
