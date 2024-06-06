use criterion::{black_box, BenchmarkId};
use criterion::{measurement::WallTime, BenchmarkGroup};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::BitVec;
use sux::traits::{Rank, Select};

mod impls;
pub use impls::*;

pub const LENS: [u64; 6] = [
    1_000_000,
    4_000_000,
    16_000_000,
    64_000_000,
    256_000_000,
    1_024_000_000,
];

pub const DENSITIES: [f64; 3] = [0.1, 0.5, 0.9];

pub const REPS: usize = 5;

pub trait Build<B> {
    fn new(bits: B) -> Self;
}

pub fn create_bitvec(
    rng: &mut SmallRng,
    len: u64,
    density: f64,
    uniform: bool,
) -> (u64, u64, BitVec) {
    let (density0, density1) = if uniform {
        (density, density)
    } else {
        (density * 0.01, density * 0.99)
    };

    let len1;
    let len2;
    if len % 2 == 0 {
        len1 = len / 2;
        len2 = len / 2;
    } else {
        len1 = len / 2 + 1;
        len2 = len / 2;
    }

    let first_half = loop {
        let b = (0..len1)
            .map(|_| rng.gen_bool(density0))
            .collect::<BitVec>();
        if b.count_ones() > 0 {
            break b;
        }
    };
    let second_half = (0..len2)
        .map(|_| rng.gen_bool(density1))
        .collect::<BitVec>();
    let num_ones_second_half = second_half.count_ones() as u64;
    let num_ones_first_half = first_half.count_ones() as u64;

    let bits = first_half
        .into_iter()
        .chain(second_half.into_iter())
        .collect::<BitVec>();

    (num_ones_first_half, num_ones_second_half, bits)
}

#[inline(always)]
pub fn fastrange(rng: &mut SmallRng, range: u64) -> u64 {
    ((rng.gen::<u64>() as u128).wrapping_mul(range as u128) >> 64) as u64
}

#[inline(always)]
pub fn fastrange_non_uniform(rng: &mut SmallRng, first_half: u64, second_half: u64) -> u64 {
    if rng.gen_bool(0.5) {
        ((rng.gen::<u64>() as u128).wrapping_mul(first_half as u128) >> 64) as u64
    } else {
        first_half + ((rng.gen::<u64>() as u128).wrapping_mul(second_half as u128) >> 64) as u64
    }
}

pub fn bench_select<S: Build<BitVec> + Select>(
    bench_group: &mut BenchmarkGroup<'_, WallTime>,
    lens: &[u64],
    densities: &[f64],
    reps: usize,
    uniform: bool,
) {
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in densities {
            // possible repetitions
            for i in 0..reps {
                let (num_ones_first_half, num_ones_second_half, bits) =
                    create_bitvec(&mut rng, *len, *density, uniform);

                let sel: S = S::new(bits);
                let mut routine = || {
                    let r =
                        fastrange_non_uniform(&mut rng, num_ones_first_half, num_ones_second_half);
                    black_box(unsafe { sel.select_unchecked(r as usize) });
                };
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", *len, *density, i)),
                    |b| b.iter(&mut routine),
                );
            }
        }
    }
}

pub fn bench_rank<R: Build<BitVec> + Rank>(
    bench_group: &mut BenchmarkGroup<'_, WallTime>,
    lens: &[u64],
    densities: &[f64],
    reps: usize,
) {
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens.iter().copied() {
        for density in densities.iter().copied() {
            // possible repetitions
            for i in 0..reps {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let rank: R = R::new(bits);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let p = fastrange(&mut rng, len) as usize;
                            black_box(unsafe { rank.rank_unchecked(p) });
                        })
                    },
                );
            }
        }
    }
}
