use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::Rank9Sel;
use sux::rank_sel::SelectFixed2;
use sux::rank_sel::SimpleSelect;
use sux::traits::Select;

const LENS: [u64; 11] = [
    (1u64 << 20) + 2,
    (1 << 21) + 2,
    (1 << 22) + 2,
    (1 << 23) + 2,
    (1 << 24) + 2,
    (1 << 25) + 2,
    (1 << 26) + 2,
    (1 << 27) + 2,
    (1 << 28) + 2,
    (1 << 29) + 2,
    (1 << 30) + 2,
];

macro_rules! bench_select_fixed2 {
    ([$($inv_size:literal),+], $subinv_size:tt, $bitvecs:ident, $bitvec_ids:ident, $bench_group:expr) => {
        $(
            bench_select_fixed2!($inv_size, $subinv_size, $bitvecs, $bitvec_ids, $bench_group);
        )+
    };
    ($inv_size:literal, [$($subinv_size:literal),+], $bitvecs:ident, $bitvec_ids:ident, $bench_group:expr) => {
        $(
            bench_select_fixed2!($inv_size, $subinv_size, $bitvecs, $bitvec_ids, $bench_group);
        )+
    };
    ($log_inv_size:literal, $log_subinv_size:literal, $bitvecs:ident, $bitvec_ids:ident, $bench_group:expr) => {{
        let mut rng = SmallRng::seed_from_u64(0);
        for (bitvec, bitvec_id) in std::iter::zip(&$bitvecs, &$bitvec_ids) {
            let bits = bitvec.clone();
            let num_ones = bits.count_ones();
            let sel: SelectFixed2<BitVec, Vec<u64>, $log_inv_size, $log_subinv_size> =
                SelectFixed2::new(bits);
            $bench_group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}_{}_{}_{}_{}",
                    $log_inv_size, $log_subinv_size, bitvec_id.0, bitvec_id.1, bitvec_id.2
                )),
                &$log_inv_size,
                |b, _| {
                    b.iter(|| {
                        // use fastrange
                        let r =  ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128) >> 64) as usize;
                        black_box(unsafe { sel.select_unchecked(r) });
                    })
                },
            );
        }
    }};
}

pub fn bench_select_fixed2(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_fixed2");

    let mut bitvecs = Vec::<BitVec>::new();
    let mut bitvec_ids = Vec::<(u64, f64, u64)>::new();
    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in [0.25, 0.5, 0.75] {
            // possible repetitions
            for i in 0..5 {
                let bitvec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                bitvecs.push(bitvec);
                bitvec_ids.push((len, density, i));
            }
        }
    }

    bench_select_fixed2!([8, 9, 10, 11, 12], [1, 2, 3], bitvecs, bitvec_ids, group);

    group.finish();
}

pub fn bench_simple_select(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("simple_select");

    let reps = 5;

    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in [0.25, 0.5, 0.75] {
            // possible repetitions
            for i in 0..reps {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let num_ones = bits.count_ones();
                let simple: SimpleSelect = SimpleSelect::new(bits, 3);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let r = ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128)
                                >> 64) as usize;
                            black_box(unsafe { simple.select_unchecked(r) });
                        })
                    },
                );
            }
        }
    }

    bench_group.finish();
}

pub fn bench_rank9sel(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank9sel");

    let reps = 5;

    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in [0.25, 0.5, 0.75] {
            // possible repetitions
            for i in 0..reps {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let num_ones = bits.count_ones();
                let rank9sel: Rank9Sel<_, _> = Rank9Sel::new(bits);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let r = ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128)
                                >> 64) as usize;
                            black_box(unsafe { rank9sel.select_unchecked(r) });
                        })
                    },
                );
            }
        }
    }

    bench_group.finish();
}

pub fn bench_simple_select_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("simple_select_non_uniform");

    let reps = 5;

    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in [0.25, 0.5, 0.75] {
            let density0 = density * 0.01;
            let density1 = density * 0.99;
            // possible repetitions
            for i in 0..reps {
                let first_half = loop {
                    let b = (0..len / 2)
                        .map(|_| rng.gen_bool(density0))
                        .collect::<BitVec>();
                    if b.count_ones() > 0 {
                        break b;
                    }
                };
                let num_ones_first_half = first_half.count_ones();
                let second_half = (0..len / 2)
                    .map(|_| rng.gen_bool(density1))
                    .collect::<BitVec>();
                let num_ones_second_half = second_half.count_ones();
                let bits = first_half
                    .into_iter()
                    .chain(second_half.into_iter())
                    .collect::<BitVec>();

                let simple: SimpleSelect = SimpleSelect::new(bits, 3);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let r = if rng.gen_bool(0.5) {
                                ((rng.gen::<u64>() as u128)
                                    .wrapping_mul(num_ones_first_half as u128)
                                    >> 64) as usize
                            } else {
                                num_ones_first_half
                                    + ((rng.gen::<u64>() as u128)
                                        .wrapping_mul(num_ones_second_half as u128)
                                        >> 64) as usize
                            };
                            black_box(unsafe { simple.select_unchecked(r) });
                        })
                    },
                );
            }
        }
    }

    bench_group.finish();
}

pub fn bench_rank9sel_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank9sel_non_uniform");

    let reps = 5;

    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in [0.25, 0.5, 0.75] {
            let density0 = density * 0.01;
            let density1 = density * 0.99;
            // possible repetitions
            for i in 0..reps {
                let first_half = loop {
                    let b = (0..len / 2)
                        .map(|_| rng.gen_bool(density0))
                        .collect::<BitVec>();
                    if b.count_ones() > 0 {
                        break b;
                    }
                };
                let num_ones_first_half = first_half.count_ones();
                let second_half = (0..len / 2)
                    .map(|_| rng.gen_bool(density1))
                    .collect::<BitVec>();
                let num_ones_second_half = second_half.count_ones();

                let bits = first_half
                    .into_iter()
                    .chain(second_half.into_iter())
                    .collect::<BitVec>();

                let rank9sel: Rank9Sel<_, _> = Rank9Sel::new(bits);
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", len, density, i)),
                    |b| {
                        b.iter(|| {
                            // use fastrange
                            let r = if rng.gen_bool(0.5) {
                                ((rng.gen::<u64>() as u128)
                                    .wrapping_mul(num_ones_first_half as u128)
                                    >> 64) as usize
                            } else {
                                num_ones_first_half
                                    + ((rng.gen::<u64>() as u128)
                                        .wrapping_mul(num_ones_second_half as u128)
                                        >> 64) as usize
                            };
                            black_box(unsafe { rank9sel.select_unchecked(r) });
                        })
                    },
                );
            }
        }
    }

    bench_group.finish();
}

criterion_group!(
    benches,
    bench_simple_select,
    bench_rank9sel,
    bench_simple_select_non_uniform,
    bench_rank9sel_non_uniform
);
criterion_main!(benches);
