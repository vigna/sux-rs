use criterion::black_box;
use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::Rank10Sel;
use sux::rank_sel::Rank9Sel;
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

const DENSITIES: [f64; 3] = [0.25, 0.5, 0.75];

const REPS: usize = 7;

trait SelStruct<B>: Select {
    fn new(bits: B) -> Self;
}
impl SelStruct<BitVec> for SimpleSelect {
    fn new(bits: BitVec) -> Self {
        SimpleSelect::new(bits, 3)
    }
}
impl SelStruct<BitVec> for Rank9Sel {
    fn new(bits: BitVec) -> Self {
        Rank9Sel::new(bits)
    }
}
impl SelStruct<BitVec> for Rank10Sel<256> {
    fn new(bits: BitVec) -> Self {
        Rank10Sel::new(bits)
    }
}
impl SelStruct<BitVec> for Rank10Sel<512> {
    fn new(bits: BitVec) -> Self {
        Rank10Sel::new(bits)
    }
}
impl SelStruct<BitVec> for Rank10Sel<1024, 11> {
    fn new(bits: BitVec) -> Self {
        Rank10Sel::new(bits)
    }
}
impl SelStruct<BitVec> for Rank10Sel<1024, 12> {
    fn new(bits: BitVec) -> Self {
        Rank10Sel::new(bits)
    }
}
impl SelStruct<BitVec> for Rank10Sel<1024, 13> {
    fn new(bits: BitVec) -> Self {
        Rank10Sel::new(bits)
    }
}

fn bench_select<S: SelStruct<BitVec>>(
    bench_group: &mut BenchmarkGroup<'_, WallTime>,
    lens: &[u64],
    densities: &[f64],
    reps: usize,
    non_uniform: bool,
) {
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in densities {
            let (density0, density1) = if non_uniform {
                (density * 0.01, density * 0.99)
            } else {
                (*density, *density)
            };
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

                let sel: S = S::new(bits);
                let mut routine = || {
                    let r = if rng.gen_bool(0.5) {
                        ((rng.gen::<u64>() as u128).wrapping_mul(num_ones_first_half as u128) >> 64)
                            as usize
                    } else {
                        num_ones_first_half
                            + ((rng.gen::<u64>() as u128)
                                .wrapping_mul(num_ones_second_half as u128)
                                >> 64) as usize
                    };
                    black_box(unsafe { sel.select_unchecked(r) });
                };
                bench_group.bench_function(
                    BenchmarkId::from_parameter(format!("{}_{}_{}", *len, *density, i)),
                    |b| b.iter(|| routine()),
                );
            }
        }
    }
}

pub fn bench_simple_select(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("simple_select");

    bench_select::<SimpleSelect>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_rank9sel(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank9sel");

    bench_select::<Rank9Sel>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_simple_select_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("simple_select_non_uniform");

    bench_select::<SimpleSelect>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}

pub fn bench_rank9sel_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank9sel_non_uniform");

    bench_select::<Rank9Sel>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}

pub fn bench_rank10sel_256(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_256");

    bench_select::<Rank10Sel<256>>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_rank10sel_512(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_512");

    bench_select::<Rank10Sel<512>>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_rank10sel_1024_11(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_1024_11");

    bench_select::<Rank10Sel<1024, 11>>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_rank10sel_1024_12(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_1024_12");

    bench_select::<Rank10Sel<1024, 12>>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_rank10sel_1024_13(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_1024_13");

    bench_select::<Rank10Sel<1024, 13>>(&mut bench_group, &LENS, &DENSITIES, REPS, false);

    bench_group.finish();
}

pub fn bench_rank10sel_256_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_256_non_uniform");

    bench_select::<Rank10Sel<256>>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}

pub fn bench_rank10sel_512_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_512_non_uniform");

    bench_select::<Rank10Sel<512>>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}

pub fn bench_rank10sel_1024_11_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_1024_11_non_uniform");

    bench_select::<Rank10Sel<1024, 11>>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}

pub fn bench_rank10sel_1024_12_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_1024_12_non_uniform");

    bench_select::<Rank10Sel<1024, 12>>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}

pub fn bench_rank10sel_1024_13_non_uniform(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank10sel_1024_13_non_uniform");

    bench_select::<Rank10Sel<1024, 13>>(&mut bench_group, &LENS, &DENSITIES, REPS, true);

    bench_group.finish();
}
