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
impl<const LOG2_LOWER_BLOCK_SIZE: usize, const LOG2_ONES_PER_INVENTORY: usize> SelStruct<BitVec>
    for Rank10Sel<LOG2_LOWER_BLOCK_SIZE, LOG2_ONES_PER_INVENTORY>
{
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

pub fn bench_rank10sel<const LOG2_LOWER_BLOCK_SIZE: usize, const LOG2_ONES_PER_INVENTORY: usize>(
    c: &mut Criterion,
    uniform: bool,
) {
    let mut name = format!(
        "rank10sel_{}_{}",
        LOG2_LOWER_BLOCK_SIZE, LOG2_ONES_PER_INVENTORY
    );
    if !uniform {
        name.push_str("_non_uniform");
    }
    let mut group = c.benchmark_group(&name);
    bench_select::<sux::rank_sel::Rank10Sel<LOG2_LOWER_BLOCK_SIZE, LOG2_ONES_PER_INVENTORY>>(
        &mut group, &LENS, &DENSITIES, REPS, uniform,
    );
    group.finish();
}

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
            $bench_group.bench_function(
                BenchmarkId::from_parameter(format!(
                    "{}_{}_{}_{}",
                    $log_inv_size, $log_subinv_size, bitvec_id.0, bitvec_id.1
                )),
                |b| {
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

pub fn compare_simple_fixed(c: &mut Criterion) {
    let lens = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000];
    let mut group = c.benchmark_group("select_fixed2");

    let mut bitvecs = Vec::<BitVec>::new();
    let mut bitvec_ids = Vec::<(u64, f64)>::new();
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in [0.5] {
            // possible repetitions
            let bitvec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            bitvecs.push(bitvec);
            bitvec_ids.push((len, density));
        }
    }

    bench_select_fixed2!([10, 11, 12], [3], bitvecs, bitvec_ids, group);
    group.finish();

    let mut group = c.benchmark_group("simple_select");
    for (bitvec, bitvec_id) in std::iter::zip(&bitvecs, &bitvec_ids) {
        let bits = bitvec.clone();
        let num_ones = bits.count_ones();
        let sel: SimpleSelect = SimpleSelect::new(bits, 3);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}_{}", bitvec_id.0, bitvec_id.1)),
            |b| {
                b.iter(|| {
                    // use fastrange
                    let r =
                        ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128) >> 64) as usize;
                    black_box(unsafe { sel.select_unchecked(r) });
                })
            },
        );
    }
}
