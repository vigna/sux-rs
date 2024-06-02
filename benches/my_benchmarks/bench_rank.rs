use criterion::measurement::WallTime;
use criterion::{black_box, BenchmarkGroup, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::{Rank10, Rank11, Rank9, RankSmall};
use sux::traits::Rank;

const LENS: [u64; 6] = [
    1_000_000,
    4_000_000,
    16_000_000,
    64_000_000,
    256_000_000,
    1_024_000_000,
];

const DENSITIES: [f64; 3] = [0.1, 0.5, 0.9];

const REPS: usize = 5;

trait RankStruct: Rank {
    fn new(bits: BitVec) -> Self;
}
impl RankStruct for Rank9 {
    fn new(bits: BitVec) -> Self {
        Rank9::new(bits)
    }
}
impl<const LOG2_LOWER_BLOCK_SIZE: usize> RankStruct for Rank10<LOG2_LOWER_BLOCK_SIZE> {
    fn new(bits: BitVec) -> Self {
        Rank10::new(bits)
    }
}
impl RankStruct for Rank11 {
    fn new(bits: BitVec) -> Self {
        Rank11::new(bits)
    }
}

impl<const NUM_U32S: usize> RankStruct for RankSmall<NUM_U32S> {
    fn new(bits: BitVec) -> Self {
        RankSmall::<NUM_U32S>::new(bits)
    }
}

fn bench_rank<R: RankStruct>(
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
                            let p = ((rng.gen::<u64>() as u128).wrapping_mul(len as u128) >> 64)
                                as usize;
                            black_box(unsafe { rank.rank_unchecked(p) });
                        })
                    },
                );
            }
        }
    }
}

pub fn bench_rank9(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank9");

    bench_rank::<Rank9>(&mut bench_group, &LENS, &DENSITIES, REPS);

    bench_group.finish();
}

pub fn bench_rank11(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank11");

    bench_rank::<Rank11>(&mut bench_group, &LENS, &DENSITIES, REPS);

    bench_group.finish();
}

pub fn bench_rank_small<const NUM_U32S: usize>(c: &mut Criterion) {
    let name = format!("rank_small_{}", NUM_U32S);
    let mut group = c.benchmark_group(name);
    bench_rank::<RankSmall<NUM_U32S>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}
