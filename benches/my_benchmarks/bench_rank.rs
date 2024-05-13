use criterion::measurement::WallTime;
use criterion::{black_box, BenchmarkGroup, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::{PoppyRevisited, Rank10, Rank11, Rank12, Rank16, Rank9};
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

trait RankStruct<B>: Rank {
    fn new(bits: B) -> Self;
}
impl RankStruct<BitVec> for Rank9 {
    fn new(bits: BitVec) -> Self {
        Rank9::new(bits)
    }
}
impl RankStruct<BitVec> for PoppyRevisited {
    fn new(bits: BitVec) -> Self {
        PoppyRevisited::new(bits)
    }
}
impl<const LOG2_UPPER_BLOCK_SIZE: usize> RankStruct<BitVec> for Rank10<LOG2_UPPER_BLOCK_SIZE> {
    fn new(bits: BitVec) -> Self {
        Rank10::new(bits)
    }
}
impl RankStruct<BitVec> for Rank11 {
    fn new(bits: BitVec) -> Self {
        Rank11::new(bits)
    }
}
impl RankStruct<BitVec> for Rank12 {
    fn new(bits: BitVec) -> Self {
        Rank12::new(bits)
    }
}
impl RankStruct<BitVec> for Rank16 {
    fn new(bits: BitVec) -> Self {
        Rank16::new(bits)
    }
}

fn bench_rank<R: RankStruct<BitVec>>(
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

pub fn bench_poppy_revisited(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("poppy_revisited");

    bench_rank::<PoppyRevisited>(&mut bench_group, &LENS, &DENSITIES, REPS);

    bench_group.finish();
}

pub fn bench_rank10<const LOG2_UPPER_BLOCK_SIZE: usize>(c: &mut Criterion) {
    let name = format!("rank10_{}", LOG2_UPPER_BLOCK_SIZE);
    let mut group = c.benchmark_group(name);
    bench_rank::<Rank10<LOG2_UPPER_BLOCK_SIZE>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}

pub fn bench_rank11(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank11");

    bench_rank::<Rank11>(&mut bench_group, &LENS, &DENSITIES, REPS);

    bench_group.finish();
}

pub fn bench_rank12(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank12");

    bench_rank::<Rank12>(&mut bench_group, &LENS, &DENSITIES, REPS);

    bench_group.finish();
}

pub fn bench_rank16(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("rank16");

    bench_rank::<Rank16>(&mut bench_group, &LENS, &DENSITIES, REPS);

    bench_group.finish();
}
