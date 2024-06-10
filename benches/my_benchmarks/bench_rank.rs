use crate::utils::*;
use criterion::Criterion;
use sux::rank_sel::{Rank11, Rank9, RankSmall};

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

pub fn bench_rank_small0(c: &mut Criterion) {
    let name = format!("rank_small_0");
    let mut group = c.benchmark_group(name);
    bench_rank::<RankSmall<2, 9>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}

pub fn bench_rank_small1(c: &mut Criterion) {
    let name = format!("rank_small_1");
    let mut group = c.benchmark_group(name);
    bench_rank::<RankSmall<1, 9>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}

pub fn bench_rank_small2(c: &mut Criterion) {
    let name = format!("rank_small_2");
    let mut group = c.benchmark_group(name);
    bench_rank::<RankSmall<1, 10>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}

pub fn bench_rank_small3(c: &mut Criterion) {
    let name = format!("rank_small_3");
    let mut group = c.benchmark_group(name);
    bench_rank::<RankSmall<1, 11>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}

pub fn bench_rank_small4(c: &mut Criterion) {
    let name = format!("rank_small_4");
    let mut group = c.benchmark_group(name);
    bench_rank::<RankSmall<3, 13>>(&mut group, &LENS, &DENSITIES, REPS);
    group.finish();
}
