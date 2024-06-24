use crate::utils::*;
use criterion::Criterion;
use sux::rank_sel::{Rank9, RankSmall};

pub fn bench_rank9(c: &mut Criterion) {
    let name = "Rank9";
    bench_rank::<Rank9>(c, name, &LENS, &DENSITIES, REPS);
}

pub fn bench_rank_small(c: &mut Criterion, rank_type: usize) {
    let name = match rank_type {
        0 => "RankSmall0",
        1 => "RankSmall1",
        2 => "RankSmall2",
        3 => "RankSmall3",
        4 => "RankSmall4",
        _ => panic!("Invalid type"),
    };

    match rank_type {
        0 => bench_rank::<RankSmall<2, 9>>(c, name, &LENS, &DENSITIES, REPS),
        1 => bench_rank::<RankSmall<1, 9>>(c, name, &LENS, &DENSITIES, REPS),
        2 => bench_rank::<RankSmall<1, 10>>(c, name, &LENS, &DENSITIES, REPS),
        3 => bench_rank::<RankSmall<1, 11>>(c, name, &LENS, &DENSITIES, REPS),
        4 => bench_rank::<RankSmall<3, 13>>(c, name, &LENS, &DENSITIES, REPS),
        _ => unreachable!(),
    }
}
