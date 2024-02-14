mod bench_rank;
mod bench_select;
use bench_rank::{bench_rank11, bench_rank9};
use bench_select::bench_select_fixed2;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, bench_rank9, bench_rank11, bench_select_fixed2);
criterion_main!(benches);
