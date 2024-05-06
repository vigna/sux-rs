mod bench_rank;
mod bench_rng_overhead;
mod bench_select;

use bench_rank::*;
use bench_rng_overhead::*;
use bench_select::*;
use criterion::Criterion;

fn main() {
    let mut criterion = Criterion::default()
        .without_plots()
        .configure_from_args()
        .with_filter("");

    let filter = std::env::args().nth(1).unwrap_or_default();

    match filter.as_str() {
        filter if filter.contains("-") || filter.is_empty() => {
            bench_rank9(&mut criterion);
            bench_rank11(&mut criterion);
            bench_simple_select(&mut criterion);
            bench_rank9sel(&mut criterion);
            bench_simple_select_non_uniform(&mut criterion);
            bench_rank9sel_non_uniform(&mut criterion);
            bench_select_fixed2(&mut criterion);
            bench_rng(&mut criterion);
            bench_rng_non_uniform(&mut criterion);
        }
        "rank9" => bench_rank9(&mut criterion),
        "rank10_256" => bench_rank10_256(&mut criterion),
        "rank10_512" => bench_rank10_512(&mut criterion),
        "rank10_1024" => bench_rank10_1024(&mut criterion),
        "rank11" => bench_rank11(&mut criterion),
        "rank12" => bench_rank12(&mut criterion),
        "rank16" => bench_rank16(&mut criterion),
        "rank" => {
            bench_rank9(&mut criterion);
            bench_rank10_256(&mut criterion);
            bench_rank10_512(&mut criterion);
            bench_rank10_1024(&mut criterion);
            bench_poppy_revisited(&mut criterion);
            bench_rank11(&mut criterion);
            bench_rank12(&mut criterion);
            bench_rank16(&mut criterion);
        }
        "rank10" => {
            bench_rank10_256(&mut criterion);
            bench_rank10_512(&mut criterion);
            bench_rank10_1024(&mut criterion);
        }
        "simple_select" => bench_simple_select(&mut criterion),
        "rank9sel" => bench_rank9sel(&mut criterion),
        "simple_select_non_uniform" => bench_simple_select_non_uniform(&mut criterion),
        "rank9sel_non_uniform" => bench_rank9sel_non_uniform(&mut criterion),
        "select_fixed2" => bench_select_fixed2(&mut criterion),
        "rng" => bench_rng(&mut criterion),
        "rng_non_uniform" => bench_rng_non_uniform(&mut criterion),
        _ => {}
    }

    criterion.final_summary();
}
