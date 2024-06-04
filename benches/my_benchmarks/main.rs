mod bench_rank;
mod bench_select;

use bench_rank::*;
use bench_select::*;
use criterion::Criterion;

/// Run benchmarks based on the provided filter.
/// Run with `cargo bench --bench my_benchmarks -- <filter> [optional criterion parameters]`.
fn main() {
    let mut criterion = Criterion::default()
        .without_plots()
        .configure_from_args()
        .with_filter("");

    let filter = std::env::args().nth(1).unwrap_or_default();

    match filter.as_str() {
        "select" => {
            bench_simple_select(&mut criterion, true);
            bench_select9(&mut criterion, true);
            bench_rank10sel::<10, 12>(&mut criterion, true);
        }
        "select_non_uniform" => {
            bench_simple_select(&mut criterion, false);
            bench_select9(&mut criterion, false);
            bench_rank10sel::<10, 12>(&mut criterion, false);
        }
        "rank" => {
            bench_rank9(&mut criterion);
            bench_rank11(&mut criterion);
            bench_rank_small0(&mut criterion);
            bench_rank_small1(&mut criterion);
            bench_rank_small2(&mut criterion);
            bench_rank_small3(&mut criterion);
            bench_rank_small4(&mut criterion);
        }
        "simple" => {
            bench_simple_select(&mut criterion, true);
            bench_simple_select(&mut criterion, false);
        }
        "select9" => {
            bench_select9(&mut criterion, true);
            bench_select9(&mut criterion, false);
        }
        "rank9" => {
            bench_rank9(&mut criterion);
        }
        "compare_simple_fixed" => compare_simple_fixed(&mut criterion),
        filter if filter.contains("-") || filter.is_empty() => {
            println!("No filter provided.");
        }
        _ => {
            println!("Invalid filter provided.");
        }
    }

    criterion.final_summary();
}
