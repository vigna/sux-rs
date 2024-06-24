mod bench_rank;
mod bench_select;
mod utils;

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
            bench_simple_select(&mut criterion, true, 0);
            bench_simple_select(&mut criterion, true, 1);
            bench_simple_select(&mut criterion, true, 2);
            bench_simple_select(&mut criterion, true, 3);
            bench_select_adapt(&mut criterion, true, 0);
            bench_select_adapt(&mut criterion, true, 1);
            bench_select_adapt(&mut criterion, true, 2);
            bench_select_adapt(&mut criterion, true, 3);
            bench_select9(&mut criterion, true);
            bench_select_small(&mut criterion, true, 0);
            bench_select_small(&mut criterion, true, 1);
            bench_select_small(&mut criterion, true, 2);
            bench_select_small(&mut criterion, true, 3);
            bench_select_small(&mut criterion, true, 4);
        }
        "select_non_uniform" => {
            bench_simple_select(&mut criterion, false, 0);
            bench_simple_select(&mut criterion, false, 1);
            bench_simple_select(&mut criterion, false, 2);
            bench_simple_select(&mut criterion, false, 3);
            bench_select_adapt(&mut criterion, false, 0);
            bench_select_adapt(&mut criterion, false, 1);
            bench_select_adapt(&mut criterion, false, 2);
            bench_select_adapt(&mut criterion, false, 3);
            bench_select9(&mut criterion, false);
            bench_select_small(&mut criterion, false, 0);
            bench_select_small(&mut criterion, false, 1);
            bench_select_small(&mut criterion, false, 2);
            bench_select_small(&mut criterion, false, 3);
            bench_select_small(&mut criterion, false, 4);
        }
        "rank" => {
            bench_rank9(&mut criterion);
            bench_rank_small(&mut criterion, 0);
            bench_rank_small(&mut criterion, 1);
            bench_rank_small(&mut criterion, 2);
            bench_rank_small(&mut criterion, 3);
            bench_rank_small(&mut criterion, 4);
        }
        "adapt" => {
            bench_select_adapt(&mut criterion, true, 3);
            bench_select_adapt(&mut criterion, false, 3);
        }
        "select9" => {
            bench_select9(&mut criterion, true);
            bench_select9(&mut criterion, false);
        }
        "rank9" => {
            bench_rank9(&mut criterion);
        }
        "select_adapt_const" => bench_select_adapt_const(&mut criterion, true),
        "select_adapt_const_mem_cost" => select_adapt_const_mem_cost(),
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
