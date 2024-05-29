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
        "rank9" => bench_rank9(&mut criterion),
        "rank10_8" => bench_rank10::<8>(&mut criterion),
        "rank10_9" => bench_rank10::<9>(&mut criterion),
        "rank10_10" => bench_rank10::<10>(&mut criterion),
        "rank11" => bench_rank11(&mut criterion),
        "simple_select" => bench_simple_select(&mut criterion),
        "rank9sel" => bench_rank9sel(&mut criterion),
        "simple_select_non_uniform" => bench_simple_select_non_uniform(&mut criterion),
        "rank9sel_non_uniform" => bench_rank9sel_non_uniform(&mut criterion),
        "rank10sel_8_11" => {
            bench_rank10sel::<8, 11>(&mut criterion, true);
            bench_rank10sel::<8, 11>(&mut criterion, false);
        }
        "rank10sel_10_12" => {
            bench_rank10sel::<10, 12>(&mut criterion, true);
            bench_rank10sel::<10, 12>(&mut criterion, false);
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
