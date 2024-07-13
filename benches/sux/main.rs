mod bench_select;
mod utils;
use bench_select::*;
use clap::{arg, Parser, ValueEnum};
use criterion::Criterion;
use sux::rank_sel::*;
use utils::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum RankSel {
    Rank9,
    RankSmall0,
    RankSmall1,
    RankSmall2,
    RankSmall3,
    RankSmall4,
    Select9,
    SelectSmall0,
    SelectSmall1,
    SelectSmall2,
    SelectSmall3,
    SelectSmall4,
    SelectAdapt0,
    SelectAdapt1,
    SelectAdapt2,
    SelectAdapt3,
    SimpleSelect0,
    SimpleSelect1,
    SimpleSelect2,
    SimpleSelect3,
    AdaptConst,
    CompareSimpleAdaptConst,
}

const MAPPING: [(&str, RankSel); 22] = [
    ("rank9", RankSel::Rank9),
    ("rank-small0", RankSel::RankSmall0),
    ("rank-small1", RankSel::RankSmall1),
    ("rank-small2", RankSel::RankSmall2),
    ("rank-small3", RankSel::RankSmall3),
    ("rank-small4", RankSel::RankSmall4),
    ("select9", RankSel::Select9),
    ("select-small0", RankSel::SelectSmall0),
    ("select-small1", RankSel::SelectSmall1),
    ("select-small2", RankSel::SelectSmall2),
    ("select-small3", RankSel::SelectSmall3),
    ("select-small4", RankSel::SelectSmall4),
    ("select-adapt0", RankSel::SelectAdapt0),
    ("select-adapt1", RankSel::SelectAdapt1),
    ("select-adapt2", RankSel::SelectAdapt2),
    ("select-adapt3", RankSel::SelectAdapt3),
    ("simple-select0", RankSel::SimpleSelect0),
    ("simple-select1", RankSel::SimpleSelect1),
    ("simple-select2", RankSel::SimpleSelect2),
    ("simple-select3", RankSel::SimpleSelect3),
    ("adapt-const", RankSel::AdaptConst),
    (
        "compare-simple-adapt-const",
        RankSel::CompareSimpleAdaptConst,
    ),
];

impl RankSel {
    fn from_str_exact(s: &str) -> Option<Self> {
        MAPPING
            .iter()
            .find_map(|(k, v)| if *k == s { Some(*v) } else { None })
    }

    fn from_str(s: &str) -> Vec<Self> {
        MAPPING
            .iter()
            .filter_map(|(k, v)| {
                if k.contains(s.to_lowercase().trim()) {
                    Some(*v)
                } else {
                    None
                }
            })
            .collect()
    }

    fn benchmark(
        &self,
        c: &mut Criterion,
        lens: &[u64],
        densities: &[f64],
        reps: usize,
        uniform: bool,
    ) {
        match self {
            RankSel::Rank9 => bench_rank::<Rank9>(c, "rank9", lens, densities, reps, uniform),
            RankSel::RankSmall0 => {
                bench_rank::<RankSmall<2, 9>>(c, "rank-small0", lens, densities, reps, uniform)
            }
            RankSel::RankSmall1 => {
                bench_rank::<RankSmall<1, 9>>(c, "rank-small1", lens, densities, reps, uniform)
            }
            RankSel::RankSmall2 => {
                bench_rank::<RankSmall<1, 10>>(c, "rank-small2", lens, densities, reps, uniform)
            }
            RankSel::RankSmall3 => {
                bench_rank::<RankSmall<1, 11>>(c, "rank-small3", lens, densities, reps, uniform)
            }
            RankSel::RankSmall4 => {
                bench_rank::<RankSmall<3, 13>>(c, "rank-small4", lens, densities, reps, uniform)
            }
            RankSel::Select9 => {
                bench_select::<Select9>(c, "select9", lens, densities, reps, uniform)
            }
            RankSel::SelectSmall0 => bench_select::<SelectSmall<2, 9>>(
                c,
                "select-small0",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall1 => bench_select::<SelectSmall<1, 9>>(
                c,
                "select-small1",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall2 => bench_select::<SelectSmall<1, 10>>(
                c,
                "select-small2",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall3 => bench_select::<SelectSmall<1, 11>>(
                c,
                "select-small3",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall4 => bench_select::<SelectSmall<3, 13>>(
                c,
                "select-small4",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectAdapt0 => {
                bench_select::<SelectAdapt0<_>>(c, "select-adapt0", lens, densities, reps, uniform)
            }
            RankSel::SelectAdapt1 => {
                bench_select::<SelectAdapt1<_>>(c, "select-adapt1", lens, densities, reps, uniform)
            }
            RankSel::SelectAdapt2 => {
                bench_select::<SelectAdapt2<_>>(c, "select-adapt2", lens, densities, reps, uniform)
            }
            RankSel::SelectAdapt3 => {
                bench_select::<SelectAdapt3<_>>(c, "select-adapt3", lens, densities, reps, uniform)
            }
            RankSel::SimpleSelect0 => bench_select::<SimpleSelect0<_>>(
                c,
                "simple-select0",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SimpleSelect1 => bench_select::<SimpleSelect1<_>>(
                c,
                "simple-select1",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SimpleSelect2 => bench_select::<SimpleSelect2<_>>(
                c,
                "simple-select2",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SimpleSelect3 => bench_select::<SimpleSelect3<_>>(
                c,
                "simple-select3",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::AdaptConst => {
                bench_select_adapt_const(c, uniform);
            }
            RankSel::CompareSimpleAdaptConst => {
                compare_simple_adapt_const(c);
            }
        }
    }
}

/// Command line arguments for the benchmarking suite.
#[derive(Parser, Debug)]
struct Cli {
    /// The lengths of the bitvectors to benchmark.
    #[arg(long, short, num_args = 1.., value_delimiter = ' ', default_value = "1000000 4000000 16000000 64000000 256000000 1024000000")]
    lens: Vec<u64>,
    /// The densities of the bitvectors to benchmark.
    #[arg(long, short, num_args = 1.., value_delimiter = ' ', default_value = "0.1 0.5 0.9")]
    densities: Vec<f64>,
    /// The number of repetitions for each benchmark.
    #[arg(long, short, default_value = "5")]
    reps: usize,
    /// Whether to use uniform or non-uniform distributions of 1s in the bitvectors.
    #[arg(short, long, default_value = "false")]
    non_uniform: bool,
    /// Flag for exact matching of the structure names to benchmark.
    #[arg(long, default_value = "false")]
    exact: bool,
    /// The rank/select structures to benchmark.
    #[arg(num_args = 1.., help = "The rank/select structures to benchmark. Without --exact, the arguments are matched as substrings. For example, 'rank' will match all rank structures. You could also give 'rank select' to benchmark all rank and select structures. Possible values: rank9, rank-small0, rank-small1, rank-small2, rank-small3, rank-small4, select9, select-small0, select-small1, select-small2, select-small3, select-small4, select-adapt0, select-adapt1, select-adapt2, select-adapt3, simple-select0, simple-select1, simple-select2, simple-select3, adapt-const, compare-simple-adapt-const")]
    rank_sel_struct: Vec<String>,
    // TODO: Add criterion arguments
    // #[arg(allow_hyphen_values = true, num_args = 1.., last = true)]
    // criterion_args: Vec<String>,
}

fn main() {
    // i don't know why but i *always* get as last argumet "--bench" so i remove it
    let mut raw_args = std::env::args().collect::<Vec<_>>();
    // check if last argument is "--bench"
    if raw_args.len() > 1 && raw_args[raw_args.len() - 1] == "--bench" {
        // remove it
        raw_args = raw_args[0..raw_args.len() - 1].to_vec();
    }

    let args = Cli::parse_from(raw_args);

    // Criterion doesn't let you parse specific arguments and using configure_from_args
    // will parse all of them, even the ones that are not for Criterion, resulting
    // in an error.
    let mut criterion = Criterion::default()
        .with_filter("")
        .with_output_color(true)
        .without_plots();

    let lens = args.lens;
    let densities = args.densities;
    let reps = args.reps;
    let uniform = !args.non_uniform;

    let rank_sel_struct: Vec<RankSel>;
    if args.exact {
        rank_sel_struct = args
            .rank_sel_struct
            .iter()
            .filter_map(|s| RankSel::from_str_exact(s))
            .collect();
    } else {
        rank_sel_struct = args
            .rank_sel_struct
            .iter()
            .flat_map(|s| RankSel::from_str(s))
            .collect();
    }

    for rank_sel in rank_sel_struct {
        rank_sel.benchmark(&mut criterion, &lens, &densities, reps, uniform);
    }

    criterion.final_summary();
}
