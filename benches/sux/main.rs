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
    SelectAdaptConst0,
    SelectAdaptConst1,
    SelectAdaptConst2,
    SelectAdaptConst3,
    CompareSimpleAdaptConst,
}

const MAPPING: [(&str, RankSel); 21] = [
    ("rank9", RankSel::Rank9),
    ("RankSmall0", RankSel::RankSmall0),
    ("RankSmall1", RankSel::RankSmall1),
    ("RankSmall2", RankSel::RankSmall2),
    ("RankSmall3", RankSel::RankSmall3),
    ("RankSmall4", RankSel::RankSmall4),
    ("select9", RankSel::Select9),
    ("SelectSmall0", RankSel::SelectSmall0),
    ("SelectSmall1", RankSel::SelectSmall1),
    ("SelectSmall2", RankSel::SelectSmall2),
    ("SelectSmall3", RankSel::SelectSmall3),
    ("SelectSmall4", RankSel::SelectSmall4),
    ("SelectAdapt0", RankSel::SelectAdapt0),
    ("SelectAdapt1", RankSel::SelectAdapt1),
    ("SelectAdapt2", RankSel::SelectAdapt2),
    ("SelectAdapt3", RankSel::SelectAdapt3),
    ("SelectAdaptConst0", RankSel::SelectAdaptConst0),
    ("SelectAdaptConst1", RankSel::SelectAdaptConst1),
    ("SelectAdaptConst2", RankSel::SelectAdaptConst2),
    ("SelectAdaptConst3", RankSel::SelectAdaptConst3),
    ("CompareSimpleAdaptConst", RankSel::CompareSimpleAdaptConst),
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
                bench_rank::<RankSmall<2, 9>>(c, "RankSmall0", lens, densities, reps, uniform)
            }
            RankSel::RankSmall1 => {
                bench_rank::<RankSmall<1, 9>>(c, "RankSmall1", lens, densities, reps, uniform)
            }
            RankSel::RankSmall2 => {
                bench_rank::<RankSmall<1, 10>>(c, "RankSmall2", lens, densities, reps, uniform)
            }
            RankSel::RankSmall3 => {
                bench_rank::<RankSmall<1, 11>>(c, "RankSmall3", lens, densities, reps, uniform)
            }
            RankSel::RankSmall4 => {
                bench_rank::<RankSmall<3, 13>>(c, "RankSmall4", lens, densities, reps, uniform)
            }
            RankSel::Select9 => {
                bench_select::<Select9>(c, "select9", lens, densities, reps, uniform)
            }
            RankSel::SelectSmall0 => bench_select::<SelectSmall<2, 9, _>>(
                c,
                "SelectSmall0",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall1 => bench_select::<SelectSmall<1, 9, _>>(
                c,
                "SelectSmall1",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall2 => bench_select::<SelectSmall<1, 10, _>>(
                c,
                "SelectSmall2",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall3 => bench_select::<SelectSmall<1, 11, _>>(
                c,
                "SelectSmall3",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectSmall4 => bench_select::<SelectSmall<3, 13, _>>(
                c,
                "SelectSmall4",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectAdapt0 => {
                bench_select::<SelectAdapt0<_>>(c, "SelectAdapt0", lens, densities, reps, uniform)
            }
            RankSel::SelectAdapt1 => {
                bench_select::<SelectAdapt1<_>>(c, "SelectAdapt1", lens, densities, reps, uniform)
            }
            RankSel::SelectAdapt2 => {
                bench_select::<SelectAdapt2<_>>(c, "SelectAdapt2", lens, densities, reps, uniform)
            }
            RankSel::SelectAdapt3 => {
                bench_select::<SelectAdapt3<_>>(c, "SelectAdapt3", lens, densities, reps, uniform)
            }
            RankSel::SelectAdaptConst0 => bench_select::<SelectAdaptConst0<_>>(
                c,
                "SelectAdaptConst0",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectAdaptConst1 => bench_select::<SelectAdaptConst1<_>>(
                c,
                "SelectAdaptConst1",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectAdaptConst2 => bench_select::<SelectAdaptConst2<_>>(
                c,
                "SelectAdaptConst2",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::SelectAdaptConst3 => bench_select::<SelectAdaptConst3<_>>(
                c,
                "SelectAdaptConst3",
                lens,
                densities,
                reps,
                uniform,
            ),
            RankSel::CompareSimpleAdaptConst => {
                compare_adapt_const(c);
            }
        }
    }
}

/// Command line arguments for the benchmarking suite.
#[derive(Parser, Debug)]
struct Cli {
    /// The lengths of the bitvectors to benchmark.
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "1000000,4000000,16000000,64000000,256000000,1024000000")]
    lengths: Vec<u64>,
    /// The densities of the bitvectors to benchmark.
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "0.1,0.5,0.9")]
    densities: Vec<f64>,
    /// The number of repetitions for each benchmark.
    #[arg(long, short, default_value = "5")]
    repeats: usize,
    /// Whether to use uniform or non-uniform distributions of 1s in the bitvectors.
    #[arg(short, long, default_value = "false")]
    non_uniform: bool,
    /// Flag for exact matching of the structure names to benchmark.
    #[arg(long, default_value = "false")]
    exact: bool,
    /// The rank/select structures to benchmark.
    #[arg(num_args = 1.., help = "The rank/select structures to benchmark. \
    Without --exact, the arguments are matched as substrings. \
    For example, 'rank' will match all rank structures. \
    You could also give 'rank select' to benchmark all rank and select structures. \
    Possible values: \
    rank9, RankSmall0, RankSmall1, RankSmall2, RankSmall3, RankSmall4, \
    select9, SelectSmall0, SelectSmall1, SelectSmall2, SelectSmall3, SelectSmall4, \
    SelectAdapt0, SelectAdapt1, SelectAdapt2, SelectAdapt3, \
    SelectAdaptConst0, SelectAdaptConst1, SelectAdaptConst2, SelectAdaptConst3, \
    CompareSimpleAdaptConst")]
    rank_sel_struct: Vec<String>,
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

    let lens = args.lengths;
    let densities = args.densities;
    let reps = args.repeats;
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
