mod bench_select;
mod utils;
use std::fmt;

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

const MAPPING: [(
    RankSel,
    fn(&mut Criterion, &str, &[u64], &[f64], usize, bool),
); 22] = [
    (RankSel::Rank9, bench_rank::<Rank9>),
    (RankSel::RankSmall0, bench_rank::<RankSmall<2, 9>>),
    (RankSel::RankSmall1, bench_rank::<RankSmall<1, 9>>),
    (RankSel::RankSmall2, bench_rank::<RankSmall<1, 10>>),
    (RankSel::RankSmall3, bench_rank::<RankSmall<1, 11>>),
    (RankSel::RankSmall4, bench_rank::<RankSmall<3, 13>>),
    (RankSel::Select9, bench_select::<Select9>),
    (RankSel::SelectSmall0, bench_select::<SelectSmall<2, 9>>),
    (RankSel::SelectSmall1, bench_select::<SelectSmall<1, 9>>),
    (RankSel::SelectSmall2, bench_select::<SelectSmall<1, 10>>),
    (RankSel::SelectSmall3, bench_select::<SelectSmall<1, 11>>),
    (RankSel::SelectSmall4, bench_select::<SelectSmall<3, 13>>),
    (RankSel::SelectAdapt0, bench_select::<SelectAdapt0<_>>),
    (RankSel::SelectAdapt1, bench_select::<SelectAdapt1<_>>),
    (RankSel::SelectAdapt2, bench_select::<SelectAdapt2<_>>),
    (RankSel::SelectAdapt3, bench_select::<SelectAdapt3<_>>),
    (RankSel::SimpleSelect0, bench_select::<SimpleSelect0<_>>),
    (RankSel::SimpleSelect1, bench_select::<SimpleSelect1<_>>),
    (RankSel::SimpleSelect2, bench_select::<SimpleSelect2<_>>),
    (RankSel::SimpleSelect3, bench_select::<SimpleSelect3<_>>),
    (RankSel::AdaptConst, bench_select_adapt_const),
    (RankSel::CompareSimpleAdaptConst, compare_simple_adapt_const),
];

impl fmt::Display for RankSel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl RankSel {
    fn from_str_exact(s: &str) -> Option<Self> {
        MAPPING
            .iter()
            .find_map(|(k, _v)| if k.to_string() == s { Some(*k) } else { None })
    }

    fn from_str(s: &str) -> Vec<Self> {
        MAPPING
            .iter()
            .filter_map(|(k, _v)| {
                if k.to_string()
                    .to_lowercase()
                    .contains(s.to_lowercase().trim())
                {
                    Some(*k)
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
        for (k, v) in MAPPING.iter() {
            if k == self {
                v(c, &k.to_string(), lens, densities, reps, uniform);
            }
        }
    }
}

/// Command line arguments for the benchmarking suite.
#[derive(Parser, Debug)]
struct Cli {
    /// The lengths of the bitvectors to benchmark.
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "1000000,4000000,16000000,64000000,256000000,1024000000")]
    lens: Vec<u64>,
    /// The densities of the bitvectors to benchmark.
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "0.1,0.5,0.9")]
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
