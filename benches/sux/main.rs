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
    SelectAdaptConst0,
    SelectAdaptConst1,
    SelectAdaptConst2,
    SelectAdaptConst3,
    SimpleSelect0,
    SimpleSelect1,
    SimpleSelect2,
    SimpleSelect3,
    AdaptConst,
    CompareAdaptConst,
}

const MAPPING: [(
    RankSel,
    fn(&mut Criterion, &str, &[u64], &[f64], usize, bool),
); 21] = [
    (RankSel::Rank9, bench_rank::<Rank9>),
    (RankSel::RankSmall0, bench_rank::<RankSmall<2, 9>>),
    (RankSel::RankSmall1, bench_rank::<RankSmall<1, 9>>),
    (RankSel::RankSmall2, bench_rank::<RankSmall<1, 10>>),
    (RankSel::RankSmall3, bench_rank::<RankSmall<1, 11>>),
    (RankSel::RankSmall4, bench_rank::<RankSmall<3, 13>>),
    (RankSel::Select9, bench_select::<Select9>),
    (RankSel::SelectSmall0, bench_select::<SelectSmall<2, 9, _>>),
    (RankSel::SelectSmall1, bench_select::<SelectSmall<1, 9, _>>),
    (RankSel::SelectSmall2, bench_select::<SelectSmall<1, 10, _>>),
    (RankSel::SelectSmall3, bench_select::<SelectSmall<1, 11, _>>),
    (RankSel::SelectSmall4, bench_select::<SelectSmall<3, 13, _>>),
    (RankSel::SelectAdapt0, bench_select::<SelectAdapt0<_>>),
    (RankSel::SelectAdapt1, bench_select::<SelectAdapt1<_>>),
    (RankSel::SelectAdapt2, bench_select::<SelectAdapt2<_>>),
    (RankSel::SelectAdapt3, bench_select::<SelectAdapt3<_>>),
    (
        RankSel::SelectAdaptConst0,
        bench_select::<SelectAdaptConst0<_>>,
    ),
    (
        RankSel::SelectAdaptConst1,
        bench_select::<SelectAdaptConst1<_>>,
    ),
    (
        RankSel::SelectAdaptConst2,
        bench_select::<SelectAdaptConst2<_>>,
    ),
    (
        RankSel::SelectAdaptConst3,
        bench_select::<SelectAdaptConst3<_>>,
    ),
    (RankSel::CompareAdaptConst, compare_adapt_const),
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
        lengths: &[u64],
        densities: &[f64],
        repeats: usize,
        uniform: bool,
    ) {
        for (k, v) in MAPPING.iter() {
            if k == self {
                v(c, &k.to_string(), lengths, densities, repeats, uniform);
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
    /// Whether to use a non-uniform distributions of 1s in the bitvectors.
    /// When using a non-uniform distribution, 1% of the bits for the provided density
    /// are located in the first half of the bitvector, and the rest in the second half.
    #[arg(short, long, default_value = "false", verbatim_doc_comment)]
    non_uniform: bool,
    /// Flag for exact matching of the structure names to benchmark.
    #[arg(long, default_value = "false")]
    exact: bool,
    /// The rank/select structures to benchmark. Possible values are:
    ///
    /// Rank9, Select9,
    /// RankSmall0, RankSmall1, RankSmall2, RankSmall3, RankSmall4,
    /// SelectSmall0, SelectSmall1, SelectSmall2, SelectSmall3, SelectSmall4,
    /// SelectAdapt0, SelectAdapt1, SelectAdapt2, SelectAdapt3,
    /// SelectAdaptConst0, SelectAdaptConst1, SelectAdaptConst2, SelectAdaptConst3,
    /// CompareAdaptConst
    ///
    /// - Without --exact, the arguments are matched as case-insensitive
    ///   substrings. For example, 'rank' will match all rank structures.
    /// - The integer after RankSmall/Select small is the index to the
    ///   rank_small! macro.
    /// - The integer after SelectAdapt/SelectAdaptConst is the number of u64's
    ///   per subinventory.
    /// - CompareAdaptConst compares a SelectAdaptConst with default constants
    ///   to a SelectAdapt with the same parameters.
    #[arg(num_args = 1.., verbatim_doc_comment)]
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
    let mut criterion = Criterion()
        .with_filter("")
        .with_output_color(true)
        .without_plots();

    let lengths = args.lengths;
    let densities = args.densities;
    let repeats = args.repeats;
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
        rank_sel.benchmark(&mut criterion, &lengths, &densities, repeats, uniform);
    }

    criterion.final_summary();
}
