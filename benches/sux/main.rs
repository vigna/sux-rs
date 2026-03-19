mod bench_select;
mod utils;
use std::fmt;

use bench_select::compare_adapt_const;
use clap::{Parser, ValueEnum};
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
    CompareAdaptConst,
}

impl fmt::Display for RankSel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl RankSel {
    fn benchmark(&self, c: &mut Criterion, lengths: &[u64], densities: &[f64], uniform: bool) {
        let name = self.to_string();
        match self {
            Self::Rank9 => bench_rank::<Rank9>(c, &name, lengths, densities, uniform),
            Self::RankSmall0 => {
                bench_rank::<RankSmall<2, 9>>(c, &name, lengths, densities, uniform)
            }
            Self::RankSmall1 => {
                bench_rank::<RankSmall<1, 9>>(c, &name, lengths, densities, uniform)
            }
            Self::RankSmall2 => {
                bench_rank::<RankSmall<1, 10>>(c, &name, lengths, densities, uniform)
            }
            Self::RankSmall3 => {
                bench_rank::<RankSmall<1, 11>>(c, &name, lengths, densities, uniform)
            }
            Self::RankSmall4 => {
                bench_rank::<RankSmall<3, 13>>(c, &name, lengths, densities, uniform)
            }
            Self::Select9 => bench_select::<Select9>(c, &name, lengths, densities, uniform),
            Self::SelectSmall0 => {
                bench_select::<SelectSmall<2, 9, _>>(c, &name, lengths, densities, uniform)
            }
            Self::SelectSmall1 => {
                bench_select::<SelectSmall<1, 9, _>>(c, &name, lengths, densities, uniform)
            }
            Self::SelectSmall2 => {
                bench_select::<SelectSmall<1, 10, _>>(c, &name, lengths, densities, uniform)
            }
            Self::SelectSmall3 => {
                bench_select::<SelectSmall<1, 11, _>>(c, &name, lengths, densities, uniform)
            }
            Self::SelectSmall4 => {
                bench_select::<SelectSmall<3, 13, _>>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt0 => {
                bench_select::<SelectAdapt0>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt1 => {
                bench_select::<SelectAdapt1>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt2 => {
                bench_select::<SelectAdapt2>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt3 => {
                bench_select::<SelectAdapt3>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst0 => {
                bench_select::<SelectAdaptConst0>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst1 => {
                bench_select::<SelectAdaptConst1>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst2 => {
                bench_select::<SelectAdaptConst2>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst3 => {
                bench_select::<SelectAdaptConst3>(c, &name, lengths, densities, uniform)
            }
            Self::CompareAdaptConst => compare_adapt_const(c, lengths, densities, uniform),
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Runs benchmarks on rank/select structures.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Cli {
    /// The lengths of the bit vectors to benchmark.​
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "1000000,4000000,16000000,64000000,256000000,1024000000")]
    lengths: Vec<u64>,
    /// The densities of the bit vectors to benchmark.​
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "0.1,0.5,0.9")]
    densities: Vec<f64>,
    /// Use a non-uniform distribution of 1s; when enabled,
    /// 1% of the bits for the provided density are in the first half and
    /// the rest in the second half.​
    #[arg(short, long, default_value = "false")]
    non_uniform: bool,
    /// Use exact matching of the structure names.​
    #[arg(long, default_value = "false")]
    exact: bool,
    /// The rank/select structures to benchmark.
    ///
    /// Rank9, Select9,
    /// RankSmall0, RankSmall1, RankSmall2, RankSmall3, RankSmall4,
    /// SelectSmall0, SelectSmall1, SelectSmall2, SelectSmall3, SelectSmall4,
    /// SelectAdapt0, SelectAdapt1, SelectAdapt2, SelectAdapt3,
    /// SelectAdaptConst0, SelectAdaptConst1, SelectAdaptConst2, SelectAdaptConst3,
    /// CompareAdaptConst
    ///
    /// Without --exact, arguments are matched as case-insensitive
    /// substrings; for example, 'rank' matches all rank structures.
    /// The integer after RankSmall/SelectSmall is the index to the
    /// rank_small! macro;
    /// the integer after SelectAdapt/SelectAdaptConst is the number
    /// of u64s per subinventory;
    /// CompareAdaptConst compares a SelectAdaptConst with default
    /// constants to a SelectAdapt with the same parameters.​
    #[arg(num_args = 1..)]
    rank_sel_struct: Vec<String>,
    /// Passed by cargo bench; ignored.​
    #[arg(long, hide = true, action = clap::ArgAction::Count)]
    bench: u8,
}

fn main() {
    let args = Cli::parse();

    // Criterion's configure_from_args cannot be used alongside a custom CLI
    // because it reads all process arguments and fails on unknown ones.
    let mut criterion = Criterion::default()
        .with_filter("")
        .with_output_color(true)
        .without_plots();

    let uniform = !args.non_uniform;

    let structures: Vec<RankSel> = if args.exact {
        args.rank_sel_struct
            .iter()
            .filter_map(|s| {
                RankSel::value_variants()
                    .iter()
                    .find(|v| v.to_string() == *s)
                    .copied()
            })
            .collect()
    } else {
        args.rank_sel_struct
            .iter()
            .flat_map(|s| {
                let lower = s.to_lowercase();
                RankSel::value_variants()
                    .iter()
                    .filter(|v| v.to_string().to_lowercase().contains(lower.trim()))
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    for rs in structures {
        rs.benchmark(&mut criterion, &args.lengths, &args.densities, uniform);
    }

    criterion.final_summary();
}
