mod bench_select;
mod utils;
use std::fmt;

use bench_select::compare_adapt_const;
use clap::{Parser, ValueEnum};
use criterion::Criterion;
use sux::bits::BitVec;
use sux::rank_sel::*;
use utils::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum RankSel {
    Rank9,
    Select9,
    // u64 RankSmall variants (rank_small![u64: 0..4])
    #[cfg(target_pointer_width = "64")]
    RankSmall64_0,
    #[cfg(target_pointer_width = "64")]
    RankSmall64_1,
    #[cfg(target_pointer_width = "64")]
    RankSmall64_2,
    #[cfg(target_pointer_width = "64")]
    RankSmall64_3,
    #[cfg(target_pointer_width = "64")]
    RankSmall64_4,
    // u64 SelectSmall variants
    #[cfg(target_pointer_width = "64")]
    SelectSmall64_0,
    #[cfg(target_pointer_width = "64")]
    SelectSmall64_1,
    #[cfg(target_pointer_width = "64")]
    SelectSmall64_2,
    #[cfg(target_pointer_width = "64")]
    SelectSmall64_3,
    #[cfg(target_pointer_width = "64")]
    SelectSmall64_4,
    // u32 RankSmall variants (rank_small![u32: 0..5])
    RankSmall32_0,
    RankSmall32_1,
    RankSmall32_2,
    RankSmall32_3,
    RankSmall32_4,
    RankSmall32_5,
    // u32 SelectSmall variants
    SelectSmall32_0,
    SelectSmall32_1,
    SelectSmall32_2,
    SelectSmall32_3,
    SelectSmall32_4,
    SelectSmall32_5,
    // SelectAdapt / SelectAdaptConst
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
            // Rank9/Select9
            Self::Rank9 => {
                bench_rank::<u64, Rank9<BitVec<Vec<u64>>>>(c, &name, lengths, densities, uniform)
            }
            Self::Select9 => bench_select::<u64, Select9<Rank9<BitVec<Vec<u64>>>>>(
                c, &name, lengths, densities, uniform,
            ),
            // u64 RankSmall (rank_small![u64: 0..4])
            #[cfg(target_pointer_width = "64")]
            Self::RankSmall64_0 => {
                bench_rank::<usize, RankSmall<64, 2, 9>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::RankSmall64_1 => {
                bench_rank::<usize, RankSmall<64, 1, 9>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::RankSmall64_2 => {
                bench_rank::<usize, RankSmall<64, 1, 10>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::RankSmall64_3 => {
                bench_rank::<usize, RankSmall<64, 1, 11>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::RankSmall64_4 => {
                bench_rank::<usize, RankSmall<64, 3, 13>>(c, &name, lengths, densities, uniform)
            }
            // u64 SelectSmall (rank_small![u64: 0..4])
            #[cfg(target_pointer_width = "64")]
            Self::SelectSmall64_0 => {
                bench_select::<usize, SelectSmall<2, 9, _>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::SelectSmall64_1 => {
                bench_select::<usize, SelectSmall<1, 9, _>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::SelectSmall64_2 => {
                bench_select::<usize, SelectSmall<1, 10, _>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::SelectSmall64_3 => {
                bench_select::<usize, SelectSmall<1, 11, _>>(c, &name, lengths, densities, uniform)
            }
            #[cfg(target_pointer_width = "64")]
            Self::SelectSmall64_4 => {
                bench_select::<usize, SelectSmall<3, 13, _>>(c, &name, lengths, densities, uniform)
            }
            // u32 RankSmall (rank_small![u32: 0..5])
            Self::RankSmall32_0 => bench_rank::<u32, RankSmall<32, 2, 8, BitVec<Vec<u32>>>>(
                c, &name, lengths, densities, uniform,
            ),
            Self::RankSmall32_1 => bench_rank::<u32, RankSmall<32, 1, 8, BitVec<Vec<u32>>>>(
                c, &name, lengths, densities, uniform,
            ),
            Self::RankSmall32_2 => bench_rank::<u32, RankSmall<32, 1, 9, BitVec<Vec<u32>>>>(
                c, &name, lengths, densities, uniform,
            ),
            Self::RankSmall32_3 => bench_rank::<u32, RankSmall<32, 1, 10, BitVec<Vec<u32>>>>(
                c, &name, lengths, densities, uniform,
            ),
            Self::RankSmall32_4 => bench_rank::<u32, RankSmall<32, 1, 11, BitVec<Vec<u32>>>>(
                c, &name, lengths, densities, uniform,
            ),
            Self::RankSmall32_5 => bench_rank::<u32, RankSmall<32, 3, 13, BitVec<Vec<u32>>>>(
                c, &name, lengths, densities, uniform,
            ),
            // u32 SelectSmall (rank_small![u32: 0..5])
            Self::SelectSmall32_0 => bench_select::<
                u32,
                SelectSmall<2, 8, RankSmall<32, 2, 8, BitVec<Vec<u32>>>>,
            >(c, &name, lengths, densities, uniform),
            Self::SelectSmall32_1 => bench_select::<
                u32,
                SelectSmall<1, 8, RankSmall<32, 1, 8, BitVec<Vec<u32>>>>,
            >(c, &name, lengths, densities, uniform),
            Self::SelectSmall32_2 => bench_select::<
                u32,
                SelectSmall<1, 9, RankSmall<32, 1, 9, BitVec<Vec<u32>>>>,
            >(c, &name, lengths, densities, uniform),
            Self::SelectSmall32_3 => bench_select::<
                u32,
                SelectSmall<1, 10, RankSmall<32, 1, 10, BitVec<Vec<u32>>>>,
            >(c, &name, lengths, densities, uniform),
            Self::SelectSmall32_4 => bench_select::<
                u32,
                SelectSmall<1, 11, RankSmall<32, 1, 11, BitVec<Vec<u32>>>>,
            >(c, &name, lengths, densities, uniform),
            Self::SelectSmall32_5 => bench_select::<
                u32,
                SelectSmall<3, 13, RankSmall<32, 3, 13, BitVec<Vec<u32>>>>,
            >(c, &name, lengths, densities, uniform),
            // SelectAdapt / SelectAdaptConst
            Self::SelectAdapt0 => {
                bench_select::<usize, SelectAdapt0>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt1 => {
                bench_select::<usize, SelectAdapt1>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt2 => {
                bench_select::<usize, SelectAdapt2>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdapt3 => {
                bench_select::<usize, SelectAdapt3>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst0 => {
                bench_select::<usize, SelectAdaptConst0>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst1 => {
                bench_select::<usize, SelectAdaptConst1>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst2 => {
                bench_select::<usize, SelectAdaptConst2>(c, &name, lengths, densities, uniform)
            }
            Self::SelectAdaptConst3 => {
                bench_select::<usize, SelectAdaptConst3>(c, &name, lengths, densities, uniform)
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
    /// Without --exact, arguments are matched as case-insensitive
    /// substrings; for example, 'ranksmall32' matches all 32-bit
    /// RankSmall variants.
    ///
    /// The integer after SelectAdapt/SelectAdaptConst is the base-2
    /// logarithm of the maximum number of usize per subinventory.
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
