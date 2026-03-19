use std::hint::black_box;

use crate::utils::*;
use criterion::{BenchmarkId, Criterion, Throughput};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sux::rank_sel::SelectAdapt;
use sux::rank_sel::SelectAdaptConst;
use sux::traits::AddNumBits;
use sux::traits::SelectUnchecked;

// Defaults for the comparison benchmark.
const LOG2_ONES_PER_INVENTORY: usize = 12;
const LOG2_WORDS_PER_SUBINVENTORY: usize = 3;

/// Compare [`SelectAdaptConst`] (compile-time parameters) against
/// [`SelectAdapt`] (runtime parameters) with the same inventory settings,
/// on the same bit vectors.
pub fn compare_adapt_const(c: &mut Criterion, lens: &[u64], densities: &[f64], uniform: bool) {
    // Pre-create all bitvecs so both structures use identical data.
    let mut rng = SmallRng::seed_from_u64(0);
    let mut bitvecs = Vec::new();
    for &len in lens {
        for &density in densities {
            let (nof, nos, bits) = create_bitvec(&mut rng, len, density, uniform);
            bitvecs.push((len, density, nof, nos, bits));
        }
    }

    // Group 1: SelectAdaptConst
    {
        let group_name = format!(
            "select_adapt_const_{}_{}",
            LOG2_ONES_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY,
        );
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Elements(1));
        let mut rng = SmallRng::seed_from_u64(0);
        let mut mem_costs = Vec::new();
        for (len, density, nof, nos, bits) in &bitvecs {
            let bits: AddNumBits<_> = bits.clone().into();
            let sel: SelectAdaptConst<
                AddNumBits<_>,
                Box<[usize]>,
                LOG2_ONES_PER_INVENTORY,
                LOG2_WORDS_PER_SUBINVENTORY,
            > = SelectAdaptConst::new(bits);
            mem_costs.push((*len, *density, mem_cost(&sel)));
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}_{}", len, density)),
                |b| {
                    b.iter(|| {
                        let r = random_rank(&mut rng, *nof, *nos);
                        black_box(unsafe { sel.select_unchecked(r as usize) })
                    })
                },
            );
        }
        group.finish();
        write_mem_costs(&group_name, &mem_costs);
    }

    // Group 2: SelectAdapt with identical inventory parameters
    {
        let group_name = format!(
            "select_adapt_{}_{}",
            LOG2_ONES_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY
        );
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Elements(1));
        let mut rng = SmallRng::seed_from_u64(0);
        let mut mem_costs = Vec::new();
        for (len, density, nof, nos, bits) in &bitvecs {
            let bits: AddNumBits<_> = bits.clone().into();
            let sel =
                SelectAdapt::with_inv(bits, LOG2_ONES_PER_INVENTORY, LOG2_WORDS_PER_SUBINVENTORY);
            mem_costs.push((*len, *density, mem_cost(&sel)));
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}_{}", len, density)),
                |b| {
                    b.iter(|| {
                        let r = random_rank(&mut rng, *nof, *nos);
                        black_box(unsafe { sel.select_unchecked(r as usize) })
                    })
                },
            );
        }
        group.finish();
        write_mem_costs(&group_name, &mem_costs);
    }
}
