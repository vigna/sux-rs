use crate::utils::*;
use criterion::black_box;
use criterion::BenchmarkId;
use criterion::Criterion;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::SelectAdapt;
use sux::rank_sel::SelectAdaptConst;
use sux::traits::AddNumBits;
use sux::traits::NumBits;
use sux::traits::SelectUnchecked;

const LOG2_ZEROS_PER_INVENTORY: usize = 10;
const LOG2_U64_PER_SUBINVENTORY: usize = 3;

pub fn compare_simple_adapt_const(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!(
        "select_adapt_const_{}_{}",
        LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY,
    ));

    let mut bitvecs = Vec::<BitVec>::new();
    let mut bitvec_ids = Vec::<(u64, f64)>::new();
    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in DENSITIES {
            let bitvec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            bitvecs.push(bitvec);
            bitvec_ids.push((len, density));
        }
    }

    let mut rng = SmallRng::seed_from_u64(0);
    for (bitvec, bitvec_id) in std::iter::zip(&bitvecs, &bitvec_ids) {
        let bits = bitvec.clone();
        let bits: AddNumBits<_> = bits.into();
        let num_ones = bits.num_ones();
        let sel: SelectAdaptConst<
            AddNumBits<_>,
            Box<[usize]>,
            LOG2_ZEROS_PER_INVENTORY,
            LOG2_U64_PER_SUBINVENTORY,
        > = SelectAdaptConst::new(bits);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}_{}_0", bitvec_id.0, bitvec_id.1)),
            |b| {
                b.iter(|| {
                    // use fastrange
                    let r =
                        ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128) >> 64) as usize;
                    black_box(unsafe { sel.select_unchecked(r) });
                })
            },
        );
    }
    group.finish();

    let mut rng = SmallRng::seed_from_u64(0);
    let mut group = c.benchmark_group(format!(
        "select_adapt_{}_{}",
        LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY
    ));
    for (bitvec, bitvec_id) in std::iter::zip(&bitvecs, &bitvec_ids) {
        let bits = bitvec.clone();
        let bits: AddNumBits<_> = bits.into();
        let num_ones = bits.num_ones();
        let sel = SelectAdapt::with_inv(bits, LOG2_ZEROS_PER_INVENTORY, LOG2_U64_PER_SUBINVENTORY);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}_{}_0", bitvec_id.0, bitvec_id.1)),
            |b| {
                b.iter(|| {
                    // use fastrange
                    let r =
                        ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128) >> 64) as usize;
                    black_box(unsafe { sel.select_unchecked(r) });
                })
            },
        );
    }
    group.finish();
}

macro_rules! bench_select_adapt_const {
    ([$($inv_size:literal),+], $subinv_size:tt, $bitvecs:ident, $bitvec_ids:ident, $c: expr) => {
        $(
            bench_select_adapt_const!($inv_size, $subinv_size, $bitvecs, $bitvec_ids, $c);
        )+
    };
    ($inv_size:literal, [$($subinv_size:literal),+], $bitvecs:ident, $bitvec_ids:ident, $c: expr) => {
        $(
            bench_select_adapt_const!($inv_size, $subinv_size, $bitvecs, $bitvec_ids, $c);
        )+
    };
    ($log_inv_size:literal, $log_subinv_size:literal, $bitvecs:ident, $bitvec_ids:ident, $c: expr) => {{
        let mut group = $c.benchmark_group(format!("select_adapt_const_{}_{}", $log_inv_size, $log_subinv_size));
        let mut rng = SmallRng::seed_from_u64(0);
        for (bitvec, bitvec_id) in std::iter::zip(&$bitvecs, &$bitvec_ids) {
            let bits = bitvec.clone();
            let bits: AddNumBits<_> = bits.into();
            let sel: SelectAdaptConst<AddNumBits<_>, Box<[usize]>, $log_inv_size, $log_subinv_size> =
                SelectAdaptConst::new(bits);
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}_{}_{}", bitvec_id.0, bitvec_id.1, bitvec_id.2
                )),
                &$log_inv_size,
                |b, _| {
                    b.iter(|| {
                        // use fastrange
                        let r =  fastrange_non_uniform(&mut rng, bitvec_id.3, bitvec_id.4) as usize;
                        black_box(unsafe { sel.select_unchecked(r) });
                    })
                },
            );
        }
        group.finish();
    }};
}

pub fn bench_select_adapt_const(c: &mut Criterion, uniform: bool) {
    let mut bitvecs = Vec::<BitVec>::new();
    let mut bitvec_ids = Vec::<(u64, f64, u64, u64, u64)>::new();
    let mut rng = SmallRng::seed_from_u64(0);
    for len in LENS {
        for density in DENSITIES {
            // possible repetitions
            for i in 0..REPS {
                let (num_ones_first_half, num_ones_second_half, bitvec) =
                    create_bitvec(&mut rng, len, density, uniform);
                bitvecs.push(bitvec);
                bitvec_ids.push((
                    len,
                    density,
                    i as u64,
                    num_ones_first_half,
                    num_ones_second_half,
                ));
            }
        }
    }

    bench_select_adapt_const!(
        [8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5],
        bitvecs,
        bitvec_ids,
        c
    );
}
