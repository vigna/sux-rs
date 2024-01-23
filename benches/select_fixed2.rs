use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::SelectFixed2;
use sux::traits::Select;

macro_rules! bench_select {
    ([$($inv_size:literal),+], $subinv_size:tt, $bitvecs:ident, $bitvec_ids:ident, $bench_group:expr) => {
        $(
            bench_select!($inv_size, $subinv_size, $bitvecs, $bitvec_ids, $bench_group);
        )+
    };
    ($inv_size:literal, [$($subinv_size:literal),+], $bitvecs:ident, $bitvec_ids:ident, $bench_group:expr) => {
        $(
            bench_select!($inv_size, $subinv_size, $bitvecs, $bitvec_ids, $bench_group);
        )+
    };
    ($log_inv_size:literal, $log_subinv_size:literal, $bitvecs:ident, $bitvec_ids:ident, $bench_group:expr) => {{
        let mut rng = SmallRng::seed_from_u64(0);
        for (bitvec, bitvec_id) in std::iter::zip(&$bitvecs, &$bitvec_ids) {
            let bits = bitvec.clone();
            let num_ones = bits.count_ones();
            let sel: SelectFixed2<BitVec, Vec<u64>, $log_inv_size, $log_subinv_size> =
                SelectFixed2::new(bits);
            $bench_group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}_{}_{}_{}_{}",
                    $log_inv_size, $log_subinv_size, bitvec_id.0, bitvec_id.1, bitvec_id.2
                )),
                &$log_inv_size,
                |b, _| {
                    b.iter(|| {
                        // use fastrange
                        let r =  ((rng.gen::<u64>() as u128).wrapping_mul(num_ones as u128) >> 64) as usize;
                        black_box(unsafe { sel.select_unchecked(r) });
                    })
                },
            );
        }
    }};
}

fn bench_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("select");

    //let lens = [1u64 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28, 1 << 30];
    let lens = [
        1u64 << 20,
        1 << 21,
        1 << 22,
        1 << 23,
        1 << 24,
        1 << 25,
        1 << 26,
        1 << 27,
        1 << 28,
        1 << 29,
        1 << 30,
    ];

    let mut bitvecs = Vec::<BitVec>::new();
    let mut bitvec_ids = Vec::<(u64, f64, u64)>::new();
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in [0.25, 0.5, 0.75] {
            // possible repetitions
            for i in 0..1 {
                let bitvec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                bitvecs.push(bitvec);
                bitvec_ids.push((len, density, i));
            }
        }
    }

    bench_select!([8, 9, 10, 11, 12], [1, 2, 3], bitvecs, bitvec_ids, group);
    //bench_select!([10, 11], [1, 2], bitvecs, bitvec_ids, group);

    group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
