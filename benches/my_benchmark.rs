use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main, AxisScale, PlotConfiguration};
use rand::Rng;
use sux::bits::bit_vec::BitVec;
use sux::rank_sel::SelectFixed2;
use sux::traits::Select;

macro_rules! bench_select {
    ([$($inv_size:literal),+], $subinv_size:tt, $bitvecs:ident, $group:expr) => {
        $(
            bench_select!($inv_size, $subinv_size, $bitvecs, $group);
        )+
    };
    ($inv_size:literal, [$($subinv_size:literal),+], $bitvecs:ident, $group:expr) => {
        $(
            bench_select!($inv_size, $subinv_size, $bitvecs, $group);
        )+
    };
    ($log_inv_size:literal, $log_subinv_size:literal, $bitvecs:ident, $group:expr) => {{
        let mut sel_result = black_box([0]);
        let mut rng = rand::thread_rng();
        for bitvec in &($bitvecs) {
            let bits = bitvec.clone();
            let num_ones = bits.count_ones();
            let size = (bitvec.len() + 63) / 64;
            let density = (bitvec.count_ones() as f32 / bitvec.len() as f32) * 100.0;
            let sel: SelectFixed2<BitVec, Vec<u64>, $log_inv_size, $log_subinv_size> =
                black_box(SelectFixed2::new(bits));
            $group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "inv{}_sub{}_size{}_dense{}",
                    $log_inv_size, $log_subinv_size, size, density
                )),
                &$log_inv_size,
                |b, _| {
                    b.iter(|| {
                        let r = rng.gen_range(0usize..num_ones);
                        sel_result[0] = unsafe { sel.select_unchecked(r) };
                    })
                },
            );
        }
    }};
}

fn bench_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("select");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    let bitvecs = unsafe {
        [
            BitVec::from_raw_parts(vec![1usize; 1000], 64 * 1000),
            BitVec::from_raw_parts(vec![1usize; 1000000], 64 * 1000000),
            BitVec::from_raw_parts(vec![0xAAAAAAAAAAAAAAAAusize; 1000], 64 * 1000),
            BitVec::from_raw_parts(vec![0xAAAAAAAAAAAAAAAAusize; 1000000], 64 * 1000000),
            BitVec::from_raw_parts(vec![0xFFFFFFFFFFFFFFFFusize; 1000], 64 * 1000),
            BitVec::from_raw_parts(vec![0xFFFFFFFFFFFFFFFFusize; 1000000], 64 * 1000000),
        ]
    };

    bench_select!(
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        [1, 2, 3],
        bitvecs,
        group
    );

    group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
