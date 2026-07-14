use core::hint::black_box;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sux::bal_paren::JacobsonBalParen;
use sux::bits::BitVec;

fn fully_nested(num_pairs: usize) -> BitVec<Vec<usize>> {
    (0..num_pairs.checked_mul(2).expect("bit length overflow"))
        .map(|position| position < num_pairs)
        .collect()
}

fn bench_jacobson_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobson_build_fully_nested");
    for num_pairs in [4_096usize, 8_192, 16_384] {
        let bits = fully_nested(num_pairs);
        group.throughput(Throughput::Elements(
            u64::try_from(bits.len()).expect("benchmark length must fit u64"),
        ));
        group.bench_with_input(BenchmarkId::from_parameter(num_pairs), &bits, |b, bits| {
            b.iter(|| JacobsonBalParen::new(black_box(bits)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_jacobson_build);
criterion_main!(benches);
