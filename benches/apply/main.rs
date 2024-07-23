//! This bench compares executing a simple operation, such as increasing two values in a BitFieldVec,
//! using either a combination of get and set or using the apply method.
//!
//! The expectation is that the apply method is faster than the get and set combination.

use criterion::BenchmarkId;
use criterion::{criterion_group, criterion_main, Criterion};
use sux::bits::BitFieldVec;
use sux::traits::BitFieldSlice;
use sux::traits::BitFieldSliceCore;
use sux::traits::BitFieldSliceMut;
use sux::traits::BitFieldSliceRW;
use sux::traits::Word;

const FRACTION: usize = 1;
const LEN: usize = 1_000_000;

fn bench_apply<W: Word>(c: &mut Criterion)
where
    BitFieldVec<W>: BitFieldSliceCore<W> + BitFieldSliceMut<W> + BitFieldSliceRW<W>,
{
    let mut group = c.benchmark_group(core::any::type_name::<W>());
    for mut bit_width in 1..=W::BITS / FRACTION {
        group.measurement_time(std::time::Duration::from_secs(10));
        bit_width *= FRACTION;
        let mut bf: BitFieldVec<W> = BitFieldVec::new(bit_width as usize, LEN);
        let mask = bf.mask();

        group.bench_with_input(
            BenchmarkId::new("get_set", bit_width),
            &bit_width,
            |b, _bw| {
                b.iter(|| {
                    for i in 0..bf.len() {
                        unsafe {
                            bf.set_unchecked(i, bf.get_unchecked(i).wrapping_add(W::ONE) & mask);
                        }
                    }
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("apply", bit_width),
            &bit_width,
            |b, _bw| {
                b.iter(|| unsafe {
                    bf.apply_inplace_unchecked(|x: W| x.wrapping_add(W::ONE) & mask);
                });
            },
        );
    }
}

fn bench_apply_u8(c: &mut Criterion) {
    bench_apply::<u8>(c);
}
fn bench_apply_u16(c: &mut Criterion) {
    bench_apply::<u16>(c);
}
fn bench_apply_u32(c: &mut Criterion) {
    bench_apply::<u32>(c);
}
fn bench_apply_u64(c: &mut Criterion) {
    bench_apply::<u64>(c);
}

criterion_group!(
    benches,
    bench_apply_u64,
    bench_apply_u32,
    bench_apply_u16,
    bench_apply_u8,
);

criterion_main!(benches);
