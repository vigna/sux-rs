//! This bench compares executing a simple operation, such as increasing two values in a BitFieldVec,
//! using either a combination of get and set or using the apply method.
//!
//! The expectation is that the apply method is faster than the get and set combination.

use criterion::{criterion_group, criterion_main, Criterion};
use sux::traits::BitFieldSlice;
use sux::traits::BitFieldSliceCore;
use sux::traits::BitFieldSliceMut;
use sux::{bits::BitFieldVec, traits::bit_field_slice::BitFieldSliceApply};

const FRACTION: u32 = 1;

fn bench_apply_u8(c: &mut Criterion) {
    for mut bit_width in 1..=u8::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("apply_u8_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u8> = BitFieldVec::new(bit_width as usize, 10_000);
                unsafe {
                    bf.apply_inplace_unchecked(|x: u8| x + 1);
                }
            });
        });
    }
}

fn bench_apply_u16(c: &mut Criterion) {
    for mut bit_width in 1..=u16::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("apply_u16_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u16> = BitFieldVec::new(bit_width as usize, 10_000);
                unsafe {
                    bf.apply_inplace_unchecked(|x: u16| x + 1);
                }
            });
        });
    }
}

fn bench_apply_u32(c: &mut Criterion) {
    for mut bit_width in 1..=u32::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("apply_u32_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u32> = BitFieldVec::new(bit_width as usize, 10_000);
                unsafe {
                    bf.apply_inplace_unchecked(|x: u32| x + 1);
                }
            });
        });
    }
}

fn bench_apply_u64(c: &mut Criterion) {
    for mut bit_width in 1..=u64::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("apply_u64_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u64> = BitFieldVec::new(bit_width as usize, 10_000);
                unsafe {
                    bf.apply_inplace_unchecked(|x: u64| x + 1);
                }
            });
        });
    }
}

fn bench_get_set_u8(c: &mut Criterion) {
    for mut bit_width in 1..=u8::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("get_set_u8_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u8> = BitFieldVec::new(bit_width as usize, 10_000);
                for i in 0..bf.len() {
                    unsafe {
                        bf.set_unchecked(i, bf.get_unchecked(i) + 1_u8);
                    }
                }
            });
        });
    }
}

fn bench_get_set_u16(c: &mut Criterion) {
    for mut bit_width in 1..=u16::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("get_set_u16_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u16> = BitFieldVec::new(bit_width as usize, 10_000);
                for i in 0..bf.len() {
                    unsafe {
                        bf.set_unchecked(i, bf.get_unchecked(i) + 1_u16);
                    }
                }
            });
        });
    }
}

fn bench_get_set_u32(c: &mut Criterion) {
    for mut bit_width in 1..=u32::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("get_set_u32_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u32> = BitFieldVec::new(bit_width as usize, 10_000);
                for i in 0..bf.len() {
                    unsafe {
                        bf.set_unchecked(i, bf.get_unchecked(i) + 1_u32);
                    }
                }
            });
        });
    }
}

fn bench_get_set_u64(c: &mut Criterion) {
    for mut bit_width in 1..=u64::BITS / FRACTION {
        bit_width *= FRACTION;
        c.bench_function(&format!("get_set_u64_{}", bit_width), |b| {
            b.iter(|| {
                let mut bf: BitFieldVec<u64> = BitFieldVec::new(bit_width as usize, 10_000);
                for i in 0..bf.len() {
                    unsafe {
                        bf.set_unchecked(i, bf.get_unchecked(i) + 1_u64);
                    }
                }
            });
        });
    }
}

criterion_group!(
    benches,
    bench_apply_u8,
    bench_apply_u16,
    bench_apply_u32,
    bench_apply_u64,
    bench_get_set_u8,
    bench_get_set_u16,
    bench_get_set_u32,
    bench_get_set_u64,
);

criterion_main!(benches);
