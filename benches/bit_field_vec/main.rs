/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use clap::Parser;
use criterion::{BenchmarkId, Criterion, Throughput};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::bits::{BitFieldVec, BitFieldVecU};
use sux::prelude::*;
use sux::traits::TryIntoUnaligned;
use value_traits::slices::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks BitFieldVec operations.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Cli {
    /// The bit widths to benchmark.​
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "1,3,8,12,32,51")]
    widths: Vec<usize>,
    /// The base-2 logarithms of the vector lengths to benchmark.​
    #[arg(long, short, num_args = 1.., value_delimiter = ',', default_value = "10,20,30")]
    log2_sizes: Vec<usize>,
    /// Use unaligned reads.​
    #[arg(long, short)]
    unaligned: bool,
    /// Passed by cargo bench; ignored.​
    #[arg(long, hide = true, action = clap::ArgAction::Count)]
    bench: u8,
}

// ── Generic read benchmarks (work with both BitFieldVec and BitFieldVecU) ──

fn bench_get_unchecked_random<D: SliceByValue<Value = usize>>(
    c: &mut Criterion,
    prefix: &str,
    widths: &[usize],
    log2_sizes: &[usize],
    make: &impl Fn(usize, usize) -> D,
) {
    let mut group = c.benchmark_group(format!("{prefix}/get_unchecked_random"));
    group.throughput(Throughput::Elements(1));
    let mut rng = SmallRng::seed_from_u64(0);
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let mask = len - 1;
            let a = make(width, len);
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| unsafe {
                        black_box(a.get_value_unchecked(rng.random::<u64>() as usize & mask))
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_get_random<D: SliceByValue<Value = usize>>(
    c: &mut Criterion,
    prefix: &str,
    widths: &[usize],
    log2_sizes: &[usize],
    make: &impl Fn(usize, usize) -> D,
) {
    let mut group = c.benchmark_group(format!("{prefix}/get_random"));
    group.throughput(Throughput::Elements(1));
    let mut rng = SmallRng::seed_from_u64(0);
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let mask = len - 1;
            let a = make(width, len);
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| b.iter(|| black_box(a.index_value(rng.random::<u64>() as usize & mask))),
            );
        }
    }
    group.finish();
}

fn bench_get_unchecked_seq<D: SliceByValue<Value = usize>>(
    c: &mut Criterion,
    prefix: &str,
    widths: &[usize],
    log2_sizes: &[usize],
    make: &impl Fn(usize, usize) -> D,
) {
    let mut group = c.benchmark_group(format!("{prefix}/get_unchecked_seq"));
    group.throughput(Throughput::Elements(1));
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let a = make(width, len);
            let mut i = 0usize;
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let v = unsafe { a.get_value_unchecked(i) };
                        i += 1;
                        if i >= len {
                            i = 0;
                        }
                        black_box(v)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_get_seq<D: SliceByValue<Value = usize>>(
    c: &mut Criterion,
    prefix: &str,
    widths: &[usize],
    log2_sizes: &[usize],
    make: &impl Fn(usize, usize) -> D,
) {
    let mut group = c.benchmark_group(format!("{prefix}/get_seq"));
    group.throughput(Throughput::Elements(1));
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let a = make(width, len);
            let mut i = 0usize;
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let v = a.index_value(i);
                        i += 1;
                        if i >= len {
                            i = 0;
                        }
                        black_box(v)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_iter_next_unchecked<D: SliceByValue<Value = usize>>(
    c: &mut Criterion,
    prefix: &str,
    widths: &[usize],
    log2_sizes: &[usize],
    make: &impl Fn(usize, usize) -> D,
) where
    for<'a> &'a D: IntoUncheckedIterator<Item = usize>,
{
    let mut group = c.benchmark_group(format!("{prefix}/iter_next_unchecked"));
    group.throughput(Throughput::Elements(1));
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let a = make(width, len);
            let mut iter = (&a).into_unchecked_iter();
            let mut i = 0usize;
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let v = unsafe { iter.next_unchecked() };
                        i += 1;
                        if i >= len {
                            iter = (&a).into_unchecked_iter();
                            i = 0;
                        }
                        black_box(v)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_iter_back_unchecked<D: SliceByValue<Value = usize>>(
    c: &mut Criterion,
    prefix: &str,
    widths: &[usize],
    log2_sizes: &[usize],
    make: &impl Fn(usize, usize) -> D,
) where
    for<'a> &'a D: IntoUncheckedBackIterator<Item = usize>,
{
    let mut group = c.benchmark_group(format!("{prefix}/iter_back_unchecked"));
    group.throughput(Throughput::Elements(1));
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let a = make(width, len);
            let mut iter = (&a).into_unchecked_iter_back();
            let mut i = 0usize;
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let v = unsafe { iter.next_unchecked() };
                        i += 1;
                        if i >= len {
                            iter = (&a).into_unchecked_iter_back();
                            i = 0;
                        }
                        black_box(v)
                    })
                },
            );
        }
    }
    group.finish();
}

// ── BitFieldVec-only benchmarks (write and checked iteration) ───

fn bench_set_unchecked_random(c: &mut Criterion, widths: &[usize], log2_sizes: &[usize]) {
    let mut group = c.benchmark_group("BitFieldVec/set_unchecked_random");
    group.throughput(Throughput::Elements(1));
    let mut rng = SmallRng::seed_from_u64(0);
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let mask = len - 1;
            let mut a = BitFieldVec::<Vec<usize>>::new(width, len);
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let x = rng.random::<u64>() as usize & mask;
                        unsafe { a.set_value_unchecked(x, 1) };
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_set_random(c: &mut Criterion, widths: &[usize], log2_sizes: &[usize]) {
    let mut group = c.benchmark_group("BitFieldVec/set_random");
    group.throughput(Throughput::Elements(1));
    let mut rng = SmallRng::seed_from_u64(0);
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let mask = len - 1;
            let mut a = BitFieldVec::<Vec<usize>>::new(width, len);
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let x = rng.random::<u64>() as usize & mask;
                        a.set_value(x, 1);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_iter_next_bfv(c: &mut Criterion, widths: &[usize], log2_sizes: &[usize]) {
    let mut group = c.benchmark_group("BitFieldVec/iter_next");
    group.throughput(Throughput::Elements(1));
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let a: BitFieldVec<Box<[usize]>> = BitFieldVec::<Vec<usize>>::new(width, len).into();
            let mut iter = (&a).into_iter();
            let mut i = 0usize;
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let v = iter.next();
                        i += 1;
                        if i >= len {
                            iter = (&a).into_iter();
                            i = 0;
                        }
                        black_box(v)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_iter_back_bfv(c: &mut Criterion, widths: &[usize], log2_sizes: &[usize]) {
    let mut group = c.benchmark_group("BitFieldVec/iter_back");
    group.throughput(Throughput::Elements(1));
    for &width in widths {
        for &log2_size in log2_sizes {
            let len = 1usize << log2_size;
            let a: BitFieldVec<Box<[usize]>> = BitFieldVec::<Vec<usize>>::new(width, len).into();
            let mut iter = (&a).into_iter();
            let mut i = 0usize;
            group.bench_function(
                BenchmarkId::new(format!("w{width}"), format!("2^{log2_size}")),
                |b| {
                    b.iter(|| {
                        let v = iter.next_back();
                        i += 1;
                        if i >= len {
                            iter = (&a).into_iter();
                            i = 0;
                        }
                        black_box(v)
                    })
                },
            );
        }
    }
    group.finish();
}

// ── Main ────────────────────────────────────────────────────────

fn main() {
    let args = Cli::parse();

    let mut criterion = Criterion::default()
        .with_filter("")
        .with_output_color(true)
        .without_plots();

    if args.unaligned {
        let prefix = "BitFieldVecU";
        let make = |width, len| -> BitFieldVecU<Box<[usize]>> {
            let bfv: BitFieldVec<Box<[usize]>> = BitFieldVec::<Vec<usize>>::new(width, len).into();
            bfv.try_into_unaligned()
                .expect("unaligned conversion failed for this bit width")
        };
        bench_get_unchecked_random(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_get_random(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_get_unchecked_seq(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_get_seq(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_iter_next_unchecked(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_iter_back_unchecked(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
    } else {
        let prefix = "BitFieldVec";
        let make = |width, len| -> BitFieldVec<Box<[usize]>> {
            BitFieldVec::<Vec<usize>>::new(width, len).into()
        };
        bench_get_unchecked_random(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_get_random(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_set_unchecked_random(&mut criterion, &args.widths, &args.log2_sizes);
        bench_set_random(&mut criterion, &args.widths, &args.log2_sizes);
        bench_get_unchecked_seq(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_get_seq(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_iter_next_unchecked(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_iter_next_bfv(&mut criterion, &args.widths, &args.log2_sizes);
        bench_iter_back_unchecked(
            &mut criterion,
            prefix,
            &args.widths,
            &args.log2_sizes,
            &make,
        );
        bench_iter_back_bfv(&mut criterion, &args.widths, &args.log2_sizes);
    }

    criterion.final_summary();
}
