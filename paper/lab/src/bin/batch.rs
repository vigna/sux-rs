//! Query throughput of PHast-R with single queries and with batches that
//! prefetch the seeds of the first level (`get_batch`), on random keys.
//!
//! Usage: batch [-n keys] [-v <S>:<log2 L>:<lambda>[:<log2 R>],...]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::PHastR;
use sux::func::phast_r::{LevelParams, SeedStore};
use sux::traits::TryIntoUnaligned;
use value_traits::slices::SliceByValue;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, default_value_t = 20_000_000)]
    queries: usize,
    #[arg(short, long, default_value_t = 5)]
    rounds: usize,
    #[arg(short, long, value_delimiter = ',', default_value = "8:10:4.5")]
    variant: Vec<String>,
}

#[inline(always)]
fn idx(x: &mut u64, n: u64) -> usize {
    *x = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*x >> 32) * n) >> 32) as usize
}

fn single<D: SeedStore, R: SliceByValue<Value = usize>>(
    f: &PHastR<GxKey, D, Box<[LevelParams]>, R>,
    keys: &[GxKey],
    q: usize,
) -> f64 {
    let n = keys.len() as u64;
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut acc = 0usize;
    let t = Instant::now();
    for _ in 0..q {
        acc = acc.wrapping_add(f.get(keys[idx(&mut x, n)]));
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / q as f64
}

fn batch<const B: usize, D: SeedStore, R: SliceByValue<Value = usize>>(
    f: &PHastR<GxKey, D, Box<[LevelParams]>, R>,
    keys: &[GxKey],
    q: usize,
) -> f64 {
    let n = keys.len() as u64;
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut acc = 0usize;
    let t = Instant::now();
    for _ in 0..q / B {
        let ks: [&GxKey; B] = std::array::from_fn(|_| &keys[idx(&mut x, n)]);
        for v in f.get_batch(ks) {
            acc = acc.wrapping_add(v);
        }
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / (q / B * B) as f64
}

fn run<D: SeedStore, R: SliceByValue<Value = usize>>(
    v: &str,
    f: &PHastR<GxKey, D, Box<[LevelParams]>, R>,
    keys: &[GxKey],
    a: &Args,
) {
    // Interleaved rounds, medians
    let mut t = vec![vec![]; 6];
    for _ in 0..a.rounds {
        t[0].push(single(f, keys, a.queries));
        t[1].push(batch::<4, D, R>(f, keys, a.queries));
        t[2].push(batch::<8, D, R>(f, keys, a.queries));
        t[3].push(batch::<16, D, R>(f, keys, a.queries));
        t[4].push(batch::<32, D, R>(f, keys, a.queries));
        t[5].push(batch::<64, D, R>(f, keys, a.queries));
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    print!("{v:20} n={}  single {:.1} ns", keys.len(), med(&mut t[0]));
    for (i, b) in [4, 8, 16, 32, 64].iter().enumerate() {
        print!("  B{b} {:.1}", med(&mut t[i + 1]));
    }
    println!();
}

fn main() {
    let a = Args::parse();
    let keys: Vec<GxKey> = (0..a.n as u64)
        .map(|i| GxKey(i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567))
        .collect();
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        let (b, sbits, _) = lab::parse_config(&p);
        if sbits <= 8 {
            let f: PHastR<GxKey> = b.try_build(&keys, no_logging![]).unwrap();
            run(v, &f, &keys, &a);
        } else {
            let f: PHastR<GxKey, BitFieldVec<Box<[usize]>>> =
                b.try_build(&keys, no_logging![]).unwrap();
            let f = f.try_into_unaligned().unwrap();
            run(v, &f, &keys, &a);
        }
    }
}
