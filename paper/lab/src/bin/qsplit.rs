//! Splits the query time of PHast-R between keys placed in the first level
//! (fast path) and bumped keys (slow path), and compares with the
//! reference PHast+ on the same keys.
//!
//! Usage: qsplit [-n keys] [-q queries] [--s8-depth d]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::{PHastSig, SeedStore, SeedStoreBuild};
use sux::func::{PHastR, PHastRBuilder};
use sux::utils::ToSig;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, default_value_t = 20_000_000)]
    queries: usize,
    #[arg(short, long, default_value_t = 5)]
    repeats: usize,
    /// Configurations: <S>:<log2 L>:<depth>:<lambda>
    #[arg(short, long, value_delimiter = ',', default_value = "8:9:1:5.0")]
    variant: Vec<String>,
}

/// Times `f` on random elements of `ks`; returns the best of `repeats`
/// averages (ns per query).
#[inline(never)]
fn time<T: Copy>(ks: &[T], queries: usize, repeats: usize, f: impl Fn(T) -> usize) -> f64 {
    let n = ks.len() as u64;
    let mut best = f64::MAX;
    for _ in 0..repeats {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut acc = 0usize;
        let t = Instant::now();
        for _ in 0..queries {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            acc = acc.wrapping_add(f(ks[(((x >> 32) * n) >> 32) as usize]));
        }
        std::hint::black_box(acc);
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / queries as f64);
    }
    best
}

fn run<D: SeedStoreBuild + SeedStore>(keys: &[GxKey], b: PHastRBuilder, a: &Args, name: &str) {
    let f: PHastR<GxKey, [u64; 1], D> = b.try_build(keys, no_logging![]).unwrap();
    let (bumped, placed): (Vec<GxKey>, Vec<GxKey>) = keys
        .iter()
        .partition(|&&k| f.is_bumped(<GxKey as ToSig<[u64; 1]>>::to_sig(k, 0).ho().0));
    let all = time(keys, a.queries, a.repeats, |k| f.get(k));
    let fast = time(&placed, a.queries, a.repeats, |k| f.get(k));
    let slow = time(&bumped, a.queries / 4, a.repeats, |k| f.get(k));
    let beta = bumped.len() as f64 / keys.len() as f64;
    println!(
        "{name:28} all {all:6.2} ns  fast {fast:6.2} ns  slow {slow:6.2} ns  bumped {:.2}%  (fast + beta * (slow - fast) = {:.2})",
        100.0 * beta,
        fast + beta * (slow - fast)
    );
}

fn main() {
    let a = Args::parse();
    let keys: Vec<GxKey> = (0..a.n as u64)
        .map(|i| GxKey(i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567))
        .collect();
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        let s: u32 = p[0].parse().unwrap();
        let ll: u32 = p[1].parse().unwrap();
        let depth: u32 = p[2].parse().unwrap();
        let lam: f64 = p[3].parse().unwrap();
        let b = PHastRBuilder::default()
            .seed_bits(s)
            .log2_slice_len(ll)
            .repair_depth(depth)
            .repair_candidates(if depth == 0 { 0 } else { 16 })
            .bucket_size(lam);
        if s <= 8 {
            run::<Box<[u8]>>(&keys, b, &a, &format!("{v} u8"));
        } else {
            run::<Box<[u16]>>(&keys, b.clone(), &a, &format!("{v} u16"));
            run::<BitFieldVec<Box<[usize]>>>(&keys, b, &a, &format!("{v} bfv"));
        }
    }
}
