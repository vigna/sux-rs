//! Splits the query time of PHast-R between keys placed in the first level
//! (fast path) and bumped keys (slow path), and compares with the
//! reference PHast+ on the same keys.
//!
//! Configurations are `<S>:<log2 L>:<depth>:<lambda>` for PHast-R (with
//! S > 8, seeds are stored as `u16`, in a `BitFieldVec`, and in a
//! `BitFieldVec` with unaligned reads), or `ref:<plus|w3>:<S>:<lambda>` for
//! the reference implementation, whose bumped keys are those whose bucket of
//! the first level has seed 0.
//!
//! Usage: qsplit [-n keys] [-q queries] [-v configurations]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use ph::BuildSeededHasher;
use ph::phast::{
    DefaultCompressedArray, Function2, Generic, GenericCore, SeedChooser, ShiftOnly,
    ShiftOnlyWrapped,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::{Bits8, BitsFast, SeedSize};
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::{SeedStore, SeedStoreBuild};
use sux::func::{PHastR, PHastRBuilder};
use sux::traits::TryIntoUnaligned;

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
    let f: PHastR<GxKey, D> = b.try_build(keys, no_logging![]).unwrap();
    let (bumped, placed): (Vec<GxKey>, Vec<GxKey>) = keys.iter().partition(|&&k| f.is_bumped(k));
    report(keys, &bumped, &placed, a, name, |k| f.get(k));
}

fn run_u(keys: &[GxKey], b: PHastRBuilder, a: &Args, name: &str) {
    let f: PHastR<GxKey, BitFieldVec<Box<[usize]>>> = b.try_build(keys, no_logging![]).unwrap();
    let (bumped, placed): (Vec<GxKey>, Vec<GxKey>) = keys.iter().partition(|&&k| f.is_bumped(k));
    let f = f.try_into_unaligned().unwrap();
    report(keys, &bumped, &placed, a, name, |k| f.get(k));
}

fn run_ref<SS: SeedSize, SC: SeedChooser>(
    keys: &[GxKey],
    ss: SS,
    sc: SC,
    lam: f64,
    a: &Args,
    name: &str,
) {
    // SAFETY: GxKey is a transparent wrapper around u64, and ph hashes u64
    // keys as sux hashes GxKey
    let keys: &[u64] = unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    let params = Generic::new(ss, (lam * 100.0).round() as u16);
    let f: Function2<GenericCore, SS, SC, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_p_hash_sc(keys, &params, BuildGxHash, sc);
    let conf = *f.level0_conf();
    let (bumped, placed): (Vec<u64>, Vec<u64>) = keys
        .iter()
        .partition(|&&k| f.level0_seed(conf.bucket_for(BuildGxHash.hash_one(k, 0))) == 0);
    report(keys, &bumped, &placed, a, name, |k| f.get(&k));
}

#[inline(always)]
fn report<T: Copy>(
    keys: &[T],
    bumped: &[T],
    placed: &[T],
    a: &Args,
    name: &str,
    get: impl Fn(T) -> usize + Copy,
) {
    let all = time(keys, a.queries, a.repeats, get);
    let fast = time(placed, a.queries, a.repeats, get);
    let slow = time(bumped, a.queries / 4, a.repeats, get);
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
        if p[0] == "ref" {
            let sbits: u8 = p[2].parse().unwrap();
            let lam: f64 = p[3].parse().unwrap();
            match (p[1], sbits) {
                ("plus", 8) => run_ref(&keys, Bits8, ShiftOnly, lam, &a, v),
                ("plus", s) => run_ref(&keys, BitsFast(s), ShiftOnly, lam, &a, v),
                ("w3", 8) => run_ref(&keys, Bits8, ShiftOnlyWrapped::<3>, lam, &a, v),
                ("w3", s) => run_ref(&keys, BitsFast(s), ShiftOnlyWrapped::<3>, lam, &a, v),
                (c, _) => panic!("unknown chooser {c}"),
            }
            continue;
        }
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
            run::<BitFieldVec<Box<[usize]>>>(&keys, b.clone(), &a, &format!("{v} bfv"));
            run_u(&keys, b, &a, &format!("{v} bfvu"));
        }
    }
}
