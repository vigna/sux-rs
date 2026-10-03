//! Compares PHast-R (sux) with the reference PHast/PHast+ implementation (ph)
//! on 64-bit integer keys, using the same hash function for both.
//!
//! Configurations are given as comma-separated specs:
//!
//! - `ref:<chooser>:<S>:<lambda>` for the reference implementation, where
//!   `<chooser>` is `plus` (PHast+), `w1`/`w2`/`w3` (PHast+ with wrapping and
//!   multiplier 1/2/3), or `phast` (regular PHast);
//!
//! - `r:<S>:<log2 L>:<depth>:<lambda>[:<log2 R>[:<storage>]]` for PHast-R,
//!   where `<depth>` is the maximum repair depth (0 disables repair) and
//!   `<storage>` is `u8`, `u16`, or `bfv` (default: `u8` if S <= 8, `bfv`
//!   otherwise).
//!
//! Human-readable results go to stdout; machine-readable lines
//! `CSV,<name>,<n>,<bits/key>,<build ns/key>,<query ns>` go to stderr.

use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{FlatType, MemSize, SizeFlags};
use ph::phast::{
    DefaultCompressedArray, Function2, Generic, GenericCore, SeedChooser, SeedOnly, ShiftOnly,
    ShiftOnlyWrapped,
};
use ph::seeds::{Bits8, BitsFast, SeedSize};
use ph::{BuildSeededHasher, GetSize};
use std::hash::Hasher;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder};

/// XXH3-64 with seed on the 8 bytes of a u64, exactly as sux's
/// `ToSig<[u64; 1]>` for `u64`, so that both implementations pay the same
/// hashing cost.
#[derive(Default, Clone, Copy)]
struct BuildX;

struct HX {
    v: u64,
    seed: u64,
}

impl Hasher for HX {
    fn write(&mut self, _: &[u8]) {
        unimplemented!("only u64 keys are supported")
    }
    #[inline(always)]
    fn write_u64(&mut self, x: u64) {
        self.v = x;
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(&self.v.to_ne_bytes(), self.seed)
    }
}

impl BuildSeededHasher for BuildX {
    type Hasher = HX;
    #[inline(always)]
    fn build_hasher(&self, seed: u64) -> HX {
        HX { v: 0, seed }
    }
}

#[derive(Parser)]
#[command(about = "Compares PHast-R with the reference PHast/PHast+ implementation")]
struct Args {
    /// The number of keys.
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    /// The number of random queries per repetition.
    #[arg(short, long, default_value_t = 10_000_000)]
    queries: usize,
    /// Comma-separated configurations (see the module documentation).
    #[arg(short, long, value_delimiter = ',')]
    variant: Vec<String>,
    /// The number of query repetitions (the best is reported).
    #[arg(short, long, default_value_t = 3)]
    repeats: usize,
    /// The number of threads for the reference construction (PHast-R uses
    /// the rayon pool; set RAYON_NUM_THREADS).
    #[arg(short, long, default_value_t = 1)]
    threads: usize,
}

fn query_bench(keys: &[u64], queries: usize, repeats: usize, f: impl Fn(u64) -> usize) -> f64 {
    let n = keys.len() as u64;
    let mut best = f64::MAX;
    for _ in 0..repeats {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut acc = 0usize;
        let t = Instant::now();
        for _ in 0..queries {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let k = keys[(((x >> 32) * n) >> 32) as usize];
            acc = acc.wrapping_add(f(k));
        }
        std::hint::black_box(acc);
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / queries as f64);
    }
    best
}

fn report(name: &str, n: usize, bits: f64, build: f64, q: f64, extra: &str) {
    println!("{name:44} {bits:.4} bits/key  build {build:7.1} ns/key  query {q:6.1} ns{extra}");
    eprintln!("CSV,{name},{n},{bits:.4},{build:.1},{q:.1}");
}

fn ref_run<SC: SeedChooser, SS: SeedSize>(
    keys: &[u64],
    ss: SS,
    lam: f64,
    sc: SC,
    a: &Args,
    name: &str,
) {
    let t = Instant::now();
    let params = Generic::new(ss, (lam * 100.0).round() as u16);
    let f: Function2<GenericCore, SS, SC, DefaultCompressedArray, BuildX> = if a.threads > 1 {
        Function2::with_slice_p_threads_hash_sc(keys, &params, a.threads, BuildX, sc)
    } else {
        Function2::with_slice_p_hash_sc(keys, &params, BuildX, sc)
    };
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.size_bytes() as f64 * 8.0 / keys.len() as f64;
    let q = query_bench(keys, a.queries, a.repeats, |k| f.get(&k));
    report(name, keys.len(), bits, build, q, "");
}

fn ref_dispatch<SS: SeedSize>(keys: &[u64], ss: SS, chooser: &str, lam: f64, a: &Args, name: &str) {
    match chooser {
        "plus" => ref_run(keys, ss, lam, ShiftOnly, a, name),
        "w1" => ref_run(keys, ss, lam, ShiftOnlyWrapped::<1>, a, name),
        "w2" => ref_run(keys, ss, lam, ShiftOnlyWrapped::<2>, a, name),
        "w3" => ref_run(keys, ss, lam, ShiftOnlyWrapped::<3>, a, name),
        "phast" => ref_run(keys, ss, lam, SeedOnly, a, name),
        _ => panic!("unknown chooser {chooser}"),
    }
}

fn sux_run<D: SeedStoreBuild + MemSize + FlatType>(
    keys: &[u64],
    b: PHastRBuilder,
    a: &Args,
    name: &str,
) {
    let t = Instant::now();
    let f: PHastR<u64, [u64; 1], D> = b.try_build(keys, no_logging![]).unwrap();
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64;
    // Verify that the function is a bijection
    let mut seen = vec![false; keys.len()];
    for k in keys {
        let v = f.get(k);
        assert!(!seen[v], "duplicate output {v}");
        seen[v] = true;
    }
    let q = query_bench(keys, a.queries, a.repeats, |k| f.get(k));
    report(
        name,
        keys.len(),
        bits,
        build,
        q,
        &format!("  levels {}", f.num_levels()),
    );
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    println!("n = {}", a.n);
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        match p[0] {
            "ref" => {
                let sbits: u8 = p[2].parse().unwrap();
                let lam: f64 = p[3].parse().unwrap();
                let name = format!("ref {} S={} l={}", p[1], sbits, lam);
                if sbits == 8 {
                    ref_dispatch(&keys, Bits8, p[1], lam, &a, &name)
                } else {
                    ref_dispatch(&keys, BitsFast(sbits), p[1], lam, &a, &name)
                }
            }
            "r" => {
                let sbits: u32 = p[1].parse().unwrap();
                let ll: u32 = p[2].parse().unwrap();
                let depth: u32 = p[3].parse().unwrap();
                let lam: f64 = p[4].parse().unwrap();
                let lr: u32 = p.get(5).map(|x| x.parse().unwrap()).unwrap_or(2);
                let storage = p
                    .get(6)
                    .copied()
                    .unwrap_or(if sbits <= 8 { "u8" } else { "bfv" });
                let b = PHastRBuilder::default()
                    .seed_bits(sbits)
                    .log2_slice_len(ll)
                    .repair_depth(depth)
                    .repair_candidates(if depth == 0 { 0 } else { 16 })
                    .bucket_size(lam)
                    .log2_patterns(lr);
                let name = format!(
                    "PHast-R S={sbits} L={} R={} d={depth} l={lam} {storage}",
                    1 << ll,
                    1 << lr
                );
                match storage {
                    "u8" => sux_run::<Box<[u8]>>(&keys, b, &a, &name),
                    "u16" => sux_run::<Box<[u16]>>(&keys, b, &a, &name),
                    "bfv" => sux_run::<BitFieldVec<Box<[usize]>>>(&keys, b, &a, &name),
                    _ => panic!("unknown storage {storage}"),
                }
            }
            _ => panic!("unknown variant {v}"),
        }
    }
}
