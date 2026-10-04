//! Compares PHast-R (sux) with the reference PHast/PHast+ implementation (ph)
//! on 64-bit integer keys, using the same hash function for both.
//!
//! Configurations are given as comma-separated specs:
//!
//! - `ref:<chooser>:<S>:<lambda>` for the reference implementation, where
//!   `<chooser>` is `plus` (PHast+), `w1`/`w2`/`w3` (PHast+ with wrapping and
//!   multiplier 1/2/3), or `phast` (regular PHast);
//!
//! - `r:<S>:<log2 L>:<depth>:<lambda>[:<log2 R>[:<storage>[:<s64|s128>]]]` for PHast-R,
//!   where `<depth>` is the maximum repair depth (0 disables repair) and
//!   `<storage>` is `u8`, `u16`, or `bfv` (default: `u8` if S <= 8, `bfv`
//!   otherwise).
//!
//! Keys are hashed with GxHash (as in the experiments of the PHast paper) or,
//! with `--hash xxh3`, with XXH3-64; both implementations use the same hash.
//!
//! Human-readable results go to stdout; machine-readable lines
//! `CSV,<name>,<n>,<bits/key>,<build ns/key>,<query ns>` go to stderr.

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use mem_dbg::{FlatType, MemSize, SizeFlags};
use ph::phast::{
    DefaultCompressedArray, Function2, Generic, GenericCore, SeedChooser, SeedOnly, ShiftOnly,
    ShiftOnlyWrapped,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::{Bits8, BitsFast, SeedSize};
use ph::{BuildSeededHasher, GetSize};
use std::hash::Hasher;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder};
use sux::utils::ToSig;

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
    /// The hash function: gx (GxHash) or xxh3 (XXH3-64).
    #[arg(long, default_value = "gx")]
    hash: String,
    /// If positive, build all structures first, and then time queries in
    /// this number of rounds, each running a batch of --queries queries on
    /// every structure in turn, reporting the median per structure; this
    /// makes comparisons immune to frequency drift.
    #[arg(long, default_value_t = 0)]
    interleave: usize,
    /// Generates a different key set (for averaging over key sets).
    #[arg(long, default_value_t = 0)]
    key_seed: u64,
}

/// A built structure: name, statistics, and a function running a batch of
/// queries (number of queries, round) and returning ns per query.
struct Entry<'a> {
    name: String,
    n: usize,
    bits: f64,
    build: f64,
    extra: String,
    bench: Box<dyn Fn(usize, u64) -> f64 + 'a>,
}

/// Runs a batch of random queries and returns ns per query.
#[inline(always)]
fn query_batch<T: Copy>(keys: &[T], queries: usize, round: u64, f: impl Fn(T) -> usize) -> f64 {
    let n = keys.len() as u64;
    let mut x = 0x9e3779b97f4a7c15u64 ^ round.wrapping_mul(0xd6e8_feb8_6659_fd93);
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
    t.elapsed().as_secs_f64() * 1e9 / queries as f64
}

fn report(name: &str, n: usize, bits: f64, build: f64, q: f64, extra: &str) {
    println!("{name:44} {bits:.4} bits/key  build {build:7.1} ns/key  query {q:6.1} ns{extra}");
    eprintln!("CSV,{name},{n},{bits:.4},{build:.1},{q:.1}");
}

fn ref_run<'a, SC: SeedChooser + 'a, SS: SeedSize + 'a>(
    keys: &'a [u64],
    ss: SS,
    lam: f64,
    sc: SC,
    a: &Args,
    name: &str,
) -> Entry<'a> {
    match a.hash.as_str() {
        "gx" => ref_run_h(keys, ss, lam, sc, a, name, BuildGxHash),
        "xxh3" => ref_run_h(keys, ss, lam, sc, a, name, BuildX),
        h => panic!("unknown hash {h}"),
    }
}

fn ref_run_h<'a, SC: SeedChooser + 'a, SS: SeedSize + 'a, H: BuildSeededHasher + Sync + 'a>(
    keys: &'a [u64],
    ss: SS,
    lam: f64,
    sc: SC,
    a: &Args,
    name: &str,
    hasher: H,
) -> Entry<'a> {
    let t = Instant::now();
    let params = Generic::new(ss, (lam * 100.0).round() as u16);
    let f: Function2<GenericCore, SS, SC, DefaultCompressedArray, H> = if a.threads > 1 {
        Function2::with_slice_p_threads_hash_sc(keys, &params, a.threads, hasher, sc)
    } else {
        Function2::with_slice_p_hash_sc(keys, &params, hasher, sc)
    };
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.size_bytes() as f64 * 8.0 / keys.len() as f64;
    Entry {
        name: name.to_string(),
        n: keys.len(),
        bits,
        build,
        extra: String::new(),
        bench: Box::new(move |q, r| query_batch(keys, q, r, |k| f.get(&k))),
    }
}

fn ref_dispatch<'a, SS: SeedSize + 'a>(
    keys: &'a [u64],
    ss: SS,
    chooser: &str,
    lam: f64,
    a: &Args,
    name: &str,
) -> Entry<'a> {
    match chooser {
        "plus" => ref_run(keys, ss, lam, ShiftOnly, a, name),
        "w1" => ref_run(keys, ss, lam, ShiftOnlyWrapped::<1>, a, name),
        "w2" => ref_run(keys, ss, lam, ShiftOnlyWrapped::<2>, a, name),
        "w3" => ref_run(keys, ss, lam, ShiftOnlyWrapped::<3>, a, name),
        "phast" => ref_run(keys, ss, lam, SeedOnly, a, name),
        _ => panic!("unknown chooser {chooser}"),
    }
}

fn sux_run<'a, D: SeedStoreBuild + MemSize + FlatType + 'a>(
    keys: &'a [u64],
    b: PHastRBuilder,
    a: &Args,
    name: &str,
    sig128: bool,
) -> Entry<'a> {
    // SAFETY: GxKey is a transparent wrapper around u64
    let gkeys: &'a [GxKey] =
        unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    match (a.hash.as_str(), sig128) {
        ("gx", false) => sux_run_k::<GxKey, [u64; 1], D>(gkeys, b, a, name),
        ("gx", true) => sux_run_k::<GxKey, [u64; 2], D>(gkeys, b, a, name),
        ("xxh3", false) => sux_run_k::<u64, [u64; 1], D>(keys, b, a, name),
        ("xxh3", true) => sux_run_k::<u64, [u64; 2], D>(keys, b, a, name),
        (h, _) => panic!("unknown hash {h}"),
    }
}

fn sux_run_k<
    'a,
    K: ToSig<SG> + Copy + Sync + 'a,
    SG: sux::func::phast_r::PHastSig + Send + Sync + 'a,
    D: SeedStoreBuild + MemSize + FlatType + 'a,
>(
    keys: &'a [K],
    b: PHastRBuilder,
    _a: &Args,
    name: &str,
) -> Entry<'a> {
    let t = Instant::now();
    let f: PHastR<K, SG, D> = b.try_build(keys, no_logging![]).unwrap();
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64;
    // Verify that the function is a bijection
    let mut seen = vec![false; keys.len()];
    for k in keys {
        let v = f.get(*k);
        assert!(!seen[v], "duplicate output {v}");
        seen[v] = true;
    }
    let bumped = keys
        .iter()
        .filter(|&&k| f.is_bumped(K::to_sig(k, 0).ho().0))
        .count();
    let extra = format!(
        "  levels {}  bumped {:.2}%",
        f.num_levels(),
        100.0 * bumped as f64 / keys.len() as f64
    );
    Entry {
        name: name.to_string(),
        n: keys.len(),
        bits,
        build,
        extra,
        bench: Box::new(move |q, r| query_batch(keys, q, r, |k| f.get(k))),
    }
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| {
            (i + a.key_seed.wrapping_mul(1 << 40)).wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567
        })
        .collect();
    // Both implementations must compute the same hashes
    for &k in keys.iter().take(1000) {
        assert_eq!(
            <GxKey as ToSig<[u64; 1]>>::to_sig(GxKey(k), 0)[0],
            BuildGxHash.hash_one(k, 0)
        );
        assert_eq!(
            <u64 as ToSig<[u64; 1]>>::to_sig(k, 0)[0],
            BuildX.hash_one(k, 0)
        );
    }
    println!("n = {} hash = {}", a.n, a.hash);
    let mut entries = vec![];
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        let e = match p[0] {
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
                let sig128 = match p.get(7).copied().unwrap_or("s64") {
                    "s64" => false,
                    "s128" => true,
                    s => panic!("unknown signature width {s}"),
                };
                let name = format!(
                    "PHast-R S={sbits} L={} R={} d={depth} l={lam} {storage}{}",
                    1 << ll,
                    1 << lr,
                    if sig128 { " s128" } else { "" }
                );
                match storage {
                    "u8" => sux_run::<Box<[u8]>>(&keys, b, &a, &name, sig128),
                    "u16" => sux_run::<Box<[u16]>>(&keys, b, &a, &name, sig128),
                    "bfv" => sux_run::<BitFieldVec<Box<[usize]>>>(&keys, b, &a, &name, sig128),
                    _ => panic!("unknown storage {storage}"),
                }
            }
            _ => panic!("unknown variant {v}"),
        };
        if a.interleave == 0 {
            // Time right away, keeping the best of the repetitions
            let q = (0..a.repeats as u64)
                .map(|r| (e.bench)(a.queries, r))
                .fold(f64::MAX, f64::min);
            report(&e.name, e.n, e.bits, e.build, q, &e.extra);
        } else {
            entries.push(e);
        }
    }
    if a.interleave > 0 {
        let mut times = vec![vec![]; entries.len()];
        for r in 0..a.interleave as u64 {
            for (e, t) in entries.iter().zip(times.iter_mut()) {
                t.push((e.bench)(a.queries, r));
            }
        }
        for (e, t) in entries.iter().zip(times.iter_mut()) {
            t.sort_by(f64::total_cmp);
            let q = t[t.len() / 2];
            report(
                &e.name,
                e.n,
                e.bits,
                e.build,
                q,
                &format!("{}  [min {:.1} max {:.1}]", e.extra, t[0], t[t.len() - 1]),
            );
        }
    }
}
