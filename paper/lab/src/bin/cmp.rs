//! Compares PHast-R (sux) with the reference PHast/PHast+ implementation (ph)
//! on 64-bit integer keys, using the same hash function for both.
//!
//! Configurations are given as comma-separated specs:
//!
//! - `ref:<chooser>:<S>:<lambda>` for the reference implementation, where
//!   `<chooser>` is `plus` (PHast+), `w1`/`w2`/`w3` (PHast+ with wrapping and
//!   multiplier 1/2/3, choosing the seed minimizing the sum of the values of
//!   the keys), `w1p`/`w2p`/`w3p` (the same, minimizing their product),
//!   `phast` (regular PHast, minimizing the product, the default of `ph`),
//!   `phastsum` (regular PHast, minimizing the sum), or `w<d>p` for
//!   d = 4, 5, 6, 7, 8, 9, 11 (PHast+ with wrapping and multiplier d, for which
//!   `ph` uses the priority weights of multiplier 3);
//!
//! - `r:<S>:<log2 L>:<lambda>[:<log2 R>[:<storage>]]` for PHast-R (bits per
//!   seed, base-2 logarithm of the slice length, expected bucket size, base-2
//!   logarithm of the number of layouts, 2 by default), where `<storage>`
//!   is `u8`, `u16`, `bfv`, or `bfvu` (a `BitFieldVec` with unaligned reads;
//!   default: `u8` if S <= 8, `bfv` otherwise);
//!
//! - `ptr:<variant>[:<lambda>]` for PtrHash, where `<variant>` is `default`
//!   (`DefaultPtrHash` with `PtrHashParams::default()`) or `compact`
//!   (`CompactPtrHash` with `PtrHashParams::default_compact()`), optionally
//!   with a different expected bucket size.
//!
//! Keys are hashed with GxHash (as in the experiments of the PHast paper) or,
//! with `--hash xxh3`, with XXH3-64; all implementations use the same hash
//! (PtrHash supports only GxHash).
//!
//! Human-readable results go to stdout; machine-readable lines
//! `CSV,<name>,<n>,<bits/key>,<build ns/key>,<query ns>,<query min>,<query
//! max>,<build min>,<build max>,<bumped %>` go to stderr. The construction
//! time is the median of `--builds` constructions; the query time is the
//! median of the interleaved rounds (or the best of the repetitions without
//! `--interleave`), and the minimum and maximum are over the same rounds; the
//! last field is the percentage of keys bumped from the first level.

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use mem_dbg::{FlatType, MemSize, SizeFlags};
use ph::phast::{
    Conf, DefaultCompressedArray, Function2, GenericCore, ProdOfValues, SeedChooserConf, SeedOnly,
    ShiftOnly, ShiftOnlyProdWrapped, ShiftOnlyWrapped, SumOfValues,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::{Bits8, BitsFast, SeedSize};
use ph::{BuildSeededHasher, GetSize};
use std::hash::Hasher;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder};
use sux::traits::TryIntoUnaligned;
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
    /// The number of constructions of each structure (the median time is
    /// reported, and the last structure is queried).
    #[arg(long, default_value_t = 1)]
    builds: usize,
    /// The order of queries: random (keys chosen uniformly at random) or
    /// sequential (consecutive keys of the set, see `query_batch`).
    #[arg(long, default_value = "random")]
    order: String,
    /// Queries PHast-R with the generic code, rather than with the code with
    /// constant shifts used with four layouts (so that all numbers of
    /// layouts are queried by the same code).
    #[arg(long)]
    generic_queries: bool,
}

/// A built structure: name, statistics, and a function running a batch of
/// queries (number of queries, round) and returning ns per query.
struct Entry<'a> {
    name: String,
    n: usize,
    bits: f64,
    /// Median, minimum and maximum construction time (ns/key).
    build: [f64; 3],
    /// Percentage of keys bumped from the first level.
    bumped: f64,
    extra: String,
    bench: Box<dyn Fn(usize, u64) -> f64 + 'a>,
}

/// Builds a structure the given number of times, dropping each structure
/// before building the next one, and returns the last structure and the
/// median, minimum and maximum construction time in ns/key.
fn timed_builds<T>(builds: usize, n: usize, mut build: impl FnMut() -> T) -> (T, [f64; 3]) {
    let mut t = vec![];
    let mut last = None;
    for _ in 0..builds.max(1) {
        drop(last.take());
        let start = Instant::now();
        last = Some(build());
        t.push(start.elapsed().as_secs_f64() * 1e9 / n as f64);
    }
    t.sort_by(f64::total_cmp);
    (last.unwrap(), [t[t.len() / 2], t[0], t[t.len() - 1]])
}

/// Runs a batch of queries and returns ns per query.
///
/// Random queries choose keys uniformly at random; sequential queries read
/// the keys of the set in order, starting from position `round * queries`
/// (modulo the number of keys). Since keys are random 64-bit values that both
/// implementations hash, the accesses to the structures are random in both
/// cases; but sequential queries read the keys in streaming fashion, so on
/// large key sets the time does not include a cache miss on the array of
/// keys, which is the same for all structures and dilutes their differences.
#[inline(always)]
fn query_batch<T: Copy>(
    keys: &[T],
    queries: usize,
    round: u64,
    sequential: bool,
    f: impl Fn(T) -> usize,
) -> f64 {
    let n = keys.len();
    let mut acc = 0usize;
    let t;
    if sequential {
        let mut i = ((round as u128 * queries as u128) % n as u128) as usize;
        t = Instant::now();
        for _ in 0..queries {
            acc = acc.wrapping_add(f(keys[i]));
            i += 1;
            if i == n {
                i = 0;
            }
        }
    } else {
        let n = n as u64;
        let mut x = 0x9e3779b97f4a7c15u64 ^ round.wrapping_mul(0xd6e8_feb8_6659_fd93);
        t = Instant::now();
        for _ in 0..queries {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let k = keys[(((x >> 32) * n) >> 32) as usize];
            acc = acc.wrapping_add(f(k));
        }
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / queries as f64
}

/// Reports a structure, given its query time and the minimum and maximum
/// query time.
fn report(e: &Entry, q: [f64; 3]) {
    let (name, n, bits, [build, bmin, bmax], bumped) = (&e.name, e.n, e.bits, e.build, e.bumped);
    let [q, qmin, qmax] = q;
    println!(
        "{name:44} {bits:.4} bits/key  build {build:7.1} ns/key [min {bmin:.1} max {bmax:.1}]  query {q:6.1} ns [min {qmin:.1} max {qmax:.1}]  bumped {bumped:.2}%{}",
        e.extra
    );
    eprintln!(
        "CSV,{name},{n},{bits:.4},{build:.1},{q:.1},{qmin:.1},{qmax:.1},{bmin:.1},{bmax:.1},{bumped:.2}"
    );
}

fn ref_run<'a, SC: SeedChooserConf + Clone + 'a, SS: SeedSize + 'a>(
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

fn ref_run_h<
    'a,
    SC: SeedChooserConf + Clone + 'a,
    SS: SeedSize + 'a,
    H: BuildSeededHasher + Sync + Copy + 'a,
>(
    keys: &'a [u64],
    ss: SS,
    lam: f64,
    sc: SC,
    a: &Args,
    name: &str,
    hasher: H,
) -> Entry<'a> {
    let bs100 = (lam * 100.0).round() as u32;
    let (f, build) = timed_builds(
        a.builds,
        keys.len(),
        || -> Function2<GenericCore, SS, SC::Core, DefaultCompressedArray, H> {
            let conf = Conf::generic_with_hash(ss, bs100, hasher);
            if a.threads > 1 {
                Function2::with_slice_conf_threads_sc(keys, conf, a.threads, sc.clone())
            } else {
                Function2::with_slice_conf_sc(keys, conf, sc.clone())
            }
        },
    );
    let bits = f.size_bytes() as f64 * 8.0 / keys.len() as f64;
    // Bumped keys are those whose bucket of the first level has seed 0
    let conf = *f.level0_conf();
    let bumped = keys
        .iter()
        .filter(|&&k| f.level0_seed(conf.bucket_for(hasher.hash_one(k, 0))) == 0)
        .count();
    let sequential = a.order == "sequential";
    Entry {
        name: name.to_string(),
        n: keys.len(),
        bits,
        build,
        bumped: 100.0 * bumped as f64 / keys.len() as f64,
        extra: String::new(),
        bench: Box::new(move |q, r| query_batch(keys, q, r, sequential, |k| f.get(&k))),
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
        "phast" => ref_run(keys, ss, lam, SeedOnly(ProdOfValues), a, name),
        "phastsum" => ref_run(keys, ss, lam, SeedOnly(SumOfValues), a, name),
        "w1p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<1>, a, name),
        "w2p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<2>, a, name),
        "w3p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<3>, a, name),
        "w4p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<4>, a, name),
        "w5p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<5>, a, name),
        "w6p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<6>, a, name),
        "w7p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<7>, a, name),
        "w8p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<8>, a, name),
        "w9p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<9>, a, name),
        "w11p" => ref_run(keys, ss, lam, ShiftOnlyProdWrapped::<11>, a, name),
        _ => panic!("unknown chooser {chooser}"),
    }
}

/// GxHash with seed on a u64, exactly as [`GxKey`] and `ph::BuildGxHash`, so
/// that PtrHash pays the same hashing cost as the other structures.
#[derive(Clone, Copy)]
struct PtrGx;

impl ptr_hash::hash::KeyHasher<u64> for PtrGx {
    type H = u64;
    #[inline(always)]
    fn hash(x: &u64, seed: u64) -> u64 {
        let mut h = gxhash::GxHasher::with_seed(seed as i64);
        h.write_u64(*x);
        h.finish()
    }
}

/// Builds PtrHash (see the module documentation for the variants).
fn ptr_run<'a>(keys: &'a [u64], variant: &str, lam: Option<f64>, a: &Args, name: &str) -> Entry<'a> {
    assert_eq!(a.hash, "gx", "PtrHash is supported only with GxHash");
    macro_rules! go {
        ($ty:ty, $params:expr) => {{
            let mut params = $params;
            if let Some(lam) = lam {
                params.lambda = lam;
            }
            let (f, build) = timed_builds(a.builds, keys.len(), || <$ty>::new(keys, params));
            // Pilots and remapping (the Elias-Fano remapping does not
            // implement MemSize; the other fields take a few bytes)
            let (pilots, remap) = f.bits_per_element();
            let bits = pilots + remap;
            // PtrHash does not bump; we report the keys remapped from the
            // slots beyond n, which need an access to the remapping
            let n = keys.len();
            let remapped = keys.iter().filter(|k| f.index_no_remap(k) >= n).count();
            let sequential = a.order == "sequential";
            Entry {
                name: name.to_string(),
                n,
                bits,
                build,
                bumped: 0.0,
                extra: format!("  remapped {:.2}%", 100.0 * remapped as f64 / n as f64),
                bench: Box::new(move |q, r| query_batch(keys, q, r, sequential, |k| f.index(&k))),
            }
        }};
    }
    match variant {
        "default" => go!(
            ptr_hash::DefaultPtrHash<PtrGx, u64>,
            ptr_hash::PtrHashParams::default()
        ),
        "compact" => go!(
            ptr_hash::CompactPtrHash<PtrGx, u64>,
            ptr_hash::PtrHashParams::default_compact()
        ),
        _ => panic!("unknown PtrHash variant {variant}"),
    }
}

fn sux_run<'a, D: SeedStoreBuild + MemSize + FlatType + 'a>(
    keys: &'a [u64],
    b: PHastRBuilder,
    a: &Args,
    name: &str,
) -> Entry<'a> {
    // SAFETY: GxKey is a transparent wrapper around u64
    let gkeys: &'a [GxKey] =
        unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    match a.hash.as_str() {
        "gx" => sux_run_k::<GxKey, D>(gkeys, b, a, name),
        "xxh3" => sux_run_k::<u64, D>(keys, b, a, name),
        h => panic!("unknown hash {h}"),
    }
}

/// Like [`sux_run`], but with seeds in a [`BitFieldVec`] converted to
/// unaligned reads.
fn sux_run_u<'a>(keys: &'a [u64], b: PHastRBuilder, a: &Args, name: &str) -> Entry<'a> {
    // SAFETY: GxKey is a transparent wrapper around u64
    let gkeys: &'a [GxKey] =
        unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    match a.hash.as_str() {
        "gx" => sux_run_k_u::<GxKey>(gkeys, b, a, name),
        "xxh3" => sux_run_k_u::<u64>(keys, b, a, name),
        h => panic!("unknown hash {h}"),
    }
}

/// Verifies that a function is a bijection, counts the bumped keys, and
/// returns the entry for the function.
macro_rules! sux_entry {
    ($keys:expr, $f:expr, $build:expr, $a:expr, $name:expr) => {{
        let (keys, f, build, a, name) = ($keys, $f, $build, $a, $name);
        let bits = f.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64;
        let mut seen = vec![false; keys.len()];
        for k in keys {
            let v = f.get(*k);
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
        drop(seen);
        let bumped = keys.iter().filter(|&&k| f.is_bumped(k)).count();
        let extra = format!("  levels {}", f.num_levels());
        let sequential = a.order == "sequential";
        Entry {
            name: name.to_string(),
            n: keys.len(),
            bits,
            build,
            bumped: 100.0 * bumped as f64 / keys.len() as f64,
            extra,
            bench: Box::new(move |q, r| query_batch(keys, q, r, sequential, |k| f.get(k))),
        }
    }};
}

fn sux_run_k<
    'a,
    K: ToSig<[u64; 1]> + Copy + Sync + 'a,
    D: SeedStoreBuild + MemSize + FlatType + 'a,
>(
    keys: &'a [K],
    b: PHastRBuilder,
    a: &Args,
    name: &str,
) -> Entry<'a> {
    let (f, build) = timed_builds(a.builds, keys.len(), || -> PHastR<K, D> {
        PHastR::try_par_new_with_builder(keys, b.clone(), no_logging![]).unwrap()
    });
    let f = if a.generic_queries { f.generic_queries() } else { f };
    sux_entry!(keys, f, build, a, name)
}

fn sux_run_k_u<'a, K: ToSig<[u64; 1]> + Copy + Sync + 'a>(
    keys: &'a [K],
    b: PHastRBuilder,
    a: &Args,
    name: &str,
) -> Entry<'a> {
    let (f, build) = timed_builds(a.builds, keys.len(), || {
        let f: PHastR<K, BitFieldVec<Box<[usize]>>> =
            PHastR::try_par_new_with_builder(keys, b.clone(), no_logging![]).unwrap();
        f.try_into_unaligned().unwrap()
    });
    let f = if a.generic_queries { f.generic_queries() } else { f };
    sux_entry!(keys, f, build, a, name)
}

fn main() {
    let a = Args::parse();
    assert!(
        a.order == "random" || a.order == "sequential",
        "unknown order {}",
        a.order
    );
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
                let (b, sbits, desc) = lab::parse_config(&p[1..p.len().min(5)]);
                let storage = p
                    .get(5)
                    .copied()
                    .unwrap_or(if sbits <= 8 { "u8" } else { "bfv" });
                let name = format!("{desc} {storage}");
                match storage {
                    "u8" => sux_run::<Box<[u8]>>(&keys, b, &a, &name),
                    "u16" => sux_run::<Box<[u16]>>(&keys, b, &a, &name),
                    "bfv" => sux_run::<BitFieldVec<Box<[usize]>>>(&keys, b, &a, &name),
                    "bfvu" => sux_run_u(&keys, b, &a, &name),
                    _ => panic!("unknown storage {storage}"),
                }
            }
            "ptr" => {
                let lam: Option<f64> = p.get(2).map(|x| x.parse().unwrap());
                let name = match lam {
                    Some(lam) => format!("PtrHash {} l={lam}", p[1]),
                    None => format!("PtrHash {}", p[1]),
                };
                ptr_run(&keys, p[1], lam, &a, &name)
            }
            _ => panic!("unknown variant {v}"),
        };
        if a.interleave == 0 {
            // Time right away, keeping the best of the repetitions
            let mut t: Vec<f64> = (0..a.repeats.max(1) as u64)
                .map(|r| (e.bench)(a.queries, r))
                .collect();
            t.sort_by(f64::total_cmp);
            report(&e, [t[0], t[0], t[t.len() - 1]]);
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
            report(e, [t[t.len() / 2], t[0], t[t.len() - 1]]);
        }
    }
}
