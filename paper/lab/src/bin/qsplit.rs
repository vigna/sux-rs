//! Splits the query time of PHast-R between keys placed in the first level
//! (fast path) and bumped keys (slow path), and compares with the
//! reference PHast+ on the same keys.
//!
//! Configurations are `<S>:<log2 L>:<lambda>[:<log2 R>]` for PHast-R (with
//! S > 8, seeds are stored as `u16`, in a `BitFieldVec`, and in a
//! `BitFieldVec` with unaligned reads), or `ref:<plus|w3|w3p|phast>:<S>:<lambda>` for
//! the reference implementation, whose bumped keys are those whose bucket of
//! the first level has seed 0.
//!
//! Usage: qsplit [-n keys] [-q queries] [-v configurations] [--order random|sequential]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use ph::BuildSeededHasher;
use ph::phast::{
    Conf, DefaultCompressedArray, Function2, GenericCore, ProdOfValues, SeedChooserConf, SeedOnly,
    ShiftOnly, ShiftOnlyProdWrapped, ShiftOnlyWrapped,
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
    #[arg(short, long, default_value_t = 5_000_000)]
    queries: usize,
    /// Number of interleaved rounds (medians are reported)
    #[arg(short, long, default_value_t = 9)]
    repeats: usize,
    /// Measure just this set of keys (0: all, 1: first level, 2: bumped),
    /// e.g., for use with performance counters
    #[arg(long)]
    set: Option<usize>,
    /// Configurations: <S>:<log2 L>:<lambda>[:<log2 R>]
    #[arg(short, long, value_delimiter = ',', default_value = "8:9:1:5.0")]
    variant: Vec<String>,
    /// The order of queries: random or sequential (as in cmp)
    #[arg(long, default_value = "random")]
    order: String,
}

/// Times `f` on elements of `ks` (random elements, or consecutive elements
/// starting from position `round * queries` modulo the length of `ks`, as in
/// cmp); returns the average time (ns per query).
#[inline(never)]
fn time<T: Copy>(
    ks: &[T],
    queries: usize,
    round: u64,
    sequential: bool,
    f: impl Fn(T) -> usize,
) -> f64 {
    let n = ks.len();
    let mut acc = 0usize;
    let t;
    if sequential {
        let mut i = ((round as u128 * queries as u128) % n as u128) as usize;
        t = Instant::now();
        for _ in 0..queries {
            acc = acc.wrapping_add(f(ks[i]));
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
            acc = acc.wrapping_add(f(ks[(((x >> 32) * n) >> 32) as usize]));
        }
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / queries as f64
}

/// A structure to be measured: its name, the fraction of bumped keys, and
/// a function timing queries (set, number of queries, round) on all keys
/// (set 0), on the keys placed in the first level (1) or on bumped keys (2).
type Entry<'a> = (String, f64, Box<dyn Fn(usize, usize, u64) -> f64 + 'a>);

fn entry<'a, T: Copy + 'a>(
    keys: &'a [T],
    bumped: Vec<T>,
    placed: Vec<T>,
    name: &str,
    sequential: bool,
    get: impl Fn(T) -> usize + Copy + 'a,
) -> Entry<'a> {
    let beta = bumped.len() as f64 / keys.len() as f64;
    (
        name.to_string(),
        beta,
        Box::new(move |set, queries, round| match set {
            0 => time(keys, queries, round, sequential, get),
            1 => time(&placed, queries, round, sequential, get),
            _ => time(&bumped, queries / 4, round, sequential, get),
        }),
    )
}

fn run<'a, D: SeedStoreBuild + SeedStore + 'static>(
    keys: &'a [GxKey],
    b: PHastRBuilder,
    name: &str,
    sequential: bool,
) -> Entry<'a> {
    let f: PHastR<GxKey, D> =
        PHastR::try_par_new_with_builder(keys, b.clone(), no_logging![]).unwrap();
    let (bumped, placed): (Vec<GxKey>, Vec<GxKey>) = keys.iter().partition(|&&k| f.is_bumped(k));
    let f: &'static PHastR<GxKey, D> = Box::leak(Box::new(f));
    entry(keys, bumped, placed, name, sequential, move |k| f.get(k))
}

fn run_u<'a>(keys: &'a [GxKey], b: PHastRBuilder, name: &str, sequential: bool) -> Entry<'a> {
    let f: PHastR<GxKey, BitFieldVec<Box<[usize]>>> =
        PHastR::try_par_new_with_builder(keys, b.clone(), no_logging![]).unwrap();
    let (bumped, placed): (Vec<GxKey>, Vec<GxKey>) = keys.iter().partition(|&&k| f.is_bumped(k));
    let f = Box::leak(Box::new(f.try_into_unaligned().unwrap()));
    let f = &*f;
    entry(keys, bumped, placed, name, sequential, move |k| f.get(k))
}

fn run_ref<'a, SS: SeedSize + 'static, SC: SeedChooserConf + 'static>(
    keys: &'a [GxKey],
    ss: SS,
    sc: SC,
    lam: f64,
    name: &str,
    sequential: bool,
) -> Entry<'a> {
    // SAFETY: GxKey is a transparent wrapper around u64, and ph hashes u64
    // keys as sux hashes GxKey
    let keys: &[u64] = unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    let conf = Conf::generic_with_hash(ss, (lam * 100.0).round() as u32, BuildGxHash);
    let f: Function2<GenericCore, SS, SC::Core, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_conf_sc(keys, conf, sc);
    let conf = *f.level0_conf();
    let (bumped, placed): (Vec<u64>, Vec<u64>) = keys
        .iter()
        .partition(|&&k| f.level0_seed(conf.bucket_for(BuildGxHash.hash_one(k, 0))) == 0);
    let f = &*Box::leak(Box::new(f));
    entry(keys, bumped, placed, name, sequential, move |k| f.get(&k))
}

fn main() {
    let a = Args::parse();
    assert!(
        a.order == "random" || a.order == "sequential",
        "unknown order {}",
        a.order
    );
    let sq = a.order == "sequential";
    let keys: Vec<GxKey> = (0..a.n as u64)
        .map(|i| GxKey(i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567))
        .collect();
    let mut entries: Vec<Entry> = vec![];
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        if p[0] == "ref" {
            let sbits: u8 = p[2].parse().unwrap();
            let lam: f64 = p[3].parse().unwrap();
            entries.push(match (p[1], sbits) {
                ("plus", 8) => run_ref(&keys, Bits8, ShiftOnly, lam, v, sq),
                ("plus", s) => run_ref(&keys, BitsFast(s), ShiftOnly, lam, v, sq),
                ("phast", 8) => run_ref(&keys, Bits8, SeedOnly(ProdOfValues), lam, v, sq),
                ("phast", s) => run_ref(&keys, BitsFast(s), SeedOnly(ProdOfValues), lam, v, sq),
                ("w3", 8) => run_ref(&keys, Bits8, ShiftOnlyWrapped::<3>, lam, v, sq),
                ("w3", s) => run_ref(&keys, BitsFast(s), ShiftOnlyWrapped::<3>, lam, v, sq),
                ("w3p", 8) => run_ref(&keys, Bits8, ShiftOnlyProdWrapped::<3>, lam, v, sq),
                ("w3p", s) => run_ref(&keys, BitsFast(s), ShiftOnlyProdWrapped::<3>, lam, v, sq),
                (c, _) => panic!("unknown chooser {c}"),
            });
            continue;
        }
        let (b, s, _) = lab::parse_config(&p);
        if s <= 8 {
            entries.push(run::<Box<[u8]>>(&keys, b, &format!("{v} u8"), sq));
        } else {
            entries.push(run::<Box<[u16]>>(&keys, b.clone(), &format!("{v} u16"), sq));
            entries.push(run_u(&keys, b, &format!("{v} bfvu"), sq));
        }
    }
    // Rounds are interleaved, so that all structures are measured in the
    // same conditions; we report medians
    let mut times = vec![[const { Vec::new() }; 3]; entries.len()];
    for r in 0..a.repeats as u64 {
        for (times, (_, _, bench)) in times.iter_mut().zip(&entries) {
            for (set, times) in times.iter_mut().enumerate() {
                times.push(if a.set.is_none_or(|s| s == set) {
                    bench(set, a.queries, r)
                } else {
                    0.0
                });
            }
        }
    }
    for (times, (name, beta, _)) in times.iter_mut().zip(&entries) {
        let [all, fast, slow] = std::array::from_fn(|set| {
            times[set].sort_by(f64::total_cmp);
            times[set][times[set].len() / 2]
        });
        println!(
            "{name:28} all {all:6.2} ns  fast {fast:6.2} ns  slow {slow:6.2} ns  bumped {:.2}%  (fast + beta * (slow - fast) = {:.2})",
            100.0 * beta,
            fast + beta * (slow - fast)
        );
    }
}
