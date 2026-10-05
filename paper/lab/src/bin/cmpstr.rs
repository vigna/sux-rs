//! Compares PHast-R with the reference PHast/PHast+ implementation (ph) on
//! random strings of 10 to 50 bytes (the standard MPHF benchmark workload),
//! with the same hash function (GxHash on the bytes of the string, exactly
//! as `std::hash::Hash` feeds a `str` to a hasher) for both.
//!
//! Configurations are as in `cmp` (`ref:<chooser>:<S>:<lambda>` and
//! `r:<S>:<log2 L>:<lambda>[:<log2 R>]`, byte seeds); query times are medians of
//! interleaved rounds.

use clap::Parser;
use dsi_progress_logger::no_logging;
use mem_dbg::{MemSize, SizeFlags};
use ph::GetSize;
use ph::phast::{
    DefaultCompressedArray, Function2, Generic, GenericCore, SeedChooser, ShiftOnly,
    ShiftOnlyWrapped,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::Bits8;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use sux::func::{PHastR, PHastRBuilder};
use sux::utils::ToSig;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, default_value_t = 2_000_000)]
    queries: usize,
    #[arg(long, default_value_t = 11)]
    interleave: usize,
    #[arg(short, long, value_delimiter = ',')]
    variant: Vec<String>,
    #[arg(long, default_value_t = 0)]
    key_seed: u64,
}

/// A string key hashed with GxHash exactly as `ph` hashes it.
#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
struct GxStr(String);

impl Hash for GxStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl ToSig<[u64; 1]> for GxStr {
    #[inline(always)]
    fn to_sig(key: impl std::borrow::Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut h = gxhash::GxHasher::with_seed(seed as i64);
        key.borrow().0.hash(&mut h);
        [h.finish()]
    }
}

struct Entry<'a> {
    name: String,
    bits: f64,
    build: f64,
    bench: Box<dyn Fn(usize, u64) -> f64 + 'a>,
}

#[inline(always)]
fn batch<'a>(keys: &'a [GxStr], q: usize, round: u64, f: impl Fn(&'a GxStr) -> usize) -> f64 {
    let n = keys.len() as u64;
    let mut x = 0x9e3779b97f4a7c15u64 ^ round.wrapping_mul(0xd6e8_feb8_6659_fd93);
    let mut acc = 0usize;
    let t = Instant::now();
    for _ in 0..q {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        acc = acc.wrapping_add(f(&keys[(((x >> 32) * n) >> 32) as usize]));
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / q as f64
}

fn reference<'a, SC: SeedChooser + 'a>(
    keys: &'a [GxStr],
    lam: f64,
    sc: SC,
    name: String,
) -> Entry<'a> {
    let t = Instant::now();
    let params = Generic::new(Bits8, (lam * 100.0).round() as u16);
    let f: Function2<GenericCore, Bits8, SC, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_p_hash_sc(keys, &params, BuildGxHash, sc);
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.size_bytes() as f64 * 8.0 / keys.len() as f64;
    Entry {
        name,
        bits,
        build,
        bench: Box::new(move |q, r| batch(keys, q, r, |k| f.get(k))),
    }
}

fn phast_r<'a>(keys: &'a [GxStr], b: PHastRBuilder, name: String) -> Entry<'a> {
    let t = Instant::now();
    let f: PHastR<GxStr, Box<[u8]>> = b.try_build(keys, no_logging![]).unwrap();
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64;
    let mut seen = vec![false; keys.len()];
    for k in keys {
        let v = f.get(k);
        assert!(!seen[v], "duplicate output {v}");
        seen[v] = true;
    }
    Entry {
        name,
        bits,
        build,
        bench: Box::new(move |q, r| batch(keys, q, r, |k| f.get(k))),
    }
}

fn main() {
    let a = Args::parse();
    // Random strings of 10 to 50 printable bytes, made distinct by a prefix
    // encoding the index
    let mut x = 0x1234_5678_9abc_def1u64 ^ a.key_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let keys: Vec<GxStr> = (0..a.n)
        .map(|i| {
            let len = 10 + (next() % 41) as usize;
            let mut s = format!("{i:x}-");
            while s.len() < len {
                s.push((b'!' + (next() % 94) as u8) as char);
            }
            GxStr(s)
        })
        .collect();
    let mut entries = vec![];
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        entries.push(match p[0] {
            "ref" => {
                let lam: f64 = p[3].parse().unwrap();
                let name = format!("ref {} S=8 l={lam}", p[1]);
                match p[1] {
                    "plus" => reference(&keys, lam, ShiftOnly, name),
                    "w3" => reference(&keys, lam, ShiftOnlyWrapped::<3>, name),
                    c => panic!("unknown chooser {c}"),
                }
            }
            "r" => {
                let (b, sbits, desc) = lab::parse_config(&p[1..]);
                assert!(sbits <= 8, "byte seeds only");
                phast_r(&keys, b, desc)
            }
            _ => panic!("unknown variant {v}"),
        });
    }
    let mut times = vec![vec![]; entries.len()];
    for r in 0..a.interleave as u64 {
        for (e, t) in entries.iter().zip(times.iter_mut()) {
            t.push((e.bench)(a.queries, r));
        }
    }
    println!("n = {} (random strings of 10-50 bytes)", a.n);
    for (e, t) in entries.iter().zip(times.iter_mut()) {
        t.sort_by(f64::total_cmp);
        let q = t[t.len() / 2];
        println!(
            "{:36} {:.4} bits/key  build {:7.1} ns/key  query {:6.1} ns",
            e.name, e.bits, e.build, q
        );
        eprintln!(
            "CSV,{},{},{:.4},{:.1},{:.1}",
            e.name, a.n, e.bits, e.build, q
        );
    }
}
