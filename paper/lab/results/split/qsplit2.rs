//! Splits query time into first-level (fast) and bumped (slow) keys for both
//! PHast-R and the reference implementation, with interleaved timing.

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use ph::phast::{
    DefaultCompressedArray, Function2, Generic, GenericCore, SeedChooser, SeedOnly, ShiftOnly,
    ShiftOnlyWrapped,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::Bits8;
use std::time::Instant;
use sux::func::{PHastR, PHastRBuilder};

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, default_value_t = 2_000_000)]
    queries: usize,
    #[arg(short, long, default_value_t = 11)]
    rounds: usize,
    /// Only run this structure (index) on this kind (0 all, 1 fast, 2 slow),
    /// for profiling.
    #[arg(long)]
    only: Option<usize>,
    #[arg(long, default_value_t = 2)]
    kind: usize,
    /// PHast-R configurations <lambda>:<depth>.
    #[arg(short, long, value_delimiter = ',', default_value = "5.0:1,4.75:2")]
    variant: Vec<String>,
}

#[inline(always)]
fn batch<T: Copy>(ks: &[T], q: usize, round: u64, f: impl Fn(T) -> usize) -> f64 {
    let n = ks.len() as u64;
    let mut x = 0x9e3779b97f4a7c15u64 ^ round.wrapping_mul(0xd6e8_feb8_6659_fd93);
    let mut acc = 0usize;
    let t = Instant::now();
    for _ in 0..q {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        acc = acc.wrapping_add(f(ks[(((x >> 32) * n) >> 32) as usize]));
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / q as f64
}

struct S<'a> {
    name: String,
    beta: f64,
    /// (kind, round) -> ns per query; kind 0 = all, 1 = fast, 2 = slow
    bench: Box<dyn Fn(usize, usize, u64) -> f64 + 'a>,
}

fn reference<'a, SC: SeedChooser + 'a>(keys: &'a [u64], lam: f64, sc: SC, name: &str) -> S<'a> {
    let params = Generic::new(Bits8, (lam * 100.0).round() as u16);
    let f: Function2<GenericCore, Bits8, SC, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_p_hash_sc(keys, &params, BuildGxHash, sc);
    let (slow, fast): (Vec<u64>, Vec<u64>) = keys.iter().partition(|&&k| f.is_bumped(&k));
    S {
        name: name.to_string(),
        beta: slow.len() as f64 / keys.len() as f64,
        bench: Box::new(move |kind, q, r| match kind {
            0 => batch(keys, q, r, |k| f.get(&k)),
            1 => batch(&fast, q, r, |k| f.get(&k)),
            _ => batch(&slow, q, r, |k| f.get(&k)),
        }),
    }
}

fn phast_r<'a>(keys: &'a [GxKey], lam: f64, depth: u32, name: &str) -> S<'a> {
    let b = PHastRBuilder::default()
        .bucket_size(lam)
        .repair_depth(depth)
        .repair_candidates(if depth == 0 { 0 } else { 16 });
    let f: PHastR<GxKey, [u64; 1], Box<[u8]>> = b.try_build(keys, no_logging![]).unwrap();
    use sux::func::phast_r::PHastSig;
    use sux::utils::ToSig;
    let (slow, fast): (Vec<GxKey>, Vec<GxKey>) = keys
        .iter()
        .partition(|&&k| f.is_bumped(<GxKey as ToSig<[u64; 1]>>::to_sig(k, 0).ho().0));
    S {
        name: name.to_string(),
        beta: slow.len() as f64 / keys.len() as f64,
        bench: Box::new(move |kind, q, r| {
            match kind {
                0 => batch(keys, q, r, |k| f.get(k)),
                1 => batch(&fast, q, r, |k| f.get(k)),
                _ => batch(&slow, q, r, |k| f.get(k)),
            }
        }),
    }
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    // SAFETY: GxKey is a transparent wrapper around u64
    let gkeys: &[GxKey] = unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    let mut ss = vec![
        reference(&keys, 5.25, ShiftOnly, "ref PHast+ S=8 l=5.25"),
        reference(&keys, 5.0, ShiftOnlyWrapped::<3>, "ref wrap3 S=8 l=5"),
        reference(&keys, 4.5, SeedOnly, "ref PHast S=8 l=4.5"),
    ];
    for v in &a.variant {
        let (l, d) = v.split_once(':').unwrap();
        let (l, d): (f64, u32) = (l.parse().unwrap(), d.parse().unwrap());
        ss.push(phast_r(gkeys, l, d, &format!("PHast-R S=8 l={l} d={d}")));
    }
    if let Some(i) = a.only {
        eprintln!("profiling {} kind {}", ss[i].name, a.kind);
        let mut tot = 0.0;
        for r in 0..a.rounds as u64 {
            tot += (ss[i].bench)(a.kind, a.queries, r);
        }
        println!("{}: {:.2} ns", ss[i].name, tot / a.rounds as f64);
        return;
    }
    let mut t = vec![[vec![], vec![], vec![]]; ss.len()];
    for r in 0..a.rounds as u64 {
        for (s, t) in ss.iter().zip(t.iter_mut()) {
            t[0].push((s.bench)(0, a.queries, r));
            t[1].push((s.bench)(1, a.queries, r));
            t[2].push((s.bench)(2, a.queries / 4, r));
        }
    }
    println!("n = {}", a.n);
    for (s, t) in ss.iter().zip(t.iter_mut()) {
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (all, fast, slow) = (med(&mut t[0]), med(&mut t[1]), med(&mut t[2]));
        println!(
            "{:26} bumped {:5.2}%  all {all:6.2}  fast {fast:6.2}  slow {slow:6.2}  beta*(slow-fast) {:5.2}",
            s.name,
            100.0 * s.beta,
            s.beta * (slow - fast)
        );
    }
}
