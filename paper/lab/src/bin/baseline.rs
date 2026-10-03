use clap::Parser;
use lab::plus::*;
use ph::phast::{
    DefaultCompressedArray, Function2, Generic, GenericCore, ShiftOnly as RefShiftOnly,
};
use ph::seeds::{Bits8, BitsFast};
use ph::{BuildDefaultSeededHasher, GetSize};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, default_value_t = 5.25)]
    lambda: f64,
    #[arg(long)]
    reference: bool,
    #[arg(long)]
    hist: bool,
}

fn main() {
    let a = Args::parse();
    let conf = Conf::plus(a.s, a.lambda);
    let t = Instant::now();
    let r = mphf(a.n, &conf, || ShiftOnly, 42);
    let el = t.elapsed().as_secs_f64();
    println!(
        "lab PHast+ S={} lambda={} L={}: {:.4} bits/key (seeds {:.4}, ef {:.4}, last {:.4}) [{:.1} ns/key]",
        a.s,
        a.lambda,
        conf.l,
        r.bits_per_key(),
        r.seed_bits / a.n as f64,
        r.ef_bits / a.n as f64,
        r.last_bits / a.n as f64,
        el * 1e9 / a.n as f64
    );
    for (i, (k, b, bumped)) in r.levels.iter().enumerate() {
        println!(
            "  level {i}: keys {k} buckets {b} bumped {bumped} ({:.3}%)",
            100.0 * *bumped as f64 / *k as f64
        );
    }
    let st = &r.stats0;
    println!("  self collisions (level 0): {}", st.self_collisions);
    println!("  size: buckets bumped rate");
    for (sz, (b, bb)) in st.by_size.iter().enumerate() {
        if *b > 0 {
            println!("   {sz:2}: {b:9} {bb:8} {:.4}", *bb as f64 / *b as f64);
        }
    }
    if a.hist {
        // entropy of seed distribution
        let tot: usize = st.seed_hist.iter().sum();
        let ent: f64 = st
            .seed_hist
            .iter()
            .filter(|&&x| x > 0)
            .map(|&x| {
                let p = x as f64 / tot as f64;
                -p * p.log2()
            })
            .sum();
        println!("  seed entropy: {ent:.3} bits (of {})", a.s);
        for (sz, h) in st.seed_hist_by_size.iter().enumerate() {
            let tot: usize = h.iter().sum();
            if tot == 0 {
                continue;
            }
            let ent: f64 = h
                .iter()
                .filter(|&&x| x > 0)
                .map(|&x| {
                    let p = x as f64 / tot as f64;
                    -p * p.log2()
                })
                .sum();
            let mean: f64 = h
                .iter()
                .enumerate()
                .skip(1)
                .map(|(s, &x)| s as f64 * x as f64)
                .sum::<f64>()
                / (tot - h[0]).max(1) as f64;
            let med = {
                let mut acc = 0;
                let half = (tot - h[0]) / 2;
                let mut m = 0;
                for (s, &x) in h.iter().enumerate().skip(1) {
                    acc += x;
                    if acc >= half {
                        m = s;
                        break;
                    }
                }
                m
            };
            println!("   size {sz:2}: n={tot:9} entropy {ent:.3} mean seed {mean:.1} median {med}");
        }
    }

    if a.reference {
        let keys: Vec<u64> = (0..a.n as u64).collect();
        let t = Instant::now();
        let b100 = (a.lambda * 100.0).round() as u16;
        let bits = if a.s == 8 {
            let f: Function2<
                GenericCore,
                Bits8,
                RefShiftOnly,
                DefaultCompressedArray,
                BuildDefaultSeededHasher,
            > = Function2::with_slice_p_hash_sc(
                &keys,
                &Generic::new(Bits8, b100),
                BuildDefaultSeededHasher::default(),
                RefShiftOnly,
            );
            f.size_bytes() as f64 * 8.0
        } else {
            let f: Function2<
                GenericCore,
                BitsFast,
                RefShiftOnly,
                DefaultCompressedArray,
                BuildDefaultSeededHasher,
            > = Function2::with_slice_p_hash_sc(
                &keys,
                &Generic::new(BitsFast(a.s as u8), b100),
                BuildDefaultSeededHasher::default(),
                RefShiftOnly,
            );
            f.size_bytes() as f64 * 8.0
        };
        let el = t.elapsed().as_secs_f64();
        println!(
            "ref PHast+ S={} lambda={}: {:.4} bits/key [{:.1} ns/key]",
            a.s,
            a.lambda,
            bits / a.n as f64,
            el * 1e9 / a.n as f64
        );
    }
}
