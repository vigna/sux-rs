//! Experiments on overloading: sweeps a single level of PHast-R with an
//! output range smaller than the number of keys, and reports the fraction of
//! bumped keys and of free positions (holes), with an estimate of the space
//! of a structure in which holes are filled through a remapping and bumped
//! keys go to further levels:
//!
//! S/λ + C·bumped + (2.06 + log2(1/holes))·1.045·holes
//!
//! where C is the cost of a bumped key at the following levels.
//!
//! Usage: overload [-n keys] [-v <S>:<log2 L>:<depth>:<pattern>] [-l lambdas] [-d overloads (%)]

use clap::Parser;
use sux::func::PHastRBuilder;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 3_000_000)]
    n: usize,
    /// <S>:<log2 L>:<depth>:<log2 R, w<M>, or g<R>>[:<log2 band>]
    #[arg(short, long, default_value = "8:10:0:w3")]
    variant: String,
    #[arg(short, long, value_delimiter = ',', default_value = "5.0")]
    lambda: Vec<f64>,
    #[arg(short, long, value_delimiter = ',', default_value = "0,1,2,3,4,6,8")]
    delta: Vec<f64>,
    /// Cost in bits of a bumped key at the following levels.
    #[arg(short, long, default_value_t = 1.9)]
    cost: f64,
    /// Relative expected sizes of the buckets of a period (uniform if
    /// empty); several profiles can be separated by `/`.
    #[arg(long, default_value = "")]
    skew: String,
    /// The number of buckets in the window of the sweep.
    #[arg(short, long, default_value_t = 256)]
    window: usize,
    /// A multiplier for the size-dependent priority weights.
    #[arg(long, default_value_t = 1.0)]
    wscale: f64,
    /// Prints the bump rate by bucket size.
    #[arg(long)]
    sizes: bool,
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    let p: Vec<&str> = a.variant.split(':').collect();
    let s: u32 = p[0].parse().unwrap();
    let depth: u32 = p[2].parse().unwrap();
    // Patterns: <log2 R> (no wrapping), w<M> (wrapping with multiplier M),
    // or g<R> (ring with R patterns)
    let (lr, wrap, ring): (u32, u32, u32) = if let Some(m) = p[3].strip_prefix('w') {
        (0, m.parse().unwrap(), 0)
    } else if let Some(r) = p[3].strip_prefix('g') {
        (0, 3, r.parse().unwrap())
    } else {
        (p[3].parse().unwrap(), 0, 0)
    };
    println!("{} n={}", a.variant, a.n);
    println!("lambda delta%  bumped%  holes%   est b/key");
    for prof in a.skew.split('/') {
        let skew: Vec<f64> = prof
            .split(',')
            .filter(|x| !x.is_empty())
            .map(|x| x.parse().unwrap())
            .collect();
        if !skew.is_empty() {
            let tot: f64 = skew.iter().sum();
            println!(
                "skew {:?}",
                skew.iter()
                    .map(|x| (x / tot * skew.len() as f64 * 100.0).round() / 100.0)
                    .collect::<Vec<_>>()
            );
        }
        for &lam in &a.lambda {
            for &d in &a.delta {
                let b = PHastRBuilder::default()
                    .seed_bits(s)
                    .log2_slice_len(p[1].parse().unwrap())
                    .repair_depth(depth)
                    .repair_candidates(if depth == 0 { 0 } else { 16 })
                    .log2_patterns(lr)
                    .wrap(wrap)
                    .ring(ring)
                    .window(a.window, a.wscale)
                    .skew(&skew)
                    .log2_band(p.get(4).map(|x| x.parse().unwrap()).unwrap_or(0))
                    .bucket_size(lam);
                let m = (a.n as f64 / (1.0 + d / 100.0)).round() as usize;
                let (bumped, holes, by_size) = b.level_stats::<u64, u64>(&keys, m);
                let (bf, hf) = (bumped as f64 / a.n as f64, holes as f64 / a.n as f64);
                let est = s as f64 / lam + a.cost * bf + (2.06 + (1.0 / hf).log2()) * 1.045 * hf;
                let sc = sux::func::phast_r::SELF_COLL.with(|c| c.get());
                println!(
                    "{lam:5.2} {d:5.1}  {:7.3} {:7.3}   {est:.4}   self-colliding: {:.3}% of keys, bumped {:.3}% of keys ({:.1}% of bumped keys)",
                    100.0 * bf,
                    100.0 * hf,
                    100.0 * sc[1] as f64 / a.n as f64,
                    100.0 * sc[3] as f64 / a.n as f64,
                    100.0 * sc[3] as f64 / bumped.max(1) as f64
                );
                if std::env::var("PHAST_FEAS").is_ok() {
                    use std::sync::atomic::Ordering::Relaxed;
                    use sux::func::phast_r::FEAS;
                    let nb: usize = FEAS.iter().map(|f| f[0].load(Relaxed)).sum();
                    let mut waste = 0.0;
                    println!("size  buckets%  mean feasible  mean log2  none%   1-2%");
                    for (k, f) in FEAS.iter().enumerate() {
                        let c = f[0].swap(0, Relaxed);
                        let (sum, lg, zero, few) = (
                            f[1].swap(0, Relaxed),
                            f[2].swap(0, Relaxed),
                            f[3].swap(0, Relaxed),
                            f[4].swap(0, Relaxed),
                        );
                        if c * 2000 > nb {
                            println!(
                                "{k:4}  {:7.2}  {:13.1}  {:9.2}  {:5.1}  {:5.1}",
                                100.0 * c as f64 / nb as f64,
                                sum as f64 / c as f64,
                                lg as f64 / 1000.0 / c as f64,
                                100.0 * zero as f64 / c as f64,
                                100.0 * few as f64 / c as f64
                            );
                        }
                        waste += lg as f64 / 1000.0;
                    }
                    println!(
                        "wasted choice: {:.3} bits/bucket = {:.3} bits/key",
                        waste / nb as f64,
                        waste / a.n as f64
                    );
                }
                if a.sizes {
                    for (k, &(nb, bb)) in by_size.iter().enumerate() {
                        if nb > 0 && k <= 14 {
                            print!("{k}:{:.3} ", bb as f64 / nb as f64);
                        }
                    }
                    println!();
                }
            }
        }
    }
}
