use clap::Parser;
use lab::plus::shift_only_weights;
use lab::thr::*;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25")]
    lambda: Vec<f64>,
    /// full-keep shifts
    #[arg(short, value_delimiter = ',', default_value = "255")]
    a: Vec<u16>,
    /// threshold classes
    #[arg(short, value_delimiter = ',', default_value = "0")]
    t: Vec<u16>,
    /// kept fractions, comma separated, per class (used in order)
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0.875,0.75,0.625,0.5,0.375,0.25,0.125"
    )]
    thetas: Vec<f64>,
    #[arg(short = 'L', long, default_value_t = 512)]
    big_l: usize,
    #[arg(long)]
    best_kept: bool,
    #[arg(long)]
    verify: bool,
    #[arg(long)]
    offset: bool,
}

fn main() {
    let x = Args::parse();
    let maxv = (1u32 << x.s) - 1;
    for &a in &x.a {
        for &t in &x.t {
            if a as u32 > maxv || (t == 0 && a as u32 != maxv) {
                continue;
            }
            let dt = if t == 0 {
                1
            } else {
                ((maxv - a as u32) / t as u32) as u16
            };
            if t > 0 && dt == 0 {
                continue;
            }
            // pick t thetas spread over the provided list
            let thetas: Vec<f64> = if t == 0 {
                vec![]
            } else {
                (0..t as usize)
                    .map(|i| x.thetas[(i * x.thetas.len()) / t as usize])
                    .collect()
            };
            for &lambda in &x.lambda {
                let tc = ThrConf {
                    s: x.s,
                    lambda,
                    l: x.big_l,
                    a,
                    t,
                    dt,
                    thetas: thetas.clone(),
                    weights: shift_only_weights(x.s, x.big_l),
                    window: 256,
                    best_kept: x.best_kept,
                    offset_mode: x.offset,
                };
                let st = Instant::now();
                let r = mphf(x.n, &tc, 42, x.verify);
                let el = st.elapsed().as_secs_f64() * 1e9 / x.n as f64;
                let l0 = r.levels[0];
                println!(
                    "S={} a={a} t={t} dt={dt} thetas={:?} lambda={lambda:.2}: {:.4} b/k (seeds {:.4} ef {:.4}) L0 bumped {:.3}% thr-buckets {:.3}% full-bumps {:.3}% [{el:.0} ns/key]",
                    x.s,
                    thetas,
                    r.bits_per_key(),
                    r.seed_bits / x.n as f64,
                    r.ef_bits / x.n as f64,
                    100.0 * l0.2 as f64 / l0.0 as f64,
                    100.0 * l0.3 as f64 / l0.1 as f64,
                    100.0 * l0.4 as f64 / l0.1 as f64,
                );
            }
        }
    }
}
