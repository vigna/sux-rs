use clap::Parser;
use lab::lattice::*;
use lab::plus::shift_only_weights;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25")]
    lambda: Vec<f64>,
    #[arg(short, long, value_delimiter = ',', default_value = "1")]
    r: Vec<u16>,
    #[arg(short, long, value_delimiter = ',', default_value = "8")]
    q: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "0,8,32,128")]
    penalty: Vec<usize>,
    #[arg(short = 'L', long, default_value_t = 512)]
    big_l: usize,
}

fn main() {
    let a = Args::parse();
    for &r in &a.r {
        for &q in &a.q {
            for &penalty in &a.penalty {
                for &lambda in &a.lambda {
                    let lc = LatConf {
                        lambda,
                        l: a.big_l,
                        r,
                        d: 255 / r,
                        q,
                        penalty,
                        weights: shift_only_weights(8, a.big_l),
                        window: 256,
                    };
                    let t = Instant::now();
                    let rep = mphf(a.n, &lc, 42);
                    let el = t.elapsed().as_secs_f64() * 1e9 / a.n as f64;
                    let l0 = rep.levels[0];
                    let nn = a.n as f64;
                    println!(
                        "R={r} q={q} pen={penalty} lambda={lambda:.2}: plain {:.4} split {:.4} b/k (seeds {:.4} ef {:.4} holes {:.4}) bumped {:.3}% holes on {:.3}% off {:.3}% [{el:.0} ns/key]",
                        (rep.seed_bits + rep.ef_plain_bits + rep.last_bits) / nn,
                        (rep.seed_bits + rep.hole_bits + rep.last_bits) / nn,
                        rep.seed_bits / nn,
                        rep.ef_plain_bits / nn,
                        rep.hole_bits / nn,
                        100.0 * l0.2 as f64 / l0.0 as f64,
                        100.0 * l0.3 as f64 / l0.0 as f64,
                        100.0 * l0.4 as f64 / l0.0 as f64,
                    );
                }
            }
        }
    }
}
