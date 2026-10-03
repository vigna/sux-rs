use clap::Parser;
use lab::evict::*;
use lab::plus::Conf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25")]
    lambda: Vec<f64>,
    #[arg(short, long, value_delimiter = ',', default_value = "1")]
    r: Vec<u16>,
    #[arg(short = 'L', long, value_delimiter = ',', default_value = "512")]
    big_l: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "0,8")]
    cand: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "1")]
    depth: Vec<u32>,
    #[arg(short = 'z', long, default_value_t = 64)]
    max_evict: usize,
    #[arg(long)]
    verify: bool,
    #[arg(long)]
    sacrifice: bool,
}

fn main() {
    let a = Args::parse();
    for &l in &a.big_l {
        for &r in &a.r {
            for &depth in &a.depth {
                for &cand in &a.cand {
                    for &lambda in &a.lambda {
                        let mut conf = Conf::plus(a.s, lambda);
                        conf.l = l;
                        let d = if r == 0 {
                            0
                        } else {
                            ((1u32 << a.s) - 1) as u16 / r
                        };
                        if r == 0 {
                            conf.weights = lab::plus::phast_weights_8_1024();
                        }
                        let ec = EvictConf {
                            r,
                            d,
                            max_cand: cand,
                            max_evict_size: a.max_evict,
                            depth,
                            sacrifice: a.sacrifice,
                        };
                        let t = Instant::now();
                        let rep = mphf(a.n, &conf, &ec, 42, a.verify);
                        let el = t.elapsed().as_secs_f64() * 1e9 / a.n as f64;
                        let l0 = rep.levels[0];
                        println!(
                            "R={r} D={d} L={l} cand={cand} depth={depth} lambda={lambda:.2}: {:.4} b/k (seeds {:.4} ef {:.4}) L0 bumped {:.3}% evict {} repaired {} sacrificed {} [{el:.0} ns/key]",
                            rep.bits_per_key(),
                            rep.seed_bits / a.n as f64,
                            rep.ef_bits / a.n as f64,
                            100.0 * l0.2 as f64 / l0.0 as f64,
                            l0.3,
                            l0.4,
                            l0.5
                        );
                    }
                }
            }
        }
    }
}
