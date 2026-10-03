use clap::Parser;
use lab::plus::*;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25")]
    lambda: Vec<f64>,
    /// patterns (R); D = (2^s - 1) / R
    #[arg(short, long, value_delimiter = ',', default_value = "4")]
    r: Vec<u16>,
    #[arg(short = 'L', long, value_delimiter = ',', default_value = "512")]
    big_l: Vec<usize>,
    /// use regular PHast
    #[arg(long)]
    phast: bool,
}

fn main() {
    let a = Args::parse();
    for &l in &a.big_l {
        for &r in &a.r {
            for &lambda in &a.lambda {
                let mut conf = Conf::plus(a.s, lambda);
                conf.l = l;
                let t = Instant::now();
                let (rep, name) = if a.phast {
                    conf.extra = 0;
                    conf.weights = phast_weights_8_1024();
                    (
                        mphf_overload(a.n, &conf, 0.0, || SeedOnly, 42),
                        "PHast".to_string(),
                    )
                } else {
                    let d = ((1u32 << a.s) - 1) as u16 / r;
                    conf.extra = d as usize - 1;
                    (
                        mphf_overload(a.n, &conf, 0.0, || MultiPattern { r, d }, 42),
                        format!("R={r} D={d}"),
                    )
                };
                let el = t.elapsed().as_secs_f64() * 1e9 / a.n as f64;
                let l0 = rep.levels[0];
                println!(
                    "{name} S={} L={l} lambda={lambda:.2}: {:.4} b/k (seeds {:.4} ef {:.4}) L0 bumped {:.3}% [{el:.0} ns/key]",
                    a.s,
                    rep.bits_per_key(),
                    rep.seed_bits / a.n as f64,
                    rep.ef_bits / a.n as f64,
                    100.0 * l0.3 as f64 / l0.0 as f64,
                );
            }
        }
    }
}
