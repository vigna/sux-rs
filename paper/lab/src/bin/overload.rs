use clap::Parser;
use lab::plus::*;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25")]
    lambda: Vec<f64>,
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    gamma: Vec<f64>,
    #[arg(short, long, default_value_t = 0)]
    big_l: usize,
}

fn main() {
    let a = Args::parse();
    for &lambda in &a.lambda {
        for &gamma in &a.gamma {
            let mut conf = Conf::plus(a.s, lambda);
            if a.big_l != 0 {
                conf.l = a.big_l;
            }
            let r = mphf_overload(a.n, &conf, gamma, || ShiftOnly, 42);
            let l0 = r.levels[0];
            println!(
                "S={} L={} lambda={lambda:.2} gamma={gamma:.3}: {:.4} b/k (seeds {:.4} ef {:.4} [{} entries]) L0 bumped {:.3}% holes {:.3}% levels {}",
                a.s,
                conf.l,
                r.bits_per_key(),
                r.seed_bits / a.n as f64,
                r.ef_bits / a.n as f64,
                r.ef_entries,
                100.0 * l0.3 as f64 / l0.0 as f64,
                100.0 * l0.4 as f64 / l0.0 as f64,
                r.levels.len()
            );
        }
    }
}
