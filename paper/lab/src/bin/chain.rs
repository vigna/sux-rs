use clap::Parser;
use lab::chain::*;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "4.5,5,5.5")]
    lambda: Vec<f64>,
    #[arg(short = 'L', long, value_delimiter = ',', default_value = "512")]
    big_l: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "0,4,16,64")]
    tries: Vec<usize>,
    #[arg(long)]
    verify: bool,
}

fn main() {
    let a = Args::parse();
    for &l in &a.big_l {
        for &tries in &a.tries {
            for chained in [false, true] {
                if !chained && tries > 0 {
                    continue;
                }
                for &lambda in &a.lambda {
                    let cc = ChainConf {
                        s: a.s,
                        lambda,
                        l,
                        tries,
                        chained,
                    };
                    let t = Instant::now();
                    let r = mphf(a.n, &cc, 42, a.verify);
                    let el = t.elapsed().as_secs_f64() * 1e9 / a.n as f64;
                    let l0 = r.levels[0];
                    println!(
                        "chain={chained} tries={tries} L={l} lambda={lambda:.2}: {:.4} b/k (seeds {:.4} ef {:.4}) L0 bumped {:.3}% backtracks/bucket {:.3} [{el:.0} ns/key]",
                        r.bits_per_key(),
                        r.seed_bits / a.n as f64,
                        r.ef_bits / a.n as f64,
                        100.0 * l0.2 as f64 / l0.0 as f64,
                        l0.3 as f64 / l0.1 as f64
                    );
                }
            }
        }
    }
}
