use clap::Parser;
use lab::plus::*;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "3")]
    q: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "6")]
    big: Vec<f64>,
    #[arg(short = 'm', long, value_delimiter = ',', default_value = "1")]
    small: Vec<f64>,
    #[arg(short = 'L', long, default_value_t = 512)]
    big_l: usize,
}

fn main() {
    let a = Args::parse();
    for &q in &a.q {
        for &big in &a.big {
            for &small in &a.small {
                let mut conf = Conf::plus(a.s, 5.0);
                conf.l = a.big_l;
                conf.mix_q = q;
                conf.mix_big = big;
                conf.mix_small = small;
                let r = mphf_overload(a.n, &conf, 0.0, || ShiftOnly, 42);
                let l0 = r.levels[0];
                println!(
                    "q={q} big={big:.2} small={small:.2} avg lambda={:.3}: {:.4} b/k (seeds {:.4} ef {:.4}) L0 bumped {:.3}%",
                    (q as f64 * big + small) / (q + 1) as f64,
                    r.bits_per_key(),
                    r.seed_bits / a.n as f64,
                    r.ef_bits / a.n as f64,
                    100.0 * l0.3 as f64 / l0.0 as f64,
                );
            }
        }
    }
}
