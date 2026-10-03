use clap::Parser;
use lab::block::*;
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
    #[arg(short, long, value_delimiter = ',', default_value = "64")]
    g: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "1,4,16")]
    t: Vec<u32>,
    #[arg(short = 'L', long, default_value_t = 512)]
    big_l: usize,
    #[arg(long)]
    early: bool,
}

fn main() {
    let a = Args::parse();
    for &g in &a.g {
        for &t in &a.t {
            for &lambda in &a.lambda {
                let mut conf = Conf::plus(a.s, lambda);
                conf.l = a.big_l;
                let bc = BlockConf {
                    g,
                    t,
                    early_exit: a.early,
                };
                let st = Instant::now();
                let r = mphf_blocks(a.n, &conf, &bc, 42);
                let el = st.elapsed().as_secs_f64() * 1e9 / a.n as f64;
                let l0 = r.levels[0];
                println!(
                    "G={g} T={t} S={} L={} lambda={lambda:.2}: {:.4} b/k (seeds {:.4} blk {:.4} ef {:.4}) L0 bumped {:.3}% trials/blk {:.2} [{el:.0} ns/key]",
                    a.s,
                    a.big_l,
                    r.bits_per_key(),
                    r.seed_bits / a.n as f64,
                    r.block_bits / a.n as f64,
                    r.ef_bits / a.n as f64,
                    100.0 * l0.2 as f64 / l0.0 as f64,
                    l0.3 as f64 / l0.1.div_ceil(g) as f64,
                );
            }
        }
    }
}
