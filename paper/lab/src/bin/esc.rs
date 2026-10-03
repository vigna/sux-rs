use clap::Parser;
use lab::escape::*;
use lab::plus::shift_only_weights;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25,6,7")]
    lambda: Vec<f64>,
    #[arg(long, default_value_t = 1)]
    r1: u16,
    #[arg(long, value_delimiter = ',', default_value = "16")]
    s2: Vec<u32>,
    /// secondary patterns (D2 = (2^s2 - 1) / r2)
    #[arg(long, value_delimiter = ',', default_value = "256")]
    r2: Vec<u16>,
    #[arg(short = 'L', long, default_value_t = 512)]
    big_l: usize,
    #[arg(long, default_value_t = 62)]
    per_line: usize,
    #[arg(long, default_value_t = 0)]
    sec_choice: u8,
    #[arg(long)]
    verify: bool,
}

fn main() {
    let x = Args::parse();
    for &s2 in &x.s2 {
        for &r2 in &x.r2 {
            for &lambda in &x.lambda {
                let d1 = (255 / x.r1) as u16;
                let d2 = if s2 == 0 {
                    1
                } else {
                    (((1u32 << s2) - 1) / r2 as u32).min(4096) as u16
                };
                let ec = EscConf {
                    lambda,
                    l: x.big_l,
                    r1: x.r1,
                    d1,
                    s2,
                    r2,
                    d2,
                    weights: shift_only_weights(8, x.big_l),
                    window: 256,
                    per_line: x.per_line,
                    sec_choice: x.sec_choice,
                };
                let t = Instant::now();
                let r = mphf(x.n, &ec, 42, x.verify);
                let el = t.elapsed().as_secs_f64() * 1e9 / x.n as f64;
                let l0 = r.levels[0];
                println!(
                    "r1={} s2={s2} r2={r2} d2={d2} lambda={lambda:.2}: {:.4} b/k (p {:.4} s {:.4} ef {:.4}) esc buckets {:.3}% esc keys {:.3}% bumped {:.3}% levels {} [{el:.0} ns/key]",
                    x.r1,
                    r.bits_per_key(),
                    r.p_bits / x.n as f64,
                    r.s_bits / x.n as f64,
                    r.ef_bits / x.n as f64,
                    100.0 * l0.2 as f64 / l0.1 as f64,
                    100.0 * l0.3 as f64 / l0.0 as f64,
                    100.0 * l0.4 as f64 / l0.0 as f64,
                    r.levels.len()
                );
            }
        }
    }
}
