use clap::Parser;
use lab::plus::shift_only_weights;
use lab::twoclass::*;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    /// filler fraction
    #[arg(short, long, value_delimiter = ',', default_value = "0,0.05,0.1")]
    p: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "5.25")]
    l0: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "1")]
    l1: Vec<f64>,
    #[arg(long, default_value_t = 8)]
    s0: u32,
    #[arg(long, default_value_t = 1)]
    r0: u16,
    #[arg(long, value_delimiter = ',', default_value = "6")]
    s1: Vec<u32>,
    #[arg(long, default_value_t = 1)]
    r1: u16,
    #[arg(long, default_value_t = 512)]
    big_l0: usize,
    #[arg(long, default_value_t = 512)]
    big_l1: usize,
    #[arg(long)]
    verify: bool,
    #[arg(long, default_value_t = 0)]
    thr: usize,
}

fn main() {
    let a = Args::parse();
    THRESHOLDS.store(a.thr, std::sync::atomic::Ordering::Relaxed);
    for &p in &a.p {
        for &l0 in &a.l0 {
            for &l1 in &a.l1 {
                for &s1 in &a.s1 {
                    let c0 = ClassConf {
                        lambda: l0,
                        l: a.big_l0,
                        r: a.r0,
                        d: (((1u32 << a.s0) - 1) / a.r0 as u32) as u16,
                        weights: shift_only_weights(a.s0, a.big_l0),
                        window: 256,
                    };
                    let c1 = ClassConf {
                        lambda: l1,
                        l: a.big_l1,
                        r: a.r1,
                        d: (((1u32 << s1) - 1) / a.r1 as u32) as u16,
                        weights: shift_only_weights(s1.max(6), a.big_l1),
                        window: 256,
                    };
                    let t = Instant::now();
                    let r = mphf(a.n, &c0, &c1, p, 42, a.verify);
                    let el = t.elapsed().as_secs_f64() * 1e9 / a.n as f64;
                    let lv = r.levels[0];
                    println!(
                        "p={p:.3} l0={l0:.2} l1={l1:.2} s0={} s1={s1} r0={} r1={}: {:.4} b/k (seeds0 {:.4} seeds1 {:.4} ef {:.4}) bumped0 {:.3}% bumped1 {:.3}% total {:.3}% [{el:.0} ns/key]",
                        a.s0,
                        a.r0,
                        a.r1,
                        r.bits_per_key(),
                        r.seed_bits0 / a.n as f64,
                        r.seed_bits1 / a.n as f64,
                        r.ef_bits / a.n as f64,
                        100.0 * lv.2 as f64 / lv.1.max(1) as f64,
                        100.0 * lv.4 as f64 / lv.3.max(1) as f64,
                        100.0 * (lv.2 + lv.4) as f64 / lv.0 as f64,
                    );
                }
            }
        }
    }
}
