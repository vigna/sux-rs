use clap::Parser;
use lab::plus::shift_only_weights;
use lab::rows::*;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u32,
    #[arg(short, long, value_delimiter = ',', default_value = "4.5,5,5.5")]
    lambda: Vec<f64>,
    #[arg(short = 'r', long, value_delimiter = ',', default_value = "8")]
    lr: Vec<usize>,
    #[arg(short, long, value_delimiter = ',', default_value = "0,1")]
    choice: Vec<u8>,
    /// weights from PHast+ table for this (S, L)
    #[arg(long, default_value_t = 512)]
    wl: usize,
    #[arg(long)]
    verify: bool,
    #[arg(long)]
    sizes: bool,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    rowmode: Vec<u8>,
}

fn main() {
    let x = Args::parse();
    for &rowmode in &x.rowmode {
        for &lr in &x.lr {
            for &choice in &x.choice {
                for &lambda in &x.lambda {
                    let rc = RowConf {
                        s: x.s,
                        lambda,
                        lr,
                        weights: shift_only_weights(8, x.wl),
                        window: 256,
                        choice,
                        rowmode,
                    };
                    let t = Instant::now();
                    let r = mphf(x.n, &rc, 42, x.verify);
                    let el = t.elapsed().as_secs_f64() * 1e9 / x.n as f64;
                    let l0 = r.levels[0];
                    println!(
                        "rows mode={rowmode} S={} A={} lr={lr} choice={choice} lambda={lambda:.2}: {:.4} b/k (seeds {:.4} ef {:.4}) L0 bumped {:.3}% [{el:.0} ns/key]",
                        x.s,
                        rc.a(),
                        r.bits_per_key(),
                        r.seed_bits / x.n as f64,
                        r.ef_bits / x.n as f64,
                        100.0 * l0.2 as f64 / l0.0 as f64
                    );
                    if x.sizes {
                        for (sz, (b, bb)) in r.by_size0.iter().enumerate() {
                            if *b > 0 {
                                println!("   {sz:2}: {b:9} {bb:8} {:.4}", *bb as f64 / *b as f64);
                            }
                        }
                    }
                }
            }
        }
    }
}
