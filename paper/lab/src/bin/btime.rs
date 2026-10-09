//! Construction time of PHast-R alone: builds the structure several times
//! and reports the minimum and median time per key, and the space (which
//! must not change when optimizing construction).
//!
//! Usage: btime [-n keys] [-r repeats] [-v <S>:<log2 L>:<lambda>[:<log2 R>],...]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use mem_dbg::{MemSize, SizeFlags};
use std::time::Instant;
use sux::func::PHastR;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, default_value_t = 5)]
    repeats: usize,
    #[arg(short, long, value_delimiter = ',', default_value = "8:10:4.25")]
    variant: Vec<String>,
}

fn main() {
    let a = Args::parse();
    let keys: Vec<GxKey> = (0..a.n as u64)
        .map(|i| GxKey(i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567))
        .collect();
    for v in &a.variant {
        let p: Vec<&str> = v.split(':').collect();
        let (b, sbits, _) = lab::parse_config(&p);
        assert!(sbits <= 8, "byte seeds only");
        let mut t = vec![];
        let mut bits = 0.0;
        for _ in 0..a.repeats {
            let start = Instant::now();
            let f: PHastR<GxKey, Box<[u8]>> =
                PHastR::try_par_new_with_builder(&keys, b.clone(), no_logging![]).unwrap();
            t.push(start.elapsed().as_secs_f64() * 1e9 / a.n as f64);
            bits = f.mem_size(SizeFlags::default()) as f64 * 8.0 / a.n as f64;
        }
        t.sort_by(f64::total_cmp);
        println!(
            "{v:16} {bits:.4} bits/key  build min {:6.1} median {:6.1} ns/key",
            t[0],
            t[t.len() / 2]
        );
    }
}
