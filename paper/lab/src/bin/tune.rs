//! Sweeps the expected bucket size of PHast-R configurations.
//!
//! Usage: tune [<keys> [<S>:<log2 L>:<depth>:<lambda>,<lambda>,...]...]
//!
//! Without configurations, sweeps the configurations of the paper.
use lab::phast_r_bits_per_key;
use std::time::Instant;
use sux::func::PHastRBuilder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .map(|x| x.parse().unwrap())
        .unwrap_or(10_000_000);
    let specs: Vec<String> = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        vec![
            "8:9:1:4.75,5.0,5.25,5.5".into(),
            "8:9:2:4.75,5.0,5.25,5.5".into(),
            "10:11:1:6.0,6.5,7.0".into(),
            "10:11:2:6.0,6.5,7.0".into(),
        ]
    };
    let keys: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    for spec in &specs {
        let p: Vec<&str> = spec.split(':').collect();
        let (s, ll, depth): (u32, u32, u32) = (
            p[0].parse().unwrap(),
            p[1].parse().unwrap(),
            p[2].parse().unwrap(),
        );
        for lam in p[3].split(',').map(|x| x.parse::<f64>().unwrap()) {
            let b = PHastRBuilder::default()
                .seed_bits(s)
                .log2_slice_len(ll)
                .repair_depth(depth)
                .repair_candidates(if depth == 0 { 0 } else { 16 })
                .bucket_size(lam);
            let t = Instant::now();
            let bits = phast_r_bits_per_key(&keys, &b, s);
            let el = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            println!(
                "S={s} L={} d={depth} lambda={lam:.2}: {bits:.4} bits/key, build {el:.1} ns/key",
                1 << ll
            );
        }
    }
}
