//! Sweeps the expected bucket size of PHast-R configurations.
//!
//! Usage: tune [<keys> [<S>:<log2 L>:<lambda>,<lambda>,...[:<log2 R>]]...]
//!
//! Without configurations, sweeps the default configurations for 8-bit and
//! 10-bit seeds.
use lab::phast_r_bits_per_key;
use std::time::Instant;

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
            "8:10:4.5,4.75,5.0,5.25".into(),
            "10:11:5.75,6.0,6.25,6.5".into(),
        ]
    };
    let keys: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    for spec in &specs {
        let p: Vec<&str> = spec.split(':').collect();
        for lam in p[2].split(',') {
            let mut fields = p.clone();
            fields[2] = lam;
            let (b, s, desc) = lab::parse_config(&fields);
            let t = Instant::now();
            let bits = phast_r_bits_per_key(&keys, &b, s);
            let el = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            println!("{desc}: {bits:.4} bits/key, build {el:.1} ns/key");
        }
    }
}
