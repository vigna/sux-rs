//! Measures the range of the empirical bridge of slice beginnings (see
//! `lab::dfs::Keys::bridge_range`) and compares it with √(πn/2), the
//! asymptotic mean (the mean of the Kuiper distribution times √n).
//!
//! Usage: bridge <instances> <n>...
use lab::dfs::Keys;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inst: u64 = args[1].parse().unwrap();
    for a in &args[2..] {
        let n: usize = a.parse::<usize>().unwrap() / 64 * 64;
        let r: Vec<f64> = (0..inst)
            .map(|i| Keys::new(n, 5.0, 1000 + i).bridge_range() as f64)
            .collect();
        let mean = r.iter().sum::<f64>() / inst as f64;
        let sd = (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (inst - 1) as f64).sqrt();
        let pred = (std::f64::consts::PI * n as f64 / 2.0).sqrt();
        println!(
            "n {n:10}  range mean {mean:9.1} (sd {sd:7.1})  sqrt(pi n / 2) {pred:9.1}  ratio {:.4}",
            mean / pred
        );
    }
}
