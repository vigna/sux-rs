use ph::phast::{Generic, Perfect, SeedOnlyK};
use ph::seeds::Bits8;
use ph::{BuildDefaultSeededHasher, GetSize};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map(|x| x.parse().unwrap())
        .unwrap_or(10_000_000);
    let keys: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    for k in [2u8, 4, 8, 16] {
        for lam in [4.0f64, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0] {
            if lam > 4.0 * k as f64 {
                continue;
            }
            let b100 = (lam * 100.0) as u16;
            let t = Instant::now();
            let f: Perfect<Bits8, SeedOnlyK, BuildDefaultSeededHasher> =
                Perfect::with_slice_p_hash_sc(
                    &keys,
                    &Generic::new(Bits8, b100),
                    BuildDefaultSeededHasher::default(),
                    SeedOnlyK(k),
                );
            let el = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            let bits = f.size_bytes() as f64 * 8.0 / n as f64;
            let minr = n.div_ceil(k as usize);
            let r = f.output_range();
            // bound: log2 e - (1/k) log2(k^k / k!)
            let kf = k as f64;
            let lgamma: f64 = (1..=k as u64).map(|x| (x as f64).ln()).sum::<f64>();
            let bound =
                std::f64::consts::LOG2_E - (kf * kf.ln() - lgamma) / kf / std::f64::consts::LN_2;
            println!(
                "k={k} lambda={lam}: {bits:.4} bits/key (bound {bound:.4}), range {:.4}x minimal, slack slots {:.3}% [{el:.0} ns/key]",
                r as f64 / minr as f64,
                100.0 * ((r * k as usize) as f64 - n as f64) / n as f64
            );
        }
    }
}
