//! Measures the fraction of keys bumped from the first level of PHast-R (as
//! implemented in sux) as a function of the repair parameters. Space-only
//! measure: the results do not depend on the hardware.
//!
//! Usage: bumps <keys> <spec>...
//!
//! A spec is `<S>:<log2 L>:<depth>:<candidates>:<lambda>`.
use dsi_progress_logger::no_logging;
use mem_dbg::{FlatType, MemSize, SizeFlags};
use sux::bits::BitFieldVec;
use sux::func::phast_r::{SeedStore, SeedStoreBuild};
use sux::func::{PHastR, PHastRBuilder};

const SEED: u64 = 0x6a09e667f3bcc908;

fn run<D: SeedStoreBuild + SeedStore + MemSize + FlatType>(
    keys: &[u64],
    b: PHastRBuilder,
) -> (f64, f64) {
    let f: PHastR<u64, D> = b.seed(SEED).try_build(keys, no_logging![]).unwrap();
    let bumped = keys.iter().filter(|k| f.is_bumped(*k)).count();
    (
        100.0 * bumped as f64 / keys.len() as f64,
        f.mem_size(SizeFlags::default()) as f64 * 8.0 / keys.len() as f64,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let keys: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    for spec in &args[2..] {
        let p: Vec<&str> = spec.split(':').collect();
        let s: u32 = p[0].parse().unwrap();
        let b = PHastRBuilder::default()
            .seed_bits(s)
            .log2_slice_len(p[1].parse().unwrap())
            .repair_depth(p[2].parse().unwrap())
            .repair_candidates(p[3].parse().unwrap())
            .bucket_size(p[4].parse().unwrap());
        let (rate, bits) = if s <= 8 {
            run::<Box<[u8]>>(&keys, b)
        } else {
            run::<BitFieldVec<Box<[usize]>>>(&keys, b)
        };
        println!("{spec:20} bumped {rate:.3}%  {bits:.4} bits/key");
    }
}
