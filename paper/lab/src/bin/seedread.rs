use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::{SeedStore, SeedStoreBuild};

#[inline(never)]
fn bench(name: &str, n: usize, q: usize, f: impl Fn(usize) -> usize) {
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut acc = 0usize;
    let t = Instant::now();
    for _ in 0..q {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let i = ((x >> 32) * n as u64 >> 32) as usize;
        acc = acc.wrapping_add(f(i));
    }
    std::hint::black_box(acc);
    println!(
        "{name}: {:.2} ns",
        t.elapsed().as_secs_f64() * 1e9 / q as f64
    );
}

fn main() {
    let n = 1_666_667usize;
    let seeds: Vec<u16> = (0..n).map(|i| (i * 7919 % 1023) as u16).collect();
    let bfv = <BitFieldVec<Box<[usize]>>>::from_seeds(&seeds, 10);
    let u16s = <Box<[u16]>>::from_seeds(&seeds, 10);
    // manual packed bytes
    let mut bytes = vec![0u8; n * 10 / 8 + 8];
    for (i, &s) in seeds.iter().enumerate() {
        let start = i * 10;
        for b in 0..10 {
            if s >> b & 1 != 0 {
                bytes[(start + b) / 8] |= 1 << ((start + b) % 8);
            }
        }
    }
    let bits = std::hint::black_box(10usize);
    for _ in 0..2 {
        bench("u16", n, 50_000_000, |i| unsafe { u16s.get_seed(i) });
        bench("bfv", n, 50_000_000, |i| unsafe { bfv.get_seed(i) });
        bench("bytes", n, 50_000_000, |i| unsafe {
            let start = i * bits;
            let w = (bytes.as_ptr().add(start / 8) as *const u32).read_unaligned();
            (w as usize >> (start % 8)) & ((1 << bits) - 1)
        });
    }
}
