use dsi_progress_logger::no_logging;
use std::time::Instant;
use sux::bits::BitFieldVec;
use sux::func::phast_r::{PHastR, PHastRBuilder, PHastSig, SeedStore};
use sux::utils::ToSig;

#[inline(never)]
fn bench<D: SeedStore>(name: &str, f: &PHastR<u64, [u64; 1], D>, hos: &[(u64, u64)], q: usize) {
    let n = hos.len() as u64;
    for _ in 0..2 {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut acc = 0usize;
        let t = Instant::now();
        for _ in 0..q {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let (h, o) = hos[((x >> 32) * n >> 32) as usize];
            acc = acc.wrapping_add(f.get_by_ho(h, o));
        }
        std::hint::black_box(acc);
        println!(
            "{name}: {:.2} ns",
            t.elapsed().as_secs_f64() * 1e9 / q as f64
        );
    }
}

struct Packed<const B: usize>(Box<[u8]>);
impl<const B: usize> SeedStore for Packed<B> {
    #[inline(always)]
    unsafe fn get_seed(&self, i: usize) -> usize {
        let start = i * B;
        let w = unsafe { (self.0.as_ptr().add(start / 8) as *const u32).read_unaligned() };
        (w as usize >> (start % 8)) & ((1 << B) - 1)
    }
}
impl<const B: usize> sux::func::phast_r::SeedStoreBuild for Packed<B> {
    const MAX_BITS: u32 = B as u32;
    fn from_seeds(seeds: &[u16], _bits: u32) -> Self {
        let mut bytes = vec![0u8; seeds.len() * B / 8 + 8];
        for (i, &s) in seeds.iter().enumerate() {
            let start = i * B;
            for b in 0..B {
                if s >> b & 1 != 0 {
                    bytes[(start + b) / 8] |= 1 << ((start + b) % 8);
                }
            }
        }
        Packed(bytes.into())
    }
}

fn main() {
    let n = 10_000_000usize;
    let keys: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    let hos: Vec<(u64, u64)> = keys
        .iter()
        .map(|k| <u64 as ToSig<[u64; 1]>>::to_sig(k, 0).ho())
        .collect();
    let b = PHastRBuilder::default()
        .seed_bits(10)
        .log2_slice_len(11)
        .bucket_size(6.0)
        .repair_depth(1);
    let f16: PHastR<u64, [u64; 1], Box<[u16]>> = b.try_build(&keys, no_logging![]).unwrap();
    let fbf: PHastR<u64, [u64; 1], BitFieldVec<Box<[usize]>>> =
        b.try_build(&keys, no_logging![]).unwrap();
    let fpk: PHastR<u64, [u64; 1], Packed<10>> = b.try_build(&keys, no_logging![]).unwrap();
    bench("u16 get_by_ho", &f16, &hos, 20_000_000);
    bench("packed<10> get_by_ho", &fpk, &hos, 20_000_000);
    bench("bfv get_by_ho", &fbf, &hos, 20_000_000);
}
