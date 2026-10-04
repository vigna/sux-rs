//! Lab for PHast+ experiments.

pub mod block;
pub mod chain;
pub mod escape;
pub mod evict;
pub mod lattice;
pub mod plus;
pub mod rows;
pub mod thr;
pub mod twoclass;

#[inline(always)]
pub fn mul_hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// SplitMix64 for generating random hashes.
#[inline(always)]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn random_hashes(n: usize, seed: u64) -> Vec<u64> {
    (0..n as u64)
        .map(|i| {
            mix64(
                i.wrapping_add(seed.wrapping_mul(0x9e3779b97f4a7c15))
                    .wrapping_mul(0x9e3779b97f4a7c15),
            )
        })
        .collect()
}

/// Cyclic bit set of 2^LOG bits.
pub struct Cyclic<const W: usize> {
    pub w: Box<[u64; W]>,
}

impl<const W: usize> Default for Cyclic<W> {
    fn default() -> Self {
        Self {
            w: Box::new([0; W]),
        }
    }
}

impl<const W: usize> Cyclic<W> {
    const MASK: usize = W * 64 - 1;
    #[inline(always)]
    pub fn get(&self, v: usize) -> bool {
        let v = v & Self::MASK;
        self.w[v / 64] >> (v % 64) & 1 != 0
    }
    #[inline(always)]
    pub fn set(&mut self, v: usize) {
        let v = v & Self::MASK;
        self.w[v / 64] |= 1 << (v % 64);
    }
    #[inline(always)]
    pub fn clear(&mut self, v: usize) {
        let v = v & Self::MASK;
        self.w[v / 64] &= !(1 << (v % 64));
    }
    /// 64 bits starting at v.
    #[inline(always)]
    pub fn get64(&self, v: usize) -> u64 {
        let c = v / 64;
        let b = v % 64;
        let lo = self.w[c & (W - 1)];
        if b == 0 {
            return lo;
        }
        let hi = self.w[(c + 1) & (W - 1)];
        (lo >> b) | (hi << (64 - b))
    }
}

/// Elias–Fano size in bits (no select index) for `n` values in `[0..u)`.
pub fn ef_bits(n: usize, u: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let l = if u > n { (u / n).ilog2() as usize } else { 0 };
    (n * l + n + (u >> l) + 1) as f64
}

/// Builds a PHast-R function on the given keys, choosing byte seeds for at
/// most 8 bits per seed and a bit-field vector otherwise, and returns its
/// space in bits per key.
pub fn phast_r_bits_per_key(keys: &[u64], b: &sux::func::PHastRBuilder, seed_bits: u32) -> f64 {
    use dsi_progress_logger::no_logging;
    use mem_dbg::{MemSize, SizeFlags};
    use sux::func::PHastR;
    let bytes = if seed_bits <= 8 {
        let f: PHastR<u64, [u64; 1], Box<[u8]>> = b.try_build(keys, no_logging![]).unwrap();
        f.mem_size(SizeFlags::default())
    } else {
        let f: PHastR<u64, [u64; 1], sux::bits::BitFieldVec<Box<[usize]>>> =
            b.try_build(keys, no_logging![]).unwrap();
        f.mem_size(SizeFlags::default())
    };
    bytes as f64 * 8.0 / keys.len() as f64
}

/// A 64-bit key hashed by sux with GxHash, exactly as `ph::BuildGxHash`
/// hashes a `u64` (`GxHasher::with_seed`, `write_u64`, `finish`), so that
/// PHast-R and the reference implementation pay the same hashing cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct GxKey(pub u64);

impl sux::utils::ToSig<[u64; 1]> for GxKey {
    #[inline(always)]
    fn to_sig(key: impl std::borrow::Borrow<Self>, seed: u64) -> [u64; 1] {
        use std::hash::Hasher;
        let mut h = gxhash::GxHasher::with_seed(seed as i64);
        h.write_u64(key.borrow().0);
        [h.finish()]
    }
}
