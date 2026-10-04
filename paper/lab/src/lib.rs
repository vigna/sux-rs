//! Lab for PHast-R experiments: comparison with the reference implementation
//! of PHast/PHast+ (`ph`), and helpers.

pub mod dfs;
pub mod walk;

/// Builds a PHast-R function on the given keys, choosing byte seeds for at
/// most 8 bits per seed and a bit-field vector otherwise, and returns its
/// space in bits per key.
pub fn phast_r_bits_per_key(keys: &[u64], b: &sux::func::PHastRBuilder, seed_bits: u32) -> f64 {
    use dsi_progress_logger::no_logging;
    use mem_dbg::{MemSize, SizeFlags};
    use sux::func::PHastR;
    let bytes = if seed_bits <= 8 {
        let f: PHastR<u64, Box<[u8]>> = b.try_build(keys, no_logging![]).unwrap();
        f.mem_size(SizeFlags::default())
    } else {
        let f: PHastR<u64, sux::bits::BitFieldVec<Box<[usize]>>> =
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
