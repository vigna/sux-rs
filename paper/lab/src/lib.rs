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
        let f: PHastR<u64, Box<[u8]>> =
            PHastR::try_par_new_with_builder(keys, b.clone(), no_logging![]).unwrap();
        f.mem_size(SizeFlags::default())
    } else {
        let f: PHastR<u64, sux::bits::BitFieldVec<Box<[usize]>>> =
            PHastR::try_par_new_with_builder(keys, b.clone(), no_logging![]).unwrap();
        f.mem_size(SizeFlags::default())
    };
    bytes as f64 * 8.0 / keys.len() as f64
}

/// Parses a PHast-R configuration `<S>:<log2 L>:<lambda>[:<log2 R>]`: bits
/// per seed, base-2 logarithm of the slice length, expected bucket size,
/// and base-2 logarithm of the number of patterns (2 if missing). Returns
/// the builder, the number of bits per seed, and a description.
pub fn parse_config(fields: &[&str]) -> (sux::func::PHastRBuilder, u32, String) {
    let s: u32 = fields[0].parse().unwrap();
    let ll: u32 = fields[1].parse().unwrap();
    let lam: f64 = fields[2].parse().unwrap();
    let lr: u32 = fields.get(3).map(|x| x.parse().unwrap()).unwrap_or(2);
    (
        sux::func::PHastRBuilder::default()
            .seed_bits(s)
            .log2_slice_len(ll)
            .bucket_size(lam)
            .log2_patterns(lr),
        s,
        format!("PHast-R S={s} L={} R={} l={lam}", 1 << ll, 1 << lr),
    )
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
