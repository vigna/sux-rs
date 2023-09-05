mod bucketer;
use crate::prelude::{VSlice, VSliceMut};
pub use bucketer::*;
use common_traits::*;

pub struct PTHash<P, B> {
    pilots: P,
    bucketer: B,
    n: usize,
}

impl<P: VSlice, B: Bucketer> PTHash<P, B> {
    pub fn get(&self, hash: u64) -> usize {
        let bucket = self.bucketer.bucket(hash);
        let pilot = self.pilots.get(bucket);
        (hash ^ pilot).fast_range(self.n as _) as usize
    }
}

pub fn external_bucketing() {}

pub fn internal_bucketing() {}

pub fn sequential_search() {}

pub fn parallel_search() {}

#[allow(dead_code)] // PTHash is a work in progress
fn search<HT, H, B>(taken: &B, hashes: &[HT]) -> u64
where
    // hash type
    HT: Word + To<usize> + FastRange,
    usize: To<HT>,
    // hasher
    H: crate::hash::Hasher<HashType = HT> + core::default::Default,
    // bitmap
    B: VSliceMut,
{
    let mut pilot = 0_u64;
    'outer: loop {
        // think about caching the hased pilots
        let mut hasher = H::default();
        hasher.write_u64(pilot);
        let hashed_pilot = hasher.finish();

        // check for collisions
        for hash in hashes.iter() {
            let idx: usize = (*hash ^ hashed_pilot).fast_range(taken.len().to()).to();
            // if there's a collision, try the next pilot
            if taken.get(idx) != 0 {
                pilot += 1;
                continue 'outer;
            }
        }

        // return it
        return pilot;
    }
}
