/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The storage of the seeds of [`PHastR`].
//!
//! [`PHastR`]: super::PHastR

use crate::bits::{BitFieldVec, BitFieldVecU};
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// Storage for the seeds of a level (query side).
///
/// Implementations are provided for `Box<[u8]>` (at most 8 bits per seed,
/// the fastest option), `Box<[u16]>`, [`BitFieldVec`] (any width up to 16
/// bits), [`BitFieldVecU`] (the same, with unaligned reads, obtained by
/// [`TryIntoUnaligned::try_into_unaligned`]), and for the corresponding
/// borrowed types obtained by ε-serde deserialization.
///
/// [`TryIntoUnaligned::try_into_unaligned`]: crate::traits::TryIntoUnaligned::try_into_unaligned
pub trait SeedStore {
    /// Returns the seed of index `i`.
    ///
    /// # Safety
    ///
    /// `i` must be smaller than the number of seeds.
    unsafe fn get_seed(&self, i: usize) -> usize;

    /// Prefetches the seed of index `i` (by default, does nothing).
    #[inline(always)]
    fn prefetch_seed(&self, _i: usize) {}
}

/// Storage for the seeds of a level (construction side).
pub trait SeedStoreBuild: SeedStore + Sized {
    /// The maximum number of bits per seed supported.
    const MAX_BITS: u32;

    /// Builds the storage from the given seeds, which use at most `bits` bits.
    fn from_seeds(seeds: &[u16], bits: u32) -> Self;
}

macro_rules! impl_seed_store_slice {
    ($ty:ty, $bits:expr) => {
        impl SeedStore for Box<[$ty]> {
            #[inline(always)]
            unsafe fn get_seed(&self, i: usize) -> usize {
                // SAFETY: by the contract of this method
                unsafe { *<[$ty]>::get_unchecked(self, i) as usize }
            }

            #[inline(always)]
            fn prefetch_seed(&self, i: usize) {
                crate::utils::prefetch_index(self, i);
            }
        }

        impl SeedStore for &[$ty] {
            #[inline(always)]
            unsafe fn get_seed(&self, i: usize) -> usize {
                // SAFETY: by the contract of this method
                unsafe { *<[$ty]>::get_unchecked(self, i) as usize }
            }
        }

        impl SeedStoreBuild for Box<[$ty]> {
            const MAX_BITS: u32 = $bits;

            fn from_seeds(seeds: &[u16], _bits: u32) -> Self {
                seeds.iter().map(|&s| s as $ty).collect()
            }
        }
    };
}

impl_seed_store_slice!(u8, 8);
impl_seed_store_slice!(u16, 16);

impl<B: crate::traits::Backend<Word = usize> + AsRef<[usize]>> SeedStore for BitFieldVec<B> {
    #[inline(always)]
    unsafe fn get_seed(&self, i: usize) -> usize {
        // SAFETY: by the contract of this method
        unsafe { self.get_value_unchecked(i) }
    }

    #[inline(always)]
    fn prefetch_seed(&self, i: usize) {
        let bw = crate::traits::BitWidth::bit_width(self);
        crate::utils::prefetch_index(self.as_slice(), i * bw / usize::BITS as usize);
    }
}

impl<B: crate::traits::Backend<Word = usize> + AsRef<[usize]>> SeedStore for BitFieldVecU<B> {
    #[inline(always)]
    unsafe fn get_seed(&self, i: usize) -> usize {
        // SAFETY: by the contract of this method (reads are unaligned)
        unsafe { self.get_value_unchecked(i) }
    }

    #[inline(always)]
    fn prefetch_seed(&self, i: usize) {
        let bw = crate::traits::BitWidth::bit_width(self);
        crate::utils::prefetch_index(self.as_ref(), i * bw / usize::BITS as usize);
    }
}

impl SeedStoreBuild for BitFieldVec<Box<[usize]>> {
    const MAX_BITS: u32 = 16;

    fn from_seeds(seeds: &[u16], bits: u32) -> Self {
        // Padded, so that conversion to unaligned reads needs no reallocation
        let mut bfv = BitFieldVec::<Box<[usize]>>::new_padded(bits as usize, seeds.len());
        for (i, &s) in seeds.iter().enumerate() {
            bfv.set_value(i, s as usize);
        }
        bfv
    }
}
