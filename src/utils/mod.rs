/*
 *
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Utility traits and implementations.
//!
//! This module provides several support facilities:
//!
//! - [`lenders`]: lending iterators over lines of (possibly compressed) files.
//! - [`sig_store`]: signature storage for static functions and filters.
//! - [`fair_chunks`]: splitting ranges into balanced chunks.
//! - [`mod2_sys`]: linear system solving over **F**₂.
//! - [`select_in_word`]: select-in-word primitives.
//! - [`Mwc192`]: a multiply-with-carry pseudo-random number generator.
//! - [`transmute_vec_into_atomic`] / [`transmute_vec_from_atomic`]: safe
//!   transmutation between non-atomic and atomic vectors.
//! - [`prefetch_index`]: cache-line prefetching for indexed data structures.

use atomic_primitive::{Atomic, AtomicPrimitive, PrimitiveAtomic};
use num_primitive::PrimitiveUnsigned;

pub mod lenders;
pub use lenders::*;

pub mod sig_store;
use rand::{Rng, SeedableRng, TryRng};
pub use sig_store::*;

pub mod fair_chunks;
pub use fair_chunks::FairChunks;

pub mod mod2_sys;
pub use mod2_sys::*;

pub mod select_in_word;
pub use select_in_word::*;

/// An error type raised when attempting to cast a non-atomic type to an atomic
/// type with incompatible alignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CannotCastToAtomicError<T>(core::marker::PhantomData<T>);

impl<T: AtomicPrimitive> Default for CannotCastToAtomicError<T> {
    fn default() -> Self {
        CannotCastToAtomicError(core::marker::PhantomData)
    }
}

impl<T: AtomicPrimitive> core::fmt::Display for CannotCastToAtomicError<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        assert_ne!(
            core::mem::align_of::<T>(),
            core::mem::align_of::<Atomic<T>>()
        );
        write!(
            f,
            "Cannot cast {} (align_of: {}) to atomic type {} (align_of: {}) because they have incompatible alignments",
            core::any::type_name::<T>(),
            core::mem::align_of::<T>(),
            core::any::type_name::<Atomic<T>>(),
            core::mem::align_of::<Atomic<T>>()
        )
    }
}

impl<T: AtomicPrimitive + core::fmt::Debug> core::error::Error for CannotCastToAtomicError<T> {}

/// Transmutes a vector of elements of non-atomic type into a vector of elements
/// of the associated atomic type.
///
/// [It is not safe to transmute a
/// vector](https://doc.rust-lang.org/std/mem/fn.transmute.html). This method
/// implements a correct transmutation of the vector content.
///
/// Since the alignment of the atomic type might be greater than that of the
/// non-atomic type, we can only perform a direct transmutation when the
/// alignment of the atomic type is greater than or equal to that of the
/// non-atomic type. In this case, we simply reinterpret the vector's pointer.
///
/// Otherwise, we fall back to a safe but less efficient method that allocates a
/// new vector and copies the elements one by one. The compiler might be able
/// to optimize this case away in some situations.
pub fn transmute_vec_into_atomic<W: AtomicPrimitive>(v: Vec<W>) -> Vec<Atomic<W>> {
    if core::mem::align_of::<Atomic<W>>() == core::mem::align_of::<W>() {
        let mut v = std::mem::ManuallyDrop::new(v);
        unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut Atomic<W>, v.len(), v.capacity()) }
    } else {
        v.into_iter().map(W::to_atomic).collect()
    }
}

/// Transmutes a vector of elements of atomic type into a vector of elements of
/// the associated non-atomic type.
///
/// [It is not safe to transmute a
/// vector](https://doc.rust-lang.org/std/mem/fn.transmute.html). This method
/// implements a correct transmutation of the vector content.
pub fn transmute_vec_from_atomic<A: PrimitiveAtomic>(v: Vec<A>) -> Vec<A::Value> {
    let mut v = std::mem::ManuallyDrop::new(v);
    // this is always safe because atomic types have bigger or equal alignment
    // than their non-atomic counterparts
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut A::Value, v.len(), v.capacity()) }
}

/// Transmutes a boxed slice of elements of non-atomic type into a boxed slice
/// of elements of the associated atomic type.
///
/// See [`transmute_vec_into_atomic`] for details.
pub fn transmute_boxed_slice_into_atomic<W: AtomicPrimitive + Copy>(
    b: Box<[W]>,
) -> Box<[Atomic<W>]> {
    if core::mem::align_of::<Atomic<W>>() == core::mem::align_of::<W>() {
        let mut b = std::mem::ManuallyDrop::new(b);
        unsafe { Box::from_raw(b.as_mut() as *mut [W] as *mut [Atomic<W>]) }
    } else {
        IntoIterator::into_iter(b).map(W::to_atomic).collect()
    }
}

/// Transmutes a boxed slice of values of atomic type into a boxed slice of
/// values of the associated non-atomic type.
///
/// [It is not safe to transmute a
/// vector](https://doc.rust-lang.org/std/mem/fn.transmute.html). This method
/// implements a correct transmutation of the vector content.
pub fn transmute_boxed_slice_from_atomic<A: PrimitiveAtomic>(b: Box<[A]>) -> Box<[A::Value]> {
    let mut b = std::mem::ManuallyDrop::new(b);
    unsafe { Box::from_raw(b.as_mut() as *mut [A] as *mut [A::Value]) }
}

/// A multiply-with-carry pseudo-random number generator with 192 bits of
/// state.
pub struct Mwc192 {
    x: u64,
    y: u64,
    c: u64,
}

impl Mwc192 {
    const MWC_A2: u64 = 0xffa04e67b3c95d86;
}

impl TryRng for Mwc192 {
    type Error = core::convert::Infallible;
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let result = self.y;
        let t = (Self::MWC_A2 as u128)
            .wrapping_mul(self.x as u128)
            .wrapping_add(self.c as u128);
        self.x = self.y;
        self.y = t as u64;
        self.c = (t >> 64) as u64;
        Ok(result)
    }

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.next_u64() as u32)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let mut left = dst;
        while left.len() >= 8 {
            let (l, r) = { left }.split_at_mut(8);
            left = r;
            let chunk: [u8; 8] = self.next_u64().to_le_bytes();
            l.copy_from_slice(&chunk);
        }
        let n = left.len();
        if n > 4 {
            let chunk: [u8; 8] = self.next_u64().to_le_bytes();
            left.copy_from_slice(&chunk[..n]);
        } else if n > 0 {
            let chunk: [u8; 4] = self.next_u32().to_le_bytes();
            left.copy_from_slice(&chunk[..n]);
        }

        Ok(())
    }
}

impl SeedableRng for Mwc192 {
    type Seed = [u8; 16];

    fn from_seed(seed: Self::Seed) -> Self {
        let mut s0 = [0; 8];
        s0.copy_from_slice(&seed[..8]);
        let mut s1 = [0; 8];
        s1.copy_from_slice(&seed[8..]);

        Mwc192 {
            x: u64::from_ne_bytes(s0),
            y: u64::from_ne_bytes(s1),
            c: 1,
        }
    }
}

/// Prefetches the cache line containing (the first byte of) `data[index]` into
/// all levels of the cache.
#[inline(always)]
pub fn prefetch_index<T>(data: impl AsRef<[T]>, index: usize) {
    let ptr = data.as_ref().as_ptr().wrapping_add(index) as *const i8;
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(all(target_arch = "x86", target_feature = "sse"))]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(all(target_arch = "aarch64", feature = "aarch64_prefetch"))]
    unsafe {
        std::arch::aarch64::_prefetch::<
            { std::arch::aarch64::_PREFETCH_READ },
            { std::arch::aarch64::_PREFETCH_LOCALITY3 },
        >(ptr);
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "x86", target_feature = "sse"),
        all(target_arch = "aarch64", feature = "aarch64_prefetch")
    )))]
    {
        let _ = ptr; // Silence unused variable warning.
    }
}

/// Extension trait for [`PrimitiveUnsigned`] types.
pub trait PrimitiveUnsignedExt {
    /// Returns the number of bits necessary to represent an unsigned integer
    /// value.
    ///
    /// This is one for zero; otherwise, it is equal to `ilog2(self) + 1`.
    ///
    /// ```
    /// # use sux::utils::PrimitiveUnsignedExt;
    /// assert_eq!(0_u64.bit_len(), 1);
    /// assert_eq!(1_u64.bit_len(), 1);
    /// assert_eq!(2_u64.bit_len(), 2);
    /// assert_eq!(3_u64.bit_len(), 2);
    /// assert_eq!(4_u64.bit_len(), 3);
    /// assert_eq!(255_u64.bit_len(), 8);
    /// assert_eq!(256_u64.bit_len(), 9);
    /// ```
    fn bit_len(self) -> u32;
}

impl<T: PrimitiveUnsigned> PrimitiveUnsignedExt for T {
    #[inline(always)]
    fn bit_len(self) -> u32 {
        if self == T::MIN { 1 } else { self.ilog2() + 1 }
    }
}
