/*
 *
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
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
//! - [`Mwc192`]: a ridiculously fast multiply-with-carry pseudo-random number generator.
//! - [`transmute_vec_into_atomic`] / [`transmute_vec_from_atomic`]: safe
//!   transmutation between non-atomic and atomic vectors.
//! - [`prefetch_index`]: cache-line prefetching for indexed data structures.
//! - [`lcp_len`]: SIMD-friendly length of the longest common prefix between
//!   two byte slices.

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

/// The types involved in a failed reinterpretation of a memory buffer, and
/// their alignments.
///
/// This is the payload of [`DifferentAlignmentError`] and of
/// [`InsufficientAlignmentError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlignmentInfo {
    from: &'static str,
    from_align: usize,
    to: &'static str,
    to_align: usize,
}

impl AlignmentInfo {
    /// Returns the information about a conversion from a buffer of elements of
    /// type `F` to a buffer of elements of type `T`.
    pub fn new<F, T>() -> Self {
        Self {
            from: core::any::type_name::<F>(),
            from_align: core::mem::align_of::<F>(),
            to: core::any::type_name::<T>(),
            to_align: core::mem::align_of::<T>(),
        }
    }

    /// Returns the name of the type the conversion started from, and its
    /// alignment.
    pub fn from_type(&self) -> (&'static str, usize) {
        (self.from, self.from_align)
    }

    /// Returns the name of the type the conversion was directed to, and its
    /// alignment.
    pub fn to_type(&self) -> (&'static str, usize) {
        (self.to, self.to_align)
    }
}

/// An error raised when an owned memory buffer cannot be reinterpreted as a
/// buffer of a different type because the two types have different alignments.
///
/// Reinterpreting the buffer of a [`Vec`] or of a boxed slice is possible only
/// if the two element types have the *same* alignment, as deallocating a buffer
/// with an alignment different from the one it was allocated with is undefined
/// behavior. When only a reference to the buffer is reinterpreted there is no
/// deallocation involved, and the weaker condition of
/// [`InsufficientAlignmentError`] applies.
///
/// Since the conversions that raise this error consume their input, the error
/// gives it back in its second field: no data is lost, and the caller can fall
/// back to copying the content. For example,
/// ```
/// # use sux::utils::transmute_vec_from_atomic;
/// # use core::sync::atomic::AtomicUsize;
/// let atomic: Vec<AtomicUsize> = (0..10).map(AtomicUsize::new).collect();
/// let v: Vec<usize> = match transmute_vec_from_atomic(atomic) {
///     // The allocation has been reused.
///     Ok(v) => v,
///     // The alignments differ: we get the vector back and copy its content.
///     Err(error) => error.1.into_iter().map(AtomicUsize::into_inner).collect(),
/// };
/// assert_eq!(v, (0..10).collect::<Vec<_>>());
/// ```
///
/// Note that the [`Debug`](core::fmt::Debug) implementation displays the
/// alignment information only, as the returned input might be very large.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DifferentAlignmentError<V = ()>(pub AlignmentInfo, pub V);

impl<V> DifferentAlignmentError<V> {
    /// Returns the error for a conversion from a buffer of elements of type `F`
    /// to a buffer of elements of type `T`, returning `value` to the caller.
    pub fn new<F, T>(value: V) -> Self {
        Self(AlignmentInfo::new::<F, T>(), value)
    }
}

impl<V> core::fmt::Debug for DifferentAlignmentError<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("DifferentAlignmentError")
            .field(&self.0)
            .finish()
    }
}

impl<V> core::fmt::Display for DifferentAlignmentError<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cannot reinterpret an owned buffer of `{}` (alignment {}) as a buffer of `{}` (alignment {}): an owned buffer must be deallocated with the same alignment it was allocated with, so the two alignments must be equal",
            self.0.from, self.0.from_align, self.0.to, self.0.to_align
        )
    }
}

impl<V> core::error::Error for DifferentAlignmentError<V> {}

/// An error raised when a memory buffer cannot be reinterpreted as a buffer of
/// a different type because the latter has a stricter alignment.
///
/// This is the condition for reinterpreting a *reference* to a buffer: since no
/// deallocation is involved, the alignment of the target type just has to be
/// less than or equal to that of the source type, whose alignment the buffer is
/// known to satisfy. For an owned buffer the stronger condition of
/// [`DifferentAlignmentError`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InsufficientAlignmentError(pub AlignmentInfo);

impl InsufficientAlignmentError {
    /// Returns the error for a conversion from a buffer of elements of type `F`
    /// to a buffer of elements of type `T`.
    pub fn new<F, T>() -> Self {
        Self(AlignmentInfo::new::<F, T>())
    }
}

impl core::fmt::Display for InsufficientAlignmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cannot reinterpret a buffer of `{}` (alignment {}) as a buffer of `{}` (alignment {}): the target type requires a stricter alignment",
            self.0.from, self.0.from_align, self.0.to, self.0.to_align
        )
    }
}

impl core::error::Error for InsufficientAlignmentError {}

/// Transmutes a vector of elements of non-atomic type into a vector of elements
/// of the associated atomic type.
///
/// [It is not safe to transmute a vector]. This method implements a correct
/// transmutation of the vector content.
///
/// The transmutation is possible only if the alignment of the atomic type is
/// equal to that of the non-atomic type: deallocating the buffer with an
/// alignment different from the one it was allocated with would be undefined
/// behavior. In this case, we simply reinterpret the vector's pointer.
/// Otherwise, the method returns a [`DifferentAlignmentError`] containing the
/// vector, so that the caller can fall back to copying its content.
///
/// [It is not safe to transmute a vector]: https://doc.rust-lang.org/std/mem/fn.transmute.html
pub fn transmute_vec_into_atomic<W: AtomicPrimitive>(
    v: Vec<W>,
) -> Result<Vec<Atomic<W>>, DifferentAlignmentError<Vec<W>>> {
    if core::mem::align_of::<Atomic<W>>() != core::mem::align_of::<W>() {
        return Err(DifferentAlignmentError::new::<W, Atomic<W>>(v));
    }
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: atomic types have the same in-memory representation as their
    // value type, and the alignments are equal (just checked)
    Ok(unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut Atomic<W>, v.len(), v.capacity()) })
}

/// Transmutes a vector of elements of atomic type into a vector of elements of
/// the associated non-atomic type.
///
/// [It is not safe to transmute a vector]. This method implements a correct
/// transmutation of the vector content.
///
/// The transmutation is possible only if the alignment of the atomic type is
/// equal to that of the non-atomic type: deallocating the buffer with an
/// alignment different from the one it was allocated with would be undefined
/// behavior. In this case, we simply reinterpret the vector's pointer.
/// Otherwise, the method returns a [`DifferentAlignmentError`] containing the
/// vector, so that the caller can fall back to copying its content.
///
/// [It is not safe to transmute a vector]: https://doc.rust-lang.org/std/mem/fn.transmute.html
pub fn transmute_vec_from_atomic<A: PrimitiveAtomic>(
    v: Vec<A>,
) -> Result<Vec<A::Value>, DifferentAlignmentError<Vec<A>>> {
    if core::mem::align_of::<A>() != core::mem::align_of::<A::Value>() {
        return Err(DifferentAlignmentError::new::<A, A::Value>(v));
    }
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: atomic types have the same in-memory representation as
    // their value type, and the alignments are equal (just checked)
    Ok(unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut A::Value, v.len(), v.capacity()) })
}

/// Transmutes a boxed slice of elements of non-atomic type into a boxed slice
/// of elements of the associated atomic type.
///
/// See [`transmute_vec_into_atomic`] for details.
pub fn transmute_boxed_slice_into_atomic<W: AtomicPrimitive>(
    b: Box<[W]>,
) -> Result<Box<[Atomic<W>]>, DifferentAlignmentError<Box<[W]>>> {
    if core::mem::align_of::<Atomic<W>>() != core::mem::align_of::<W>() {
        return Err(DifferentAlignmentError::new::<W, Atomic<W>>(b));
    }
    let mut b = std::mem::ManuallyDrop::new(b);
    // SAFETY: atomic types have the same in-memory representation as their
    // value type, and the alignments are equal (just checked)
    Ok(unsafe { Box::from_raw(b.as_mut() as *mut [W] as *mut [Atomic<W>]) })
}

/// Transmutes a boxed slice of values of atomic type into a boxed slice of
/// values of the associated non-atomic type.
///
/// See [`transmute_vec_from_atomic`] for details.
pub fn transmute_boxed_slice_from_atomic<A: PrimitiveAtomic>(
    b: Box<[A]>,
) -> Result<Box<[A::Value]>, DifferentAlignmentError<Box<[A]>>> {
    if core::mem::align_of::<A>() != core::mem::align_of::<A::Value>() {
        return Err(DifferentAlignmentError::new::<A, A::Value>(b));
    }
    let mut b = std::mem::ManuallyDrop::new(b);
    // SAFETY: atomic types have the same in-memory representation as
    // their value type, and the alignments are equal (just checked)
    Ok(unsafe { Box::from_raw(b.as_mut() as *mut [A] as *mut [A::Value]) })
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

/// Returns the length of the longest common prefix (in bytes) between two
/// byte slices. This is the byte position of the first mismatch, or the
/// length of the shorter slice if one is a prefix of the other.
///
/// Uses `chunks_exact(128)` to enable SIMD auto-vectorization in the common
/// case where both slices share a long common prefix.
#[inline]
pub fn lcp_len(xs: &[u8], ys: &[u8]) -> usize {
    let off = std::iter::zip(xs.chunks_exact(128), ys.chunks_exact(128))
        .take_while(|(x, y)| x == y)
        .count()
        * 128;
    off + std::iter::zip(&xs[off..], &ys[off..])
        .take_while(|(x, y)| x == y)
        .count()
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
