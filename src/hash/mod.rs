/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Test with http://burtleburtle.net/bob/c/froggy.cpp
//! Hashes to consider:
//! - murmur
//! - blake3
//! - highway
//!
//! highway has a good rust libray:
//! https://github.com/google/highwayhash
//! https://arxiv.org/pdf/1612.06257.pdf
//! Just use https://github.com/nickbabcock/highway-rs

pub mod spooky;

/// Automatically implement our generalized Hasher for anything that already
/// implements [`core::hash::Hasher`].
impl<T: core::hash::Hasher> Hasher for T {
    type HashType = u64;

    #[inline(always)]
    /// Just fowrard to the underlying hasher.
    fn finish(&self) -> Self::HashType {
        <Self as core::hash::Hasher>::finish(self)
    }

    #[inline(always)]
    /// Just fowrard to the underlying hasher.
    fn write(&mut self, msg: &[u8]) {
        <Self as core::hash::Hasher>::write(self, msg)
    }
}

/// A hasher that can be seeded.
pub trait SeedableHasher: core::default::Default {
    type SeedType;

    /// Creates a new hasher with the given seed.
    fn new_with_seed(seed: Self::SeedType) -> Self;
}

/// A generalization of the [`core::hash::Hasher`] trait that
/// does not require the hash to be `u64`.
pub trait Hasher {
    /// The type of the hash that this hasher produces.
    type HashType;

    /// Writes some data into this `Hasher`.
    ///
    /// See [`core::hash::Hasher`] for semantics.
    fn write(&mut self, msg: &[u8]);

    /// Returns the hash value for the values written so far.
    ///
    /// See [`core::hash::Hasher`] for semantics.
    fn finish(&self) -> Self::HashType;

    /// Writes a single `u8` into this hasher.
    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.write(&[i])
    }
    /// Writes a single `u16` into this hasher.
    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `u32` into this hasher.
    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `u64` into this hasher.
    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `u128` into this hasher.
    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `usize` into this hasher.
    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.write(&i.to_ne_bytes())
    }

    /// Writes a single `i8` into this hasher.
    #[inline(always)]
    fn write_i8(&mut self, i: i8) {
        self.write_u8(i as u8)
    }
    /// Writes a single `i16` into this hasher.
    #[inline(always)]
    fn write_i16(&mut self, i: i16) {
        self.write_u16(i as u16)
    }
    /// Writes a single `i32` into this hasher.
    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.write_u32(i as u32)
    }
    /// Writes a single `i64` into this hasher.
    #[inline(always)]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64)
    }
    /// Writes a single `i128` into this hasher.
    #[inline(always)]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128)
    }
    /// Writes a single `isize` into this hasher.
    #[inline(always)]
    fn write_isize(&mut self, i: isize) {
        self.write_usize(i as usize)
    }

    /// Writes a length prefix into this hasher, as part of being prefix-free.
    ///
    /// See [`core::hash::Hasher`] for semantics.
    #[inline(always)]
    fn write_length_prefix(&mut self, len: usize) {
        self.write_usize(len);
    }

    /// Writes a single `str` into this hasher.
    ///
    /// See [`core::hash::Hasher`] for semantics.
    #[inline(always)]
    fn write_str(&mut self, s: &str) {
        self.write(s.as_bytes());
        self.write_u8(0xff);
    }
}
