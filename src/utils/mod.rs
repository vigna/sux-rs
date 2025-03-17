/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Utility traits and implementations.

pub mod lenders;
pub use lenders::*;

pub mod sig_store;
use rand::{RngCore, SeedableRng};
pub use sig_store::*;

pub mod fair_chunks;
pub use fair_chunks::FairChunks;

pub mod mod2_sys;
pub use mod2_sys::*;

/// Transmutes a vector of one type into a vector of another type.
///
/// [It is not safe to transmute a
/// vector](https://doc.rust-lang.org/std/mem/fn.transmute.html). This method
/// implements a correct transmutation of the vector content.
///
/// # Safety
///
/// The caller must ensure the source and destination types have the same size
/// and memory layout.
pub unsafe fn transmute_vec<S, D>(v: Vec<S>) -> Vec<D> {
    // Ensure the original vector is not dropped
    let mut v = std::mem::ManuallyDrop::new(v);
    Vec::from_raw_parts(v.as_mut_ptr() as *mut D, v.len(), v.capacity())
}

/// Transmutes a boxed slice of one type into a boxed slice of another type.
///
/// # Safety
///
/// The caller must ensure the source and destination types have
/// the same size and memory layout.
pub unsafe fn transmute_boxed_slice<S, D>(b: Box<[S]>) -> Box<[D]> {
    // Ensure the original boxed value is not dropped
    let mut b = std::mem::ManuallyDrop::new(b);
    Box::from_raw(b.as_mut() as *mut [S] as *mut [D])
}

pub struct Mwc192 {
    x: u64,
    y: u64,
    c: u64,
}

impl Mwc192 {
    const MWC_A2: u64 = 0xffa04e67b3c95d86;
}

impl RngCore for Mwc192 {
    fn next_u64(&mut self) -> u64 {
        let result = self.y;
        let t = (Self::MWC_A2 as u128)
            .wrapping_mul(self.x as u128)
            .wrapping_add(self.c as u128);
        self.x = self.y;
        self.y = t as u64;
        self.c = (t >> 64) as u64;
        result
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
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
