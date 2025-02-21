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
