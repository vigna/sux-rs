/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Utility traits and implementations.

pub mod lenders;
pub use crate::utils::lenders::*;

pub mod sig_store;
pub use crate::utils::sig_store::*;

pub mod spooky;
pub use crate::utils::spooky::*;

pub unsafe fn transmute_vec<S, D>(v: Vec<S>) -> Vec<D> {
    // Ensure the original vector is not dropped
    let mut v = std::mem::ManuallyDrop::new(v);
    Vec::from_raw_parts(v.as_mut_ptr() as *mut D, v.len(), v.capacity())
}

pub unsafe fn transmute_boxed_slice<S, D>(b: Box<[S]>) -> Box<[D]> {
    // Ensure the original boxed value is not dropped
    let mut b = std::mem::ManuallyDrop::new(b);
    Box::from_raw(b.as_mut() as *mut [S] as *mut [D])
}
