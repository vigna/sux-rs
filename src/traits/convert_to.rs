/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! # Traits
//! This modules contains basic traits related to succinct data structures.
//!
//! Traits are collected into a module so you can do `use sux::traits::*;`
//! for ease of use.

use anyhow::Result;
use std::sync::atomic::{AtomicU64, AtomicUsize};

/// Like [`Into`], but we need to avoid the orphan rule and error
/// [E0210](https://github.com/rust-lang/rust/blob/master/compiler/rustc_error_codes/src/error_codes/E0210.md)
///
/// Reference: <https://rust-lang.github.io/chalk/book/clauses/coherence.html>
pub trait ConvertTo<B> {
    fn convert_to(self) -> Result<B>;
}

impl ConvertTo<usize> for usize {
    #[inline(always)]
    fn convert_to(self) -> Result<Self> {
        Ok(self)
    }
}
impl ConvertTo<AtomicUsize> for AtomicUsize {
    #[inline(always)]
    fn convert_to(self) -> Result<Self> {
        Ok(self)
    }
}
impl ConvertTo<Vec<u64>> for Vec<u64> {
    #[inline(always)]
    fn convert_to(self) -> Result<Self> {
        Ok(self)
    }
}

impl ConvertTo<Vec<u64>> for Vec<AtomicU64> {
    #[inline(always)]
    fn convert_to(self) -> Result<Vec<u64>> {
        Ok(unsafe { std::mem::transmute::<Vec<AtomicU64>, Vec<u64>>(self) })
    }
}
impl ConvertTo<Vec<AtomicU64>> for Vec<u64> {
    #[inline(always)]
    fn convert_to(self) -> Result<Vec<AtomicU64>> {
        Ok(unsafe { std::mem::transmute::<Vec<u64>, Vec<AtomicU64>>(self) })
    }
}
impl<'a> ConvertTo<&'a [AtomicU64]> for &'a [u64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a [AtomicU64]> {
        Ok(unsafe { std::mem::transmute::<&'a [u64], &'a [AtomicU64]>(self) })
    }
}
impl<'a> ConvertTo<&'a [u64]> for &'a [AtomicU64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a [u64]> {
        Ok(unsafe { std::mem::transmute::<&'a [AtomicU64], &'a [u64]>(self) })
    }
}
impl<'a> ConvertTo<&'a mut [AtomicU64]> for &'a mut [u64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a mut [AtomicU64]> {
        Ok(unsafe { std::mem::transmute::<&'a mut [u64], &'a mut [AtomicU64]>(self) })
    }
}
impl<'a> ConvertTo<&'a mut [u64]> for &'a mut [AtomicU64] {
    #[inline(always)]
    fn convert_to(self) -> Result<&'a mut [u64]> {
        Ok(unsafe { std::mem::transmute::<&'a mut [AtomicU64], &'a mut [u64]>(self) })
    }
}
