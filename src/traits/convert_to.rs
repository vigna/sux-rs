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

macro_rules! convert_to {
    ($std:ty, $atomic:ty) => {
        impl ConvertTo<Vec<$std>> for Vec<$std> {
            #[inline(always)]
            fn convert_to(self) -> Result<Self> {
                Ok(self)
            }
        }

        impl ConvertTo<Vec<$std>> for Vec<$atomic> {
            #[inline(always)]
            fn convert_to(self) -> Result<Vec<$std>> {
                Ok(unsafe { std::mem::transmute::<Vec<$atomic>, Vec<$std>>(self) })
            }
        }
        impl ConvertTo<Vec<$atomic>> for Vec<$std> {
            #[inline(always)]
            fn convert_to(self) -> Result<Vec<$atomic>> {
                Ok(unsafe { std::mem::transmute::<Vec<$std>, Vec<$atomic>>(self) })
            }
        }
        impl<'a> ConvertTo<&'a [$atomic]> for &'a [$std] {
            #[inline(always)]
            fn convert_to(self) -> Result<&'a [$atomic]> {
                Ok(unsafe { std::mem::transmute::<&'a [$std], &'a [$atomic]>(self) })
            }
        }
        impl<'a> ConvertTo<&'a [$std]> for &'a [$atomic] {
            #[inline(always)]
            fn convert_to(self) -> Result<&'a [$std]> {
                Ok(unsafe { std::mem::transmute::<&'a [$atomic], &'a [$std]>(self) })
            }
        }
        impl<'a> ConvertTo<&'a mut [$atomic]> for &'a mut [$std] {
            #[inline(always)]
            fn convert_to(self) -> Result<&'a mut [$atomic]> {
                Ok(unsafe { std::mem::transmute::<&'a mut [$std], &'a mut [$atomic]>(self) })
            }
        }
        impl<'a> ConvertTo<&'a mut [$std]> for &'a mut [$atomic] {
            #[inline(always)]
            fn convert_to(self) -> Result<&'a mut [$std]> {
                Ok(unsafe { std::mem::transmute::<&'a mut [$atomic], &'a mut [$std]>(self) })
            }
        }
    };
}

convert_to!(u64, AtomicU64);
convert_to!(usize, AtomicUsize);
