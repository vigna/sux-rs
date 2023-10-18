/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Custom conversion trait.

use anyhow::Result;
use std::sync::atomic::*;

/// Like [`Into`], but we need to avoid the orphan rule and error
/// [E0210](https://github.com/rust-lang/rust/blob/master/compiler/rustc_error_codes/src/error_codes/E0210.md).
///
/// We provide implementations performing conversions between vectors and (mutable) references
/// to slices of atomic and non-atomic unsigned integers.
///
/// Other structures, such as [`EliasFano`](crate::dict::elias_fano::EliasFano),
/// use this trait to add features to a basic implementation.
///
/// Note that this trait is intentionally non-reflexive, that is, it does not provide
/// a blanket implementation of `ConvertTo<A>` for `A`.
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

convert_to!(u8, AtomicU8);
convert_to!(u16, AtomicU16);
convert_to!(u32, AtomicU32);
convert_to!(u64, AtomicU64);
convert_to!(usize, AtomicUsize);
