/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Utilities for selecting bits in a word.

/// Select the i-th 1-bit or 0-bit in a word of memory.
/// ```
/// use sux::utils::SelectInWord;
///
/// assert_eq!(0b1_u64.select_in_word(0), 0);
/// assert_eq!(0b11_u64.select_in_word(1), 1);
/// assert_eq!(0b101_u64.select_in_word(1), 2);
/// assert_eq!(0x8000_0000_0000_0000_u64.select_in_word(0), 63);
/// assert_eq!(0x8000_0000_8000_0000_u64.select_in_word(0), 31);
/// assert_eq!(0x8000_0000_8000_0000_u64.select_in_word(1), 63);
/// ```
pub trait SelectInWord: core::ops::Not<Output = Self> + Sized + Copy {
    fn select_in_word(&self, rank: usize) -> usize;

    #[inline(always)]
    fn select_zero_in_word(&self, rank: usize) -> usize {
        (!*self).select_in_word(rank)
    }
}

impl SelectInWord for u8 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        debug_assert!(rank < self.count_ones() as _);
        let index = *self as usize | (rank << 8);
        SELECT_IN_BYTE[index] as usize
    }
}

#[cfg(not(target_feature = "bmi2"))]
macro_rules! popcount_check_byte {
    ($bytes:expr, $remaining:ident, $idx:expr) => {
        let ones = POPCOUNT[$bytes[$idx] as usize] as usize;
        if $remaining < ones {
            return $idx * 8 + SELECT_IN_BYTE[$bytes[$idx] as usize | ($remaining << 8)] as usize;
        }
        $remaining -= ones;
    };
}

#[cfg(not(target_feature = "bmi2"))]
const POPCOUNT: [u8; 256] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
];

impl SelectInWord for u16 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        debug_assert!(rank < self.count_ones() as _);
        #[cfg(target_feature = "bmi2")]
        {
            (*self as u32).select_in_word(rank)
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            let bytes = self.to_le_bytes();
            let mut remaining = rank;
            popcount_check_byte!(bytes, remaining, 0);
            8 + SELECT_IN_BYTE[bytes[1] as usize | (remaining << 8)] as usize
        }
    }
}

impl SelectInWord for u32 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        debug_assert!(rank < self.count_ones() as _);
        #[cfg(target_feature = "bmi2")]
        {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_pdep_u32;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_pdep_u32;
            let mask = 1 << rank;
            let one = unsafe { _pdep_u32(mask, *self) };
            one.trailing_zeros() as usize
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            let bytes = self.to_le_bytes();
            let mut remaining = rank;
            popcount_check_byte!(bytes, remaining, 0);
            popcount_check_byte!(bytes, remaining, 1);
            popcount_check_byte!(bytes, remaining, 2);
            24 + SELECT_IN_BYTE[bytes[3] as usize | (remaining << 8)] as usize
        }
    }
}

impl SelectInWord for u64 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        debug_assert!(rank < self.count_ones() as _);
        #[cfg(all(target_feature = "bmi2", target_arch = "x86_64"))]
        {
            use core::arch::x86_64::_pdep_u64;
            let mask = 1 << rank;
            let one = unsafe { _pdep_u64(mask, *self) };
            one.trailing_zeros() as usize
        }
        #[cfg(all(target_feature = "bmi2", not(target_arch = "x86_64")))]
        {
            let lower_ones = (*self as u32).count_ones() as usize;
            if rank < lower_ones {
                (*self as u32).select_in_word(rank)
            } else {
                32 + ((*self >> 32) as u32).select_in_word(rank - lower_ones)
            }
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            let bytes = self.to_le_bytes();
            let lower_ones = (*self as u32).count_ones() as usize;
            if rank < lower_ones {
                let mut remaining = rank;
                popcount_check_byte!(bytes, remaining, 0);
                popcount_check_byte!(bytes, remaining, 1);
                popcount_check_byte!(bytes, remaining, 2);
                24 + SELECT_IN_BYTE[bytes[3] as usize | (remaining << 8)] as usize
            } else {
                let mut remaining = rank - lower_ones;
                popcount_check_byte!(bytes, remaining, 4);
                popcount_check_byte!(bytes, remaining, 5);
                popcount_check_byte!(bytes, remaining, 6);
                56 + SELECT_IN_BYTE[bytes[7] as usize | (remaining << 8)] as usize
            }
        }
    }
}

impl SelectInWord for u128 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        debug_assert!(rank < self.count_ones() as _);
        #[cfg(target_feature = "bmi2")]
        {
            let ones = (*self as u64).count_ones() as usize;
            if ones > rank {
                (*self as u64).select_in_word(rank)
            } else {
                64 + ((*self >> 64) as u64).select_in_word(rank - ones)
            }
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            let bytes = self.to_le_bytes();
            let lower_ones = (*self as u64).count_ones() as usize;
            if rank < lower_ones {
                let lower_lower_ones = (*self as u32).count_ones() as usize;
                if rank < lower_lower_ones {
                    let mut remaining = rank;
                    popcount_check_byte!(bytes, remaining, 0);
                    popcount_check_byte!(bytes, remaining, 1);
                    popcount_check_byte!(bytes, remaining, 2);
                    24 + SELECT_IN_BYTE[bytes[3] as usize | (remaining << 8)] as usize
                } else {
                    let mut remaining = rank - lower_lower_ones;
                    popcount_check_byte!(bytes, remaining, 4);
                    popcount_check_byte!(bytes, remaining, 5);
                    popcount_check_byte!(bytes, remaining, 6);
                    56 + SELECT_IN_BYTE[bytes[7] as usize | (remaining << 8)] as usize
                }
            } else {
                let upper = (*self >> 64) as u64;
                let upper_lower_ones = (upper as u32).count_ones() as usize;
                let remaining_in_upper = rank - lower_ones;
                if remaining_in_upper < upper_lower_ones {
                    let mut remaining = remaining_in_upper;
                    popcount_check_byte!(bytes, remaining, 8);
                    popcount_check_byte!(bytes, remaining, 9);
                    popcount_check_byte!(bytes, remaining, 10);
                    88 + SELECT_IN_BYTE[bytes[11] as usize | (remaining << 8)] as usize
                } else {
                    let mut remaining = remaining_in_upper - upper_lower_ones;
                    popcount_check_byte!(bytes, remaining, 12);
                    popcount_check_byte!(bytes, remaining, 13);
                    popcount_check_byte!(bytes, remaining, 14);
                    120 + SELECT_IN_BYTE[bytes[15] as usize | (remaining << 8)] as usize
                }
            }
        }
    }
}

#[cfg(target_pointer_width = "16")]
type Usize = u16;
#[cfg(target_pointer_width = "32")]
type Usize = u32;
#[cfg(target_pointer_width = "64")]
type Usize = u64;

impl SelectInWord for usize {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        (*self as Usize).select_in_word(rank)
    }
}

#[allow(clippy::all)]
const SELECT_IN_BYTE: [u8; 2048] = [
    8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    8, 8, 8, 1, 8, 2, 2, 1, 8, 3, 3, 1, 3, 2, 2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 7, 7, 1, 7, 2, 2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2,
    8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 7, 8, 7, 7, 2, 8, 7, 7, 3, 7, 3, 3, 2, 8, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2,
    8, 7, 7, 5, 7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2,
    7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6, 6, 4, 6, 4, 4, 3,
    8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3,
    8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4, 6, 4, 4, 3,
    8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 4,
    8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5, 8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4,
    8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6,
    8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
];
