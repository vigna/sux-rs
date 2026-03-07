/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

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

impl SelectInWord for u16 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        #[cfg(target_feature = "bmi2")]
        {
            (*self as u64).select_in_word(rank)
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            const ONES_STEP_1: u16 = 0x1111;
            const ONES_STEP_2: u16 = 0x0101;
            const LAMBDAS_STEP_2: u16 = 0x80 * ONES_STEP_2;

            let mut s = *self;
            s = s - ((s & (0xA * ONES_STEP_1)) >> 1);
            s = (s & (0x3 * ONES_STEP_1)) + ((s >> 2) & (0x3 * ONES_STEP_1));
            s = (s + (s >> 4)) & (0xF * ONES_STEP_2);
            let byte_sums: u16 = s.wrapping_mul(ONES_STEP_2);

            let rank_step_8: u16 = rank as u16 * ONES_STEP_2;
            let geq_rank_step_8: u16 =
                ((rank_step_8 | LAMBDAS_STEP_2) - byte_sums) & LAMBDAS_STEP_2;
            let place = (geq_rank_step_8.count_ones() * 8) as usize;
            let byte_rank: u16 = rank as u16 - (((byte_sums << 8) >> place) & 0xFF_u16);
            let index = ((*self >> place) & 0xFF) | (byte_rank << 8);
            place + SELECT_IN_BYTE[index as usize] as usize
        }
    }
}

impl SelectInWord for u32 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        #[cfg(target_feature = "bmi2")]
        {
            (*self as u64).select_in_word(rank)
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            const ONES_STEP_2: u32 = 0x11111111;
            const ONES_STEP_4: u32 = 0x01010101;
            const LAMBDAS_STEP_4: u32 = 0x80 * ONES_STEP_4;

            let mut s = *self;
            s = s - ((s & (0xA * ONES_STEP_2)) >> 1);
            s = (s & (0x3 * ONES_STEP_2)) + ((s >> 2) & (0x3 * ONES_STEP_2));
            s = (s + (s >> 4)) & (0xF * ONES_STEP_4);
            let byte_sums: u32 = s.wrapping_mul(ONES_STEP_4);

            let rank_step_8: u32 = rank as u32 * ONES_STEP_4;
            let geq_rank_step_8: u32 =
                ((rank_step_8 | LAMBDAS_STEP_4) - byte_sums) & LAMBDAS_STEP_4;
            let place = (geq_rank_step_8.count_ones() * 8) as usize;
            let byte_rank: u32 = rank as u32 - (((byte_sums << 8) >> place) & 0xFF_u32);
            let index = ((*self >> place) & 0xFF) | (byte_rank << 8);
            place + SELECT_IN_BYTE[index as usize] as usize
        }
    }
}

impl SelectInWord for u64 {
    #[inline(always)]
    fn select_in_word(&self, rank: usize) -> usize {
        debug_assert!(rank < self.count_ones() as _);
        #[cfg(target_feature = "bmi2")]
        {
            use core::arch::x86_64::_pdep_u64;
            let mask = 1 << rank;
            let one = unsafe { _pdep_u64(mask, *self) };
            one.trailing_zeros() as usize
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            const ONES_STEP_4: u64 = 0x1111111111111111;
            const ONES_STEP_8: u64 = 0x0101010101010101;
            const LAMBDAS_STEP_8: u64 = 0x80 * ONES_STEP_8;

            let mut s = *self;
            s = s - ((s & (0xA * ONES_STEP_4)) >> 1);
            s = (s & (0x3 * ONES_STEP_4)) + ((s >> 2) & (0x3 * ONES_STEP_4));
            s = (s + (s >> 4)) & (0xF * ONES_STEP_8);
            let byte_sums: u64 = s.wrapping_mul(ONES_STEP_8);

            let rank_step_8: u64 = rank as u64 * ONES_STEP_8;
            let geq_rank_step_8: u64 =
                ((rank_step_8 | LAMBDAS_STEP_8) - byte_sums) & LAMBDAS_STEP_8;
            let place = (geq_rank_step_8.count_ones() * 8) as usize;
            let byte_rank: u64 = rank as u64 - (((byte_sums << 8) >> place) & 0xFF_u64);
            let index = ((*self >> place) & 0xFF) | (byte_rank << 8);
            place + SELECT_IN_BYTE[index as usize] as usize
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
            const ONES_STEP_8: u128 = 0x11111111111111111111111111111111;
            const ONES_STEP_16: u128 = 0x01010101010101010101010101010101;
            const LAMBDAS_STEP_16: u128 = 0x80 * ONES_STEP_16;

            let mut s = *self;
            s = s - ((s & (0xA * ONES_STEP_8)) >> 1);
            s = (s & (0x3 * ONES_STEP_8)) + ((s >> 2) & (0x3 * ONES_STEP_8));
            s = (s + (s >> 4)) & (0xF * ONES_STEP_16);
            let byte_sums: u128 = s.wrapping_mul(ONES_STEP_16);

            let rank_step_8: u128 = rank as u128 * ONES_STEP_16;
            let geq_rank_step_8: u128 =
                ((rank_step_8 | LAMBDAS_STEP_16) - byte_sums) & LAMBDAS_STEP_16;
            let place = (geq_rank_step_8.count_ones() * 8) as usize;
            let byte_rank: u128 = rank as u128 - (((byte_sums << 8) >> place) & 0xFF_u128);
            let index = ((*self >> place) & 0xFF) | (byte_rank << 8);
            place + SELECT_IN_BYTE[index as usize] as usize
        }
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

macro_rules! impl_usize {
    ($ty:ty, $pw:literal) => {
        #[cfg(target_pointer_width = $pw)]
        impl SelectInWord for usize {
            #[inline(always)]
            fn select_in_word(&self, rank: usize) -> usize {
                (*self as $ty).select_in_word(rank) as usize
            }
        }
    };
}

impl_usize!(u16, "16");
impl_usize!(u32, "32");
impl_usize!(u64, "64");
