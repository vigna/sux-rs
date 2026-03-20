/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::utils::SelectInWord;

/// Verifies `select_in_word` and `select_zero_in_word` for a single value,
/// given its u128 representation.
fn verify(w: &impl SelectInWord, v128: u128, bits: u32) {
    let ones = v128.count_ones() as usize;
    let zeros = bits as usize - ones;

    for rank in 0..ones {
        let pos = w.select_in_word(rank);
        assert!(
            (pos as u32) < bits,
            "select_in_word({v128:#x}, {rank}) = {pos} out of range"
        );
        let mask128 = if pos == 0 { 0u128 } else { (1u128 << pos) - 1 };
        let preceding = (v128 & mask128).count_ones() as usize;
        assert_eq!(
            preceding, rank,
            "select_in_word({v128:#x}, {rank}) = {pos}: preceding ones = {preceding}"
        );
        assert_ne!(
            v128 & (1u128 << pos),
            0,
            "select_in_word({v128:#x}, {rank}) = {pos}: bit not set"
        );
    }

    for rank in 0..zeros {
        let pos = w.select_zero_in_word(rank);
        assert!(
            (pos as u32) < bits,
            "select_zero_in_word({v128:#x}, {rank}) = {pos} out of range"
        );
        let mask128 = if pos == 0 { 0u128 } else { (1u128 << pos) - 1 };
        let preceding_zeros = pos - (v128 & mask128).count_ones() as usize;
        assert_eq!(
            preceding_zeros, rank,
            "select_zero_in_word({v128:#x}, {rank}) = {pos}: preceding zeros = {preceding_zeros}"
        );
        assert_eq!(
            v128 & (1u128 << pos),
            0,
            "select_zero_in_word({v128:#x}, {rank}) = {pos}: bit is set"
        );
    }
}

#[test]
fn test_select_in_word_u8() {
    for v in 0..=u8::MAX {
        verify(&v, v as u128, u8::BITS);
    }
}

#[test]
fn test_select_in_word_u16() {
    for v in 0..=u16::MAX {
        verify(&v, v as u128, u16::BITS);
    }
}

#[test]
fn test_select_in_word_u32() {
    let mut values: Vec<u32> = vec![
        0, 1, 2,
        0x8000_0000,
        u32::MAX,
        0xAAAA_AAAA,
        0x5555_5555,
        0x0F0F_0F0F,
        0xF0F0_F0F0,
        0x0000_FFFF,
        0xFFFF_0000,
        0x8000_0001,
        0x1234_5678,
        0xDEAD_BEEF,
        0x7FFF_FFFF,
    ];
    for i in 0..32 {
        values.push(1u32 << i);
        values.push(!(1u32 << i));
    }
    for v in values {
        verify(&v, v as u128, u32::BITS);
    }
}

#[test]
fn test_select_in_word_u64() {
    let mut values: Vec<u64> = vec![
        0, 1, 2,
        0x8000_0000_0000_0000,
        u64::MAX,
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0x0F0F_0F0F_0F0F_0F0F,
        0xF0F0_F0F0_F0F0_F0F0,
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x8000_0000_0000_0001,
        0x1234_5678_9ABC_DEF0,
        0xDEAD_BEEF_CAFE_BABE,
        0x8000_0000_8000_0000,
    ];
    for i in 0..64 {
        values.push(1u64 << i);
        values.push(!(1u64 << i));
    }
    for v in values {
        verify(&v, v as u128, u64::BITS);
    }
}

#[test]
fn test_select_in_word_u128() {
    let mut values: Vec<u128> = vec![
        0, 1, 2,
        1u128 << 127,
        u128::MAX,
        0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555_5555_5555_5555_5555,
        0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F,
        0xF0F0_F0F0_F0F0_F0F0_F0F0_F0F0_F0F0_F0F0,
        0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF_0000_0000_0000_0000,
        // Cross the u64 boundary
        0x0000_0000_0000_0001_8000_0000_0000_0000,
        0x8000_0000_0000_0000_0000_0000_0000_0001,
        0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0,
    ];
    for i in 0..128 {
        values.push(1u128 << i);
        values.push(!(1u128 << i));
    }
    for v in values {
        verify(&v, v, u128::BITS);
    }
}

#[test]
fn test_select_in_word_usize() {
    let mut values: Vec<usize> = vec![
        0, 1, 2,
        1usize << (usize::BITS - 1),
        usize::MAX,
        0x5555_5555_5555_5555usize,
        0xAAAA_AAAA_AAAA_AAAAusize,
        0x0F0F_0F0F_0F0F_0F0Fusize,
    ];
    for i in 0..usize::BITS {
        values.push(1usize << i);
        values.push(!(1usize << i));
    }
    for v in values {
        verify(&v, v as u128, usize::BITS);
    }
}
