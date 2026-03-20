/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Benchmarks for select-in-word implementations.
//!
//! Each implementation family (e.g., "broadword") gets its own set of
//! benchmarks, one per word type (u8, u16, u32, u64, u128). Functions are
//! copied here rather than called through the [`SelectInWord`] trait so
//! that multiple implementations can be compared side by side.

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

const NUM_VALUES: usize = 1024;

// ──────────────────────────────────────────────────────────────────────
// Broadword select implementations
// ──────────────────────────────────────────────────────────────────────

#[inline(always)]
fn broadword_select_u16(word: u16, rank: usize) -> usize {
    const ONES_STEP_1: u16 = 0x1111;
    const ONES_STEP_2: u16 = 0x0101;
    const LAMBDAS_STEP_2: u16 = 0x80 * ONES_STEP_2;

    let mut s = word;
    s = s - ((s & (0xA * ONES_STEP_1)) >> 1);
    s = (s & (0x3 * ONES_STEP_1)) + ((s >> 2) & (0x3 * ONES_STEP_1));
    s = (s + (s >> 4)) & (0xF * ONES_STEP_2);
    let byte_sums: u16 = s.wrapping_mul(ONES_STEP_2);

    let rank_step_8: u16 = rank as u16 * ONES_STEP_2;
    let geq_rank_step_8: u16 = ((rank_step_8 | LAMBDAS_STEP_2) - byte_sums) & LAMBDAS_STEP_2;
    let place = (geq_rank_step_8.count_ones() * 8) as usize;
    let byte_rank: u16 = rank as u16 - (((byte_sums << 8) >> place) & 0xFF_u16);
    let index = ((word >> place) & 0xFF) | (byte_rank << 8);
    place + SELECT_IN_BYTE[index as usize] as usize
}

#[inline(always)]
fn broadword_select_u32(word: u32, rank: usize) -> usize {
    const ONES_STEP_2: u32 = 0x11111111;
    const ONES_STEP_4: u32 = 0x01010101;
    const LAMBDAS_STEP_4: u32 = 0x80 * ONES_STEP_4;

    let mut s = word;
    s = s - ((s & (0xA * ONES_STEP_2)) >> 1);
    s = (s & (0x3 * ONES_STEP_2)) + ((s >> 2) & (0x3 * ONES_STEP_2));
    s = (s + (s >> 4)) & (0xF * ONES_STEP_4);
    let byte_sums: u32 = s.wrapping_mul(ONES_STEP_4);

    let rank_step_8: u32 = rank as u32 * ONES_STEP_4;
    let geq_rank_step_8: u32 = ((rank_step_8 | LAMBDAS_STEP_4) - byte_sums) & LAMBDAS_STEP_4;
    let place = (geq_rank_step_8.count_ones() * 8) as usize;
    let byte_rank: u32 = rank as u32 - (((byte_sums << 8) >> place) & 0xFF_u32);
    let index = ((word >> place) & 0xFF) | (byte_rank << 8);
    place + SELECT_IN_BYTE[index as usize] as usize
}

#[inline(always)]
fn broadword_select_u64(word: u64, rank: usize) -> usize {
    const ONES_STEP_4: u64 = 0x1111111111111111;
    const ONES_STEP_8: u64 = 0x0101010101010101;
    const LAMBDAS_STEP_8: u64 = 0x80 * ONES_STEP_8;

    let mut s = word;
    s = s - ((s & (0xA * ONES_STEP_4)) >> 1);
    s = (s & (0x3 * ONES_STEP_4)) + ((s >> 2) & (0x3 * ONES_STEP_4));
    s = (s + (s >> 4)) & (0xF * ONES_STEP_8);
    let byte_sums: u64 = s.wrapping_mul(ONES_STEP_8);

    let rank_step_8: u64 = rank as u64 * ONES_STEP_8;
    let geq_rank_step_8: u64 = ((rank_step_8 | LAMBDAS_STEP_8) - byte_sums) & LAMBDAS_STEP_8;
    let place = (geq_rank_step_8.count_ones() * 8) as usize;
    let byte_rank: u64 = rank as u64 - (((byte_sums << 8) >> place) & 0xFF_u64);
    let index = ((word >> place) & 0xFF) | (byte_rank << 8);
    place + SELECT_IN_BYTE[index as usize] as usize
}

#[inline(always)]
fn broadword_select_u128(word: u128, rank: usize) -> usize {
    const ONES_STEP_8: u128 = 0x11111111111111111111111111111111;
    const ONES_STEP_16: u128 = 0x01010101010101010101010101010101;
    const LAMBDAS_STEP_16: u128 = 0x80 * ONES_STEP_16;

    let mut s = word;
    s = s - ((s & (0xA * ONES_STEP_8)) >> 1);
    s = (s & (0x3 * ONES_STEP_8)) + ((s >> 2) & (0x3 * ONES_STEP_8));
    s = (s + (s >> 4)) & (0xF * ONES_STEP_16);
    let byte_sums: u128 = s.wrapping_mul(ONES_STEP_16);

    let rank_step_8: u128 = rank as u128 * ONES_STEP_16;
    let geq_rank_step_8: u128 = ((rank_step_8 | LAMBDAS_STEP_16) - byte_sums) & LAMBDAS_STEP_16;
    let place = (geq_rank_step_8.count_ones() * 8) as usize;
    let byte_rank: u128 = rank as u128 - (((byte_sums << 8) >> place) & 0xFF_u128);
    let index = ((word >> place) & 0xFF) | (byte_rank << 8);
    place + SELECT_IN_BYTE[index as usize] as usize
}

// ──────────────────────────────────────────────────────────────────────
// Popcount-based select implementations (byte-scanning with POPCOUNT table)
// ──────────────────────────────────────────────────────────────────────

/// Check byte at index `$idx`: if the remaining rank falls within this byte,
/// return immediately via `SELECT_IN_BYTE`. Otherwise subtract and continue.
macro_rules! popcount_check_byte {
    ($bytes:expr, $remaining:ident, $idx:expr) => {
        let ones = POPCOUNT[$bytes[$idx] as usize] as usize;
        if $remaining < ones {
            return $idx * 8 + SELECT_IN_BYTE[$bytes[$idx] as usize | ($remaining << 8)] as usize;
        }
        $remaining -= ones;
    };
}

#[inline(always)]
fn popcount_select_u8(word: u8, rank: usize) -> usize {
    SELECT_IN_BYTE[word as usize | (rank << 8)] as usize
}

#[inline(always)]
fn popcount_select_u16(word: u16, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let mut remaining = rank;
    popcount_check_byte!(bytes, remaining, 0);
    8 + SELECT_IN_BYTE[bytes[1] as usize | (remaining << 8)] as usize
}

#[inline(always)]
fn popcount_select_u32(word: u32, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let mut remaining = rank;
    popcount_check_byte!(bytes, remaining, 0);
    popcount_check_byte!(bytes, remaining, 1);
    popcount_check_byte!(bytes, remaining, 2);
    24 + SELECT_IN_BYTE[bytes[3] as usize | (remaining << 8)] as usize
}

#[inline(always)]
fn popcount_select_u64(word: u64, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let mut remaining = rank;
    popcount_check_byte!(bytes, remaining, 0);
    popcount_check_byte!(bytes, remaining, 1);
    popcount_check_byte!(bytes, remaining, 2);
    popcount_check_byte!(bytes, remaining, 3);
    popcount_check_byte!(bytes, remaining, 4);
    popcount_check_byte!(bytes, remaining, 5);
    popcount_check_byte!(bytes, remaining, 6);
    56 + SELECT_IN_BYTE[bytes[7] as usize | (remaining << 8)] as usize
}

#[inline(always)]
fn popcount_select_u128(word: u128, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let mut remaining = rank;
    popcount_check_byte!(bytes, remaining, 0);
    popcount_check_byte!(bytes, remaining, 1);
    popcount_check_byte!(bytes, remaining, 2);
    popcount_check_byte!(bytes, remaining, 3);
    popcount_check_byte!(bytes, remaining, 4);
    popcount_check_byte!(bytes, remaining, 5);
    popcount_check_byte!(bytes, remaining, 6);
    popcount_check_byte!(bytes, remaining, 7);
    popcount_check_byte!(bytes, remaining, 8);
    popcount_check_byte!(bytes, remaining, 9);
    popcount_check_byte!(bytes, remaining, 10);
    popcount_check_byte!(bytes, remaining, 11);
    popcount_check_byte!(bytes, remaining, 12);
    popcount_check_byte!(bytes, remaining, 13);
    popcount_check_byte!(bytes, remaining, 14);
    120 + SELECT_IN_BYTE[bytes[15] as usize | (remaining << 8)] as usize
}

// ──────────────────────────────────────────────────────────────────────
// Popcount2 select: split word in half via count_ones, then byte-scan
// ──────────────────────────────────────────────────────────────────────

#[inline(always)]
fn popcount2_select_u64(word: u64, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let lower_ones = (word as u32).count_ones() as usize;
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

#[inline(always)]
fn popcount2_select_u128(word: u128, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let lower_ones = (word as u64).count_ones() as usize;
    if rank < lower_ones {
        let mut remaining = rank;
        popcount_check_byte!(bytes, remaining, 0);
        popcount_check_byte!(bytes, remaining, 1);
        popcount_check_byte!(bytes, remaining, 2);
        popcount_check_byte!(bytes, remaining, 3);
        popcount_check_byte!(bytes, remaining, 4);
        popcount_check_byte!(bytes, remaining, 5);
        popcount_check_byte!(bytes, remaining, 6);
        56 + SELECT_IN_BYTE[bytes[7] as usize | (remaining << 8)] as usize
    } else {
        let mut remaining = rank - lower_ones;
        popcount_check_byte!(bytes, remaining, 8);
        popcount_check_byte!(bytes, remaining, 9);
        popcount_check_byte!(bytes, remaining, 10);
        popcount_check_byte!(bytes, remaining, 11);
        popcount_check_byte!(bytes, remaining, 12);
        popcount_check_byte!(bytes, remaining, 13);
        popcount_check_byte!(bytes, remaining, 14);
        120 + SELECT_IN_BYTE[bytes[15] as usize | (remaining << 8)] as usize
    }
}

// ──────────────────────────────────────────────────────────────────────
// Popcount3 select (u128 only): two-level decision tree via count_ones
// ──────────────────────────────────────────────────────────────────────

#[inline(always)]
fn popcount3_select_u128(word: u128, rank: usize) -> usize {
    let bytes = word.to_le_bytes();
    let lower_ones = (word as u64).count_ones() as usize;
    if rank < lower_ones {
        let lower32_ones = (word as u32).count_ones() as usize;
        if rank < lower32_ones {
            let mut remaining = rank;
            popcount_check_byte!(bytes, remaining, 0);
            popcount_check_byte!(bytes, remaining, 1);
            popcount_check_byte!(bytes, remaining, 2);
            24 + SELECT_IN_BYTE[bytes[3] as usize | (remaining << 8)] as usize
        } else {
            let mut remaining = rank - lower32_ones;
            popcount_check_byte!(bytes, remaining, 4);
            popcount_check_byte!(bytes, remaining, 5);
            popcount_check_byte!(bytes, remaining, 6);
            56 + SELECT_IN_BYTE[bytes[7] as usize | (remaining << 8)] as usize
        }
    } else {
        let upper = (word >> 64) as u64;
        let upper_lower32_ones = (upper as u32).count_ones() as usize;
        let remaining_rank = rank - lower_ones;
        if remaining_rank < upper_lower32_ones {
            let mut remaining = remaining_rank;
            popcount_check_byte!(bytes, remaining, 8);
            popcount_check_byte!(bytes, remaining, 9);
            popcount_check_byte!(bytes, remaining, 10);
            88 + SELECT_IN_BYTE[bytes[11] as usize | (remaining << 8)] as usize
        } else {
            let mut remaining = remaining_rank - upper_lower32_ones;
            popcount_check_byte!(bytes, remaining, 12);
            popcount_check_byte!(bytes, remaining, 13);
            popcount_check_byte!(bytes, remaining, 14);
            120 + SELECT_IN_BYTE[bytes[15] as usize | (remaining << 8)] as usize
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Cancellation select: cancel lowest bits via x &= x - 1, then trailing_zeros
// ──────────────────────────────────────────────────────────────────────

macro_rules! impl_cancellation_select {
    ($name:ident, $ty:ty) => {
        #[inline(always)]
        fn $name(word: $ty, rank: usize) -> usize {
            let mut x = word;
            for _ in 0..rank {
                x &= x - 1;
            }
            x.trailing_zeros() as usize
        }
    };
}

impl_cancellation_select!(cancellation_select_u8, u8);
impl_cancellation_select!(cancellation_select_u16, u16);
impl_cancellation_select!(cancellation_select_u32, u32);
impl_cancellation_select!(cancellation_select_u64, u64);
impl_cancellation_select!(cancellation_select_u128, u128);

// ──────────────────────────────────────────────────────────────────────
// BMI2 select implementations (using x86_64 PDEP)
// ──────────────────────────────────────────────────────────────────────

#[cfg(target_feature = "bmi2")]
#[inline(always)]
fn bmi_select_u8(word: u8, rank: usize) -> usize {
    bmi_select_u32(word as u32, rank)
}

#[cfg(target_feature = "bmi2")]
#[inline(always)]
fn bmi_select_u16(word: u16, rank: usize) -> usize {
    bmi_select_u32(word as u32, rank)
}

#[cfg(target_feature = "bmi2")]
#[inline(always)]
fn bmi_select_u32(word: u32, rank: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_pdep_u32;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_pdep_u32;
    let mask = 1u32 << rank;
    let one = unsafe { _pdep_u32(mask, word) };
    one.trailing_zeros() as usize
}

#[cfg(all(target_feature = "bmi2", target_arch = "x86_64"))]
#[inline(always)]
fn bmi_select_u64(word: u64, rank: usize) -> usize {
    use core::arch::x86_64::_pdep_u64;
    let mask = 1u64 << rank;
    let one = unsafe { _pdep_u64(mask, word) };
    one.trailing_zeros() as usize
}

// On 32-bit x86 with bmi2, split into 2x _pdep_u32
#[cfg(all(target_feature = "bmi2", not(target_arch = "x86_64")))]
#[inline(always)]
fn bmi_select_u64(word: u64, rank: usize) -> usize {
    let lower_ones = (word as u32).count_ones() as usize;
    if rank < lower_ones {
        bmi_select_u32(word as u32, rank)
    } else {
        32 + bmi_select_u32((word >> 32) as u32, rank - lower_ones)
    }
}

#[cfg(target_feature = "bmi2")]
#[inline(always)]
fn bmi_select_u128(word: u128, rank: usize) -> usize {
    let ones = (word as u64).count_ones() as usize;
    if ones > rank {
        bmi_select_u64(word as u64, rank)
    } else {
        64 + bmi_select_u64((word >> 64) as u64, rank - ones)
    }
}

// ──────────────────────────────────────────────────────────────────────
// Benchmarks
// ──────────────────────────────────────────────────────────────────────

/// Generate `NUM_VALUES` random (value, rank) pairs for a given word type.
///
/// Each value has at least one bit set, and the rank is valid for that value.
macro_rules! gen_pairs {
    ($rng:expr, $ty:ty) => {{
        let mut pairs = Vec::with_capacity(NUM_VALUES);
        while pairs.len() < NUM_VALUES {
            let v: $ty = $rng.random();
            if v != 0 {
                let rank = $rng.random_range(0..v.count_ones() as usize);
                pairs.push((v, rank));
            }
        }
        pairs
    }};
}

/// Bench a single select call per iteration, cycling through `$pairs`.
macro_rules! bench_one {
    ($c:expr, $name:expr, $pairs:expr, $func:expr) => {
        let pairs = &$pairs;
        let mut i = 0usize;
        $c.bench_function($name, |b| {
            b.iter(|| {
                let (v, r) = pairs[i % NUM_VALUES];
                i = i.wrapping_add(1);
                black_box($func(v, r))
            })
        });
    };
}

fn bench_broadword(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);

    let u16_pairs = gen_pairs!(rng, u16);
    let u32_pairs = gen_pairs!(rng, u32);
    let u64_pairs = gen_pairs!(rng, u64);
    let u128_pairs = gen_pairs!(rng, u128);

    bench_one!(c, "broadword_u16", u16_pairs, broadword_select_u16);
    bench_one!(c, "broadword_u32", u32_pairs, broadword_select_u32);
    bench_one!(c, "broadword_u64", u64_pairs, broadword_select_u64);
    bench_one!(c, "broadword_u128", u128_pairs, broadword_select_u128);
}

fn bench_popcount(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);

    let u8_pairs = gen_pairs!(rng, u8);
    let u16_pairs = gen_pairs!(rng, u16);
    let u32_pairs = gen_pairs!(rng, u32);
    let u64_pairs = gen_pairs!(rng, u64);
    let u128_pairs = gen_pairs!(rng, u128);

    bench_one!(c, "popcount_u8", u8_pairs, popcount_select_u8);
    bench_one!(c, "popcount_u16", u16_pairs, popcount_select_u16);
    bench_one!(c, "popcount_u32", u32_pairs, popcount_select_u32);
    bench_one!(c, "popcount_u64", u64_pairs, popcount_select_u64);
    bench_one!(c, "popcount_u128", u128_pairs, popcount_select_u128);
    bench_one!(c, "popcount2_u64", u64_pairs, popcount2_select_u64);
    bench_one!(c, "popcount2_u128", u128_pairs, popcount2_select_u128);
    bench_one!(c, "popcount3_u128", u128_pairs, popcount3_select_u128);
}

fn bench_cancellation(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);

    let u8_pairs = gen_pairs!(rng, u8);
    let u16_pairs = gen_pairs!(rng, u16);
    let u32_pairs = gen_pairs!(rng, u32);
    let u64_pairs = gen_pairs!(rng, u64);
    let u128_pairs = gen_pairs!(rng, u128);

    bench_one!(c, "cancellation_u8", u8_pairs, cancellation_select_u8);
    bench_one!(c, "cancellation_u16", u16_pairs, cancellation_select_u16);
    bench_one!(c, "cancellation_u32", u32_pairs, cancellation_select_u32);
    bench_one!(c, "cancellation_u64", u64_pairs, cancellation_select_u64);
    bench_one!(c, "cancellation_u128", u128_pairs, cancellation_select_u128);
}

#[cfg(target_feature = "bmi2")]
fn bench_bmi(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);

    let u8_pairs = gen_pairs!(rng, u8);
    let u16_pairs = gen_pairs!(rng, u16);
    let u32_pairs = gen_pairs!(rng, u32);
    let u64_pairs = gen_pairs!(rng, u64);
    let u128_pairs = gen_pairs!(rng, u128);

    bench_one!(c, "bmi_u8", u8_pairs, bmi_select_u8);
    bench_one!(c, "bmi_u16", u16_pairs, bmi_select_u16);
    bench_one!(c, "bmi_u32", u32_pairs, bmi_select_u32);
    bench_one!(c, "bmi_u64", u64_pairs, bmi_select_u64);
    bench_one!(c, "bmi_u128", u128_pairs, bmi_select_u128);
}

#[cfg(target_feature = "bmi2")]
criterion_group!(
    benches,
    bench_broadword,
    bench_popcount,
    bench_cancellation,
    bench_bmi
);
#[cfg(not(target_feature = "bmi2"))]
criterion_group!(benches, bench_broadword, bench_popcount, bench_cancellation);
criterion_main!(benches);

// ──────────────────────────────────────────────────────────────────────
// Lookup tables
// ──────────────────────────────────────────────────────────────────────

#[allow(clippy::all)]
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
