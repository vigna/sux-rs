/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Test that compares the additional space occupancy of select_adapt
//! data structures relative to the underlying bit vector. The test runs
//! identically on 32-bit and 64-bit platforms, allowing comparison of
//! overhead across architectures.

use mem_dbg::{MemSize, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::prelude::*;
use sux::traits::PlatformWord;

const LEN: usize = 1_000_000;

/// Returns the additional space of `s` as a percentage of the bit vector length.
fn overhead(s: &(impl MemSize + BitLength)) -> f64 {
    let total_bits = s.mem_size(SizeFlags::default()) * 8;
    let bit_vec_bits = s.len();
    ((total_bits - bit_vec_bits) * 100) as f64 / bit_vec_bits as f64
}

/// Returns the theoretical overhead percentage for SelectAdapt-family structures:
/// (1 + M) * PlatformWord::BITS / L, where L is the target inventory span and
/// M = 2^max_log2_words_per_subinv.
fn theoretical_overhead(target_inventory_span: usize, max_log2_words_per_subinv: usize) -> f64 {
    let m = 1usize << max_log2_words_per_subinv;
    (1 + m) as f64 * PlatformWord::BITS as f64 / target_inventory_span as f64 * 100.0
}

#[test]
fn test_space_select_adapt() {
    let mut rng = SmallRng::seed_from_u64(0);

    eprintln!(
        "\n{:<30} {:>8} {:>10} {:>12} {:>10}",
        "Structure", "Density", "Overhead%", "Theoretical", "PW bits"
    );
    eprintln!("{}", "-".repeat(75));

    for density in [0.1, 0.5, 0.9] {
        let bits: AddNumBits<_> = (0..LEN)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();

        // SelectAdapt with M=8 (max_log2=3), default span 8192
        let sel = SelectAdapt::new(bits.clone(), 3);
        let ov = overhead(&sel);
        let th = theoretical_overhead(SelectAdapt::<AddNumBits<BitVec>>::DEFAULT_TARGET_INVENTORY_SPAN, 3);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectAdapt(M=3)", density, ov, th, PlatformWord::BITS
        );
        // Overhead should be at most 3× theoretical (accounting for quantization,
        // spill, and struct fields).
        assert!(
            ov < th * 3.0,
            "SelectAdapt(M=3) d={density}: overhead {ov:.2}% > 3× theoretical {th:.2}%"
        );

        // SelectAdapt with M=1 (max_log2=0), default span 8192
        let sel = SelectAdapt::new(bits.clone(), 0);
        let ov = overhead(&sel);
        let th = theoretical_overhead(SelectAdapt::<AddNumBits<BitVec>>::DEFAULT_TARGET_INVENTORY_SPAN, 0);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectAdapt(M=0)", density, ov, th, PlatformWord::BITS
        );
        assert!(
            ov < th * 3.0,
            "SelectAdapt(M=0) d={density}: overhead {ov:.2}% > 3× theoretical {th:.2}%"
        );
    }
}

#[test]
fn test_space_select_adapt_const() {
    let mut rng = SmallRng::seed_from_u64(0);

    eprintln!(
        "\n{:<30} {:>8} {:>10} {:>10}",
        "Structure", "Density", "Overhead%", "PW bits"
    );
    eprintln!("{}", "-".repeat(62));

    for density in [0.1, 0.5, 0.9] {
        let bits: AddNumBits<_> = (0..LEN)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();

        // Default parameters (platform-adjusted LOG2_ONES_PER_INVENTORY)
        let sel = SelectAdaptConst::<_, _>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectAdaptConst<default>", density, ov, PlatformWord::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectAdaptConst::<_, _, 12, 2>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectAdaptConst<12,2>", density, ov, PlatformWord::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectAdaptConst::<_, _, 13, 0>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectAdaptConst<13,0>", density, ov, PlatformWord::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");
    }
}

#[test]
fn test_space_select_zero_adapt() {
    let mut rng = SmallRng::seed_from_u64(0);

    eprintln!(
        "\n{:<30} {:>8} {:>10} {:>12} {:>10}",
        "Structure", "Density", "Overhead%", "Theoretical", "PW bits"
    );
    eprintln!("{}", "-".repeat(75));

    for density in [0.1, 0.5, 0.9] {
        let bits: AddNumBits<_> = (0..LEN)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();

        let sel = SelectZeroAdapt::new(bits.clone(), 3);
        let ov = overhead(&sel);
        let th = theoretical_overhead(SelectZeroAdapt::<AddNumBits<BitVec>>::DEFAULT_TARGET_INVENTORY_SPAN, 3);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectZeroAdapt(M=3)", density, ov, th, PlatformWord::BITS
        );
        assert!(
            ov < th * 3.0,
            "SelectZeroAdapt(M=3) d={density}: overhead {ov:.2}% > 3× theoretical {th:.2}%"
        );

        let sel = SelectZeroAdapt::new(bits.clone(), 0);
        let ov = overhead(&sel);
        let th = theoretical_overhead(SelectZeroAdapt::<AddNumBits<BitVec>>::DEFAULT_TARGET_INVENTORY_SPAN, 0);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectZeroAdapt(M=0)", density, ov, th, PlatformWord::BITS
        );
        assert!(
            ov < th * 3.0,
            "SelectZeroAdapt(M=0) d={density}: overhead {ov:.2}% > 3× theoretical {th:.2}%"
        );
    }
}

#[test]
fn test_space_select_zero_adapt_const() {
    let mut rng = SmallRng::seed_from_u64(0);

    eprintln!(
        "\n{:<30} {:>8} {:>10} {:>10}",
        "Structure", "Density", "Overhead%", "PW bits"
    );
    eprintln!("{}", "-".repeat(62));

    for density in [0.1, 0.5, 0.9] {
        let bits: AddNumBits<_> = (0..LEN)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();

        // Default parameters (platform-adjusted LOG2_ZEROS_PER_INVENTORY)
        let sel = SelectZeroAdaptConst::<_, _>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectZeroAdaptConst<def>", density, ov, PlatformWord::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectZeroAdaptConst::<_, _, 12, 2>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectZeroAdaptConst<12,2>", density, ov, PlatformWord::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectZeroAdaptConst::<_, _, 13, 0>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectZeroAdaptConst<13,0>", density, ov, PlatformWord::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");
    }
}
