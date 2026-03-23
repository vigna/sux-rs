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
use sux::rank_sel::select_adapt;

const LEN: usize = 1_000_000;

/// Returns the additional space of `s` as a percentage of the bit vector length.
fn overhead(s: &(impl MemSize + BitLength)) -> f64 {
    let total_bits = s.mem_size(SizeFlags::default()) * 8;
    let bit_vec_bits = s.len();
    ((total_bits - bit_vec_bits) * 100) as f64 / bit_vec_bits as f64
}

/// Returns the theoretical overhead percentage for SelectAdapt-family structures:
/// (1 + M) * usize::BITS / L, where L is the target inventory span and
/// M = 2^max_log2_words_per_subinv.
fn theoretical_overhead(target_inventory_span: usize, max_log2_words_per_subinv: usize) -> f64 {
    let m = 1usize << max_log2_words_per_subinv;
    (1 + m) as f64 * usize::BITS as f64 / target_inventory_span as f64 * 100.0
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
        let sel = SelectAdapt::new(bits.clone());
        let ov = overhead(&sel);
        let th = theoretical_overhead(select_adapt::default_target_inventory_span(3), 3);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectAdapt(M=3)",
            density,
            ov,
            th,
            usize::BITS
        );
        // Overhead should be at most 3× theoretical (accounting for quantization,
        // spill, and struct fields).
        assert!(
            ov < th * 3.0,
            "SelectAdapt(M=3) d={density}: overhead {ov:.2}% > 3× theoretical {th:.2}%"
        );

        // SelectAdapt with M=1 (max_log2=0)
        let sel = SelectAdapt::with_span(
            bits.clone(),
            select_adapt::default_target_inventory_span(0),
            0,
        );
        let ov = overhead(&sel);
        let th = theoretical_overhead(select_adapt::default_target_inventory_span(0), 0);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectAdapt(M=0)",
            density,
            ov,
            th,
            usize::BITS
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
            "SelectAdaptConst<default>",
            density,
            ov,
            usize::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectAdaptConst::<_, _, 12, 2>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectAdaptConst<12,2>",
            density,
            ov,
            usize::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectAdaptConst::<_, _, 13, 0>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectAdaptConst<13,0>",
            density,
            ov,
            usize::BITS
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

        let sel = SelectZeroAdapt::new(bits.clone());
        let ov = overhead(&sel);
        let th = theoretical_overhead(select_adapt::default_target_inventory_span(3), 3);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectZeroAdapt(M=3)",
            density,
            ov,
            th,
            usize::BITS
        );
        assert!(
            ov < th * 3.0,
            "SelectZeroAdapt(M=3) d={density}: overhead {ov:.2}% > 3× theoretical {th:.2}%"
        );

        let sel = SelectZeroAdapt::with_span(
            bits.clone(),
            select_adapt::default_target_inventory_span(0),
            0,
        );
        let ov = overhead(&sel);
        let th = theoretical_overhead(select_adapt::default_target_inventory_span(0), 0);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10.2}% {:>10}",
            "SelectZeroAdapt(M=0)",
            density,
            ov,
            th,
            usize::BITS
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
            "SelectZeroAdaptConst<def>",
            density,
            ov,
            usize::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectZeroAdaptConst::<_, _, 12, 2>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectZeroAdaptConst<12,2>",
            density,
            ov,
            usize::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");

        let sel = SelectZeroAdaptConst::<_, _, 13, 0>::new(bits.clone());
        let ov = overhead(&sel);
        eprintln!(
            "{:<30} {:>8.1} {:>9.2}% {:>10}",
            "SelectZeroAdaptConst<13,0>",
            density,
            ov,
            usize::BITS
        );
        assert!(ov < 30.0, "overhead {ov:.2}% too high");
    }
}

#[test]
fn test_with_overhead() {
    let mut rng = SmallRng::seed_from_u64(0);

    eprintln!(
        "\n{:<40} {:>8} {:>8} {:>10}",
        "Structure", "Density", "Target%", "Actual%"
    );
    eprintln!("{}", "-".repeat(70));

    for density in [0.1, 0.5, 0.9] {
        let bits: AddNumBits<_> = (0..LEN)
            .map(|_| rng.random_bool(density))
            .collect::<BitVec>()
            .into();

        for target in [3.0, 7.0, 15.0, 30.0] {
            let sel = SelectAdapt::with_overhead(bits.clone(), target, 3);
            let ov = overhead(&sel);
            eprintln!(
                "{:<40} {:>8.1} {:>7.0}% {:>9.2}%",
                "SelectAdapt::with_overhead", density, target, ov
            );
            // Due to ilog2 rounding, actual overhead can be up to 2x the
            // target. It should never exceed 3x (accounting for struct
            // fields and spill).
            assert!(
                ov < target * 3.0,
                "d={density} target={target}%: overhead {ov:.2}% > 3× target"
            );

            let sel_zero = SelectZeroAdapt::with_overhead(bits.clone(), target, 3);
            let ov_zero = overhead(&sel_zero);
            eprintln!(
                "{:<40} {:>8.1} {:>7.0}% {:>9.2}%",
                "SelectZeroAdapt::with_overhead", density, target, ov_zero
            );
            assert!(
                ov_zero < target * 3.0,
                "zero d={density} target={target}%: overhead {ov_zero:.2}% > 3× target"
            );
        }
    }

    // Verify the k_min cap: with a very high overhead request, the
    // constructor should cap K so worst-case scan is at least 1 word.
    // Build a dense vector and request 200% overhead.
    let bits: AddNumBits<_> = (0..LEN)
        .map(|_| rng.random_bool(0.5))
        .collect::<BitVec>()
        .into();

    let sel_capped = SelectAdapt::with_overhead(bits.clone(), 200.0, 3);
    let ov_capped = overhead(&sel_capped);
    eprintln!("\nCapped case (target=200%): actual={ov_capped:.2}%");
    // Should be significantly less than 200% due to the cap.
    assert!(
        ov_capped < 200.0,
        "capped overhead {ov_capped:.2}% should be less than 200%"
    );
}
