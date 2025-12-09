/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]
use anyhow::Result;
use dsi_progress_logger::*;
use sux::{
    bits::BitFieldVec,
    dict::{BitSignedVFunc, SignedVFunc},
    func::{VBuilder, VFunc, shard_edge::FuseLge3Shards},
    utils::FromCloneableIntoIterator,
};

#[test]
fn test_signed_vfunc() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();
    let n = 1000;
    let func: SignedVFunc<
        VFunc<usize, usize, BitFieldVec, [u64; 2], FuseLge3Shards>,
        Box<[usize]>,
    > = VBuilder::default()
        .expected_num_keys(n)
        .try_build_sig_index(FromCloneableIntoIterator::from(0..n), &mut pl)?;
    for i in 0..n {
        assert_eq!(Some(i), func.get(i));
    }
    for i in 0..n {
        assert_eq!(None, func.get(i + n));
    }

    Ok(())
}

#[test]
fn test_bit_signed_vfunc() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();
    let n = 1000;
    let func: BitSignedVFunc<
        VFunc<usize, usize, BitFieldVec, [u64; 2], FuseLge3Shards>,
        BitFieldVec,
    > = VBuilder::default()
        .expected_num_keys(n)
        .try_build_bit_sig_index(FromCloneableIntoIterator::from(0..n), 31, &mut pl)?;
    for i in 0..n {
        assert_eq!(Some(i), func.get(i));
    }
    for i in 0..n {
        assert_eq!(None, func.get(i + n));
    }

    Ok(())
}
