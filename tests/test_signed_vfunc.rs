/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]
#![cfg(feature = "rayon")]
use anyhow::Result;
use dsi_progress_logger::*;
use sux::{
    bits::BitFieldVec,
    func::{SignedFunc, VFunc, shard_edge::FuseLge3Shards},
    traits::TryIntoUnaligned,
    utils::FromCloneableIntoIterator,
};

type MySignedVFunc =
    SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], FuseLge3Shards>, Box<[usize]>>;

type MyBitSignedVFunc =
    SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[usize]>>>;

#[test]
fn test_signed_vfunc() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();
    let n = 1000;
    let func: MySignedVFunc =
        MySignedVFunc::try_new(FromCloneableIntoIterator::from(0..n), &mut pl)?;
    let func = func.try_into_unaligned()?;
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
    let hash_bits = if cfg!(target_pointer_width = "64") {
        31
    } else {
        23
    };

    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();
    let n = 1000;
    let func: MyBitSignedVFunc =
        MyBitSignedVFunc::try_new(FromCloneableIntoIterator::from(0..n), hash_bits, &mut pl)?;
    let func = func.try_into_unaligned()?;
    for i in 0..n {
        assert_eq!(Some(i), func.get(i));
    }
    for i in 0..n {
        assert_eq!(None, func.get(i + n));
    }

    Ok(())
}

#[test]
fn test_invalid_hash_width_is_error() {
    let mut pl = ProgressLogger::default();
    for hash_width in [0, 65] {
        let result =
            MyBitSignedVFunc::try_new(FromCloneableIntoIterator::from(0..1), hash_width, &mut pl);
        assert!(result.is_err(), "hash width {hash_width} must be rejected");
    }
}

#[test]
fn test_u128_hash_storage_is_queryable() -> Result<()> {
    type U128Signed = SignedFunc<VFunc<usize, BitFieldVec<Box<[usize]>>>, Box<[u128]>>;
    let mut pl = ProgressLogger::default();
    let func = U128Signed::try_new(FromCloneableIntoIterator::from(0..10), &mut pl)?;
    for key in 0..10 {
        assert_eq!(func.get(key), Some(key));
    }
    Ok(())
}
