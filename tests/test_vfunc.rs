/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(feature = "rayon")]
use anyhow::Result;
use dsi_progress_logger::*;
use mem_dbg::{FlatType, MemSize};
use rdst::RadixKey;
use std::ops::{BitXor, BitXorAssign};
use sux::traits::TryIntoUnaligned;
use sux::{
    bits::BitFieldVec,
    dict::VFilter,
    func::{
        VBuilder, VFunc,
        shard_edge::{Fuse3NoShards, Fuse3Shards, FuseLge3FullSigs, FuseLge3Shards, ShardEdge},
    },
    utils::{EmptyVal, FromCloneableIntoIterator, Sig, SigVal, ToSig},
};

fn _test_vfunc<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>(
    sizes: &[usize],
    offline: bool,
    low_mem: bool,
) -> Result<()>
where
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
{
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();

    for &n in sizes {
        dbg!(offline, n);
        let func = <VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>>::try_new_with_builder(
            FromCloneableIntoIterator::from(0..n),
            FromCloneableIntoIterator::from(0_usize..),
            VBuilder::default().offline(offline).low_mem(low_mem),
            &mut pl,
        )?;
        let func = func.try_into_unaligned().unwrap();
        pl.start("Querying...");
        for i in 0..n {
            assert_eq!(i, func.get(i));
        }
        pl.done_with_count(n);
    }

    Ok(())
}

#[test]
fn test_vfunc_lge() -> Result<()> {
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[0, 10, 1000, 100_000], true, false)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], true, false)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], true, false)?;
    _test_vfunc::<[u64; 2], Fuse3Shards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfunc::<[u64; 2], Fuse3Shards>(&[0, 10, 1000, 100_000], true, false)?;
    Ok(())
}

#[test]
fn test_vfunc_peeling_by_sig_vals() -> Result<()> {
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, false)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, true)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], Fuse3NoShards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], Fuse3NoShards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], Fuse3Shards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], Fuse3Shards>(&[1_000_000], false, true)?;
    Ok(())
}

fn _test_vfilter<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>(
    sizes: &[usize],
    offline: bool,
    low_mem: bool,
) -> Result<()>
where
    usize: ToSig<S>,
    SigVal<S, EmptyVal>: RadixKey,
    SigVal<E::LocalSig, EmptyVal>: BitXor + BitXorAssign,
{
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();

    for &n in sizes {
        dbg!(offline, n);
        let filter = <VFilter<usize, Box<[u8]>, S, E>>::try_new_with_builder(
            FromCloneableIntoIterator::from(0..n),
            VBuilder::default().offline(offline).low_mem(low_mem),
            &mut pl,
        )?;
        pl.start("Querying (positive)...");
        for i in 0..n {
            assert!(filter.contains(i), "Contains failed for {}", i);
        }
        pl.done_with_count(n);

        if n > 1_000_000 {
            pl.start("Querying (negative)...");
            let mut c = 0;
            for i in 0..n {
                c += filter.contains(i + n) as usize;
            }
            pl.done_with_count(n);

            let failure_rate = (c as f64) / n as f64;
            assert!(
                failure_rate < 1. / 254.,
                "Failure rate is too high: 1 / {}",
                1. / failure_rate
            );
        }
    }

    Ok(())
}

#[test]
fn test_vfilter_lge() -> Result<()> {
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[0, 10, 1000, 100_000], true, false)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], true, false)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[0, 10, 1000, 100_000], true, false)?;
    _test_vfilter::<[u64; 2], Fuse3Shards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfilter::<[u64; 2], Fuse3Shards>(&[0, 10, 1000, 100_000], true, false)?;
    Ok(())
}

#[test]
fn test_vfilter_peeling_by_sig_vals() -> Result<()> {
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, false)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, true)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], Fuse3NoShards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], Fuse3NoShards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 1], Fuse3NoShards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], Fuse3Shards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], Fuse3Shards>(&[1_000_000], false, true)?;
    Ok(())
}

#[test]
fn test_dup_key() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    assert!(
        <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
            FromCloneableIntoIterator::from(std::iter::repeat_n(0, 10)),
            FromCloneableIntoIterator::from(0..),
            VBuilder::default().check_dups(true),
            &mut ProgressLogger::default(),
        )
        .is_err()
    );

    Ok(())
}

#[test]
fn test_mismatched_keys_and_values() -> Result<()> {
    use sux::func::BuildError;
    let mut pl = ProgressLogger::default();

    // Lender-based construction: fewer values than keys is an error
    let result = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new(
        FromCloneableIntoIterator::from(0..100_usize),
        FromCloneableIntoIterator::from(0..50_usize),
        &mut pl,
    );
    assert!(matches!(
        result.unwrap_err().downcast_ref::<BuildError>(),
        Some(BuildError::MismatchedKeysAndValues { .. })
    ));

    // Lender-based construction: extra values are allowed (the values
    // lender may be infinite)
    let func = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new(
        FromCloneableIntoIterator::from(0..100_usize),
        FromCloneableIntoIterator::from(0_usize..),
        &mut pl,
    )?;
    assert_eq!(func.get(50), 50);

    // Slice-based construction: lengths must match exactly
    let keys: Vec<usize> = (0..100).collect();
    for len in [50_usize, 200] {
        let values: Vec<usize> = (0..len).collect();
        let result =
            <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_par_new(&keys, &values, &mut pl);
        assert!(matches!(
            result.unwrap_err().downcast_ref::<BuildError>(),
            Some(BuildError::MismatchedKeysAndValues { .. })
        ));
    }

    Ok(())
}

/// The parallel builder does not support offline (disk-based) construction and
/// must reject it rather than silently ignoring the flag.
#[test]
fn test_par_builder_rejects_offline() {
    let keys: Vec<usize> = (0..1000).collect();
    let values: Vec<usize> = (0..1000).collect();
    let result = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
        &keys,
        &values,
        VBuilder::default().offline(true),
        &mut ProgressLogger::default(),
    );
    let err = result.expect_err("parallel builder must reject offline mode instead of ignoring it");
    assert!(
        err.to_string().contains("offline"),
        "expected an offline-rejection error, got: {err}"
    );
}

/// A non-finite or negative `eps` is rejected up front rather than triggering an
/// unbounded shard-resizing retry loop.
#[test]
fn test_builder_rejects_invalid_eps() {
    for bad in [-1.0_f64, f64::NAN, f64::INFINITY] {
        let result = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
            FromCloneableIntoIterator::from(0..1000),
            FromCloneableIntoIterator::from(0_usize..),
            VBuilder::default().eps(bad),
            &mut ProgressLogger::default(),
        );
        let err = result.expect_err("builder must reject invalid eps");
        assert!(
            err.to_string().contains("must be finite"),
            "expected an eps-validation error for {bad}, got: {err}"
        );
    }
}
