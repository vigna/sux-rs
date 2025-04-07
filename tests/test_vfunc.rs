/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::ops::{BitXor, BitXorAssign};

use anyhow::Result;
use dsi_progress_logger::*;
use rdst::RadixKey;
use sux::{
    bits::BitFieldVec,
    func::shard_edge::{FuseLge3FullSigs, FuseLge3NoShards, FuseLge3Shards, ShardEdge},
    prelude::VBuilder,
    utils::{EmptyVal, FromIntoIterator, Sig, SigVal, ToSig},
};

fn _test_vfunc<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
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
        let func = VBuilder::<_, BitFieldVec<_>, S, E>::default()
            .expected_num_keys(n)
            .offline(offline)
            .low_mem(low_mem)
            .try_build_func(
                FromIntoIterator::from(0..n),
                FromIntoIterator::from(0_usize..),
                &mut pl,
            )?;
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
    _test_vfunc::<[u64; 1], FuseLge3NoShards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfunc::<[u64; 1], FuseLge3NoShards>(&[0, 10, 1000, 100_000], true, false)?;
    Ok(())
}

#[test]
fn test_vfunc_peeling_by_sig_vals() -> Result<()> {
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, false)?;
    _test_vfunc::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, true)?;
    _test_vfunc::<[u64; 1], FuseLge3NoShards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 1], FuseLge3NoShards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], FuseLge3NoShards>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], FuseLge3NoShards>(&[1_000_000], false, true)?;
    _test_vfunc::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, false)?;
    _test_vfunc::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, true)?;
    Ok(())
}

fn _test_vfilter<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
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
        let filter = VBuilder::<_, Box<[u8]>, S, E>::default()
            .expected_num_keys(n)
            .offline(offline)
            .low_mem(low_mem)
            .try_build_filter(FromIntoIterator::from(0..n), &mut pl)?;
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
    _test_vfilter::<[u64; 1], FuseLge3NoShards>(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfilter::<[u64; 1], FuseLge3NoShards>(&[0, 10, 1000, 100_000], true, false)?;
    Ok(())
}

#[test]
fn test_vfilter_peeling_by_sig_vals() -> Result<()> {
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, false)?;
    _test_vfilter::<[u64; 2], FuseLge3Shards>(&[1_000_000], true, true)?;
    _test_vfilter::<[u64; 1], FuseLge3NoShards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 1], FuseLge3NoShards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], FuseLge3NoShards>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], FuseLge3NoShards>(&[1_000_000], false, true)?;
    _test_vfilter::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, false)?;
    _test_vfilter::<[u64; 2], FuseLge3FullSigs>(&[1_000_000], false, true)?;
    Ok(())
}

#[test]
fn test_dup_key() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    assert!(VBuilder::<usize, BitFieldVec<usize>>::default()
        .check_dups(true)
        .try_build_func(
            FromIntoIterator::from(std::iter::repeat_n(0, 10)),
            FromIntoIterator::from(0..),
            &mut ProgressLogger::default(),
        )
        .is_err());

    Ok(())
}
