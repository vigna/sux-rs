/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use dsi_progress_logger::*;
use epserde::prelude::*;
use sux::{
    bits::BitFieldVec,
    dict::VFilter,
    func::{shard_edge::FuseLge3Shards, VFunc},
    prelude::VBuilder,
    utils::FromIntoIterator,
};

#[test]
fn test_vfunc_lge() -> Result<()> {
    _test_vfunc(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfunc(&[0, 10, 1000, 100_000], true, false)?;
    Ok(())
}

#[test]
fn test_vfunc_peeling_by_sig_vals() -> Result<()> {
    _test_vfunc(&[1_000_000], false, false)?;
    _test_vfunc(&[1_000_000], false, true)?;
    _test_vfunc(&[1_000_000], true, false)?;
    _test_vfunc(&[1_000_000], true, true)?;
    Ok(())
}

fn _test_vfunc(sizes: &[usize], offline: bool, low_mem: bool) -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();

    for &n in sizes {
        dbg!(offline, n);
        let func = VBuilder::<_, BitFieldVec<_>, [u64; 2], FuseLge3Shards>::default()
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
            assert_eq!(i, func.get(&i));
        }
        pl.done_with_count(n);
    }

    Ok(())
}

fn _test_vfilter(sizes: &[usize], offline: bool, low_mem: bool) -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();

    for &n in sizes {
        dbg!(offline, n);
        let filter = VBuilder::<_, Box<[u8]>>::default()
            .expected_num_keys(n)
            .offline(offline)
            .low_mem(low_mem)
            .try_build_filter(FromIntoIterator::from(0..n), &mut pl)?;
        let mut cursor = <AlignedCursor<maligned::A16>>::new();
        filter.serialize(&mut cursor)?;
        cursor.set_position(0);
        let filter = VFilter::<u8, VFunc<usize, _, Box<[u8]>>>::deserialize_eps(cursor.as_bytes())?;
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
    _test_vfilter(&[0, 10, 1000, 100_000], false, false)?;
    _test_vfilter(&[0, 10, 1000, 100_000], true, false)?;
    Ok(())
}

#[test]
fn test_vfilter_peeling_by_sig_vals() -> Result<()> {
    _test_vfilter(&[1_000_000], false, false)?;
    _test_vfilter(&[1_000_000], false, true)?;
    _test_vfilter(&[1_000_000], true, false)?;
    _test_vfilter(&[1_000_000], true, true)?;
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
            FromIntoIterator::from(std::iter::repeat(0).take(10)),
            FromIntoIterator::from(0..),
            &mut ProgressLogger::default(),
        )
        .is_err());

    Ok(())
}
