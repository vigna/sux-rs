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
    func::{Fuse3Shards, VFilter, VFunc},
    prelude::VBuilder,
    utils::FromIntoIterator,
};

#[test]
fn test_vfunc() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();

    for offline in [false, true] {
        for n in [0, 10, 1000, 100_000, 3_000_000] {
            dbg!(offline, n);
            let func = VBuilder::<_, BitFieldVec<_>, [u64; 2], Fuse3Shards>::default()
                .log2_buckets(4)
                .offline(offline)
                .try_build_func(
                    FromIntoIterator::from(0..n),
                    FromIntoIterator::from(0_usize..),
                    &mut pl,
                )?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            func.serialize(&mut cursor)?;
            cursor.set_position(0);
            let func = VFunc::<usize, _, BitFieldVec<_>, [u64; 2], Fuse3Shards>::deserialize_eps(
                cursor.as_bytes(),
            )?;
            pl.start("Querying...");
            for i in 0..n {
                assert_eq!(i, func.get(&i));
            }
            pl.done_with_count(n);
        }
    }

    for offline in [false, true] {
        for n in [0, 10, 1000, 100_000, 3_000_000] {
            dbg!(offline, n);
            let func = VBuilder::<usize, Box<[usize]>, [u64; 2], Fuse3Shards>::default()
                .log2_buckets(4)
                .offline(offline)
                .try_build_func(
                    FromIntoIterator::from(0..n),
                    FromIntoIterator::from(0_usize..),
                    &mut pl,
                )?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            func.serialize(&mut cursor)?;
            cursor.set_position(0);
            let func = VFunc::<usize, _, Box<[_]>, [u64; 2], Fuse3Shards>::deserialize_eps(
                cursor.as_bytes(),
            )?;
            pl.start("Querying...");
            for i in 0..n {
                assert_eq!(i, func.get(&i));
            }
            pl.done_with_count(n);
        }
    }

    Ok(())
}

#[test]
fn test_vfilter() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let mut pl = ProgressLogger::default();

    for offline in [false, true] {
        for n in [0, 10, 1000, 100_000, 3_000_000] {
            dbg!(offline, n);
            let filter = VBuilder::<_, BitFieldVec<usize>, [u64; 2], Fuse3Shards>::default()
                .log2_buckets(4)
                .offline(offline)
                .try_build_filter(FromIntoIterator::from(0..n), 10, &mut pl)?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            filter.serialize(&mut cursor)?;
            cursor.set_position(0);
            let filter = VFilter::<
                usize,
                VFunc<usize, _, BitFieldVec<usize>, [u64; 2], Fuse3Shards>,
            >::deserialize_eps(cursor.as_bytes())?;
            pl.start("Querying (positive)...");
            for i in 0..n {
                assert!(filter.contains(&i), "Contains failed for {}", i);
            }
            pl.done_with_count(n);

            if n > 1_000_000 {
                pl.start("Querying (negative)...");
                let mut c = 0;
                for i in 0..n {
                    c += filter.contains(&(i + n)) as usize;
                }
                pl.done_with_count(n);

                let failure_rate = (c as f64) / n as f64;
                assert!(
                    failure_rate < 1. / 1023.,
                    "Failure rate is too high: 1 / {}",
                    1. / failure_rate
                );
            }
        }
    }

    for offline in [false, true] {
        for n in [0, 10, 1000, 100_000, 3_000_000] {
            dbg!(offline, n);
            let func = VBuilder::<_, Box<[u8]>, [u64; 2], Fuse3Shards>::default()
                .log2_buckets(4)
                .offline(offline)
                .try_build_filter(FromIntoIterator::from(0..n), &mut pl)?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            func.serialize(&mut cursor)?;
            cursor.set_position(0);
            let filter =
                VFilter::<u8, VFunc<usize, _, Vec<u8>, [u64; 2], Fuse3Shards>>::deserialize_eps(
                    cursor.as_bytes(),
                )?;
            pl.start("Querying (positive)...");
            for i in 0..n {
                assert!(filter.contains(&i), "Contains failed for {}", i);
            }
            pl.done_with_count(n);
            if n > 1_000_000 {
                pl.start("Querying (negative)...");
                let mut c = 0;
                for i in 0..n {
                    c += filter.contains(&(i + n)) as usize;
                }
                //pl.done_with_count(n); Or it fails

                let failure_rate = (c as f64) / n as f64;
                assert!(
                    failure_rate < 1. / 255.,
                    "Failure rate is too high: 1 / {}",
                    1. / failure_rate
                );
            }
        }
    }

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

#[test]
fn test_broken() -> Result<()> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let n = 3_000_000;
    let filter = VBuilder::<usize, BitFieldVec<usize>, [u64; 2], Fuse3Shards>::default()
        .try_build_filter(FromIntoIterator::from(0..n), 10, no_logging![])?;

    let mut pl = ProgressLogger::default();
    pl.start("Querying (negative)...");
    let mut c = 0;
    for i in 0..n {
        c += filter.contains(&(i + n)) as usize;
    }
    //pl.done_with_count(n);

    dbg!(c);
    dbg!(n as f64 / c as f64);

    Ok(())
}
