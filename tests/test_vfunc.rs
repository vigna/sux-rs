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
    func::{VFilter, VFunc},
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
            let func = VBuilder::<_, _, BitFieldVec<_>, [u64; 2], true>::default()
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
            let func =
                VFunc::<_, _, BitFieldVec<_>, [u64; 2], true>::deserialize_eps(cursor.as_bytes())?;
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
            let func = VBuilder::<_, _, Vec<_>, [u64; 2], true>::default()
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
            let func = VFunc::<_, _, Vec<_>, [u64; 2], true>::deserialize_eps(cursor.as_bytes())?;
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
            let filter = VBuilder::<_, _, BitFieldVec<usize>, [u64; 2], true, ()>::default()
                .log2_buckets(4)
                .offline(offline)
                .seed(1)
                .try_build_filter(FromIntoIterator::from(0..n), 10, &mut pl)?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            filter.serialize(&mut cursor)?;
            cursor.set_position(0);
            let filter =
                VFilter::<usize, VFunc<_, _, BitFieldVec<usize>, [u64; 2], true>>::deserialize_eps(
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
            let func = VBuilder::<_, _, Vec<u8>, [u64; 2], true, ()>::default()
                .log2_buckets(4)
                .seed(1)
                .offline(offline)
                .try_build_filter(FromIntoIterator::from(0..n), &mut pl)?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            func.serialize(&mut cursor)?;
            cursor.set_position(0);
            let filter = VFilter::<u8, VFunc<_, _, Vec<u8>, [u64; 2], true>>::deserialize_eps(
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
                pl.done_with_count(n);

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

    assert!(VBuilder::<usize, usize>::default()
        .try_build_func(
            FromIntoIterator::from(std::iter::repeat(0).take(10)),
            FromIntoIterator::from(0..),
            &mut ProgressLogger::default(),
        )
        .is_err());

    Ok(())
}
