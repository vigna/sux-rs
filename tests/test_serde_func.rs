/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Round trips through serde of the structures based on a ShardEdge.

#![cfg(all(feature = "rayon", feature = "serde"))]

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::dict::VFilter;
use sux::func::shard_edge::{Fuse3NoShards, Fuse3Shards, FuseAug3FullSigs};
use sux::func::{CompVFunc, Lcp2MmphfInt, LcpMmphfStr, VFunc, VFunc2};

/// Serializes and deserializes `t` with serde, checking that the result
/// has the same debug representation.
fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug>(
    t: &T,
) -> Result<T> {
    let json = serde_json::to_string(t)?;
    let back: T = serde_json::from_str(&json)?;
    assert_eq!(format!("{back:?}"), format!("{t:?}"));
    Ok(back)
}

#[test]
fn test_serde_vfunc() -> Result<()> {
    let keys: Vec<usize> = (0..10_000).collect();
    let values: Vec<usize> = (0..10_000).map(|i| i * 3 % 1000).collect();

    let func =
        <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_par_new(&keys, &values, no_logging![])?;
    let back = round_trip(&func)?;
    for (&key, &value) in keys.iter().zip(&values) {
        assert_eq!(back.get(key), value);
    }

    let values16: Vec<u16> = values.iter().map(|&v| v as u16).collect();
    let func = <VFunc<usize, Box<[u16]>, [u64; 1], Fuse3NoShards>>::try_par_new(
        &keys,
        &values16,
        no_logging![],
    )?;
    let back = round_trip(&func)?;
    for (&key, &value) in keys.iter().zip(&values16) {
        assert_eq!(back.get(key), value);
    }

    let func = <VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], Fuse3Shards>>::try_par_new(
        &keys,
        &values,
        no_logging![],
    )?;
    round_trip(&func)?;

    let func = <VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], FuseAug3FullSigs>>::try_par_new(
        &keys,
        &values,
        no_logging![],
    )?;
    round_trip(&func)?;

    let skeys: Vec<String> = (0..1000).map(|i| format!("key{i}")).collect();
    let func = <VFunc<str, BitFieldVec<Box<[usize]>>>>::try_par_new(
        &skeys,
        &values[..1000],
        no_logging![],
    )?;
    let back = round_trip(&func)?;
    for (key, &value) in skeys.iter().zip(&values) {
        assert_eq!(back.get(key.as_str()), value);
    }

    Ok(())
}

#[test]
fn test_serde_vfilter() -> Result<()> {
    let keys: Vec<usize> = (0..10_000).collect();

    let filter = <VFilter<usize, Box<[u8]>>>::try_par_new(&keys, no_logging![])?;
    let back = round_trip(&filter)?;
    for i in 0..2 * keys.len() {
        assert_eq!(back.contains(i), filter.contains(i));
    }

    let filter = <VFilter<usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>>::try_par_new(
        &keys,
        10,
        no_logging![],
    )?;
    let back = round_trip(&filter)?;
    for i in 0..2 * keys.len() {
        assert_eq!(back.contains(i), filter.contains(i));
    }

    Ok(())
}

#[test]
fn test_serde_other_funcs() -> Result<()> {
    let keys: Vec<u64> = (0..10_000).map(|i| i * 7).collect();
    let values: Vec<usize> = (0..10_000)
        .map(|i: usize| i.trailing_zeros() as usize)
        .collect();

    let func =
        <VFunc2<u64, BitFieldVec<Box<[usize]>>>>::try_par_new(&keys, &values, no_logging![])?;
    let back = round_trip(&func)?;
    for (&key, &value) in keys.iter().zip(&values) {
        assert_eq!(back.get(key), value);
    }

    let func = <CompVFunc<u64>>::try_par_new(&keys, &values, no_logging![])?;
    let back = round_trip(&func)?;
    for (&key, &value) in keys.iter().zip(&values) {
        assert_eq!(back.get(key), value);
    }

    let func = <Lcp2MmphfInt<u64>>::try_par_new(&keys, no_logging![])?;
    let back = round_trip(&func)?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(back.get(key), i);
    }

    let skeys: Vec<String> = (0..1000).map(|i| format!("{i:05}")).collect();
    let func = <LcpMmphfStr>::try_par_new(&skeys, no_logging![])?;
    let back = round_trip(&func)?;
    for (i, key) in skeys.iter().enumerate() {
        assert_eq!(back.get(key.as_str()), i);
    }

    Ok(())
}

#[cfg(feature = "mwhc")]
#[test]
fn test_serde_mwhc() -> Result<()> {
    use sux::func::shard_edge::{Mwhc3NoShards, Mwhc3Shards};

    let keys: Vec<usize> = (0..10_000).collect();
    let values: Vec<usize> = (0..10_000).collect();

    let func = <VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], Mwhc3Shards>>::try_par_new(
        &keys,
        &values,
        no_logging![],
    )?;
    let back = round_trip(&func)?;
    for (&key, &value) in keys.iter().zip(&values) {
        assert_eq!(back.get(key), value);
    }

    let func = <VFunc<usize, BitFieldVec<Box<[usize]>>, [u64; 2], Mwhc3NoShards>>::try_par_new(
        &keys,
        &values,
        no_logging![],
    )?;
    round_trip(&func)?;

    Ok(())
}
