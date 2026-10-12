/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::type_complexity)]
#![cfg(feature = "rayon")]
use anyhow::Result;
use dsi_progress_logger::*;
use sux::{
    bits::BitFieldVec,
    func::{PHastR, PHastRBuilder, SignedFunc},
    traits::TryIntoUnaligned,
    utils::{FromCloneableIntoIterator, FromSlice},
};

type MySignedPHastR = SignedFunc<PHastR<usize>, Box<[usize]>>;
type MyBitSignedPHastR = SignedFunc<PHastR<usize>, BitFieldVec<Box<[usize]>>>;

/// Checks that the keys [0 . . n) are mapped bijectively onto [0 . . n),
/// and that the keys [n . . 2n) are rejected.
fn check(n: usize, get: impl Fn(usize) -> Option<usize>) {
    let mut seen = vec![false; n];
    for i in 0..n {
        let v = get(i).unwrap_or_else(|| panic!("key {i} rejected"));
        assert!(!seen[v], "output {v} of key {i} already seen");
        seen[v] = true;
    }
    for i in n..2 * n {
        assert_eq!(None, get(i), "key {i} accepted");
    }
}

#[test]
fn test_signed_phast_r() -> Result<()> {
    let n = 100_000;
    let func = MySignedPHastR::try_new(FromCloneableIntoIterator::from(0..n), no_logging![])?;
    check(n, |i| func.get(i));
    let func = func.try_into_unaligned()?;
    check(n, |i| func.get(i));
    Ok(())
}

#[test]
fn test_bit_signed_phast_r() -> Result<()> {
    let hash_bits = if cfg!(target_pointer_width = "64") {
        31
    } else {
        23
    };
    let n = 100_000;
    let func = MyBitSignedPHastR::try_new(
        FromCloneableIntoIterator::from(0..n),
        hash_bits,
        no_logging![],
    )?;
    check(n, |i| func.get(i));
    let func = func.try_into_unaligned()?;
    check(n, |i| func.get(i));
    Ok(())
}

/// The parallel constructors build the same function as the sequential
/// ones, and sign it in the same way.
#[test]
fn test_par_signed_phast_r() -> Result<()> {
    let n = 100_000;
    let keys: Vec<usize> = (0..n).collect();
    let seq = MySignedPHastR::try_new(FromCloneableIntoIterator::from(0..n), no_logging![])?;
    let par = MySignedPHastR::try_par_new(&keys, no_logging![])?;
    check(n, |i| par.get(i));
    for i in 0..2 * n {
        assert_eq!(seq.get(i), par.get(i));
    }

    let seq = MyBitSignedPHastR::try_new(FromCloneableIntoIterator::from(0..n), 13, no_logging![])?;
    let par = MyBitSignedPHastR::try_par_new(&keys, 13, no_logging![])?;
    for i in 0..n {
        assert_eq!(seq.get(i), par.get(i));
        assert!(par.get(i).is_some());
    }
    Ok(())
}

#[test]
fn test_offline_signed_phast_r() -> Result<()> {
    let n = 100_000;
    let func = MySignedPHastR::try_new_with_builder(
        FromCloneableIntoIterator::from(0..n),
        PHastRBuilder::default().offline(true),
        no_logging![],
    )?;
    check(n, |i| func.get(i));
    Ok(())
}

/// Seeds of more than eight bits are stored in a bit-field vector, which can
/// be read with unaligned reads.
#[test]
fn test_wide_seeds_signed_phast_r() -> Result<()> {
    type F = SignedFunc<PHastR<usize, BitFieldVec<Box<[usize]>>>, Box<[u32]>>;
    let n = 100_000;
    let keys: Vec<usize> = (0..n).collect();
    let builder = PHastRBuilder::default()
        .seed_bits(10)
        .log2_slice_len(11)
        .bucket_size(5.75);
    let func = F::try_par_new_with_builder(&keys, builder, no_logging![])?;
    check(n, |i| func.get(i));
    let func = func.try_into_unaligned()?;
    check(n, |i| func.get(i));
    let func: F = func.into();
    check(n, |i| func.get(i));
    Ok(())
}

#[test]
fn test_str_signed_phast_r() -> Result<()> {
    let n = 10_000;
    let keys: Vec<String> = (0..2 * n).map(|i| format!("key {i}")).collect();
    let func =
        <SignedFunc<PHastR<str>, Box<[u64]>>>::try_new(FromSlice::new(&keys[..n]), no_logging![])?;
    let mut seen = vec![false; n];
    for key in &keys[..n] {
        let v = func.get(key.as_str()).unwrap();
        assert!(!seen[v]);
        seen[v] = true;
    }
    for key in &keys[n..] {
        assert_eq!(None, func.get(key.as_str()));
    }
    Ok(())
}

/// With 8-bit hashes, about one key outside the original set out of 256 is
/// accepted.
#[test]
fn test_false_positives_signed_phast_r() -> Result<()> {
    let n = 100_000;
    let keys: Vec<usize> = (0..n).collect();
    let func = <SignedFunc<PHastR<usize>, Box<[u8]>>>::try_par_new(&keys, no_logging![])?;
    let fp = (n..11 * n).filter(|&i| func.get(i).is_some()).count();
    let expected = 10 * n / 256;
    assert!(
        fp > expected * 3 / 4 && fp < expected * 5 / 4,
        "{fp} false positives, expected about {expected}"
    );
    Ok(())
}

#[test]
fn test_invalid_hash_width_is_error() {
    for hash_width in [0, 65] {
        let result = MyBitSignedPHastR::try_new(
            FromCloneableIntoIterator::from(0..1),
            hash_width,
            no_logging![],
        );
        assert!(result.is_err(), "hash width {hash_width} must be rejected");
        let result = MyBitSignedPHastR::try_par_new(&[0_usize], hash_width, no_logging![]);
        assert!(result.is_err(), "hash width {hash_width} must be rejected");
    }
}

#[cfg(feature = "epserde")]
#[test]
fn test_signed_phast_r_mapped_query() -> Result<()> {
    use epserde::deser::Deserialize;
    use epserde::prelude::Aligned64;
    use epserde::ser::Serialize;
    use epserde::utils::AlignedCursor;

    let n = 100_000;
    let func = MySignedPHastR::try_new(FromCloneableIntoIterator::from(0..n), no_logging![])?;

    let mut cursor = <AlignedCursor<Aligned64>>::new();
    // SAFETY: func was built by its safe constructor, so all epserde
    // representation invariants hold.
    unsafe {
        func.serialize(&mut cursor).expect("Could not serialize");
    }
    let len = cursor.len();
    cursor.set_position(0);
    // SAFETY: we just serialized a valid MySignedPHastR into this buffer.
    let mapped =
        unsafe { <MySignedPHastR>::read_mem(&mut cursor, len).expect("Could not deserialize") };
    let mapped = mapped.uncase();
    check(n, |i| mapped.get(i));
    for i in 0..n {
        assert_eq!(func.get(i), mapped.get(i));
    }
    Ok(())
}
