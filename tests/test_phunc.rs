/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![cfg(feature = "rayon")]

use anyhow::Result;
use dsi_progress_logger::no_logs;
use std::io::Cursor;
use sux::bits::BitFieldVec;
use sux::func::{
    Lcp2MmphfInt, Lcp2MmphfStr, LcpMmphfInt, LcpMmphfStr, Mphf, PHastR, PHastRBuilder, PHunc,
};
use sux::traits::TryIntoUnaligned;
use sux::utils::{FromSlice, LineLender};
use value_traits::slices::{SliceByValue, SliceByValueMut};

// The output of a PHastR is in range for all keys
const _: () = assert!(<PHastR<u64> as Mphf>::CLAMPED);

/// The value of a key.
fn value(key: u64) -> u64 {
    key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20
}

/// Builds functions on the first `n` integers from slices and from
/// lenders, with values in a boxed slice and in a bit-field vector, and
/// checks their values.
fn check(n: usize, builder: impl Fn() -> PHastRBuilder) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<u64> = keys.iter().map(|&k| value(k)).collect();

    macro_rules! check_values {
        ($func:expr) => {{
            let func = $func;
            assert_eq!(func.len(), n);
            assert_eq!(func.is_empty(), n == 0);
            for (key, &value) in keys.iter().zip(&values) {
                assert_eq!(func.get(key), value, "key {key} of {n}");
            }
        }};
    }

    check_values!(<PHunc<PHastR<u64>, Box<[u64]>>>::try_par_new_with_builder(
        &keys,
        &values,
        builder(),
        no_logs![]
    )?);
    check_values!(
        <PHunc<PHastR<u64>, BitFieldVec<Box<[u64]>>>>::try_par_new_with_builder(
            &keys,
            &values,
            builder(),
            no_logs![]
        )?
    );
    for offline in [false, true] {
        check_values!(<PHunc<PHastR<u64>, Box<[u64]>>>::try_new_with_builder(
            FromSlice::new(&keys),
            FromSlice::new(&values),
            builder().offline(offline),
            no_logs![]
        )?);
        check_values!(
            <PHunc<PHastR<u64>, BitFieldVec<Box<[u64]>>>>::try_new_with_builder(
                FromSlice::new(&keys),
                FromSlice::new(&values),
                builder().offline(offline),
                no_logs![]
            )?
        );
    }
    Ok(())
}

#[test]
fn test_small() -> Result<()> {
    for n in [0, 1, 2, 3, 10, 100, 1000, 4096, 4097, 10_000] {
        check(n, PHastRBuilder::default)?;
    }
    Ok(())
}

#[test]
fn test_medium() -> Result<()> {
    check(300_000, PHastRBuilder::default)
}

/// Seeds in a bit-field vector.
#[test]
fn test_seeds() -> Result<()> {
    let keys: Vec<u64> = (0..50_000).collect();
    let values: Vec<u64> = keys.iter().map(|&k| value(k)).collect();
    let builder = || PHastRBuilder::default().seed_bits(10).log2_slice_len(11);
    let func =
        <PHunc<PHastR<u64, BitFieldVec<Box<[usize]>>>, Box<[u64]>>>::try_par_new_with_builder(
            &keys,
            &values,
            builder(),
            no_logs![],
        )?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), value);
    }
    let func = <PHunc<PHastR<u64, BitFieldVec<Box<[usize]>>>, BitFieldVec<Box<[u64]>>>>::try_new_with_builder(
        FromSlice::new(&keys),
        FromSlice::new(&values),
        builder(),
        no_logs![],
    )?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), value);
    }
    Ok(())
}

#[test]
fn test_strings() -> Result<()> {
    let keys: Vec<String> = (0..50_000).map(|i| format!("key{i}")).collect();
    let values: Vec<u32> = (0..50_000).map(|i| i * 3).collect();
    let func = <PHunc<PHastR<str>, Box<[u32]>>>::try_par_new(&keys, &values, no_logs![])?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(func.get(key.as_str()), value);
    }
    let func = <PHunc<PHastR<str>, BitFieldVec<Box<[u32]>>>>::try_new(
        FromSlice::new(&keys),
        FromSlice::new(&values),
        no_logs![],
    )?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(func.get(key.as_str()), value);
    }
    Ok(())
}

/// Values that are not integers.
#[test]
fn test_value_types() -> Result<()> {
    let keys: Vec<u64> = (0..20_000).collect();
    let values: Vec<[u16; 3]> = keys
        .iter()
        .map(|&k| [k as u16, (k >> 3) as u16, (k >> 7) as u16])
        .collect();
    let func = <PHunc<PHastR<u64>, Box<[[u16; 3]]>>>::try_par_new(&keys, &values, no_logs![])?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), value);
    }
    let values: Vec<f64> = keys.iter().map(|&k| k as f64 / 3.0).collect();
    let func = <PHunc<PHastR<u64>, Box<[f64]>>>::try_new(
        FromSlice::new(&keys),
        FromSlice::new(&values),
        no_logs![],
    )?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(func.get(key), value);
    }
    Ok(())
}

/// The output of a key outside the original set must be smaller than the
/// number of keys (zero, if there are none), as [`PHunc`] reads its values
/// without bounds checks.
#[test]
fn test_foreign_keys() -> Result<()> {
    for n in [0, 1, 2, 3, 10, 100, 1000, 4097, 100_000] {
        let keys: Vec<u64> = (0..n as u64).collect();
        let values: Vec<u64> = keys.iter().map(|&k| value(k)).collect();
        for builder in [
            PHastRBuilder::default(),
            PHastRBuilder::default().log2_slice_len(6),
        ] {
            let func = <PHunc<PHastR<u64>, Box<[u64]>>>::try_par_new_with_builder(
                &keys,
                &values,
                builder,
                no_logs![],
            )?;
            let (phf, data) = func.into_parts();
            assert_eq!(data.len(), n.max(1));
            for key in n as u64..n as u64 + 200_000 {
                let index = phf.index(&key);
                assert!(index < n.max(1), "index {index} with {n} keys");
            }
            let func = PHunc::from_parts(phf, data);
            for key in n as u64..n as u64 + 1000 {
                let _ = func.get(key);
            }
        }
    }
    Ok(())
}

/// Checks a function based on a monotone minimal perfect hash function on
/// the given sorted keys: the array contains the values in the order of the
/// keys. The output of a monotone function for a key outside the original
/// set can be larger than the number of keys, so the function clamps it.
fn check_monotone<F: Mphf<Key: ToOwned>>(
    phf: F,
    keys: &[<F::Key as ToOwned>::Owned],
    foreign: impl Iterator<Item = <F::Key as ToOwned>::Owned>,
) where
    <F::Key as ToOwned>::Owned: std::borrow::Borrow<F::Key>,
{
    use std::borrow::Borrow;
    assert!(!F::CLAMPED);
    let n = keys.len();
    assert_eq!(phf.len(), n);
    // One padding value if there are no keys
    let data: Vec<u64> = (0..n.max(1) as u64).map(value).collect();
    let foreign: Vec<_> = foreign.collect();
    let indices: Vec<usize> = foreign.iter().map(|key| phf.index(key.borrow())).collect();
    let func = PHunc::from_parts(phf, data.clone());
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.borrow()), data[i], "key {i} of {n}");
        // The position of a key is its rank
        assert_eq!(func.index(key.borrow()), i);
        assert_eq!(func.get_with_index(key.borrow()), (i, data[i]));
    }
    for (key, &index) in foreign.iter().zip(&indices) {
        let clamped = index.min(data.len() - 1);
        assert_eq!(func.get(key.borrow()), data[clamped]);
        assert_eq!(func.index(key.borrow()), clamped);
        assert_eq!(func.get_with_index(key.borrow()), (clamped, data[clamped]));
    }
    // Otherwise, this test does not check the clamping (tiny functions can
    // have just in-range outputs)
    if n >= 100 {
        assert!(indices.iter().any(|&index| index >= n), "{n} keys");
    }
}

#[test]
fn test_monotone() -> Result<()> {
    for n in [0, 1, 2, 100, 10_000, 100_000] {
        let keys: Vec<u64> = (0..n as u64).map(|i| i * 7 + 3).collect();
        let foreign = || (0..100_000_u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 << 40);
        check_monotone(
            <LcpMmphfInt<u64>>::try_new(FromSlice::new(&keys), n, no_logs![])?,
            &keys,
            foreign(),
        );
        check_monotone(
            <Lcp2MmphfInt<u64>>::try_new(FromSlice::new(&keys), n, no_logs![])?,
            &keys,
            foreign(),
        );

        let keys: Vec<String> = (0..n).map(|i| format!("key{i:08}")).collect();
        let lender = || LineLender::new(Cursor::new(keys.join("\n").into_bytes()));
        let foreign = || (0..100_000).map(|i| format!("foreign{i}"));
        check_monotone(
            <LcpMmphfStr>::try_new(lender(), n, no_logs![])?,
            &keys,
            foreign(),
        );
        check_monotone(
            <Lcp2MmphfStr>::try_new(lender(), n, no_logs![])?,
            &keys,
            foreign(),
        );
    }
    Ok(())
}

/// The positions of the values can be used to access other arrays.
#[test]
fn test_index() -> Result<()> {
    let keys: Vec<u64> = (0..50_000).collect();
    let values: Vec<u64> = keys.iter().map(|&k| value(k)).collect();
    let func = <PHunc<PHastR<u64>, BitFieldVec<Box<[u64]>>>>::try_new(
        FromSlice::new(&keys),
        FromSlice::new(&values),
        no_logs![],
    )?;
    // Ancillary data, indexed like the values
    let mut names = vec![String::new(); keys.len()];
    for &key in &keys {
        let index = func.index(key);
        assert!(names[index].is_empty(), "position {index} is not unique");
        names[index] = format!("name{key}");
    }
    for &key in &keys {
        let (index, v) = func.get_with_index(key);
        assert_eq!(index, func.index(key));
        assert_eq!(index, func.mphf().index(&key));
        assert_eq!(v, value(key));
        assert_eq!(func.values().index_value(index), v);
        assert_eq!(names[index], format!("name{key}"));
    }
    // Keys outside the original set still have a valid position
    for key in 50_000..100_000_u64 {
        let (index, v) = func.get_with_index(key);
        assert!(index < keys.len());
        assert_eq!(func.values().index_value(index), v);
    }
    Ok(())
}

/// Functions can be assembled from references to their parts, so several
/// functions can share the same minimal perfect hash function.
#[test]
fn test_from_parts_refs() -> Result<()> {
    let keys: Vec<u64> = (0..10_000).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logs![])?;
    let mut values = vec![0_u64; keys.len()];
    let mut squares = BitFieldVec::<Box<[usize]>>::new_padded(40, keys.len());
    for &key in &keys {
        let index = phf.index(&key);
        values[index] = value(key);
        squares.set_value(index, (key * key) as usize);
    }

    // References to a vector, to a slice, and to a bit-field vector
    let by_vec = PHunc::from_parts(&phf, &values);
    let by_slice = PHunc::from_parts(&phf, values.as_slice());
    let by_bits = PHunc::from_parts(&phf, &squares);
    for &key in &keys {
        assert_eq!(by_vec.get(key), value(key));
        assert_eq!(by_slice.get(key), value(key));
        assert_eq!(by_bits.get(key), (key * key) as usize);
        assert_eq!(by_vec.index(key), by_bits.index(key));
    }

    // From the parts of another function
    let owned = PHunc::from_parts(&phf, values.clone());
    let borrowed = PHunc::from_parts(owned.mphf(), owned.values());
    // A boxed function
    let boxed = PHunc::from_parts(Box::new(&phf), values.as_slice());
    for &key in &keys {
        assert_eq!(borrowed.get(key), value(key));
        assert_eq!(boxed.get(key), value(key));
    }
    Ok(())
}

#[test]
fn test_from_parts() -> Result<()> {
    let keys: Vec<u64> = (0..10_000).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logs![])?;
    let mut data = vec![0_u64; keys.len()];
    for &key in &keys {
        data[phf.index(&key)] = value(key);
    }
    let func = PHunc::from_parts(phf, data);
    for &key in &keys {
        assert_eq!(func.get(key), value(key));
    }
    Ok(())
}

#[test]
#[should_panic(expected = "the number of values")]
fn test_from_parts_wrong_len() {
    let keys: Vec<u64> = (0..100).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logs![]).unwrap();
    let _ = PHunc::from_parts(phf, vec![0_u64; 99]);
}

#[test]
fn test_unaligned() -> Result<()> {
    let keys: Vec<u64> = (0..100_000).collect();
    let values: Vec<usize> = keys.iter().map(|&k| value(k) as usize % 100_000).collect();
    let func =
        <PHunc<PHastR<u64>, BitFieldVec<Box<[usize]>>>>::try_par_new(&keys, &values, no_logs![])?;
    let unaligned = func.clone().try_into_unaligned()?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(unaligned.get(key), value);
    }
    let aligned: PHunc<PHastR<u64>, BitFieldVec<Box<[usize]>>> = unaligned.into();
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(aligned.get(key), value);
    }
    Ok(())
}

/// Functions deserialized with ε-serde borrow their data, and give the same
/// values.
#[cfg(feature = "epserde")]
#[test]
fn test_epserde() -> Result<()> {
    use epserde::prelude::*;
    use epserde::utils::AlignedCursor;

    macro_rules! check_epserde {
        ($t:ty, $n:expr) => {{
            let keys: Vec<u64> = (0..$n).collect();
            let values: Vec<u64> = keys.iter().map(|&k| value(k)).collect();
            let func = <$t>::try_par_new(&keys, &values, no_logs![])?;
            let mut cursor = <AlignedCursor<Aligned64>>::new();
            // SAFETY: func was built by its constructor
            unsafe { func.serialize(&mut cursor) }?;
            let len = cursor.len();
            cursor.set_position(0);
            // SAFETY: we just serialized a valid structure into this buffer
            let case = unsafe { <$t>::read_mem(&mut cursor, len) }?;
            let des = case.uncase();
            assert_eq!(des.len(), keys.len());
            for (key, &value) in keys.iter().zip(&values) {
                assert_eq!(des.get(key), value);
            }
        }};
    }

    for n in [0, 1, 100_000] {
        check_epserde!(PHunc<PHastR<u64>, Box<[u64]>>, n);
        check_epserde!(PHunc<PHastR<u64>, BitFieldVec<Box<[u64]>>>, n);
    }
    Ok(())
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() -> Result<()> {
    let keys: Vec<u64> = (0..10_000).collect();
    let values: Vec<u64> = keys.iter().map(|&k| value(k)).collect();
    let func =
        <PHunc<PHastR<u64>, BitFieldVec<Box<[u64]>>>>::try_par_new(&keys, &values, no_logs![])?;
    let json = serde_json::to_string(&func)?;
    let des: PHunc<PHastR<u64>, BitFieldVec<Box<[u64]>>> = serde_json::from_str(&json)?;
    for (key, &value) in keys.iter().zip(&values) {
        assert_eq!(des.get(key), value);
    }
    Ok(())
}
