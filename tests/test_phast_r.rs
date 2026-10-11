/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use std::borrow::Borrow;

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::func::phast_r::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder};
use sux::traits::TryIntoUnaligned;
use sux::utils::{FromSlice, ToSig};

/// Builds a function on the first `n` integers from a slice, and checks that
/// it is a minimal perfect hash function.
fn check<D: SeedStoreBuild>(n: usize, builder: PHastRBuilder) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let phf = <PHastR<u64, D>>::try_par_new_with_builder(&keys, builder, no_logging![])?;
    assert_eq!(phf.len(), n);
    let mut seen = vec![false; n];
    for key in &keys {
        let v = phf.get(key);
        assert!(v < n, "{v} >= {n}");
        assert!(!seen[v], "duplicate output {v}");
        seen[v] = true;
    }
    Ok(())
}

#[test]
fn test_small() -> Result<()> {
    for n in [0, 1, 2, 3, 10, 100, 1000, 5000, 10000] {
        check::<Box<[u8]>>(n, PHastRBuilder::default())?;
        check::<BitFieldVec<Box<[usize]>>>(n, PHastRBuilder::default().seed_bits(10))?;
        for log2_layouts in 0..=3 {
            check::<Box<[u8]>>(
                n,
                PHastRBuilder::default()
                    .log2_layouts(log2_layouts)
                    .log2_slice_len(8),
            )?;
        }
    }
    Ok(())
}

#[test]
fn test_medium() -> Result<()> {
    check::<Box<[u8]>>(1_000_000, PHastRBuilder::default())?;
    check::<Box<[u8]>>(300_000, PHastRBuilder::default().bucket_size(5.0))?;
    // Rings of four words (10-bit seeds, or a single layout) and two words
    // (two layouts); with the default parameters rings have one word
    check::<BitFieldVec<Box<[usize]>>>(
        300_000,
        PHastRBuilder::default()
            .seed_bits(10)
            .log2_slice_len(11)
            .bucket_size(6.0),
    )?;
    check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_layouts(1))?;
    check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_layouts(0))?;
    check::<BitFieldVec<Box<[usize]>>>(
        300_000,
        PHastRBuilder::default()
            .seed_bits(11)
            .log2_slice_len(12)
            .bucket_size(6.75),
    )?;
    // Longer slices, and thus a larger stride
    check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(12))?;
    check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(16))?;
    // Rings shorter than a word: a layout has four rotations
    check::<Box<[u8]>>(
        100_000,
        PHastRBuilder::default()
            .seed_bits(4)
            .log2_slice_len(8)
            .bucket_size(2.0),
    )?;
    // Slices with fewer slots than there are seeds: rotations are redundant
    check::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(6))
}

#[test]
fn test_large_buckets() -> Result<()> {
    // Large buckets (kept in a heap, and with several groups of keys) and
    // very long rings
    for log2_layouts in [0, 2] {
        check::<BitFieldVec<Box<[usize]>>>(
            20_000,
            PHastRBuilder::default()
                .seed_bits(16)
                .log2_layouts(log2_layouts)
                .log2_slice_len(16)
                .bucket_size(64.0),
        )?;
    }
    check::<Box<[u8]>>(100_000, PHastRBuilder::default().bucket_size(64.0))
}

#[cfg(feature = "rayon")]
#[test]
fn test_many_chunks() -> Result<()> {
    // Many threads, and hence many chunks and gaps, with long slices
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build()?;
    pool.install(|| {
        check::<BitFieldVec<Box<[usize]>>>(
            2_000_000,
            PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        )?;
        check::<Box<[u8]>>(2_000_000, PHastRBuilder::default())
    })
}

#[test]
fn test_parameter_checks() {
    let keys: Vec<u64> = (0..1000).collect();
    let build = |builder: PHastRBuilder| {
        <PHastR<u64>>::try_par_new_with_builder(&keys, builder, no_logging![])
    };
    // Seeds too large for the storage
    assert!(build(PHastRBuilder::default().seed_bits(10)).is_err());
    // Too many layouts for the slice length
    assert!(build(PHastRBuilder::default().log2_layouts(3)).is_err());
    // Too many layouts for the number of seed bits
    assert!(build(PHastRBuilder::default().seed_bits(2)).is_err());
    // Too few seed bits
    assert!(build(PHastRBuilder::default().seed_bits(1).log2_layouts(0)).is_err());
    // Offline mode needs a lender
    assert!(build(PHastRBuilder::default().offline(true)).is_err());
    // Expected bucket sizes out of range (at most an eighth of the slice
    // length, and at most 64)
    for bucket_size in [f64::NAN, f64::INFINITY, -1.0, 0.0, 0.5, 64.5, 1000.0] {
        assert!(build(PHastRBuilder::default().bucket_size(bucket_size)).is_err());
    }
    assert!(build(PHastRBuilder::default().log2_slice_len(8).bucket_size(33.0)).is_err());
    assert!(build(PHastRBuilder::default().log2_slice_len(2)).is_err());
    assert!(build(PHastRBuilder::default().bucket_size(1.0)).is_ok());
    assert!(build(PHastRBuilder::default().bucket_size(64.0)).is_ok());
    // Weights out of range
    assert!(build(PHastRBuilder::default().weights([1 << 33, 0, 0, 0, 0, 0, 0])).is_err());
}

#[test]
fn test_construction_failures() {
    // Construction can fail with very short slices or very large buckets
    let keys: Vec<u64> = (0..20_000).collect();
    let err = <PHastR<u64>>::try_par_new_with_builder(
        &keys,
        PHastRBuilder::default()
            .log2_layouts(0)
            .log2_slice_len(4)
            .bucket_size(2.0),
        no_logging![],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("Could not build the last level"),
        "{err}"
    );
    let keys: Vec<u64> = (0..100_000).collect();
    let err = <PHastR<u64>>::try_par_new_with_builder(
        &keys,
        PHastRBuilder::default()
            .seed_bits(2)
            .log2_layouts(0)
            .log2_slice_len(9)
            .bucket_size(64.0),
        no_logging![],
    )
    .unwrap_err();
    assert!(err.to_string().contains("could be placed"), "{err}");
}

#[test]
fn test_weights_and_generic_queries() -> Result<()> {
    // Explicit default weights give the same function, and the generic
    // query code gives the same outputs as the code with constant shifts
    let keys: Vec<u64> = (0..300_000).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
    let weighted = <PHastR<u64>>::try_par_new_with_builder(
        &keys,
        PHastRBuilder::default().weights([-52000, 53315, 105861, 138629, 156944, 175176, 186591]),
        no_logging![],
    )?;
    assert_eq!(format!("{weighted:?}"), format!("{phf:?}"));
    // Other weights give a valid function
    let other = <PHastR<u64>>::try_par_new_with_builder(
        &keys,
        PHastRBuilder::default().weights([0, 1 << 16, 2 << 16, 3 << 16, 4 << 16, 5 << 16, 6 << 16]),
        no_logging![],
    )?;
    let generic = phf.clone().generic_queries();
    let mut seen = vec![false; keys.len()];
    for key in &keys {
        assert_eq!(generic.get(key), phf.get(key));
        let v = other.get(key);
        assert!(!seen[v]);
        seen[v] = true;
    }
    Ok(())
}

/// Serialization with serde and deserialization give the same function.
#[cfg(feature = "serde")]
#[test]
fn test_serde() -> Result<()> {
    fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug>(
        t: &T,
    ) -> Result<()> {
        let json = serde_json::to_string(t)?;
        let back: T = serde_json::from_str(&json)?;
        assert_eq!(format!("{back:?}"), format!("{t:?}"));
        Ok(())
    }
    let keys: Vec<u64> = (0..100_000).collect();
    round_trip(&<PHastR<u64>>::try_par_new(&keys, no_logging![])?)?;
    let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
        &keys,
        PHastRBuilder::default().seed_bits(10).log2_slice_len(11),
        no_logging![],
    )?;
    round_trip(&phf)?;
    round_trip(&phf.try_into_unaligned()?)?;
    let keys: Vec<String> = (0..10_000).map(|i| format!("key{i}")).collect();
    round_trip(&<PHastR<str>>::try_par_new(&keys, no_logging![])?)?;
    Ok(())
}

#[test]
fn test_clone_unsized() -> Result<()> {
    // Functions are Send and Sync whatever the type of the keys
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PHastR<str>>();
    send_sync::<PHastR<u64, BitFieldVec<Box<[usize]>>>>();
    // Clone and Debug do not require the type of the keys to implement them
    let keys = ["a", "b", "c"];
    let phf = <PHastR<str>>::try_par_new(&keys, no_logging![])?;
    let clone = phf.clone();
    assert_eq!(format!("{phf:?}"), format!("{clone:?}"));
    for key in keys {
        assert_eq!(clone.get(key), phf.get(key));
    }
    Ok(())
}

#[test]
fn test_unaligned_round_trip() -> Result<()> {
    // Conversion to unaligned reads and back, for both seed storages (seeds
    // and remapping sequence are already padded, so nothing changes)
    let keys: Vec<u64> = (0..100_000).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
    let back: PHastR<u64> = phf.clone().try_into_unaligned()?.into();
    assert_eq!(format!("{back:?}"), format!("{phf:?}"));
    for key in &keys {
        assert_eq!(back.get(key), phf.get(key));
    }
    let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
        &keys,
        PHastRBuilder::default().seed_bits(10).log2_slice_len(11),
        no_logging![],
    )?;
    let back: PHastR<u64, BitFieldVec<Box<[usize]>>> = phf.clone().try_into_unaligned()?.into();
    assert_eq!(format!("{back:?}"), format!("{phf:?}"));
    for key in &keys {
        assert_eq!(back.get(key), phf.get(key));
    }
    Ok(())
}

#[test]
fn test_lender() -> Result<()> {
    // The constructors reading keys from a lender, in memory and offline
    let keys: Vec<u64> = (0..100_000).collect();
    for builder in [
        PHastRBuilder::default(),
        PHastRBuilder::default().offline(true),
    ] {
        let phf =
            <PHastR<u64>>::try_new_with_builder(FromSlice::new(&keys), builder, no_logging![])?;
        assert_eq!(phf.len(), keys.len());
        let mut seen = vec![false; keys.len()];
        for key in &keys {
            let v = phf.get(key);
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
    }
    let phf = <PHastR<u64>>::try_new(FromSlice::new(&keys), no_logging![])?;
    assert_eq!(phf.len(), keys.len());
    Ok(())
}

#[test]
fn test_unaligned() -> Result<()> {
    for (bits, log2_slice_len, n) in [
        (10, 11, 300_000),
        (12, 12, 300_000),
        (16, 12, 10_000),
        (10, 11, 0),
    ] {
        let keys: Vec<u64> = (0..n as u64).collect();
        let phf = <PHastR<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
            &keys,
            PHastRBuilder::default()
                .seed_bits(bits)
                .log2_slice_len(log2_slice_len)
                .bucket_size(6.0),
            no_logging![],
        )?;
        let expected: Vec<usize> = keys.iter().map(|k| phf.get(k)).collect();
        let phf = phf.try_into_unaligned()?;
        for (k, &e) in keys.iter().zip(&expected) {
            assert_eq!(phf.get(k), e);
        }
        let phf: PHastR<u64, BitFieldVec<Box<[usize]>>> = phf.into();
        for (k, &e) in keys.iter().zip(&expected) {
            assert_eq!(phf.get(k), e);
        }
    }
    Ok(())
}

#[test]
fn test_batch() -> Result<()> {
    let keys: Vec<u64> = (0..100_000).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
    for chunk in keys.chunks_exact(8) {
        let batch: [&u64; 8] = std::array::from_fn(|i| &chunk[i]);
        let values = phf.get_batch(batch);
        for (key, value) in chunk.iter().zip(values) {
            assert_eq!(value, phf.get(key));
        }
    }
    Ok(())
}

#[test]
fn test_duplicates() {
    let keys: Vec<u64> = vec![1, 2, 3, 2];
    assert!(<PHastR<u64>>::try_par_new(&keys, no_logging![]).is_err());
}

/// A key whose 64-bit signature with seed 0 collides with that of another key
/// for the first 2 · `COLLIDING` keys (pairs 2*i*, 2*i* + 1); other seeds give
/// independent signatures.
#[derive(Clone, Copy)]
struct CollidingKey(u64);
const COLLIDING: u64 = 2000;

impl ToSig<[u64; 1]> for CollidingKey {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let k = key.borrow().0;
        if seed == 0 && k < 2 * COLLIDING {
            <u64 as ToSig<[u64; 1]>>::to_sig(k / 2, 0)
        } else {
            <u64 as ToSig<[u64; 1]>>::to_sig(k, seed)
        }
    }
}

#[test]
fn test_signature_collisions() -> Result<()> {
    // Colliding signatures are bumped and separated by the signatures of the
    // following levels, with all configurations of the following levels (in
    // particular, with only the last level)
    for n in [5_000u64, 300_000] {
        let keys: Vec<CollidingKey> = (0..n).map(CollidingKey).collect();
        let phf = <PHastR<CollidingKey>>::try_par_new(&keys, no_logging![])?;
        let mut seen = vec![false; keys.len()];
        for &key in &keys {
            if key.0 < 2 * COLLIDING {
                assert!(phf.is_bumped(key));
            }
            let v = phf.get(key);
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
    }
    // Equal keys are still detected
    let mut keys: Vec<CollidingKey> = (0..100_000).map(CollidingKey).collect();
    keys.push(CollidingKey(50_000));
    assert!(<PHastR<CollidingKey>>::try_par_new(&keys, no_logging![]).is_err());
    Ok(())
}

/// A memory-mapped function must take little memory beyond the mapping: with
/// the default flags mem_dbg does not follow references, so it counts only
/// the fields of the structure, whereas following references it must count
/// the same data as the original.
#[cfg(feature = "mmap")]
#[test]
fn test_mmap_mem_size() -> Result<()> {
    use epserde::prelude::*;
    use mem_dbg::{MemSize, SizeFlags};
    let keys: Vec<u64> = (0..1_000_000).collect();
    let phf = <PHastR<u64>>::try_par_new(&keys, no_logging![])?;
    let file = tempfile::NamedTempFile::new()?;
    // SAFETY: phf was built by its constructor
    unsafe { phf.store(file.path()) }?;
    // SAFETY: we just stored a valid structure into the file
    let case = unsafe { <PHastR<u64>>::mmap(file.path(), Flags::empty()) }?;
    let mapped = case.uncase();
    let owned_size = phf.mem_size(SizeFlags::default());
    let mapped_size = mapped.mem_size(SizeFlags::default());
    let mapped_followed = mapped.mem_size(SizeFlags::FOLLOW_REFS);
    assert!(
        mapped_size < 1024,
        "mapped structure takes {mapped_size} bytes"
    );
    assert!(
        mapped_followed >= owned_size - 1024 && mapped_followed <= owned_size + 1024,
        "owned {owned_size} bytes, mapped following references {mapped_followed} bytes"
    );
    for key in &keys {
        assert_eq!(mapped.get(key), phf.get(key));
    }
    Ok(())
}

#[test]
fn test_strings() -> Result<()> {
    let keys: Vec<String> = (0..200_000).map(|i| format!("key{i}")).collect();
    let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let phf = <PHastR<str>>::try_par_new(&refs, no_logging![])?;
    let mut seen = vec![false; keys.len()];
    for key in &keys {
        let v = phf.get(key.as_str());
        assert!(!seen[v], "duplicate output {v}");
        seen[v] = true;
    }
    Ok(())
}
