/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![cfg(feature = "rayon")]
use anyhow::Result;
use core::borrow::Borrow;
use core::sync::atomic::{AtomicUsize, Ordering};
use dsi_progress_logger::no_logging;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::bits::BitFieldVec;
use sux::func::{Lcp2MmphfInt, Lcp2MmphfStr, LcpMmphf, LcpMmphfInt, SignedFunc};
use sux::traits::TryIntoUnaligned;
use sux::utils::{FromSlice, ToSig};

type SignedLcpMmphfStr = SignedFunc<LcpMmphf<str>, Box<[u64]>>;
type SignedLcpMmphfSliceU8 = SignedFunc<LcpMmphf<[u8]>, Box<[u64]>>;
type SignedLcpMmphfInt<T> = SignedFunc<LcpMmphfInt<T>, Box<[u64]>>;
type BitSignedLcpMmphfInt<T> = SignedFunc<LcpMmphfInt<T>, BitFieldVec<Box<[usize]>>>;
type BitSignedLcpMmphfStr = SignedFunc<LcpMmphf<str>, BitFieldVec<Box<[usize]>>>;

static SIGNATURE_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CountingKey(Vec<u8>);

impl AsRef<[u8]> for CountingKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl ToSig<[u64; 2]> for CountingKey {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        SIGNATURE_CALLS.fetch_add(1, Ordering::Relaxed);
        <[u8] as ToSig<[u64; 2]>>::to_sig(key.borrow().as_ref(), seed)
    }
}

// ── String tests ────────────────────────────────────────────────────

#[test]
fn test_str_small() -> Result<()> {
    let keys = vec![
        "alpha".to_owned(),
        "beta".to_owned(),
        "delta".to_owned(),
        "gamma".to_owned(),
    ];
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            func.get(key.as_str()),
            Some(i),
            "key {key:?} at position {i}"
        );
    }
    assert_eq!(func.get("not_a_key"), None);
    assert_eq!(func.get(""), None);
    Ok(())
}

#[test]
fn test_str_single() -> Result<()> {
    let keys = vec!["hello".to_owned()];
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), 1, no_logging![])?;
    let func = func.try_into_unaligned()?;
    assert_eq!(func.get("hello"), Some(0));
    assert_eq!(func.get("world"), None);
    assert_eq!(func.len(), 1);
    assert!(!func.is_empty());
    Ok(())
}

#[test]
fn test_str_empty() -> Result<()> {
    let keys: Vec<String> = vec![];
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), 0, no_logging![])?;
    let func = func.try_into_unaligned()?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    assert_eq!(func.get("anything"), None);
    Ok(())
}

#[test]
fn test_str_1000() -> Result<()> {
    let mut keys: Vec<String> = (0..1000).map(|i| format!("key_{:06}", i)).collect();
    keys.sort();
    let func: SignedLcpMmphfStr =
        SignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            func.get(key.as_str()),
            Some(i),
            "key {key:?} at position {i}"
        );
    }
    assert_eq!(func.get("zzz_not_present"), None);
    assert_eq!(func.get("key_999999"), None);
    Ok(())
}

// ── [u8] tests ──────────────────────────────────────────────────────

#[test]
fn test_slice_u8_small() -> Result<()> {
    let keys: Vec<Vec<u8>> = vec![
        b"alpha".to_vec(),
        b"beta".to_vec(),
        b"delta".to_vec(),
        b"gamma".to_vec(),
    ];
    let func: SignedLcpMmphfSliceU8 =
        SignedLcpMmphfSliceU8::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_slice()), Some(i));
    }
    assert_eq!(func.get(b"not_a_key".as_slice()), None);
    Ok(())
}

// ── Integer tests ───────────────────────────────────────────────────

#[test]
fn test_u64_small() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: SignedLcpMmphfInt<u64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(999), None);
    assert_eq!(func.get(0), None);
    Ok(())
}

#[test]
fn test_u64_empty() -> Result<()> {
    let keys: Vec<u64> = vec![];
    let func: SignedLcpMmphfInt<u64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), 0, no_logging![])?;
    let func = func.try_into_unaligned()?;
    assert_eq!(func.len(), 0);
    assert!(func.is_empty());
    assert_eq!(func.get(42), None);
    Ok(())
}

#[test]
fn test_u64_1000() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();

    let func: SignedLcpMmphfInt<u64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), n, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(0), None);
    Ok(())
}

#[test]
fn test_signed_i64() -> Result<()> {
    let keys: Vec<i64> = vec![-100, -10, -1, 0, 1, 10, 100];
    let func: SignedLcpMmphfInt<i64> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(999), None);
    assert_eq!(func.get(-999), None);
    Ok(())
}

#[test]
fn test_u32() -> Result<()> {
    let keys: Vec<u32> = vec![1, 100, 1000, 10000, 100000];
    let func: SignedLcpMmphfInt<u32> =
        SignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(42), None);
    Ok(())
}

// ── BitSigned tests (sub-word-width hashes) ─────────────────────────

#[test]
fn test_bit_signed_u64_8bit() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: BitSignedLcpMmphfInt<u64> =
        BitSignedLcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(999), None);
    Ok(())
}

#[test]
fn test_bit_signed_str_12bit() -> Result<()> {
    let keys = vec![
        "alpha".to_owned(),
        "beta".to_owned(),
        "delta".to_owned(),
        "gamma".to_owned(),
    ];
    let func: BitSignedLcpMmphfStr =
        BitSignedLcpMmphfStr::try_new(FromSlice::new(&keys), keys.len(), 12, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            func.get(key.as_str()),
            Some(i),
            "key {key:?} at position {i}"
        );
    }
    assert_eq!(func.get("not_a_key"), None);
    Ok(())
}

#[test]
fn test_bit_signed_u64_1000() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();

    let func: BitSignedLcpMmphfInt<u64> =
        BitSignedLcpMmphfInt::try_new(FromSlice::new(&keys), n, 16, no_logging![])?;
    let func = func.try_into_unaligned()?;

    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key} at position {i}");
    }
    assert_eq!(func.get(0), None);
    Ok(())
}

// ── Signed Lcp2Mmphf tests ──────────────────────────────────────────

type SignedLcp2MmphfStr = SignedFunc<Lcp2MmphfStr, Box<[u64]>>;
type SignedLcp2MmphfInt<T> = SignedFunc<Lcp2MmphfInt<T>, Box<[u64]>>;
type BitSignedLcp2MmphfInt<T> = SignedFunc<Lcp2MmphfInt<T>, BitFieldVec<Box<[usize]>>>;

#[test]
fn test_lcp2_u64_small() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: SignedLcp2MmphfInt<u64> =
        SignedLcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key}");
    }
    assert_eq!(func.get(999), None);
    Ok(())
}

#[test]
fn test_lcp2_u64_1000() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut keys: Vec<u64> = (0..1000).map(|_| rng.random::<u64>()).collect();
    keys.sort();
    keys.dedup();
    let n = keys.len();
    let func: SignedLcp2MmphfInt<u64> =
        SignedLcp2MmphfInt::try_new(FromSlice::new(&keys), n, no_logging![])?;
    let func = func.try_into_unaligned()?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key}");
    }
    assert_eq!(func.get(u64::MAX), None);
    Ok(())
}

#[test]
fn test_lcp2_str_small() -> Result<()> {
    let mut keys = vec![
        "alpha".to_owned(),
        "beta".to_owned(),
        "delta".to_owned(),
        "gamma".to_owned(),
    ];
    keys.sort();
    let func: SignedLcp2MmphfStr =
        SignedLcp2MmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
    let func = func.try_into_unaligned()?;
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(func.get(key.as_str()), Some(i), "key {key}");
    }
    assert_eq!(func.get("absent"), None);
    Ok(())
}

#[test]
fn test_lcp2_bit_signed_u64() -> Result<()> {
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    let func: BitSignedLcp2MmphfInt<u64> =
        BitSignedLcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), 8, no_logging![])?;
    let func = func.try_into_unaligned()?;
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!(func.get(key), Some(i), "key {key}");
    }
    Ok(())
}

#[test]
fn test_signed_lcp_queries_hash_key_once() -> Result<()> {
    let keys = ["alpha", "beta", "delta", "gamma"].map(|key| CountingKey(key.as_bytes().to_vec()));

    type CountingLcp = SignedFunc<LcpMmphf<CountingKey>, Box<[u64]>>;
    let lcp = CountingLcp::try_par_new(&keys, no_logging![])?;
    SIGNATURE_CALLS.store(0, Ordering::Relaxed);
    assert_eq!(lcp.get(&keys[2]), Some(2));
    assert_eq!(SIGNATURE_CALLS.load(Ordering::Relaxed), 1);

    type CountingLcp2 = SignedFunc<sux::func::Lcp2Mmphf<CountingKey>, Box<[u64]>>;
    let lcp2 = CountingLcp2::try_par_new(&keys, no_logging![])?;
    SIGNATURE_CALLS.store(0, Ordering::Relaxed);
    assert_eq!(lcp2.get(&keys[2]), Some(2));
    assert_eq!(SIGNATURE_CALLS.load(Ordering::Relaxed), 1);
    Ok(())
}
