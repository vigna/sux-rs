/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Round trips through serde of rear-coded lists, plain and mapped.

#![cfg(feature = "serde")]

use anyhow::Result;
use sux::bits::BitFieldVec;
use sux::dict::{MappedRearCodedListStr, RearCodedListBuilder, RearCodedListStr};
use sux::traits::IndexedSeq;

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

fn build_rcl(strings: &[&str]) -> RearCodedListStr {
    let mut rclb = RearCodedListBuilder::<str, true>::new(4);
    for s in strings {
        rclb.push(*s);
    }
    rclb.build()
}

#[test]
fn test_serde_rcl_str() -> Result<()> {
    let strings = ["aa", "aab", "abc", "abdd", "abde", "abdf"];
    let back = round_trip(&build_rcl(&strings))?;
    for (i, s) in strings.iter().enumerate() {
        assert_eq!(back.get(i), *s);
    }
    Ok(())
}

#[test]
fn test_serde_mapped_rcl_str() -> Result<()> {
    let strings = ["aa", "aab", "abc", "abdd", "abde", "abdf"];
    let mut map = BitFieldVec::<Vec<usize>>::new(4, 0);
    // Reverse permutation
    for v in (0..strings.len()).rev() {
        map.push(v);
    }
    let mrcl = MappedRearCodedListStr::from_parts(build_rcl(&strings), map.into());
    let back = round_trip(&mrcl)?;
    for i in 0..strings.len() {
        assert_eq!(back.get(i), strings[strings.len() - 1 - i]);
    }
    Ok(())
}
