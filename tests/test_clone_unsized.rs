/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Structures with unsized keys (e.g., `str`) must be `Clone`, `Send`, and
//! `Sync`: derived implementations would require the key type to be `Clone`,
//! too, and a marker owning the key type would make the auto traits depend
//! on it.

use sux::bits::{BitFieldVec, BitVec};
use sux::dict::{MappedRearCodedListStr, RearCodedListStr, VFilter};
use sux::func::{CompVFunc, Lcp2MmphfStr, LcpMmphfStr, PHastR, SignedFunc, VFunc, VFunc2};

fn assert_clone<T: Clone>() {}

#[test]
fn test_unsized_keys_are_clone() {
    assert_clone::<VFunc<str, BitFieldVec<Box<[usize]>>>>();
    assert_clone::<VFunc<[u8], BitFieldVec<Box<[usize]>>>>();
    assert_clone::<VFunc2<str>>();
    assert_clone::<CompVFunc<str, BitVec<Box<[usize]>>>>();
    assert_clone::<VFilter<str, BitFieldVec<Box<[usize]>>>>();
    assert_clone::<LcpMmphfStr>();
    assert_clone::<Lcp2MmphfStr>();
    assert_clone::<RearCodedListStr>();
    assert_clone::<MappedRearCodedListStr>();
    assert_clone::<SignedFunc<VFunc<str, BitFieldVec<Box<[usize]>>>, Box<[u64]>>>();
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn test_unsized_keys_are_send_sync() {
    assert_send_sync::<VFunc<str, BitFieldVec<Box<[usize]>>>>();
    assert_send_sync::<CompVFunc<str, BitVec<Box<[usize]>>>>();
    assert_send_sync::<VFilter<str, BitFieldVec<Box<[usize]>>>>();
    assert_send_sync::<LcpMmphfStr>();
    assert_send_sync::<Lcp2MmphfStr>();
    assert_send_sync::<PHastR<str>>();
}
