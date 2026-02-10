/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::prelude::*;
use sux::traits::{IndexedSeq, Pred, Succ, Types};

// Tests for IndexedSeq on slices
#[test]
fn test_indexed_seq_slice() {
    let slice: &[u32] = &[10, 20, 30, 40, 50];
    assert_eq!(IndexedSeq::len(slice), 5);
    assert!(!IndexedSeq::is_empty(slice));
    assert_eq!(IndexedSeq::get(slice, 0), 10);
    assert_eq!(IndexedSeq::get(slice, 4), 50);

    unsafe {
        assert_eq!(IndexedSeq::get_unchecked(slice, 1), 20);
        assert_eq!(IndexedSeq::get_unchecked(slice, 3), 40);
    }

    let empty: &[u32] = &[];
    assert!(IndexedSeq::is_empty(empty));
    assert_eq!(IndexedSeq::len(empty), 0);
}

#[test]
fn test_indexed_seq_vec() {
    let vec: Vec<u64> = vec![100, 200, 300];
    assert_eq!(IndexedSeq::len(&vec), 3);
    assert!(!IndexedSeq::is_empty(&vec));
    assert_eq!(IndexedSeq::get(&vec, 0), 100);
    assert_eq!(IndexedSeq::get(&vec, 2), 300);

    unsafe {
        assert_eq!(IndexedSeq::get_unchecked(&vec, 1), 200);
    }

    let empty: Vec<u64> = vec![];
    assert!(IndexedSeq::is_empty(&empty));
}

#[test]
fn test_indexed_seq_array() {
    let arr: [u16; 4] = [5, 10, 15, 20];
    assert_eq!(IndexedSeq::len(&arr), 4);
    assert!(!IndexedSeq::is_empty(&arr));
    assert_eq!(IndexedSeq::get(&arr, 0), 5);
    assert_eq!(IndexedSeq::get(&arr, 3), 20);

    unsafe {
        assert_eq!(IndexedSeq::get_unchecked(&arr, 2), 15);
    }

    let empty: [u16; 0] = [];
    assert!(IndexedSeq::is_empty(&empty));
}

// Tests for different integer types
#[test]
fn test_indexed_seq_various_types() {
    // u8
    let slice_u8: &[u8] = &[1, 2, 3];
    assert_eq!(IndexedSeq::get(slice_u8, 0), 1);

    // i32
    let slice_i32: &[i32] = &[-10, 0, 10];
    assert_eq!(IndexedSeq::get(slice_i32, 0), -10);
    assert_eq!(IndexedSeq::get(slice_i32, 2), 10);

    // i64
    let vec_i64: Vec<i64> = vec![-100, 100];
    assert_eq!(IndexedSeq::get(&vec_i64, 0), -100);

    // usize
    let arr_usize: [usize; 2] = [42, 84];
    assert_eq!(IndexedSeq::get(&arr_usize, 1), 84);
}

// Tests for Succ and Pred using EliasFano
#[test]
fn test_succ_with_elias_fano() {
    let values = [10usize, 20, 30, 40, 50];
    let mut efb = EliasFanoBuilder::new(values.len(), *values.last().unwrap());
    efb.extend(values);
    let ef = efb.build_with_seq_and_dict();

    // Test succ
    let (idx, val) = Succ::succ(&ef, 15).unwrap();
    assert_eq!(val, 20);
    assert_eq!(idx, 1);

    let (idx, val) = Succ::succ(&ef, 10).unwrap();
    assert_eq!(val, 10);
    assert_eq!(idx, 0);

    // Test succ_strict
    let (idx, val) = Succ::succ_strict(&ef, 10).unwrap();
    assert_eq!(val, 20);
    assert_eq!(idx, 1);

    // Test succ with value equal to max
    let result = Succ::succ(&ef, 50);
    assert!(result.is_some());
    assert_eq!(result.unwrap().1, 50);

    // Test succ_strict with value equal to max
    let result = Succ::succ_strict(&ef, 50);
    assert!(result.is_none());

    // Test succ with value greater than max
    let result = Succ::succ(&ef, 51);
    assert!(result.is_none());
}

#[test]
fn test_pred_with_elias_fano() {
    let values = [10usize, 20, 30, 40, 50];
    let mut efb = EliasFanoBuilder::new(values.len(), *values.last().unwrap());
    efb.extend(values);
    let ef = efb.build_with_seq_and_dict();

    // Test pred
    let (idx, val) = Pred::pred(&ef, 25).unwrap();
    assert_eq!(val, 20);
    assert_eq!(idx, 1);

    let (idx, val) = Pred::pred(&ef, 50).unwrap();
    assert_eq!(val, 50);
    assert_eq!(idx, 4);

    // Test pred_strict
    let (idx, val) = Pred::pred_strict(&ef, 50).unwrap();
    assert_eq!(val, 40);
    assert_eq!(idx, 3);

    // Test pred with value equal to min
    let result = Pred::pred(&ef, 10);
    assert!(result.is_some());
    assert_eq!(result.unwrap().1, 10);

    // Test pred_strict with value equal to min
    let result = Pred::pred_strict(&ef, 10);
    assert!(result.is_none());

    // Test pred with value less than min
    let result = Pred::pred(&ef, 5);
    assert!(result.is_none());
}

#[test]
fn test_succ_pred_delegation() {
    let values = [10usize, 20, 30];
    let mut efb = EliasFanoBuilder::new(values.len(), *values.last().unwrap());
    efb.extend(values);
    let ef = efb.build_with_seq_and_dict();
    let ef_ref = &ef;

    // Test delegation through reference
    let (idx, val) = Succ::succ(ef_ref, 15).unwrap();
    assert_eq!(val, 20);
    assert_eq!(idx, 1);

    let (idx, val) = Succ::succ_strict(ef_ref, 10).unwrap();
    assert_eq!(val, 20);
    assert_eq!(idx, 1);

    let (idx, val) = Pred::pred(ef_ref, 25).unwrap();
    assert_eq!(val, 20);
    assert_eq!(idx, 1);

    let (idx, val) = Pred::pred_strict(ef_ref, 20).unwrap();
    assert_eq!(val, 10);
    assert_eq!(idx, 0);
}

#[test]
fn test_succ_pred_empty() {
    // Create empty Elias-Fano
    let efb = EliasFanoBuilder::new(0, 0);
    let ef = efb.build_with_seq_and_dict();

    // Succ and pred on empty should return None
    assert!(Succ::succ(&ef, 10).is_none());
    assert!(Succ::succ_strict(&ef, 10).is_none());
    assert!(Pred::pred(&ef, 10).is_none());
    assert!(Pred::pred_strict(&ef, 10).is_none());
}

#[test]
fn test_indexed_seq_get_default_impl() {
    // Test the default get implementation that uses get_unchecked
    let slice: &[u32] = &[1, 2, 3, 4, 5];
    // This tests the default implementation path
    for i in 0..5 {
        assert_eq!(IndexedSeq::get(slice, i), (i + 1) as u32);
    }
}

#[test]
fn test_types_autoimpl() {
    // Test that Types works with references
    let _slice: &[u32] = &[1, 2, 3];
    let _: <&[u32] as Types>::Output<'_> = 1u32;

    let _vec: Vec<u32> = vec![1, 2, 3];
    let _: <&Vec<u32> as Types>::Output<'_> = 1u32;
}
