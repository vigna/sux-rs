/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::prelude::*;
use sux::traits::{IndexedSeq, IntoIteratorFrom};

#[test]
fn test_slice_seq_new() {
    let data = vec![1usize, 2, 3, 4, 5];
    let seq = SliceSeq::new(&data);
    assert_eq!(seq.len(), 5);
}

#[test]
fn test_slice_seq_from() {
    let data = vec![10usize, 20, 30];
    let seq: SliceSeq<usize, _> = SliceSeq::from(&data);
    assert_eq!(seq.len(), 3);
}

#[test]
fn test_slice_seq_get_unchecked() {
    let data = vec![100usize, 200, 300, 400];
    let seq = SliceSeq::new(&data);
    unsafe {
        assert_eq!(seq.get_unchecked(0), 100);
        assert_eq!(seq.get_unchecked(1), 200);
        assert_eq!(seq.get_unchecked(2), 300);
        assert_eq!(seq.get_unchecked(3), 400);
    }
}

#[test]
fn test_slice_seq_len() {
    let empty: Vec<usize> = vec![];
    let seq_empty = SliceSeq::new(&empty);
    assert_eq!(seq_empty.len(), 0);
    assert!(seq_empty.is_empty());

    let data = vec![1usize, 2, 3];
    let seq = SliceSeq::new(&data);
    assert_eq!(seq.len(), 3);
    assert!(!seq.is_empty());
}

#[test]
fn test_slice_seq_iter() {
    let data = vec![5usize, 10, 15, 20];
    let seq = SliceSeq::new(&data);
    let collected: Vec<_> = seq.iter().collect();
    assert_eq!(collected, vec![5, 10, 15, 20]);
}

#[test]
fn test_slice_seq_into_iterator() {
    let data = vec![1usize, 2, 3];
    let seq = SliceSeq::new(&data);
    let collected: Vec<_> = (&seq).into_iter().collect();
    assert_eq!(collected, vec![1, 2, 3]);
}

#[test]
fn test_slice_seq_into_iterator_from() {
    let data = vec![10usize, 20, 30, 40, 50];
    let seq = SliceSeq::new(&data);
    let collected: Vec<_> = (&seq).into_iter_from(2).collect();
    assert_eq!(collected, vec![30, 40, 50]);
}

#[test]
fn test_slice_seq_into_iterator_from_zero() {
    let data = vec![1usize, 2, 3];
    let seq = SliceSeq::new(&data);
    let collected: Vec<_> = (&seq).into_iter_from(0).collect();
    assert_eq!(collected, vec![1, 2, 3]);
}

#[test]
fn test_slice_seq_into_iterator_from_end() {
    let data = vec![1usize, 2, 3];
    let seq = SliceSeq::new(&data);
    let collected: Vec<_> = (&seq).into_iter_from(3).collect();
    assert!(collected.is_empty());
}

#[test]
fn test_slice_seq_with_array() {
    let data = [1usize, 2, 3, 4];
    let seq = SliceSeq::new(&data);
    assert_eq!(seq.len(), 4);
    unsafe {
        assert_eq!(seq.get_unchecked(0), 1);
        assert_eq!(seq.get_unchecked(3), 4);
    }
}

#[test]
fn test_slice_seq_equality() {
    let data1 = vec![1usize, 2, 3];
    let data2 = vec![1usize, 2, 3];
    let data3 = vec![1usize, 2, 4];
    let seq1 = SliceSeq::new(&data1);
    let seq2 = SliceSeq::new(&data2);
    let seq3 = SliceSeq::new(&data3);
    // Equality compares the underlying slice content
    assert_eq!(seq1, seq1);
    assert_eq!(seq1, seq2); // Same contents = equal
    assert_ne!(seq1, seq3); // Different contents
}

#[test]
fn test_slice_seq_clone() {
    let data = vec![1usize, 2, 3];
    let seq = SliceSeq::new(&data);
    let cloned = seq;
    assert_eq!(seq.len(), cloned.len());
}

#[test]
fn test_slice_seq_debug() {
    let data = vec![1usize, 2, 3];
    let seq = SliceSeq::new(&data);
    let debug_str = format!("{:?}", seq);
    assert!(debug_str.contains("SliceSeq"));
}
