/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::prelude::*;
use sux::utils::FairChunks;

#[test]
fn test_fair_chunks_basic() {
    // Weights: 10, 20, 30, 40
    let cwf = vec![0usize, 10, 30, 60, 100];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(50, &ef).collect();
    // With target 50, should split into chunks
    assert!(!chunks.is_empty());
    // First chunk should start at 0
    assert_eq!(chunks[0].start, 0);
    // Last chunk should end at num_weights (4)
    assert_eq!(chunks.last().unwrap().end, 4);
}

#[test]
fn test_fair_chunks_new_with() {
    // Same test but using new_with constructor
    let cwf = vec![0usize, 10, 30, 60, 100];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_dict();
    let chunks: Vec<_> = FairChunks::new_with(50, &ef, 4, 100).collect();
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].start, 0);
    assert_eq!(chunks.last().unwrap().end, 4);
}

#[test]
fn test_fair_chunks_single_element() {
    // Single element with weight 100, target 50
    let cwf = vec![0usize, 100];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(50, &ef).collect();
    // Returns 0..1 (the element) then 1..1 (empty final chunk since we haven't exhausted target_weight)
    assert_eq!(chunks, vec![0..1, 1..1]);
}

#[test]
fn test_fair_chunks_large_target() {
    // Target larger than total weight - should return single chunk
    let cwf = vec![0usize, 10, 20, 30];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(1000, &ef).collect();
    assert_eq!(chunks, vec![0..3]);
}

#[test]
fn test_fair_chunks_small_target() {
    // Very small target - should create many chunks
    let cwf = vec![0usize, 10, 20, 30, 40, 50];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(10, &ef).collect();
    // Should have multiple chunks
    assert!(chunks.len() >= 2);
    // All ranges should be contiguous
    for i in 1..chunks.len() {
        assert_eq!(chunks[i - 1].end, chunks[i].start);
    }
}

#[test]
fn test_fair_chunks_fused_iterator() {
    let cwf = vec![0usize, 10, 20, 30];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let mut chunks = FairChunks::new(100, &ef);
    // Exhaust the iterator
    while chunks.next().is_some() {}
    // FusedIterator should return None repeatedly
    assert!(chunks.next().is_none());
    assert!(chunks.next().is_none());
    assert!(chunks.next().is_none());
}

#[test]
fn test_fair_chunks_example_from_docs() {
    // Test the exact example from the documentation
    let weights = [
        15usize, 27, 20, 26, 4, 22, 10, 25, 7, 13, 0, 11, 5, 28, 23, 1, 12, 24, 3, 30, 8, 29, 17,
        2, 14, 9, 16, 18, 21, 19,
    ];
    let mut cwf = vec![0usize];
    cwf.extend(weights.iter().scan(0, |acc, x| {
        *acc += x;
        Some(*acc)
    }));
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(50, &ef).collect();
    assert_eq!(
        chunks,
        vec![
            0..3,   // weight 62
            3..6,   // weight 52
            6..10,  // weight 55
            10..15, // weight 67
            15..20, // weight 70
            20..23, // weight 54
            23..28, // weight 59
            28..30, // weight 40
        ],
    );
}

#[test]
fn test_fair_chunks_contiguous_ranges() {
    // Verify all returned ranges are contiguous and cover the full range
    let cwf = vec![0usize, 5, 15, 30, 50, 75, 100];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(25, &ef).collect();

    // First chunk starts at 0
    assert_eq!(chunks[0].start, 0);
    // Last chunk ends at num_weights
    assert_eq!(chunks.last().unwrap().end, 6);
    // All ranges are contiguous
    for i in 1..chunks.len() {
        assert_eq!(
            chunks[i - 1].end,
            chunks[i].start,
            "Ranges not contiguous at index {}",
            i
        );
    }
}

#[test]
fn test_fair_chunks_uniform_weights() {
    // All weights are equal (10 each)
    let cwf = vec![0usize, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
    efb.extend(cwf.iter().copied());
    let ef = efb.build_with_seq_and_dict();
    let chunks: Vec<_> = FairChunks::new(30, &ef).collect();
    // With uniform weights of 10 and target 30, expect chunks of ~3 elements each
    // Should cover all 10 elements
    assert_eq!(chunks[0].start, 0);
    assert_eq!(chunks.last().unwrap().end, 10);
}
