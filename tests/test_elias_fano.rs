/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]
use std::iter::zip;

use anyhow::Result;
use indexed_dict::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use sux::prelude::*;

#[test]
#[cfg(feature = "rayon")]
fn test_elias_fano_concurrent() -> Result<()> {
    use rayon::prelude::*;
    use sux::dict::elias_fano::EliasFanoConcurrentBuilder;
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let efb = EliasFanoConcurrentBuilder::new(n, u);
        // push concurrently the values
        values
            .par_iter()
            .enumerate()
            .for_each(|(index, value)| unsafe { efb.set(index, *value) });
        // Finish the creation of elias-fano
        let _ef = efb.build();
    }
    Ok(())
}

#[test]
fn test_elias_fano() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(n, u);
        // push the values
        for value in values.iter() {
            efb.push(*value);
        }
        // Finish the creation of elias-fano
        let ef = efb.build();

        // Add an index on ones (will make it into an IndexedSeq)
        let ef = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _>::new) };
        // Add an index on zeros (will make it into an IndexedDict + Succ + Pred)
        let ef = unsafe { ef.map_high_bits(SelectZeroAdaptConst::<_, _>::new) };

        for v in 0..u {
            let res = values.binary_search(&v);
            let contains = res.is_ok();
            assert_eq!(ef.contains(v), contains);

            if contains {
                let i = res.unwrap();
                assert_eq!(ef.get(i), v);
                assert_eq!(ef.get(ef.index_of(v).unwrap()), v);
            } else {
                assert_eq!(ef.index_of(v), None);
            }
        }

        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!({ ef.get(i) }, *v);
        }

        let mut iterator = ef.iter().enumerate();
        while let Some((i, v)) = iterator.next() {
            assert_eq!(v, values[i]);
            assert_eq!(iterator.len(), ef.len() - i - 1);
        }

        for from in 0..ef.len() {
            let mut iterator = ef.iter_from(from).enumerate();
            while let Some((i, v)) = iterator.next() {
                assert_eq!(v, values[i + from]);
                assert_eq!(iterator.len(), ef.len() - i - from - 1);
            }
        }

        let last = values.last().unwrap();
        let mut lower_bound = 0;
        for (i, v) in values.iter().enumerate() {
            if lower_bound > *v {
                continue;
            }
            loop {
                assert!(ef.succ(lower_bound).unwrap() == (i, *v));
                lower_bound += 1;
                if lower_bound > values[i] {
                    break;
                }
            }
        }

        assert_eq!(None, ef.succ(last + 1));

        let mut lower_bound = 0;
        for (i, v) in values.iter().enumerate() {
            if lower_bound >= *v {
                continue;
            }
            loop {
                assert!(ef.succ_strict(lower_bound).unwrap() == (i, *v));
                lower_bound += 1;
                if lower_bound >= values[i] {
                    break;
                }
            }
        }

        assert_eq!(None, ef.succ_strict(last));

        let first = *values.first().unwrap();
        let mut upper_bound = first + 1;
        for (i, v) in values.iter().enumerate() {
            if upper_bound <= *v {
                continue;
            }
            // Skip repeated items as we return the index of the last one
            if i + 1 < values.len() && values[i] == values[i + 1] {
                continue;
            }
            loop {
                assert!(ef.pred_strict(upper_bound).unwrap() == (i, *v));
                upper_bound += 1;
                if i + 1 == values.len() || upper_bound > values[i + 1] {
                    break;
                }
            }
        }

        for upper_bound in 0..first + 1 {
            assert_eq!(None, ef.pred_strict(upper_bound));
        }

        let first = *values.first().unwrap();
        let mut upper_bound = first;
        for (i, v) in values.iter().enumerate() {
            if upper_bound < *v {
                continue;
            }
            // Skip repeated items as we return the index of the last one
            if i + 1 < values.len() && values[i] == values[i + 1] {
                continue;
            }
            loop {
                assert!(ef.pred(upper_bound).unwrap() == (i, *v));
                upper_bound += 1;
                if i + 1 == values.len() || upper_bound >= values[i + 1] {
                    break;
                }
            }
        }

        for upper_bound in 0..first {
            assert_eq!(None, ef.pred(upper_bound));
        }
    }

    Ok(())
}

#[test]
fn test_empty() {
    let efb = EliasFanoBuilder::new(0, 10);
    let ef = efb.build_with_seq_and_dict();
    assert_eq!(ef.len(), 0);
}

#[test]
#[should_panic]
fn test_empty_access() {
    let efb = EliasFanoBuilder::new(0, 10);
    let ef = efb.build_with_seq_and_dict();
    assert_eq!(ef.len(), 0);
    ef.get(0);
}

#[test]
#[should_panic]
fn test_too_many_values() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(0);
    efb.push(1);
    efb.push(2);
}

#[test]
#[should_panic]
fn test_too_few_values() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(0);
    efb.build();
}

#[test]
#[should_panic]
fn test_non_monotone() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(1);
    efb.push(0);
}

#[test]
#[should_panic]
fn test_too_large() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(11);
}

#[test]
#[should_panic]
fn test_from_non_monotone() {
    let _ef: EliasFano = vec![1, 0].into();
}

#[test]
fn test_extend() {
    let mut efb = EliasFanoBuilder::new(3, 10);
    let v = vec![0, 1, 2];
    efb.extend(v.clone());
    let ef = efb.build();
    zip(ef.iter(), v.iter()).for_each(|(a, b)| assert_eq!(a, *b));
}

#[cfg(feature = "epserde")]
#[test]
fn test_epserde() -> Result<()> {
    use epserde::prelude::Aligned16;
    use epserde::ser::Serialize;

    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(100, 1000), (100, 100), (1000, 100)] {
        use epserde::utils::AlignedCursor;

        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(n, u);
        // push the values
        for value in values.iter() {
            efb.push(*value);
        }
        // Finish the creation of elias-fano
        let ef = unsafe { efb.build().map_high_bits(SelectAdaptConst::<_, _>::new) };

        let mut cursor = <AlignedCursor<Aligned16>>::new();
        let schema = unsafe { ef.serialize_with_schema(&mut cursor) }?;
        println!("{}", schema.to_csv());

        let len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            use epserde::deser::Deserialize;

            <EliasFano<
                SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>>,
                BitFieldVec<usize, Box<[usize]>>,
            >>::read_mmap(&mut cursor, len, epserde::deser::Flags::empty())
        }?;

        for i in 0..n {
            assert_eq!(ef.get(i), c.uncase().get(i));
        }
    }
    Ok(())
}

#[test]
fn test_convenience_methods() {
    let mut efb = EliasFanoBuilder::new(10, 100);
    for i in 0..10 {
        efb.push(i * 10);
    }
    let ef = efb.build_with_seq();
    for i in 0..10 {
        assert_eq!(ef.get(i), i * 10);
    }

    let mut efb = EliasFanoBuilder::new(10, 100);
    for i in 0..10 {
        efb.push(i * 10);
    }
    let ef = efb.build_with_dict();
    for i in 0..10 {
        assert_eq!(unsafe { ef.succ_unchecked::<false>(i * 10) }, (i, i * 10));
    }

    let mut efb = EliasFanoBuilder::new(10, 100);
    for i in 0..10 {
        efb.push(i * 10);
    }
    let ef = efb.build_with_seq_and_dict();
    for i in 0..10 {
        assert_eq!(ef.get(i), i * 10);
        assert_eq!(ef.succ(i * 10).unwrap(), (i, i * 10));
    }
}

#[test]
fn test_iter_from_succ() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();
        values.sort();

        // Test with EfSeqDict (has both SelectUnchecked and SelectZeroUnchecked)
        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef = efb.build_with_seq_and_dict();

        // Test iter_from_succ: results match succ + iter_from
        for v in 0..=u {
            let succ_result = ef.succ(v);
            let iter_succ_result = ef.iter_from_succ(v);
            match (succ_result, iter_succ_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, iter))) => {
                    assert_eq!(idx, idx2);
                    // The iterator should yield the successor and all subsequent values
                    let collected: Vec<usize> = iter.collect();
                    assert_eq!(collected.len(), n - idx);
                    assert_eq!(collected[0], val);
                    for (i, &cv) in collected.iter().enumerate() {
                        assert_eq!(cv, values[idx + i]);
                    }
                }
                _ => panic!("succ and iter_from_succ disagree for value {v}"),
            }
        }

        // Test iter_from_succ_strict
        for v in 0..=u {
            let succ_result = ef.succ_strict(v);
            let iter_succ_result = ef.iter_from_succ_strict(v);
            match (succ_result, iter_succ_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, iter))) => {
                    assert_eq!(idx, idx2);
                    let collected: Vec<usize> = iter.collect();
                    assert_eq!(collected.len(), n - idx);
                    assert_eq!(collected[0], val);
                    for (i, &cv) in collected.iter().enumerate() {
                        assert_eq!(cv, values[idx + i]);
                    }
                }
                _ => panic!("succ_strict and iter_from_succ_strict disagree for value {v}"),
            }
        }

        // Test with EfDict (only SelectZeroUnchecked, no SelectUnchecked)
        // This tests the inherent unchecked method's weaker bounds
        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef_dict = efb.build_with_dict();

        for v in 0..=u {
            // Use succ_unchecked on EfDict (no Succ trait since no SelectUnchecked)
            // and compare with iter_from_succ_unchecked
            let succ_result = values.partition_point(|&x| x < v);
            if succ_result < n {
                let (idx, iter) = unsafe { ef_dict.iter_from_succ_unchecked::<false>(v) };
                assert_eq!(idx, succ_result);
                let collected: Vec<usize> = iter.collect();
                assert_eq!(collected.len(), n - idx);
                assert_eq!(collected[0], values[succ_result]);
                for (i, &cv) in collected.iter().enumerate() {
                    assert_eq!(cv, values[idx + i]);
                }
            }
        }

        // Test unchecked version directly
        for v in 0..=u {
            if let Some((idx, val)) = ef.succ(v) {
                let (idx2, iter) = unsafe { ef.iter_from_succ_unchecked::<false>(v) };
                assert_eq!(idx, idx2);
                let collected: Vec<usize> = iter.collect();
                assert_eq!(collected[0], val);
            }
        }
    }

    // Test empty sequence
    let efb = EliasFanoBuilder::new(0, 10);
    let ef = efb.build_with_seq_and_dict();
    assert!(ef.iter_from_succ(0).is_none());
    assert!(ef.iter_from_succ_strict(0).is_none());

    Ok(())
}

#[test]
fn test_rev_iter() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();
        values.sort();

        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef = efb.build_with_seq_and_dict();

        // 1. rev_iter() yields all elements in reverse order
        let rev: Vec<usize> = ef.rev_iter().collect();
        let mut expected = values.clone();
        expected.reverse();
        assert_eq!(rev, expected);

        // 2. rev_iter_from(k) yields elements k-1, k-2, ..., 0
        for from in 0..=n {
            let rev: Vec<usize> = ef.rev_iter_from(from).collect();
            let expected: Vec<usize> = values[..from].iter().copied().rev().collect();
            assert_eq!(rev, expected);
        }

        // 3. iter().rev() at position 0 is empty; iter_from(n).rev() yields all in reverse
        let rev: Vec<usize> = ef.iter().rev().collect();
        assert!(rev.is_empty());

        let rev: Vec<usize> = ef.iter_from(n).rev().collect();
        let mut expected = values.clone();
        expected.reverse();
        assert_eq!(rev, expected);

        // 4. Round-trip: iter_from(k).rev().rev() yields same as iter_from(k)
        for from in 0..=n {
            let roundtrip: Vec<usize> = ef.iter_from(from).rev().rev().collect();
            let direct: Vec<usize> = ef.iter_from(from).collect();
            assert_eq!(roundtrip, direct);
        }

        // 5. rev_iter().rev() yields forward from cursor n (empty)
        let roundtrip: Vec<usize> = ef.rev_iter().rev().collect();
        assert!(roundtrip.is_empty());

        // Consuming rev_iter fully then rev gives forward from cursor 0 = iter()
        let mut rev = ef.rev_iter();
        while rev.next().is_some() {}
        let roundtrip: Vec<usize> = rev.rev().collect();
        let direct: Vec<usize> = ef.iter().collect();
        assert_eq!(roundtrip, direct);

        // 6. ExactSizeIterator::len() is correct throughout iteration
        let mut rev = ef.rev_iter();
        for remaining in (0..n).rev() {
            assert_eq!(rev.len(), remaining + 1);
            rev.next();
            assert_eq!(rev.len(), remaining);
        }
        assert_eq!(rev.len(), 0);
        assert!(rev.next().is_none());

        // Test len for iter_from().rev()
        for from in 0..=n {
            let mut rev = ef.iter_from(from).rev();
            assert_eq!(rev.len(), from);
            for remaining in (0..from).rev() {
                rev.next();
                assert_eq!(rev.len(), remaining);
            }
        }
    }

    // 7. Works with EfDict (only SelectZeroUnchecked, no SelectUnchecked)
    let mut rng = SmallRng::seed_from_u64(42);
    for (n, u) in [(10, 1000), (100, 100)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();
        values.sort();

        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef = efb.build_with_dict();

        // EfDict has no SelectUnchecked, so iter_from/rev_iter_from won't work,
        // but rev_iter() should work since it only needs AsRef<[usize]>
        let rev: Vec<usize> = ef.rev_iter().collect();
        let mut expected = values.clone();
        expected.reverse();
        assert_eq!(rev, expected);
    }

    // Test empty sequence
    let efb = EliasFanoBuilder::new(0, 10);
    let ef = efb.build_with_seq_and_dict();
    let rev: Vec<usize> = ef.rev_iter().collect();
    assert!(rev.is_empty());
    assert_eq!(ef.rev_iter().len(), 0);

    Ok(())
}

#[test]
fn test_rev_iter_from_pred() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();
        values.sort();

        // Test with EfSeqDict (has both SelectUnchecked and SelectZeroUnchecked)
        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef = efb.build_with_seq_and_dict();

        // Test rev_iter_from_pred: results match pred + reverse iteration
        for v in 0..=u {
            let pred_result = ef.pred(v);
            let iter_pred_result = ef.rev_iter_from_pred(v);
            match (pred_result, iter_pred_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, iter))) => {
                    assert_eq!(idx, idx2);
                    // The iterator should yield the predecessor and all preceding values
                    let collected: Vec<usize> = iter.collect();
                    assert_eq!(collected.len(), idx + 1);
                    assert_eq!(collected[0], val);
                    for (i, &cv) in collected.iter().enumerate() {
                        assert_eq!(cv, values[idx - i]);
                    }
                }
                _ => panic!("pred and rev_iter_from_pred disagree for value {v}"),
            }
        }

        // Test rev_iter_from_pred_strict
        for v in 0..=u {
            let pred_result = ef.pred_strict(v);
            let iter_pred_result = ef.rev_iter_from_pred_strict(v);
            match (pred_result, iter_pred_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, iter))) => {
                    assert_eq!(idx, idx2);
                    let collected: Vec<usize> = iter.collect();
                    assert_eq!(collected.len(), idx + 1);
                    assert_eq!(collected[0], val);
                    for (i, &cv) in collected.iter().enumerate() {
                        assert_eq!(cv, values[idx - i]);
                    }
                }
                _ => panic!(
                    "pred_strict and rev_iter_from_pred_strict disagree for value {v}"
                ),
            }
        }

        // Test with EfDict (only SelectZeroUnchecked, no SelectUnchecked)
        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef_dict = efb.build_with_dict();

        for v in 0..=u {
            let pred_idx = values.partition_point(|&x| x <= v);
            if pred_idx > 0 {
                let (idx, iter) =
                    unsafe { ef_dict.rev_iter_from_pred_unchecked::<false>(v) };
                assert_eq!(idx, pred_idx - 1);
                let collected: Vec<usize> = iter.collect();
                assert_eq!(collected.len(), idx + 1);
                assert_eq!(collected[0], values[idx]);
                for (i, &cv) in collected.iter().enumerate() {
                    assert_eq!(cv, values[idx - i]);
                }
            }
        }

        // Test unchecked version directly
        for v in 0..=u {
            if let Some((idx, val)) = ef.pred(v) {
                let (idx2, iter) =
                    unsafe { ef.rev_iter_from_pred_unchecked::<false>(v) };
                assert_eq!(idx, idx2);
                let collected: Vec<usize> = iter.collect();
                assert_eq!(collected[0], val);
            }
        }
    }

    // Test empty sequence
    let efb = EliasFanoBuilder::new(0, 10);
    let ef = efb.build_with_seq_and_dict();
    assert!(ef.rev_iter_from_pred(0).is_none());
    assert!(ef.rev_iter_from_pred_strict(0).is_none());

    Ok(())
}

#[test]
fn test_bidi_iter() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();
        values.sort();

        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef = efb.build_with_seq_and_dict();

        // Test bidi_iter_from_succ: matches succ, then next/prev work correctly
        for v in 0..=u {
            let succ_result = ef.succ(v);
            let bidi_result = ef.bidi_iter_from_succ(v);
            match (succ_result, bidi_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, mut bidi))) => {
                    assert_eq!(idx, idx2);
                    assert_eq!(bidi.len(), n - idx);
                    // next() yields the successor first
                    assert_eq!(bidi.next(), Some(val));
                    // prev() goes back to the successor
                    assert_eq!(bidi.prev(), Some(val));
                    // next() again yields the successor
                    assert_eq!(bidi.next(), Some(val));
                    // Collect the rest via next()
                    let rest: Vec<usize> = bidi.collect();
                    for (i, &cv) in rest.iter().enumerate() {
                        assert_eq!(cv, values[idx + 1 + i]);
                    }
                }
                _ => panic!("succ and bidi_iter_from_succ disagree for value {v}"),
            }
        }

        // Test bidi_iter_from_succ_strict
        for v in 0..=u {
            let succ_result = ef.succ_strict(v);
            let bidi_result = ef.bidi_iter_from_succ_strict(v);
            match (succ_result, bidi_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, mut bidi))) => {
                    assert_eq!(idx, idx2);
                    assert_eq!(bidi.next(), Some(val));
                }
                _ => panic!("succ_strict and bidi_iter_from_succ_strict disagree for value {v}"),
            }
        }

        // Test bidi_iter_from_pred: matches pred, then prev/next work correctly
        for v in 0..=u {
            let pred_result = ef.pred(v);
            let bidi_result = ef.bidi_iter_from_pred(v);
            match (pred_result, bidi_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, mut bidi))) => {
                    assert_eq!(idx, idx2);
                    // prev() yields the predecessor first
                    assert_eq!(bidi.prev(), Some(val));
                    // next() goes forward to the predecessor again
                    assert_eq!(bidi.next(), Some(val));
                    // prev() back to the predecessor
                    assert_eq!(bidi.prev(), Some(val));
                    // Collect the rest via prev()
                    let mut prev_vals = Vec::new();
                    while let Some(pv) = bidi.prev() {
                        prev_vals.push(pv);
                    }
                    for (i, &cv) in prev_vals.iter().enumerate() {
                        assert_eq!(cv, values[idx - 1 - i]);
                    }
                }
                _ => panic!("pred and bidi_iter_from_pred disagree for value {v}"),
            }
        }

        // Test bidi_iter_from_pred_strict
        for v in 0..=u {
            let pred_result = ef.pred_strict(v);
            let bidi_result = ef.bidi_iter_from_pred_strict(v);
            match (pred_result, bidi_result) {
                (None, None) => {}
                (Some((idx, val)), Some((idx2, mut bidi))) => {
                    assert_eq!(idx, idx2);
                    assert_eq!(bidi.prev(), Some(val));
                }
                _ => panic!(
                    "pred_strict and bidi_iter_from_pred_strict disagree for value {v}"
                ),
            }
        }

        // Test full traversal forward then backward
        if let Some((_, mut bidi)) = ef.bidi_iter_from_succ(0) {
            // Forward
            for i in 0..n {
                assert_eq!(bidi.next(), Some(values[i]));
            }
            assert_eq!(bidi.next(), None);
            // Backward
            for i in (0..n).rev() {
                assert_eq!(bidi.prev(), Some(values[i]));
            }
            assert_eq!(bidi.prev(), None);
        }

        // Test with EfDict (only SelectZeroUnchecked)
        let mut efb = EliasFanoBuilder::new(n, u);
        for &v in &values {
            efb.push(v);
        }
        let ef_dict = efb.build_with_dict();

        for v in 0..=u {
            let succ_idx = values.partition_point(|&x| x < v);
            if succ_idx < n {
                let (idx, mut bidi) =
                    unsafe { ef_dict.bidi_iter_from_succ_unchecked::<false>(v) };
                assert_eq!(idx, succ_idx);
                assert_eq!(bidi.next(), Some(values[succ_idx]));
            }

            let pred_idx = values.partition_point(|&x| x <= v);
            if pred_idx > 0 {
                let (idx, mut bidi) =
                    unsafe { ef_dict.bidi_iter_from_pred_unchecked::<false>(v) };
                assert_eq!(idx, pred_idx - 1);
                assert_eq!(bidi.prev(), Some(values[pred_idx - 1]));
            }
        }
    }

    // Test empty sequence
    let efb = EliasFanoBuilder::new(0, 10);
    let ef = efb.build_with_seq_and_dict();
    assert!(ef.bidi_iter_from_succ(0).is_none());
    assert!(ef.bidi_iter_from_succ_strict(0).is_none());
    assert!(ef.bidi_iter_from_pred(0).is_none());
    assert!(ef.bidi_iter_from_pred_strict(0).is_none());

    Ok(())
}
