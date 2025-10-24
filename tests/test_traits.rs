/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::traits::{IndexedSeq, IntoIteratorFrom};

#[test]
fn test_slice_delegations() {
    let v = [1, 2, 3, 4, 5];
    let s = &v[..];
    // Check that IndexedSeq is implemented for slices
    assert_eq!(IndexedSeq::len(s), v.len());

    // Ambiguity check
    assert_eq!(s.len(), v.len());
    assert_eq!(IndexedSeq::get(&s, 0), v[0]);

    for zip in s.iter().zip(v.iter()) {
        assert_eq!(zip.0, zip.1);
    }

    // Check that IntoIterFrom is implemented for slices
    for zip in s.into_iter_from(1).zip(v.iter().skip(1)) {
        assert_eq!(zip.0, zip.1);
    }
}

#[test]
fn test_into_iter_from() {
    let v = vec![1, 2, 3, 4, 5];
    (&v).into_iter_from(2)
        .zip([3, 4, 5].iter())
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });
    v.into_iter_from(2)
        .zip(vec![3, 4, 5])
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });

    let v = vec![1, 2, 3, 4, 5].into_boxed_slice();
    (&v).into_iter_from(2)
        .zip([3, 4, 5].iter())
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });
    v.into_iter_from(2)
        .zip(vec![3, 4, 5])
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });
}
