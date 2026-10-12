/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
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
    v.into_iter_from(2).zip(vec![3, 4, 5]).for_each(|(a, b)| {
        assert_eq!(a, b);
    });

    let v = vec![1, 2, 3, 4, 5].into_boxed_slice();
    (&v).into_iter_from(2)
        .zip([3, 4, 5].iter())
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });
    v.into_iter_from(2).zip(vec![3, 4, 5]).for_each(|(a, b)| {
        assert_eq!(a, b);
    });
}

#[test]
fn test_hinted_forwarding() {
    use sux::bit_vec;
    use sux::traits::{RankHinted, SelectHinted, SelectZeroHinted};

    // Generic functions make sure that the forwarding implementations are used
    fn rank<T: RankHinted>(t: T, pos: usize) -> usize {
        unsafe { t.rank_hinted::<{ usize::MAX }>(pos, 0, 0) }
    }
    fn select<T: SelectHinted>(t: T, rank: usize) -> usize {
        unsafe { t.select_hinted::<{ usize::MAX }>(rank, 0, 0) }
    }
    fn select_zero<T: SelectZeroHinted>(t: T, rank: usize) -> usize {
        unsafe { t.select_zero_hinted::<{ usize::MAX }>(rank, 0, 0) }
    }

    let mut bits = bit_vec![1, 0, 1, 1, 0, 1, 0, 0, 1];
    assert_eq!(rank(&bits, 5), 3);
    assert_eq!(rank(&mut bits, 5), 3);
    assert_eq!(select(&bits, 3), 5);
    assert_eq!(select(&mut bits, 3), 5);
    assert_eq!(select_zero(&bits, 2), 6);
    assert_eq!(select_zero(&mut bits, 2), 6);
    let bits = Box::new(bits);
    assert_eq!(rank(bits.clone(), 5), 3);
    assert_eq!(select(bits.clone(), 3), 5);
    assert_eq!(select_zero(bits, 2), 6);
}

#[test]
fn test_word_backends() {
    use sux::traits::{Backend, TryIntoUnaligned};

    let _: <[u8] as Backend>::Word = 0_u8;
    let _: <Vec<u16> as Backend>::Word = 0_u16;
    let _: <[u32; 3] as Backend>::Word = 0_u32;

    assert_eq!(vec![1_u64, 2].try_into_unaligned().unwrap(), vec![1, 2]);
    assert_eq!([1_u8, 2].try_into_unaligned().unwrap(), [1, 2]);
    let b: Box<[u16]> = Box::new([1, 2]);
    assert_eq!(&*b.try_into_unaligned().unwrap(), &[1, 2]);
}
