use sux::bits::BitVec;
use sux::prelude::*;
use sux::traits::Word;
use value_traits::slices::SliceByValue;

/// Converts a `CompIntList` with default (EfSeq) delimiters to one backed by
/// `Vec<usize>`, for testing with a different delimiter type.
fn to_vec_delimiters<V: Word>(
    list: CompIntList<BitVec<Box<[V]>>>,
) -> CompIntList<BitVec<Box<[V]>>, Vec<u64>> {
    unsafe {
        list.map_delimiters(|d| {
            let n = d.len();
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(d.index_value(i));
            }
            v
        })
    }
}

/// Builds a `CompIntList` with the given `min` and `values`, then verifies
/// every element is read back correctly—first with EfSeq delimiters, then
/// with Vec<usize> delimiters.
fn check<V: Word>(min: V, values: Vec<V>) {
    let list = CompIntList::new(min, &values);
    assert_eq!(list.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(list.index_value(i), v, "EfSeq: mismatch at index {i}");
    }

    let vec_list = to_vec_delimiters(CompIntList::new(min, &values));
    assert_eq!(vec_list.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(vec_list.index_value(i), v, "Vec: mismatch at index {i}");
    }
}

// ────────────────────── value tests ──────────────────────

#[test]
fn test_u32() {
    check(1u32, vec![1, 2, 3, 4, 5, 6, 7]);
    check(1u32, vec![42, 1000, 123456, u32::MAX / 2, u32::MAX]);
    check(1u32, (0..31).map(|k| 1u32 << k).collect());
    check(1u32, vec![u32::MAX]);
    check(1u32, vec![1; 100]);
    check(1u32, vec![7, 7, 7, 42, 42, 1, 1]);
    check(0u32, vec![0, 0, 0, 5, 42, u32::MAX - 1]);
    check(1u32, (1..=10000).collect());
}

#[test]
fn test_u64() {
    check(1u64, vec![1, 2, 3, 4, 5, 6, 7]);
    check(1u64, vec![42, 1000, 123456789, u64::MAX / 2, u64::MAX]);
    check(1u64, (0..63).map(|k| 1u64 << k).collect());
    check(1u64, vec![u64::MAX]);
    check(1u64, vec![1; 100]);
    check(1u64, vec![7, 7, 7, 42, 42, 1, 1]);
    check(1u64, vec![42]);
    // Mixed widths: 0, 1, 2, 7, 15, 31, 63
    check(1u64, vec![1, 3, 7, 255, 65535, (1u64 << 32) - 1, u64::MAX]);
}

#[test]
fn test_u128() {
    check(1u128, vec![1, 2, 3, 4, 5, 6, 7]);
    check(1u128, vec![1, 42, u128::MAX / 3, u128::MAX / 2, u128::MAX]);
    check(1u128, (0..127).map(|k| 1u128 << k).collect());
    check(1u128, vec![u128::MAX]);
    check(1u128, vec![1; 100]);
    // Values beyond usize range
    let big = (1u128 << 100) + 42;
    check(
        1u128,
        vec![
            (1u128 << 64) + 1,
            (1u128 << 100) + 12345,
            u128::MAX - 1,
            u128::MAX,
        ],
    );
    check(1u128, vec![big, big, big, 1, 1]);
    check(1u128, vec![(1u128 << 120) + 999]);
}

// ────────────────────── min offset tests ──────────────────────

#[test]
fn test_min_offset() {
    check(0usize, vec![0, 1, 2, 3, 100, usize::MAX - 1]);
    check(100usize, vec![100, 101, 200, 1000]);
    // All values equal to min (zero-width storage)
    check(42usize, vec![42; 5]);
}

// ────────────────────── empty / single ──────────────────────

#[test]
fn test_empty() {
    let empty: Vec<usize> = vec![];
    let list = CompIntList::new(0, &empty);
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());

    let vec_list = to_vec_delimiters(CompIntList::new(0, &empty));
    assert_eq!(vec_list.len(), 0);
    assert!(vec_list.is_empty());
}

// ────────────────────── panic tests ──────────────────────

#[test]
#[should_panic(expected = "lower bound")]
fn test_value_below_min_panics() {
    let values = vec![0usize, 1, 2];
    let _ = CompIntList::new(1, &values);
}
