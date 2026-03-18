use sux::list::CompIntListBuilder;
use sux::prelude::*;
use sux::traits::bit_field_slice::Word;
use value_traits::slices::SliceByValue;

/// Converts a `CompIntList` with default (EfSeq) delimiters to one backed by
/// `Vec<u64>`, for testing with a different delimiter type.
fn to_vec_delimiters<V: Word>(list: CompIntList<V>) -> CompIntList<V, Vec<u64>> {
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
/// with Vec<u64> delimiters.
fn check<V: Word>(min: V, values: Vec<V>) {
    let list = CompIntList::new(min, values.clone());
    assert_eq!(list.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(list.index_value(i), v, "EfSeq: mismatch at index {i}");
    }

    let vec_list = to_vec_delimiters(CompIntList::new(min, values.clone()));
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
    // Values beyond u64 range
    let big = (1u128 << 100) + 42;
    check(1u128, vec![(1u128 << 64) + 1, (1u128 << 100) + 12345, u128::MAX - 1, u128::MAX]);
    check(1u128, vec![big, big, big, 1, 1]);
    check(1u128, vec![(1u128 << 120) + 999]);
}

// ────────────────────── min offset tests ──────────────────────

#[test]
fn test_min_offset() {
    check(0u64, vec![0, 1, 2, 3, 100, u64::MAX - 1]);
    check(100u64, vec![100, 101, 200, 1000]);
    // All values equal to min (zero-width storage)
    check(42u64, vec![42; 5]);
}

// ────────────────────── empty / single ──────────────────────

#[test]
fn test_empty() {
    let ef = CompIntList::<u64>::new(0, Vec::new());
    assert_eq!(ef.len(), 0);
    assert!(ef.is_empty());

    let vec_list = to_vec_delimiters(CompIntList::<u64>::new(0, Vec::new()));
    assert_eq!(vec_list.len(), 0);
    assert!(vec_list.is_empty());
}

// ────────────────────── builder tests ──────────────────────

#[test]
fn test_builder_push() {
    let mut builder = CompIntListBuilder::new(1u64);
    builder.push(1);
    builder.push(3);
    builder.push(7);
    builder.push(42);
    builder.push(100);

    let ef = builder.build();
    assert_eq!(ef.len(), 5);
    assert_eq!(ef.index_value(0), 1);
    assert_eq!(ef.index_value(2), 7);
    assert_eq!(ef.index_value(4), 100);
}

#[test]
fn test_builder_extend() {
    let mut builder = CompIntListBuilder::new(1u32);
    builder.push(1);
    builder.extend([2u32, 3, 4, 5]);

    let ef = builder.build();
    assert_eq!(ef.len(), 5);
    for i in 0..5 {
        assert_eq!(ef.index_value(i), (i + 1) as u32);
    }
}

#[test]
fn test_builder_empty() {
    let builder = CompIntListBuilder::<u64>::new(0);
    let ef = builder.build();
    assert_eq!(ef.len(), 0);
    assert!(ef.is_empty());
}

// ────────────────────── panic tests ──────────────────────

#[test]
#[should_panic(expected = "lower bound")]
fn test_value_below_min_panics() {
    CompIntList::new(1, vec![0u64, 1, 2]);
}

#[test]
#[should_panic(expected = "lower bound")]
fn test_builder_below_min_panics() {
    let mut builder = CompIntListBuilder::new(10u64);
    builder.push(10);
    builder.push(9);
}
