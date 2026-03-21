use sux::list::prefix_sum_int_list::PrefixSumIntList;
use value_traits::slices::SliceByValue;

/// Converts a `PrefixSumIntList` with default (EfSeq) prefix sums to one
/// backed by `Vec<u64>`, for testing with a different backend.
fn to_vec_backend(list: PrefixSumIntList) -> PrefixSumIntList<u64, Vec<u64>> {
    unsafe {
        list.map_prefix_sums(|d| {
            let n = d.len();
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(d.index_value(i));
            }
            v
        })
    }
}

/// Builds a `PrefixSumIntList`, then verifies every element and prefix sum—
/// first with EfSeq backend, then with Vec<u64> backend.
fn check(values: Vec<u64>) {
    let list = PrefixSumIntList::<u64>::new(&values);
    assert_eq!(list.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(list.index_value(i), v, "EfSeq: value mismatch at index {i}");
    }

    // Check prefix sums
    let mut prefix = 0u64;
    assert_eq!(list.prefix_sum(0), 0);
    for (i, &v) in values.iter().enumerate() {
        prefix += v;
        assert_eq!(
            list.prefix_sum(i + 1),
            prefix,
            "EfSeq: prefix_sum mismatch at index {}",
            i + 1
        );
    }

    // Check with Vec<u64> backend
    let vec_list = to_vec_backend(PrefixSumIntList::<u64>::new(&values));
    assert_eq!(vec_list.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(
            vec_list.index_value(i),
            v,
            "Vec: value mismatch at index {i}"
        );
    }
}

// ────────────────────── value tests ──────────────────────

#[test]
fn test_basic() {
    check(vec![3, 1, 4, 1, 5, 9, 2, 6]);
    check(vec![1, 2, 3, 4, 5, 6, 7]);
    check(vec![42, 1000, 123456, u64::MAX / 4]);
}

#[test]
fn test_zeros() {
    check(vec![0, 0, 0, 0, 0]);
    check(vec![0, 1, 0, 2, 0, 3]);
}

#[test]
fn test_large_values() {
    check(vec![1u64 << 40, 1 << 50, 1 << 60]);
    check(vec![u64::MAX / 4, u64::MAX / 4]);
}

#[test]
fn test_constant() {
    check(vec![42; 100]);
    check(vec![1; 1000]);
}

#[test]
fn test_single() {
    check(vec![0]);
    check(vec![1]);
    check(vec![u64::MAX / 2]);
}

#[test]
fn test_range() {
    check((0..10000).collect());
    check((1..=100).collect());
}

// ────────────────────── empty ──────────────────────

#[test]
fn test_empty() {
    let empty: Vec<u64> = vec![];
    let list = PrefixSumIntList::<u64>::new(&empty);
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());
    assert_eq!(list.prefix_sum(0), 0);

    let vec_list = to_vec_backend(PrefixSumIntList::<u64>::new(&empty));
    assert_eq!(vec_list.len(), 0);
    assert!(vec_list.is_empty());
}

// ────────────────────── prefix_sum boundary ──────────────────────

#[test]
#[should_panic(expected = "index out of bounds")]
fn test_prefix_sum_out_of_bounds() {
    let values = vec![1u64, 2, 3];
    let list = PrefixSumIntList::<u64>::new(&values);
    list.prefix_sum(4); // n = 3, max valid is 3
}
