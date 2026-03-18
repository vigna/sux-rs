use sux::dict::CompIntListBuilder;
use sux::prelude::*;
use value_traits::slices::SliceByValue;

// ────────────────────── u32 tests ──────────────────────

#[test]
fn test_u32_small_values() {
    let values = vec![1u32, 2, 3, 4, 5, 6, 7];
    let ef = CompIntList::new(values.clone());
    assert_eq!(ef.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u32_large_values() {
    let values = vec![42u32, 1000, 123456, u32::MAX / 2, u32::MAX];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u32_powers_of_two() {
    let values: Vec<u32> = (0..31).map(|k| 1u32 << k).collect();
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u32_max() {
    let ef = CompIntList::new(vec![u32::MAX]);
    assert_eq!(ef.index_value(0), u32::MAX);
}

#[test]
fn test_u32_ones() {
    let ef = CompIntList::new(vec![1u32; 100]);
    assert_eq!(ef.len(), 100);
    for i in 0..100 {
        assert_eq!(ef.index_value(i), 1);
    }
}

#[test]
fn test_u32_duplicates() {
    let values = vec![7u32, 7, 7, 42, 42, 1, 1];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
#[should_panic(expected = "strictly positive")]
fn test_u32_zero_panics() {
    CompIntList::new(vec![0u32, 1, 2]);
}

// ────────────────────── u64 tests ──────────────────────

#[test]
fn test_u64_small_values() {
    let values = vec![1u64, 2, 3, 4, 5, 6, 7];
    let ef = CompIntList::new(values.clone());
    assert_eq!(ef.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u64_large_values() {
    let values = vec![42u64, 1000, 123456789, u64::MAX / 2, u64::MAX];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u64_powers_of_two() {
    let values: Vec<u64> = (0..63).map(|k| 1u64 << k).collect();
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u64_max() {
    let ef = CompIntList::new(vec![u64::MAX]);
    assert_eq!(ef.index_value(0), u64::MAX);
}

#[test]
fn test_u64_ones() {
    let ef = CompIntList::new(vec![1u64; 100]);
    assert_eq!(ef.len(), 100);
    for i in 0..100 {
        assert_eq!(ef.index_value(i), 1);
    }
}

#[test]
fn test_u64_duplicates() {
    let values = vec![7u64, 7, 7, 42, 42, 1, 1];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u64_empty() {
    let ef = CompIntList::<u64>::new(Vec::new());
    assert_eq!(ef.len(), 0);
    assert!(ef.is_empty());
}

#[test]
#[should_panic(expected = "strictly positive")]
fn test_u64_zero_panics() {
    CompIntList::new(vec![0u64, 1, 2]);
}

// ────────────────────── u128 tests ──────────────────────

#[test]
fn test_u128_small_values() {
    let values = vec![1u128, 2, 3, 4, 5, 6, 7];
    let ef = CompIntList::new(values.clone());
    assert_eq!(ef.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u128_large_values() {
    let big = u128::MAX;
    let values = vec![1u128, 42, big / 3, big / 2, big];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u128_powers_of_two() {
    let values: Vec<u128> = (0..127).map(|k| 1u128 << k).collect();
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u128_max() {
    let ef = CompIntList::new(vec![u128::MAX]);
    assert_eq!(ef.index_value(0), u128::MAX);
}

#[test]
fn test_u128_ones() {
    let ef = CompIntList::new(vec![1u128; 100]);
    assert_eq!(ef.len(), 100);
    for i in 0..100 {
        assert_eq!(ef.index_value(i), 1);
    }
}

#[test]
fn test_u128_beyond_u64() {
    // Values that don't fit in u64
    let values = vec![
        (1u128 << 64) + 1,
        (1u128 << 100) + 12345,
        u128::MAX - 1,
        u128::MAX,
    ];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u128_duplicates() {
    let big = (1u128 << 100) + 42;
    let values = vec![big, big, big, 1, 1];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
#[should_panic(expected = "strictly positive")]
fn test_u128_zero_panics() {
    CompIntList::new(vec![0u128, 1, 2]);
}

// ────────────────────── builder tests ──────────────────────

#[test]
fn test_builder_push() {
    let mut builder = CompIntListBuilder::new();
    builder.push(1u64);
    builder.push(3);
    builder.push(7);
    builder.push(42);
    builder.push(100);

    let ef = builder.build();
    assert_eq!(ef.len(), 5);
    assert_eq!(ef.index_value(0), 1);
    assert_eq!(ef.index_value(1), 3);
    assert_eq!(ef.index_value(2), 7);
    assert_eq!(ef.index_value(3), 42);
    assert_eq!(ef.index_value(4), 100);
}

#[test]
fn test_builder_extend() {
    let mut builder = CompIntListBuilder::new();
    builder.push(1u32);
    builder.extend([2u32, 3, 4, 5]);

    let ef = builder.build();
    assert_eq!(ef.len(), 5);
    for i in 0..5 {
        assert_eq!(ef.index_value(i), (i + 1) as u32);
    }
}

#[test]
fn test_builder_empty() {
    let builder = CompIntListBuilder::<u64>::new();
    let ef = builder.build();
    assert_eq!(ef.len(), 0);
    assert!(ef.is_empty());
}

#[test]
#[should_panic(expected = "strictly positive")]
fn test_builder_zero_panics() {
    let mut builder = CompIntListBuilder::new();
    builder.push(0u64);
}

// ────────────────────── cross-type / stress ──────────────────────

#[test]
fn test_u64_mixed_widths() {
    // Mix values with very different bit lengths to stress
    // the variable-width packing
    let values = vec![
        1u64,             // width 0
        3,                // width 1
        7,                // width 2
        255,              // width 7
        65535,            // width 15
        (1u64 << 32) - 1, // width 31
        u64::MAX,         // width 63
    ];
    let ef = CompIntList::new(values.clone());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u32_many_values() {
    // Test with a larger number of values
    let values: Vec<u32> = (1..=10000).collect();
    let ef = CompIntList::new(values.clone());
    assert_eq!(ef.len(), 10000);
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}

#[test]
fn test_u64_single_value() {
    let ef = CompIntList::new(vec![42u64]);
    assert_eq!(ef.len(), 1);
    assert_eq!(ef.index_value(0), 42);
}

#[test]
fn test_u128_single_value() {
    let v = (1u128 << 120) + 999;
    let ef = CompIntList::new(vec![v]);
    assert_eq!(ef.len(), 1);
    assert_eq!(ef.index_value(0), v);
}

// ────────────────────── map_delimiters tests ──────────────────────

#[test]
fn test_map_delimiters_to_vec() {
    let values = vec![1u64, 3, 7, 42, 100];
    let ef = CompIntList::new(values.clone());

    // Map delimiters from EfSeq to Vec<u64>
    let ef = unsafe {
        ef.map_delimiters(|d| {
            let mut v = Vec::with_capacity(d.len());
            for i in 0..d.len() {
                v.push(d.index_value(i));
            }
            v
        })
    };

    assert_eq!(ef.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ef.index_value(i), v, "mismatch at index {i}");
    }
}
