/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::array::PartialArrayBuilder;

#[test]
fn test_empty_array() {
    let builder = PartialArrayBuilder::<i32>::new(0);
    let array = builder.build();

    assert_eq!(array.len(), 0);
    assert!(array.is_empty());
    assert_eq!(array.num_values(), 0);
}

#[test]
fn test_basic_operations() {
    let mut builder = PartialArrayBuilder::new(10);

    builder.set(1, "foo");
    builder.set(2, "hello");
    builder.set(7, "world");

    let array = builder.build();

    assert_eq!(array.len(), 10);
    assert_eq!(array.num_values(), 3);

    assert_eq!(array.get(0), None);
    assert_eq!(array.get(1), Some(&"foo"));
    assert_eq!(array.get(2), Some(&"hello"));
    assert_eq!(array.get(3), None);
    assert_eq!(array.get(4), None);
    assert_eq!(array.get(5), None);
    assert_eq!(array.get(6), None);
    assert_eq!(array.get(7), Some(&"world"));
    assert_eq!(array.get(8), None);
    assert_eq!(array.get(9), None);
}

#[test]
#[should_panic(expected = "Positions must be set in increasing order: got 2 after 2")]
fn test_replacement() {
    let mut builder = PartialArrayBuilder::new(5);

    builder.set(2, "first");
    builder.set(2, "second");
}

#[test]
fn test_iterator() {
    let mut builder = PartialArrayBuilder::new(8);

    builder.set(1, 10);
    builder.set(2, 20);
    builder.set(5, 50);
    builder.set(7, 70);

    let array = builder.build();

    let pairs: Vec<_> = array.iter().collect();
    assert_eq!(pairs, vec![(1, &10), (2, &20), (5, &50), (7, &70)]);

    let values: Vec<_> = array.values().cloned().collect();
    assert_eq!(values, vec![10, 20, 50, 70]);

    let positions: Vec<_> = array.positions().collect();
    assert_eq!(positions, vec![1, 2, 5, 7]);
}

#[test]
#[should_panic(expected = "Position 10 is out of bounds for array of size 10")]
fn test_builder_bounds_check() {
    let mut builder = PartialArrayBuilder::new(10);
    builder.set(10, "oops");
}

#[test]
#[should_panic(expected = "Bit index out of bounds: 5 >= 5")]
fn test_array_bounds_check() {
    let builder = PartialArrayBuilder::<i32>::new(5);
    let array = builder.build();
    array.get(5);
}

#[test]
fn test_dense_array() {
    let mut builder = PartialArrayBuilder::new(5);

    for i in 0..5 {
        builder.set(i, i * i);
    }

    let array = builder.build();

    for i in 0..5 {
        assert_eq!(array.get(i), Some(&(i * i)));
    }

    assert_eq!(array.num_values(), 5);
}

#[test]
fn test_single_element() {
    let mut builder = PartialArrayBuilder::new(1000);
    builder.set(500, "middle");

    let array = builder.build();

    assert_eq!(array.len(), 1000);
    assert_eq!(array.num_values(), 1);
    assert_eq!(array.get(500), Some(&"middle"));

    for i in 0..1000 {
        if i != 500 {
            assert_eq!(array.get(i), None);
        }
    }
}
