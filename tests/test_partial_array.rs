/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::array::partial_array;

#[test]
fn test_empty_array() {
    let builder = partial_array::new_sparse::<i32>(0, 0);
    let array = builder.build();

    assert_eq!(array.len(), 0);
    assert!(array.is_empty());
    assert_eq!(array.num_values(), 0);

    let builder = partial_array::new_dense::<i32>(0);
    let array = builder.build();

    assert_eq!(array.len(), 0);
    assert!(array.is_empty());
    assert_eq!(array.num_values(), 0);
}

#[test]
fn test_basic_operations() {
    let mut builder = partial_array::new_sparse(10, 3);

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

    let mut builder = partial_array::new_dense(10);

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
fn test_replacement_sparse() {
    let mut builder = partial_array::new_sparse(5, 1);

    builder.set(2, "first");
    builder.set(2, "second");
}

#[test]
#[should_panic(expected = "Positions must be set in increasing order: got 2 after 2")]
fn test_replacement_dense() {
    let mut builder = partial_array::new_dense(5);

    builder.set(2, "first");
    builder.set(2, "second");
}

#[test]
#[should_panic(expected = "Position 10 is out of bounds for array of len 10")]
fn test_builder_bounds_check_sparse() {
    let mut builder = partial_array::new_sparse::<&str>(10, 1);
    builder.set(10, "oops");
}

#[test]
#[should_panic(expected = "Position 10 is out of bounds for array of len 10")]
fn test_builder_bounds_check_dense() {
    let mut builder = partial_array::new_dense::<&str>(10);
    builder.set(10, "oops");
}

#[test]
#[should_panic(expected = "Position 5 is out of bounds for array of len 5")]
fn test_array_bounds_check_sparse() {
    let builder = partial_array::new_sparse::<usize>(5, 0);
    let array = builder.build();
    array.get(5);
}

#[test]
#[should_panic(expected = "Position 5 is out of bounds for array of len 5")]
fn test_array_bounds_check_dense() {
    let builder = partial_array::new_dense::<usize>(5);
    let array = builder.build();
    array.get(5);
}

#[test]
fn test_single_element() {
    let mut builder = partial_array::new_sparse(1000, 1);
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

    let mut builder = partial_array::new_dense(1000);
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
