/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::sync::atomic::{AtomicU32, Ordering};

use sux::prelude::*;

#[test]
fn test_is_empty() {
    assert!(BitFieldVec::<usize, _>::new(0, 0).is_empty());
}

#[test]
fn test_set() {
    let mut s = BitFieldVec::<usize, _>::new(3, 10);
    s.set(0, 1);
    assert_eq!(s.get(0), 1);

    let mut s = BitFieldVec::<usize, _>::new(0, 10);
    s.set(0, 0);
    assert_eq!(s.get(0), 0);
}

#[test]
#[should_panic]
fn test_set_too_large() {
    let mut s = BitFieldVec::<usize, _>::new(3, 10);
    s.set(0, 10);
}

#[test]
#[should_panic]
fn test_set_ouf_of_bounds() {
    let mut s = BitFieldVec::<usize, _>::new(3, 10);
    s.set(10, 4);
}

#[test]
fn test_set_atomic() {
    let s = AtomicBitFieldVec::<usize, _>::new(3, 10);
    s.set_atomic(0, 1, Ordering::Relaxed);
    assert_eq!(s.get_atomic(0, Ordering::Relaxed), 1);

    use sux::traits::bit_field_slice::AtomicHelper;
    s.set(0, 1, Ordering::Relaxed);
    assert_eq!(s.get(0, Ordering::Relaxed), 1);
    unsafe {
        s.set_unchecked(0, 1, Ordering::Relaxed);
        assert_eq!(s.get_unchecked(0, Ordering::Relaxed), 1);
    }

    let s = AtomicBitFieldVec::<usize, _>::new(0, 10);
    s.set_atomic(0, 0, Ordering::Relaxed);
    assert_eq!(s.get_atomic(0, Ordering::Relaxed), 0);
}

#[test]
#[should_panic]
fn test_set_atomic_too_large() {
    let s = AtomicBitFieldVec::<usize, _>::new(3, 10);
    s.set_atomic(0, 10, Ordering::Relaxed);
}

#[test]
#[should_panic]
fn test_set_atomic_ouf_of_bounds() {
    let s = AtomicBitFieldVec::<usize, _>::new(3, 10);
    s.set_atomic(10, 4, Ordering::Relaxed);
}

#[test]
fn test_iterator() {
    let mut s = BitFieldVec::<usize, _>::new(3, 0);
    let t = [0, 1, 2, 2, 1, 0, 3, 0];
    s.extend(t);
    for i in s.iter() {
        assert_eq!(s.get(i), t[i]);
    }
}

#[test]
fn test_slices() {
    let mut s = vec![0_u32, 1, 2, 3];
    assert_eq!(s.bit_width(), 32);
    assert_eq!(BitFieldSliceCore::len(&s), 4);

    s.set(0, 1);
    assert_eq!(s.get(0), 1);
    s.reset();
    assert_eq!(s.get(0), 0);

    unsafe {
        s.set_unchecked(0, 1);
        assert_eq!(s.get_unchecked(0), 1);
        s.reset();
        assert_eq!(s.get_unchecked(0), 0);
    }
}

#[test]
fn test_slices_atomic() {
    let mut s = vec![
        AtomicU32::new(0),
        AtomicU32::new(1),
        AtomicU32::new(2),
        AtomicU32::new(3),
    ];
    assert_eq!(s.bit_width(), 32);
    assert_eq!(BitFieldSliceCore::len(&s), 4);

    s.set_atomic(0, 1, Ordering::Relaxed);
    assert_eq!(s.get_atomic(0, Ordering::Relaxed), 1);
    s.reset_atomic(Ordering::Relaxed);
    assert_eq!(s.get_atomic(0, Ordering::Relaxed), 0);

    unsafe {
        s.set_atomic_unchecked(0, 1, Ordering::Relaxed);
        assert_eq!(s.get_atomic_unchecked(0, Ordering::Relaxed), 1);
        s.reset_atomic(Ordering::Relaxed);
        assert_eq!(s.get_atomic_unchecked(0, Ordering::Relaxed), 0);
    }
}
