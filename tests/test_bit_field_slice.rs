/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use bit_field_slice::*;
use std::sync::atomic::{AtomicU32, Ordering};
use sux::prelude::*;
use value_traits::slices::{SliceByValue, SliceByValueMut};

#[test]
fn test_is_empty() {
    assert!(BitFieldVec::<usize, _>::new(0, 0).is_empty());
}

#[test]
fn test_set_value() {
    let mut s = BitFieldVec::<usize, _>::new(3, 10);
    s.set_value(0, 1);
    assert_eq!(s.index_value(0), 1);

    let mut s = BitFieldVec::<usize, _>::new(0, 10);
    s.set_value(0, 0);
    assert_eq!(s.index_value(0), 0);
}

#[test]
#[should_panic]
fn test_set_too_large() {
    let mut s = BitFieldVec::<usize, _>::new(3, 10);
    s.set_value(0, 10);
}

#[test]
#[should_panic]
fn test_set_out_of_bounds() {
    let mut s = BitFieldVec::<usize, _>::new(3, 10);
    s.set_value(10, 4);
}

#[test]
fn test_set_atomic() {
    let s = AtomicBitFieldVec::<usize, _>::new(3, 10);
    s.set_atomic(0, 1, Ordering::Relaxed);
    assert_eq!(s.get_atomic(0, Ordering::Relaxed), 1);

    s.set_atomic(0, 1, Ordering::Relaxed);
    assert_eq!(s.get_atomic(0, Ordering::Relaxed), 1);
    unsafe {
        s.set_atomic_unchecked(0, 1, Ordering::Relaxed);
        assert_eq!(s.get_atomic_unchecked(0, Ordering::Relaxed), 1);
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
fn test_set_atomic_out_of_bounds() {
    let s = AtomicBitFieldVec::<usize, _>::new(3, 10);
    s.set_atomic(10, 4, Ordering::Relaxed);
}

#[test]
fn test_iterator() {
    let mut s = BitFieldVec::<usize, _>::new(3, 0);
    let t = [0, 1, 2, 2, 1, 0, 3, 0];
    s.extend(t);
    for i in s.iter() {
        assert_eq!(s.index_value(i), t[i]);
    }
}

#[test]
fn test_slices() {
    let mut s = vec![0_u32, 1, 2, 3];
    assert_eq!(s.bit_width(), 32);
    assert_eq!(SliceByValue::len(&s), 4);

    s.set_value(0, 1);
    assert_eq!(s.index_value(0), 1);
    s.reset();
    assert_eq!(s.index_value(0), 0);

    unsafe {
        s.set_value_unchecked(0, 1);
        assert_eq!(s.get_value_unchecked(0), 1);
        s.reset();
        assert_eq!(s.get_value_unchecked(0), 0);
    }

    s.fill(10);
    for i in 0..s.len() {
        assert_eq!(s.index_value(i), 10);
    }
    s.reset();
    for i in 0..s.len() {
        assert_eq!(s.index_value(i), 0);
    }
    #[cfg(feature = "rayon")]
    {
        s.fill(10);
        for i in 0..s.len() {
            assert_eq!(s.index_value(i), 10);
        }
        s.par_reset();
        for i in 0..s.len() {
            assert_eq!(s.index_value(i), 0);
        }
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
    assert_eq!(AtomicBitFieldSlice::len(&s), 4);

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

// Tests for BitWidth on various types

#[test]
fn test_bit_width_slice() {
    let slice: &[u8] = &[1, 2, 3];
    assert_eq!(BitWidth::bit_width(&slice), 8);

    let slice: &[u16] = &[1, 2, 3];
    assert_eq!(BitWidth::bit_width(&slice), 16);

    let slice: &[u32] = &[1, 2, 3];
    assert_eq!(BitWidth::bit_width(&slice), 32);

    let slice: &[u64] = &[1, 2, 3];
    assert_eq!(BitWidth::bit_width(&slice), 64);

    let slice: &[usize] = &[1, 2, 3];
    assert_eq!(BitWidth::bit_width(&slice), usize::BITS as usize);
}

#[test]
fn test_bit_width_vec() {
    let vec: Vec<u8> = vec![1, 2, 3];
    assert_eq!(BitWidth::bit_width(&vec), 8);

    let vec: Vec<u32> = vec![1, 2, 3];
    assert_eq!(BitWidth::bit_width(&vec), 32);

    let vec: Vec<u64> = vec![1, 2, 3];
    assert_eq!(BitWidth::bit_width(&vec), 64);
}

#[test]
fn test_bit_width_array() {
    let arr: [u8; 3] = [1, 2, 3];
    assert_eq!(BitWidth::bit_width(&arr), 8);

    let arr: [u32; 3] = [1, 2, 3];
    assert_eq!(BitWidth::bit_width(&arr), 32);
}

#[test]
fn test_bit_width_delegation() {
    let vec: Vec<u32> = vec![1, 2, 3];
    let ref_vec = &vec;
    assert_eq!(BitWidth::bit_width(&ref_vec), 32);

    let boxed: Box<[u32]> = vec![1, 2, 3].into_boxed_slice();
    assert_eq!(BitWidth::bit_width(&boxed), 32);
}

// Tests for BitFieldSlice

#[test]
fn test_bit_field_slice_as_slice() {
    let slice: &[u32] = &[1, 2, 3];
    assert_eq!(BitFieldSlice::as_slice(&slice), &[1, 2, 3]);

    let vec: Vec<u32> = vec![4, 5, 6];
    assert_eq!(BitFieldSlice::as_slice(&vec), &[4, 5, 6]);

    let arr: [u32; 3] = [7, 8, 9];
    assert_eq!(BitFieldSlice::as_slice(&arr), &[7, 8, 9]);
}

#[test]
fn test_bit_field_slice_delegation() {
    let vec: Vec<u32> = vec![1, 2, 3];
    let ref_vec = &vec;
    assert_eq!(BitFieldSlice::as_slice(&ref_vec), &[1, 2, 3]);

    let boxed: Box<[u32]> = vec![4, 5, 6].into_boxed_slice();
    assert_eq!(BitFieldSlice::as_slice(&boxed), &[4, 5, 6]);
}

// Tests for BitFieldSliceMut

#[test]
fn test_bit_field_slice_mut_reset() {
    let mut slice: Vec<u32> = vec![1, 2, 3, 4, 5];
    BitFieldSliceMut::reset(&mut slice[..]);
    assert_eq!(slice, vec![0, 0, 0, 0, 0]);

    let mut vec: Vec<u32> = vec![10, 20, 30];
    BitFieldSliceMut::reset(&mut vec);
    assert_eq!(vec, vec![0, 0, 0]);

    let mut arr: [u32; 3] = [100, 200, 300];
    BitFieldSliceMut::reset(&mut arr);
    assert_eq!(arr, [0, 0, 0]);
}

#[cfg(feature = "rayon")]
#[test]
fn test_bit_field_slice_mut_par_reset() {
    let mut slice: Vec<u32> = vec![1, 2, 3, 4, 5];
    BitFieldSliceMut::par_reset(&mut slice[..]);
    assert_eq!(slice, vec![0, 0, 0, 0, 0]);

    let mut vec: Vec<u32> = vec![10, 20, 30];
    BitFieldSliceMut::par_reset(&mut vec);
    assert_eq!(vec, vec![0, 0, 0]);

    let mut arr: [u32; 3] = [100, 200, 300];
    BitFieldSliceMut::par_reset(&mut arr);
    assert_eq!(arr, [0, 0, 0]);
}

#[test]
fn test_bit_field_slice_mut_as_mut_slice() {
    let mut slice: Vec<u32> = vec![1, 2, 3];
    {
        let mslice = BitFieldSliceMut::as_mut_slice(&mut slice[..]);
        mslice[0] = 100;
    }
    assert_eq!(slice, vec![100, 2, 3]);

    let mut vec: Vec<u32> = vec![1, 2, 3];
    {
        let mslice = BitFieldSliceMut::as_mut_slice(&mut vec);
        mslice[1] = 200;
    }
    assert_eq!(vec, vec![1, 200, 3]);

    let mut arr: [u32; 3] = [1, 2, 3];
    {
        let mslice = BitFieldSliceMut::as_mut_slice(&mut arr);
        mslice[2] = 300;
    }
    assert_eq!(arr, [1, 2, 300]);
}

#[test]
fn test_bit_field_slice_mut_delegation() {
    let mut vec: Vec<u32> = vec![1, 2, 3];
    let mut_ref = &mut vec;
    BitFieldSliceMut::reset(mut_ref);
    assert_eq!(vec, vec![0, 0, 0]);

    let mut boxed: Box<[u32]> = vec![10, 20, 30].into_boxed_slice();
    BitFieldSliceMut::reset(&mut boxed);
    assert!(boxed.iter().all(|&x| x == 0));
}

// Tests for AtomicBitFieldSlice

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize};

#[test]
fn test_atomic_bit_width() {
    let slice: &[AtomicU8] = &[];
    assert_eq!(BitWidth::bit_width(&slice), 8);

    let slice: &[AtomicU32] = &[];
    assert_eq!(BitWidth::bit_width(&slice), 32);

    let slice: &[AtomicU64] = &[];
    assert_eq!(BitWidth::bit_width(&slice), 64);

    let slice: &[AtomicUsize] = &[];
    assert_eq!(BitWidth::bit_width(&slice), usize::BITS as usize);

    // Vec
    let vec: Vec<AtomicU32> = Vec::new();
    assert_eq!(BitWidth::bit_width(&vec), 32);

    // Array
    let arr: [AtomicU32; 0] = [];
    assert_eq!(BitWidth::bit_width(&arr), 32);
}

#[test]
fn test_atomic_bit_field_slice_basic() {
    let slice: Vec<AtomicU32> = vec![AtomicU32::new(10), AtomicU32::new(20), AtomicU32::new(30)];
    assert_eq!(AtomicBitFieldSlice::len(&slice[..]), 3);
    assert!(!AtomicBitFieldSlice::is_empty(&slice[..]));
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&slice[..], 0, Ordering::Relaxed),
        10
    );
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&slice[..], 1, Ordering::Relaxed),
        20
    );
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&slice[..], 2, Ordering::Relaxed),
        30
    );
}

#[test]
fn test_atomic_bit_field_slice_empty() {
    let slice: Vec<AtomicU32> = Vec::new();
    assert_eq!(AtomicBitFieldSlice::len(&slice[..]), 0);
    assert!(AtomicBitFieldSlice::is_empty(&slice[..]));
}

#[test]
fn test_atomic_bit_field_slice_set_atomic() {
    let slice: Vec<AtomicU32> = vec![AtomicU32::new(0), AtomicU32::new(0)];
    AtomicBitFieldSlice::set_atomic(&slice[..], 0, 42, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&slice[..], 0, Ordering::Relaxed),
        42
    );

    AtomicBitFieldSlice::set_atomic(&slice[..], 1, 100, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&slice[..], 1, Ordering::Relaxed),
        100
    );
}

#[test]
fn test_atomic_bit_field_slice_unchecked() {
    let slice: Vec<AtomicU32> = vec![AtomicU32::new(5), AtomicU32::new(15)];
    unsafe {
        assert_eq!(
            AtomicBitFieldSlice::get_atomic_unchecked(&slice[..], 0, Ordering::Relaxed),
            5
        );
        AtomicBitFieldSlice::set_atomic_unchecked(&slice[..], 0, 50, Ordering::Relaxed);
        assert_eq!(
            AtomicBitFieldSlice::get_atomic_unchecked(&slice[..], 0, Ordering::Relaxed),
            50
        );
    }
}

#[test]
fn test_atomic_bit_field_slice_reset() {
    let mut slice: Vec<AtomicU32> = vec![AtomicU32::new(1), AtomicU32::new(2), AtomicU32::new(3)];
    AtomicBitFieldSlice::reset_atomic(&mut slice[..], Ordering::Relaxed);
    for i in 0..3 {
        assert_eq!(
            AtomicBitFieldSlice::get_atomic(&slice[..], i, Ordering::Relaxed),
            0
        );
    }
}

#[cfg(feature = "rayon")]
#[test]
fn test_atomic_bit_field_slice_par_reset() {
    let mut slice: Vec<AtomicU32> =
        vec![AtomicU32::new(10), AtomicU32::new(20), AtomicU32::new(30)];
    AtomicBitFieldSlice::par_reset_atomic(&mut slice[..], Ordering::Relaxed);
    for i in 0..3 {
        assert_eq!(
            AtomicBitFieldSlice::get_atomic(&slice[..], i, Ordering::Relaxed),
            0
        );
    }
}

#[test]
fn test_atomic_bit_field_slice_vec() {
    let mut vec: Vec<AtomicU64> = vec![AtomicU64::new(100), AtomicU64::new(200)];
    assert_eq!(AtomicBitFieldSlice::len(&vec), 2);
    assert!(!AtomicBitFieldSlice::is_empty(&vec));

    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&vec, 0, Ordering::Relaxed),
        100
    );
    AtomicBitFieldSlice::set_atomic(&vec, 0, 999, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&vec, 0, Ordering::Relaxed),
        999
    );

    AtomicBitFieldSlice::reset_atomic(&mut vec, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&vec, 0, Ordering::Relaxed),
        0
    );
}

#[cfg(feature = "rayon")]
#[test]
fn test_atomic_bit_field_slice_vec_par_reset() {
    let mut vec: Vec<AtomicU64> = vec![AtomicU64::new(100), AtomicU64::new(200)];
    AtomicBitFieldSlice::par_reset_atomic(&mut vec, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&vec, 0, Ordering::Relaxed),
        0
    );
}

#[test]
fn test_atomic_bit_field_slice_array() {
    let mut arr: [AtomicUsize; 3] = [
        AtomicUsize::new(1),
        AtomicUsize::new(2),
        AtomicUsize::new(3),
    ];
    assert_eq!(AtomicBitFieldSlice::len(&arr), 3);
    assert!(!AtomicBitFieldSlice::is_empty(&arr));

    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&arr, 1, Ordering::Relaxed),
        2
    );
    AtomicBitFieldSlice::set_atomic(&arr, 1, 222, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(&arr, 1, Ordering::Relaxed),
        222
    );

    unsafe {
        assert_eq!(
            AtomicBitFieldSlice::get_atomic_unchecked(&arr, 2, Ordering::Relaxed),
            3
        );
        AtomicBitFieldSlice::set_atomic_unchecked(&arr, 2, 333, Ordering::Relaxed);
        assert_eq!(
            AtomicBitFieldSlice::get_atomic_unchecked(&arr, 2, Ordering::Relaxed),
            333
        );
    }

    AtomicBitFieldSlice::reset_atomic(&mut arr, Ordering::Relaxed);
    for i in 0..3 {
        assert_eq!(
            AtomicBitFieldSlice::get_atomic(&arr, i, Ordering::Relaxed),
            0
        );
    }
}

#[cfg(feature = "rayon")]
#[test]
fn test_atomic_bit_field_slice_array_par_reset() {
    let mut arr: [AtomicUsize; 3] = [
        AtomicUsize::new(1),
        AtomicUsize::new(2),
        AtomicUsize::new(3),
    ];
    AtomicBitFieldSlice::par_reset_atomic(&mut arr, Ordering::Relaxed);
    for i in 0..3 {
        assert_eq!(
            AtomicBitFieldSlice::get_atomic(&arr, i, Ordering::Relaxed),
            0
        );
    }
}

#[test]
fn test_atomic_delegation() {
    let mut vec: Vec<AtomicU32> = vec![AtomicU32::new(10), AtomicU32::new(20)];
    let mut_ref = &mut vec;
    assert_eq!(AtomicBitFieldSlice::len(mut_ref), 2);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(mut_ref, 0, Ordering::Relaxed),
        10
    );
    AtomicBitFieldSlice::set_atomic(mut_ref, 0, 100, Ordering::Relaxed);
    AtomicBitFieldSlice::reset_atomic(mut_ref, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(mut_ref, 0, Ordering::Relaxed),
        0
    );
}

#[cfg(feature = "rayon")]
#[test]
fn test_atomic_delegation_par_reset() {
    let mut vec: Vec<AtomicU32> = vec![AtomicU32::new(5), AtomicU32::new(15)];
    let mut_ref = &mut vec;
    AtomicBitFieldSlice::par_reset_atomic(mut_ref, Ordering::Relaxed);
    assert_eq!(
        AtomicBitFieldSlice::get_atomic(mut_ref, 0, Ordering::Relaxed),
        0
    );
}

#[test]
fn test_various_integer_types() {
    // u8 tests
    let mut slice_u8: Vec<u8> = vec![1, 2, 3];
    assert_eq!(BitWidth::bit_width(&slice_u8[..]), 8);
    BitFieldSliceMut::reset(&mut slice_u8[..]);
    assert_eq!(slice_u8, vec![0, 0, 0]);

    // u16 tests
    let mut slice_u16: Vec<u16> = vec![100, 200, 300];
    assert_eq!(BitWidth::bit_width(&slice_u16[..]), 16);
    BitFieldSliceMut::reset(&mut slice_u16[..]);
    assert_eq!(slice_u16, vec![0, 0, 0]);

    // u64 tests
    let mut slice_u64: Vec<u64> = vec![1000, 2000, 3000];
    assert_eq!(BitWidth::bit_width(&slice_u64[..]), 64);
    BitFieldSliceMut::reset(&mut slice_u64[..]);
    assert_eq!(slice_u64, vec![0, 0, 0]);

    // u128 tests
    let mut slice_u128: Vec<u128> = vec![10000, 20000, 30000];
    assert_eq!(BitWidth::bit_width(&slice_u128[..]), 128);
    BitFieldSliceMut::reset(&mut slice_u128[..]);
    assert_eq!(slice_u128, vec![0, 0, 0]);
}
