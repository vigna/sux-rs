/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use core::sync::atomic::Ordering;
use epserde::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use std::sync::atomic::AtomicUsize;
use sux::prelude::*;

#[test]
fn test() {
    let n = 50;
    let n2 = 100;
    let u = 1000;

    let mut rng = SmallRng::seed_from_u64(0);

    let bm = BitVec::with_value(u, true);

    assert_eq!(bm.len(), u);
    assert_eq!(bm.count_ones(), u);

    // Dirty vector
    let ones = [usize::MAX; 2];
    assert_eq!(unsafe { BitVec::from_raw_parts(&ones, 0) }.count_ones(), 0);
    assert_eq!(unsafe { BitVec::from_raw_parts(&ones, 1) }.count_ones(), 1);

    for i in 0..u {
        assert!(bm[i]);
    }

    let mut bm = BitVec::new(u);

    for _ in 0..10 {
        let mut values = (0..u).collect::<Vec<_>>();
        let (indices, _) = values.partial_shuffle(&mut rng, n2);

        for i in indices[..n].iter().copied() {
            bm.set(i, true);
        }

        for i in 0..u {
            assert_eq!(bm.get(i), indices[..n].contains(&i));
            assert_eq!(bm[i], indices[..n].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i, true);
        }

        for i in 0..u {
            assert_eq!(bm.get(i), indices.contains(&i));
        }

        for i in indices[..n].iter().copied() {
            bm.set(i, false);
        }

        for i in 0..u {
            assert_eq!(bm.get(i), indices[n..].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i, false);
        }

        for i in 0..u {
            assert!(!bm.get(i));
        }
    }

    let bm: AtomicBitVec = bm.into();

    // Dirty vector
    let ones = [AtomicUsize::new(usize::MAX), AtomicUsize::new(usize::MAX)];
    assert_eq!(
        unsafe { AtomicBitVec::from_raw_parts(&ones, 0) }.count_ones(),
        0
    );
    assert_eq!(
        unsafe { AtomicBitVec::from_raw_parts(&ones, 1) }.count_ones(),
        1
    );

    for _ in 0..10 {
        let mut values = (0..u).collect::<Vec<_>>();
        let (indices, _) = values.partial_shuffle(&mut rng, n2);

        for i in indices[..n].iter().copied() {
            bm.set(i, true, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(bm.get(i, Ordering::Relaxed), indices[..n].contains(&i));
            assert_eq!(bm[i], indices[..n].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i, true, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(bm.get(i, Ordering::Relaxed), indices.contains(&i));
        }

        for i in indices[..n].iter().copied() {
            bm.set(i, false, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(bm.get(i, Ordering::Relaxed), indices[n..].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i, false, Ordering::Relaxed);
        }

        for i in 0..u {
            assert!(!bm.get(i, Ordering::Relaxed));
        }
    }

    let bm = AtomicBitVec::with_value(u, true);

    assert_eq!(bm.len(), u);
    assert_eq!(bm.count_ones(), u);

    for i in 0..u {
        assert!(bm.get(i, Ordering::Relaxed));
    }
}

#[test]
fn test_atomic_swap() {
    let b = AtomicBitVec::new(10);
    assert!(!b.get(1, Ordering::Relaxed));
    assert!(!b.swap(1, true, Ordering::Relaxed));
    assert!(b.get(1, Ordering::Relaxed));
    assert!(b.swap(1, true, Ordering::Relaxed));
    assert!(b.swap(1, false, Ordering::Relaxed));
    assert!(!b.get(1, Ordering::Relaxed));
}

#[test]
fn test_push_pop() {
    let mut b = BitVec::new(0);
    b.push(true);
    b.push(false);
    assert!(b.get(0));
    assert!(!b.get(1));
    for i in 2..200 {
        b.push(i % 2 == 0);
    }
    for i in 2..200 {
        assert_eq!(b.get(i), i % 2 == 0);
    }
    for i in 0..200 {
        assert_eq!(b.pop(), Some(i % 2 != 0));
    }
    assert_eq!(b.pop(), None);
}

#[test]
fn test_resize() {
    let mut c = BitVec::new(0);
    c.resize(100, true);
    for i in 0..100 {
        assert!(c.get(i));
    }
    c.resize(50, false);
    for i in 0..50 {
        assert!(c.get(i));
    }
    assert_eq!(c.len(), 50);
}

#[test]
fn test_fill() {
    for len in [0, 1, 64, 65, 100, 127, 128, 1000] {
        let mut c = BitVec::new(len);
        c.fill(true);
        for (i, b) in c.into_iter().enumerate() {
            assert!(b, "{}", i);
        }

        if len != c.capacity() {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize] & 1 << (len % usize::BITS as usize),
                0
            );
        }

        c.fill(false);
        for (i, b) in c.into_iter().enumerate() {
            assert!(!b, "{}", i);
        }
    }
}

#[test]
fn test_atomic_fill() {
    for len in [0, 1, 64, 65, 100, 127, 128, 1000] {
        let mut c = AtomicBitVec::new(len);
        c.fill(true, Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(c.get(i, Ordering::Relaxed), "{}", i);
        }

        if len % usize::BITS as usize != 0 {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize].load(Ordering::Relaxed)
                    & 1 << (len % usize::BITS as usize),
                0
            );
        }

        c.fill(false, Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(!c.get(i, Ordering::Relaxed), "{}", i);
        }
    }
}

#[test]
fn test_flip() {
    for len in [0, 1, 64, 65, 100, 127, 128, 1000] {
        let mut c = BitVec::new(len);
        c.flip();
        for (i, b) in c.into_iter().enumerate() {
            assert!(b, "{}", i);
        }

        if len != c.capacity() {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize] & 1 << (len % usize::BITS as usize),
                0
            );
        }

        c.flip();
        for (i, b) in c.into_iter().enumerate() {
            assert!(!b, "{}", i);
        }
    }
}

#[test]
fn test_atomic_flip() {
    for len in [0, 1, 64, 65, 100, 127, 128, 1000] {
        let mut c = AtomicBitVec::new(len);
        c.flip(Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(c.get(i, Ordering::Relaxed), "{}", i);
        }

        if len % usize::BITS as usize != 0 {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize].load(Ordering::Relaxed)
                    & 1 << (len % usize::BITS as usize),
                0
            );
        }

        c.flip(Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(!c.get(i, Ordering::Relaxed), "{}", i);
        }
    }
}

#[test]
fn test_iter() {
    let mut c = BitVec::new(100);
    for i in 0..100 {
        c.set(i, i % 2 == 0);
    }

    for (i, b) in c.iter().enumerate() {
        assert_eq!(b, i % 2 == 0);
    }
}

#[test]
fn test_iter_ones_alternate() {
    let mut c = BitVec::new(200);
    for i in 0..200 {
        c.set(i, i % 2 == 0);
    }

    for (i, p) in c.iter_ones().enumerate() {
        assert_eq!(p, i * 2);
    }
}

#[test]
fn test_iter_ones_empty() {
    let c = BitVec::new(200);
    assert_eq!(c.iter_ones().next(), None);
}

#[test]
fn test_iter_ones_one() {
    let mut c = BitVec::new(200);
    c.set(1, true);
    let mut i = c.iter_ones();
    assert_eq!(i.next(), Some(1));
    assert_eq!(i.next(), None);
}

#[test]
fn test_iter_zeros_alternate() {
    let mut c = BitVec::new(200);
    for i in 0..200 {
        c.set(i, i % 2 != 0);
    }

    for (i, p) in c.iter_zeros().enumerate() {
        assert_eq!(p, i * 2);
    }
}

#[test]
fn test_iter_zeros_full() {
    let mut c = BitVec::new(200);
    c.flip();
    assert_eq!(c.iter_zeros().next(), None);
}

#[test]
fn test_iter_zeros_one() {
    let mut c = BitVec::new(200);
    c.set(1, true);
    c.flip();
    let mut i = c.iter_zeros();
    assert_eq!(i.next(), Some(1));
    assert_eq!(i.next(), None);
}

#[test]
fn test_atomic_iter() {
    let mut c = AtomicBitVec::new(100);
    for i in 0..100 {
        c.set(i, i % 2 == 0, Ordering::Relaxed);
    }

    for (i, b) in c.iter().enumerate() {
        assert_eq!(b, i % 2 == 0);
    }
}

#[test]
fn test_eq() {
    let mut b = BitVec::new(0);
    let mut c = BitVec::new(0);
    assert_eq!(b, c);

    b.push(true);
    assert_ne!(b, c);
    c.push(true);
    assert_eq!(b, c);

    for i in 0..64 {
        b.push(i % 2 == 0);
        c.push(i % 2 == 0);
        assert_eq!(b, c);
    }

    let c: BitVec<Box<[usize]>> = c.into();
    assert_eq!(b, c);
    let (bits, l) = c.into_raw_parts();
    let d = unsafe { BitVec::from_raw_parts(bits.as_ref(), l) };
    assert_eq!(b, d);
}

#[test]
fn test_epserde() {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut b = BitVec::new(200);
    for i in 0..200 {
        b.set(i, rng.next_u64() % 2 != 0);
    }

    let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file).unwrap());
    b.serialize(&mut file).unwrap();
    drop(file);

    let c = <BitVec<Vec<usize>>>::mmap(&tmp_file, epserde::deser::Flags::empty()).unwrap();

    for i in 0..200 {
        assert_eq!(b.get(i), c.get(i));
    }
}

#[test]
fn test_from() {
    // Vec to atomic vec
    let mut b = BitVec::<Vec<usize>>::new(10);
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let b: AtomicBitVec<Vec<AtomicUsize>> = b.into();
    let b: BitVec<Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
        assert_eq!(b[i], i % 2 == 0);
    }

    // Boxed slice to atomic boxed slice
    let bits = vec![0; 10].into_boxed_slice();
    let mut b = unsafe { BitVec::<Box<[usize]>>::from_raw_parts(bits, 10) };
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let b: AtomicBitVec<Box<[AtomicUsize]>> = b.into();
    let b: BitVec<Box<[usize]>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
    }

    // Reference to atomic reference
    let bits = vec![0; 10].into_boxed_slice();
    let mut b = unsafe { BitVec::<Box<[usize]>>::from_raw_parts(bits, 10) };
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let (bits, l) = b.into_raw_parts();
    let b = unsafe { BitVec::<&[usize]>::from_raw_parts(bits.as_ref(), l) };
    let b: AtomicBitVec<&[AtomicUsize]> = b.into();
    let (bits, l) = b.into_raw_parts();
    let b = unsafe { AtomicBitVec::<&[AtomicUsize]>::from_raw_parts(bits.as_ref(), l) };
    let b: BitVec<&[usize]> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
    }

    // Mutable reference to mutable reference
    let mut bits = vec![0; 10].into_boxed_slice();
    let mut b = unsafe { BitVec::<&mut [usize]>::from_raw_parts(bits.as_mut(), 10) };
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let b: AtomicBitVec<&mut [AtomicUsize]> = b.into();
    let b: BitVec<&mut [usize]> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
    }

    // Vec to boxed slice
    let mut b = BitVec::<Vec<usize>>::new(10);
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let b: BitVec<Box<[usize]>> = b.into();
    let b: BitVec<Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
    }
}

#[test]
fn test_iter_ones_zeros() {
    // Exit on bit found beyond bit length (diry vector)
    let v = unsafe { BitVec::from_raw_parts(vec![1 << 63], 10) };
    assert_eq!(v.iter_ones().next(), None);

    let v = unsafe { BitVec::from_raw_parts(vec![!(1 << 63)], 10) };
    assert_eq!(v.iter_zeros().next(), None);

    // Exit on last word
    let v = unsafe { BitVec::from_raw_parts(vec![0], 10) };
    assert_eq!(v.iter_ones().next(), None);

    let v = unsafe { BitVec::from_raw_parts(vec![!0], 10) };
    assert_eq!(v.iter_zeros().next(), None);
}

#[test]
fn test_macro() {
    // Empty bit vector
    let b = bit_vec![];
    assert_eq!(b.len(), 0);

    // 10 bits set to true
    let b = bit_vec![true; 10];
    assert_eq!(b.len(), 10);
    assert_eq!(b.iter().all(|x| x), true);
    let b = bit_vec![1; 10];
    assert_eq!(b.len(), 10);
    assert_eq!(b.iter().all(|x| x), true);

    // 10 bits set to false
    let b = bit_vec![false; 10];
    assert_eq!(b.len(), 10);
    assert_eq!(b.iter().any(|x| x), false);
    let b = bit_vec![0; 10];
    assert_eq!(b.len(), 10);
    assert_eq!(b.iter().any(|x| x), false);

    // Bit list
    let b = bit_vec![0, 1, 0, 1, 0, 0];
    assert_eq!(b.len(), 6);
    assert_eq!(b[0], false);
    assert_eq!(b[1], true);
    assert_eq!(b[2], false);
    assert_eq!(b[3], true);
    assert_eq!(b[4], false);
    assert_eq!(b[5], false);
}
