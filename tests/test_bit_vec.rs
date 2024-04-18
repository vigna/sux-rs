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
use sux::bits::bit_vec::BitVec;
use sux::prelude::AtomicBitVec;

#[test]
fn test_bit_vec() {
    let n = 50;
    let n2 = 100;
    let u = 1000;

    let mut rng = SmallRng::seed_from_u64(0);

    let bm = BitVec::with_value(u, true);

    assert_eq!(bm.len(), u);
    assert_eq!(bm.count_ones(), u);

    for i in 0..u {
        assert_eq!(bm[i], true);
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
    for _ in 0..10 {
        let mut values = (0..u).collect::<Vec<_>>();
        let (indices, _) = values.partial_shuffle(&mut rng, n2);

        for i in indices[..n].iter().copied() {
            bm.set(i, true, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(bm.get(i, Ordering::Relaxed), indices[..n].contains(&i));
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
        assert_eq!(bm.get(i, Ordering::Relaxed), true);
    }
}

#[test]
fn test_atomic_swap() {
    let b = AtomicBitVec::new(10);
    assert_eq!(b.get(1, Ordering::Relaxed), false);
    assert_eq!(b.swap(1, true, Ordering::Relaxed), false);
    assert_eq!(b.get(1, Ordering::Relaxed), true);
    assert_eq!(b.swap(1, true, Ordering::Relaxed), true);
    assert_eq!(b.swap(1, false, Ordering::Relaxed), true);
    assert_eq!(b.get(1, Ordering::Relaxed), false);
}

#[test]
fn test_push() {
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
