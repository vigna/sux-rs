/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use atomic_primitive::Atomic;
use core::sync::atomic::Ordering;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use sux::prelude::*;

use sux::traits::BitVecValueOps;
use sux::traits::bit_vec_ops::*;

#[test]
fn test() {
    let n = 50;
    let n2 = 100;
    let u = 1000;

    let mut rng = SmallRng::seed_from_u64(0);

    let bm: BitVec = BitVec::with_value(u, true);

    assert_eq!(bm.len(), u);
    assert_eq!(bm.count_ones(), u);

    // Dirty vector
    let ones = [usize::MAX; 2];
    assert_eq!(
        unsafe { BitVec::from_raw_parts(ones.as_slice(), 0) }.count_ones(),
        0
    );
    assert_eq!(
        unsafe { BitVec::from_raw_parts(ones.as_slice(), 1) }.count_ones(),
        1
    );

    for i in 0..u {
        assert!(bm[i]);
    }

    let mut bm: BitVec = BitVec::new(u);

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
    let ones = [
        Atomic::<usize>::new(usize::MAX),
        Atomic::<usize>::new(usize::MAX),
    ];
    assert_eq!(
        unsafe { AtomicBitVec::from_raw_parts(&ones[..], 0) }.count_ones(),
        0
    );
    assert_eq!(
        unsafe { AtomicBitVec::from_raw_parts(&ones[..], 1) }.count_ones(),
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

    let bm: AtomicBitVec = AtomicBitVec::with_value(u, true);

    assert_eq!(bm.len(), u);
    assert_eq!(bm.count_ones(), u);

    for i in 0..u {
        assert!(bm.get(i, Ordering::Relaxed));
    }
}

#[test]
fn test_atomic_swap() {
    let b: AtomicBitVec = AtomicBitVec::new(10);
    assert!(!b.get(1, Ordering::Relaxed));
    assert!(!b.swap(1, true, Ordering::Relaxed));
    assert!(b.get(1, Ordering::Relaxed));
    assert!(b.swap(1, true, Ordering::Relaxed));
    assert!(b.swap(1, false, Ordering::Relaxed));
    assert!(!b.get(1, Ordering::Relaxed));
}

#[test]
fn test_push_pop() {
    let mut b: BitVec = BitVec::new(0);
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
    let mut c: BitVec = BitVec::new(0);
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
        let mut c: BitVec = BitVec::new(len);
        c.fill(true);
        for (i, b) in c.into_iter().enumerate() {
            assert!(b, "{}", i);
        }
        c.fill(false);
        for (i, b) in c.into_iter().enumerate() {
            assert!(!b, "{}", i);
        }

        #[cfg(feature = "rayon")]
        {
            c.par_fill(true);
            for (i, b) in c.into_iter().enumerate() {
                assert!(b, "{}", i);
            }
            c.par_fill(false);
            for (i, b) in c.into_iter().enumerate() {
                assert!(!b, "{}", i);
            }
        }

        if len != c.capacity() {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize] & (1 << (len % usize::BITS as usize)),
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
        let mut c: AtomicBitVec = AtomicBitVec::new(len);
        c.fill(true, Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(c.get(i, Ordering::Relaxed), "{}", i);
        }
        c.fill(false, Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(!c.get(i, Ordering::Relaxed), "{}", i);
        }

        #[cfg(feature = "rayon")]
        {
            c.par_fill(true, Ordering::Relaxed);
            for i in 0..c.len() {
                assert!(c.get(i, Ordering::Relaxed), "{}", i);
            }
            c.par_fill(false, Ordering::Relaxed);
            for i in 0..c.len() {
                assert!(!c.get(i, Ordering::Relaxed), "{}", i);
            }
        }

        if len % usize::BITS as usize != 0 {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize].load(Ordering::Relaxed)
                    & (1 << (len % usize::BITS as usize)),
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
        let mut c: BitVec = BitVec::new(len);
        c.flip();
        for (i, b) in c.into_iter().enumerate() {
            assert!(b, "{}", i);
        }

        if len != c.capacity() {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize] & (1 << (len % usize::BITS as usize)),
                0
            );
        }

        c.flip();
        for (i, b) in c.into_iter().enumerate() {
            assert!(!b, "{}", i);
        }

        #[cfg(feature = "rayon")]
        {
            c.par_flip();
            for (i, b) in c.into_iter().enumerate() {
                assert!(b, "{}", i);
            }

            c.par_flip();
            for (i, b) in c.into_iter().enumerate() {
                assert!(!b, "{}", i);
            }
        }
    }
}

#[test]
fn test_atomic_flip() {
    for len in [0, 1, 64, 65, 100, 127, 128, 1000] {
        let mut c: AtomicBitVec = AtomicBitVec::new(len);
        c.flip(Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(c.get(i, Ordering::Relaxed), "{}", i);
        }

        if len % usize::BITS as usize != 0 {
            assert_eq!(
                c.as_ref()[len / usize::BITS as usize].load(Ordering::Relaxed)
                    & (1 << (len % usize::BITS as usize)),
                0
            );
        }

        c.flip(Ordering::Relaxed);
        for i in 0..c.len() {
            assert!(!c.get(i, Ordering::Relaxed), "{}", i);
        }

        #[cfg(feature = "rayon")]
        {
            c.par_flip(Ordering::Relaxed);
            for i in 0..c.len() {
                assert!(c.get(i, Ordering::Relaxed), "{}", i);
            }

            c.par_flip(Ordering::Relaxed);
            for i in 0..c.len() {
                assert!(!c.get(i, Ordering::Relaxed), "{}", i);
            }
        }
    }
}

#[test]
fn test_iter() {
    let mut c: BitVec = BitVec::new(100);
    for i in 0..100 {
        c.set(i, i % 2 == 0);
    }

    for (i, b) in c.iter().enumerate() {
        assert_eq!(b, i % 2 == 0);
    }
}

#[test]
fn test_iter_ones_alternate() {
    let mut c: BitVec = BitVec::new(200);
    for i in 0..200 {
        c.set(i, i % 2 == 0);
    }

    for (i, p) in c.iter_ones().enumerate() {
        assert_eq!(p, i * 2);
    }
}

#[test]
fn test_iter_ones_empty() {
    let c: BitVec = BitVec::new(200);
    assert_eq!(c.iter_ones().next(), None);
}

#[test]
fn test_iter_ones_one() {
    let mut c: BitVec = BitVec::new(200);
    c.set(1, true);
    let mut i = c.iter_ones();
    assert_eq!(i.next(), Some(1));
    assert_eq!(i.next(), None);
}

#[test]
fn test_iter_zeros_alternate() {
    let mut c: BitVec = BitVec::new(200);
    for i in 0..200 {
        c.set(i, i % 2 != 0);
    }

    for (i, p) in c.iter_zeros().enumerate() {
        assert_eq!(p, i * 2);
    }
}

#[test]
fn test_iter_zeros_full() {
    let mut c: BitVec = BitVec::new(200);
    c.flip();
    assert_eq!(c.iter_zeros().next(), None);
}

#[test]
fn test_iter_zeros_one() {
    let mut c: BitVec = BitVec::new(200);
    c.set(1, true);
    c.flip();
    let mut i = c.iter_zeros();
    assert_eq!(i.next(), Some(1));
    assert_eq!(i.next(), None);
}

#[test]
fn test_atomic_iter() {
    let c: AtomicBitVec = AtomicBitVec::new(100);
    for i in 0..100 {
        c.set(i, i % 2 == 0, Ordering::Relaxed);
    }

    for (i, b) in c.iter().enumerate() {
        assert_eq!(b, i % 2 == 0);
    }
}

#[test]
fn test_eq() {
    let mut b: BitVec = BitVec::new(0);
    let mut c: BitVec = BitVec::new(0);
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

#[cfg(feature = "epserde")]
#[test]
fn test_epserde() -> Result<()> {
    use epserde::prelude::Aligned16;
    use epserde::utils::AlignedCursor;
    use rand::{Rng, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(0);
    let mut b: BitVec = BitVec::new(200);
    for i in 0..200 {
        b.set(i, rng.next_u64() % 2 != 0);
    }

    let mut cursor = <AlignedCursor<Aligned16>>::new();
    unsafe {
        use epserde::ser::Serialize;
        b.serialize(&mut cursor)?
    };

    let len = cursor.len();
    cursor.set_position(0);
    let c = unsafe {
        use epserde::deser::Deserialize;
        <BitVec<Vec<usize>>>::read_mem(&mut cursor, len)?
    };

    for i in 0..200 {
        assert_eq!(b.get(i), c.uncase().get(i));
    }
    Ok(())
}

#[test]
fn test_from() {
    // Vec to atomic vec
    let mut b = BitVec::<Vec<usize>>::new(10);
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let b: AtomicBitVec<Box<[Atomic<usize>]>> = b.into();
    let b: BitVec<Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
        assert_eq!(b[i], i % 2 == 0);
    }

    // Boxed slice to atomic boxed slice
    let bits = vec![0_usize; 10].into_boxed_slice();
    let mut b = unsafe { BitVec::<Box<[usize]>>::from_raw_parts(bits, 10) };
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    let b: AtomicBitVec<Box<[Atomic<usize>]>> = b.into();
    let b: BitVec<Box<[usize]>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i % 2 == 0);
    }

    // Reference to atomic reference
    let bits = vec![0_usize; 10].into_boxed_slice();
    let mut b = unsafe { BitVec::<Box<[usize]>>::from_raw_parts(bits, 10) };
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }

    // Mutable reference to mutable reference
    let mut bits = vec![0_usize; 10].into_boxed_slice();
    let mut b = unsafe { BitVec::<&mut [usize]>::from_raw_parts(bits.as_mut(), 10) };
    for i in 0..10 {
        b.set(i, i % 2 == 0);
    }
    if let Result::<AtomicBitVec<&mut [Atomic<usize>]>, _>::Ok(b) = b.try_into() {
        let b: BitVec<&mut [usize]> = b.into();
        for i in 0..10 {
            assert_eq!(b.get(i), i % 2 == 0);
        }
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
    // Exit on bit found beyond bit length (dirty vector)
    let v = unsafe { BitVec::from_raw_parts(vec![1_u64 << 63], 10) };
    assert_eq!(v.iter_ones().next(), None);

    let v = unsafe { BitVec::from_raw_parts(vec![!(1_u64 << 63)], 10) };
    assert_eq!(v.iter_zeros().next(), None);

    // Exit on last word
    let v = unsafe { BitVec::from_raw_parts(vec![0_u64], 10) };
    assert_eq!(v.iter_ones().next(), None);

    let v = unsafe { BitVec::from_raw_parts(vec![!0_u64], 10) };
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
    assert!(b.iter().all(|x| x));
    let b = bit_vec![1; 10];
    assert_eq!(b.len(), 10);
    assert!(b.iter().all(|x| x));

    // 10 bits set to false
    let b = bit_vec![false; 10];
    assert_eq!(b.len(), 10);
    assert!(!b.iter().any(|x| x));
    let b = bit_vec![0; 10];
    assert_eq!(b.len(), 10);
    assert!(!b.iter().any(|x| x));

    // Bit list
    let b = bit_vec![0, 1, 0, 1, 0, 0];
    assert_eq!(b.len(), 6);
    assert!(!b[0]);
    assert!(b[1]);
    assert!(!b[2]);
    assert!(b[3]);
    assert!(!b[4]);
    assert!(!b[5]);
}

/// Test BitVec with all word types using a macro.
macro_rules! test_word_type {
    ($W:ty) => {{
        use rand::RngExt;
        let mut rng = SmallRng::seed_from_u64(0);

        let u = 1000;
        let n = 50;
        let n2 = 100;

        // Test with_value + count_ones
        let bm: BitVec<Vec<$W>> = BitVec::with_value(u, true);
        assert_eq!(bm.len(), u);
        assert_eq!(BitCount::count_ones(&bm), u);
        for i in 0..u {
            assert!(BitVecOps::<$W>::get(&bm, i));
        }

        // Test new (all zeros) + set/get
        let mut bm: BitVec<Vec<$W>> = BitVec::new(u);
        assert_eq!(BitCount::count_ones(&bm), 0);

        for _ in 0..10 {
            let mut values = (0..u).collect::<Vec<_>>();
            let (indices, _) = values.partial_shuffle(&mut rng, n2);

            for &i in &indices[..n] {
                BitVecOpsMut::<$W>::set(&mut bm, i, true);
            }
            for i in 0..u {
                assert_eq!(
                    BitVecOps::<$W>::get(&bm, i),
                    indices[..n].contains(&i)
                );
            }

            for &i in &indices[n..] {
                BitVecOpsMut::<$W>::set(&mut bm, i, true);
            }
            for i in 0..u {
                assert_eq!(
                    BitVecOps::<$W>::get(&bm, i),
                    indices.contains(&i)
                );
            }

            for &i in &indices[..n] {
                BitVecOpsMut::<$W>::set(&mut bm, i, false);
            }
            for i in 0..u {
                assert_eq!(
                    BitVecOps::<$W>::get(&bm, i),
                    indices[n..].contains(&i)
                );
            }

            for &i in &indices[n..] {
                BitVecOpsMut::<$W>::set(&mut bm, i, false);
            }
            for i in 0..u {
                assert!(!BitVecOps::<$W>::get(&bm, i));
            }
        }

        // Test push/pop
        let mut b: BitVec<Vec<$W>> = BitVec::new(0);
        b.push(true);
        b.push(false);
        assert!(BitVecOps::<$W>::get(&b, 0));
        assert!(!BitVecOps::<$W>::get(&b, 1));
        for i in 2..200 {
            b.push(i % 2 == 0);
        }
        for i in 2..200 {
            assert_eq!(BitVecOps::<$W>::get(&b, i), i % 2 == 0);
        }
        for i in 0..200 {
            assert_eq!(b.pop(), Some(i % 2 != 0));
        }
        assert_eq!(b.pop(), None);

        // Test resize
        let mut c: BitVec<Vec<$W>> = BitVec::new(0);
        c.resize(100, true);
        for i in 0..100 {
            assert!(BitVecOps::<$W>::get(&c, i));
        }
        c.resize(50, false);
        for i in 0..50 {
            assert!(BitVecOps::<$W>::get(&c, i));
        }
        assert_eq!(c.len(), 50);

        // Test fill
        for len in [0, 1, 33, 64, 65, 100, 127, 128, 1000] {
            let mut c: BitVec<Vec<$W>> = BitVec::new(len);
            BitVecOpsMut::<$W>::fill(&mut c, true);
            for b in BitVecOps::<$W>::iter(&c) {
                assert!(b);
            }
            BitVecOpsMut::<$W>::fill(&mut c, false);
            for b in BitVecOps::<$W>::iter(&c) {
                assert!(!b);
            }
        }

        // Test flip
        for len in [0, 1, 33, 64, 65, 100, 127, 128, 1000] {
            let mut c: BitVec<Vec<$W>> = BitVec::new(len);
            BitVecOpsMut::<$W>::flip(&mut c);
            for b in BitVecOps::<$W>::iter(&c) {
                assert!(b);
            }
            BitVecOpsMut::<$W>::flip(&mut c);
            for b in BitVecOps::<$W>::iter(&c) {
                assert!(!b);
            }
        }

        // Test iter_ones / iter_zeros
        for len in [0, 1, 33, 100, 200] {
            let mut c: BitVec<Vec<$W>> = BitVec::new(len);
            for i in 0..len {
                BitVecOpsMut::<$W>::set(&mut c, i, i % 2 == 0);
            }
            for (j, p) in BitVecOps::<$W>::iter_ones(&c).enumerate() {
                assert_eq!(p, j * 2);
            }
            for (j, p) in BitVecOps::<$W>::iter_zeros(&c).enumerate() {
                assert_eq!(p, j * 2 + 1);
            }
        }

        // Test count_ones with random data
        for _ in 0..10 {
            let len = 100 + (rng.random_range(0..500));
            let bits = (0..len)
                .map(|_| rng.random_bool(0.5))
                .collect::<BitVec<Vec<$W>>>();
            let expected: usize = BitVecOps::<$W>::iter(&bits).filter(|&b| b).count();
            assert_eq!(BitCount::count_ones(&bits), expected);
        }

        // Test bit_vec! macro with word type
        let b = bit_vec![$W: 0, 1, 0, 1];
        assert_eq!(b.len(), 4);
        assert!(!BitVecOps::<$W>::get(&b, 0));
        assert!(BitVecOps::<$W>::get(&b, 1));

        let b = bit_vec![$W: false; 10];
        assert_eq!(b.len(), 10);
        assert_eq!(BitCount::count_ones(&b), 0);

        let b = bit_vec![$W: true; 10];
        assert_eq!(b.len(), 10);
        assert_eq!(BitCount::count_ones(&b), 10);

        let b = bit_vec![$W];
        assert_eq!(b.len(), 0);

        // Test Index<usize>
        let b = bit_vec![$W: 0, 1, 0, 1, 1, 0];
        assert_eq!(b[0], false);
        assert_eq!(b[1], true);
        assert_eq!(b[2], false);
        assert_eq!(b[3], true);
        assert_eq!(b[4], true);
        assert_eq!(b[5], false);

        // Test PartialEq (same backend type)
        let b1 = bit_vec![$W: 0, 1, 0, 1];
        let b2 = bit_vec![$W: 0, 1, 0, 1];
        let b3 = bit_vec![$W: 1, 0, 1, 0];
        assert_eq!(b1, b2);
        assert_ne!(b1, b3);

        // Test PartialEq (cross-backend: Vec vs Box)
        let b_box: BitVec<Box<[$W]>> = b1.clone().into();
        assert_eq!(b1, b_box);

        // Test IntoIterator
        let b = bit_vec![$W: 1, 0, 1, 1, 0];
        let collected: Vec<bool> = (&b).into_iter().collect();
        assert_eq!(collected, vec![true, false, true, true, false]);

        // Test Display
        let b = bit_vec![$W: 1, 0, 1];
        let s = format!("{}", b);
        assert_eq!(s, "[101]");
    }};
}

#[test]
fn test_word_u8() {
    test_word_type!(u8);
}

#[test]
fn test_word_u16() {
    test_word_type!(u16);
}

#[test]
fn test_word_u32() {
    test_word_type!(u32);
}

#[test]
fn test_word_u64() {
    test_word_type!(u64);
}

#[test]
fn test_word_u128() {
    test_word_type!(u128);
}

/// Test [`BitVec::append`] with all word types using a macro.
macro_rules! test_append_word_type {
    ($W:ty) => {{
        let bpw = <$W>::BITS as usize;

        // Test many combinations of self_len and other_len to exercise
        // aligned, unaligned, single-word, multi-word, and empty cases.
        let lengths = [
            0,
            1,
            bpw / 2 - 1,
            bpw / 2,
            bpw - 1,
            bpw,
            bpw + 1,
            2 * bpw - 1,
            2 * bpw,
            2 * bpw + 1,
            3 * bpw + 7,
        ];

        for &self_len in &lengths {
            for &other_len in &lengths {
                // Create self with a known pattern
                let mut a: BitVec<Vec<$W>> = BitVec::new(0);
                for i in 0..self_len {
                    a.push(i % 3 == 0);
                }

                // Create other with a different pattern
                let mut b: BitVec<Vec<$W>> = BitVec::new(0);
                for i in 0..other_len {
                    b.push(i % 5 == 0);
                }

                // Reference: push bit by bit
                let mut expected = a.clone();
                for i in 0..other_len {
                    expected.push(b[i]);
                }

                // Test append
                a.append(&b);

                assert_eq!(
                    a.len(),
                    expected.len(),
                    "length mismatch for self_len={self_len}, other_len={other_len}"
                );
                assert_eq!(
                    a, expected,
                    "content mismatch for self_len={self_len}, other_len={other_len}"
                );
            }
        }
    }};
}

#[test]
fn test_append_usize() {
    test_append_word_type!(usize);
}

#[test]
fn test_append_u8() {
    test_append_word_type!(u8);
}

#[test]
fn test_append_u16() {
    test_append_word_type!(u16);
}

#[test]
fn test_append_u32() {
    test_append_word_type!(u32);
}

#[test]
fn test_append_u64() {
    test_append_word_type!(u64);
}

#[test]
fn test_append_u128() {
    test_append_word_type!(u128);
}

#[test]
fn test_append_cross_backend() {
    // Append from a Box<[usize]>-backed BitVec to a Vec<usize>-backed one.
    let mut a: BitVec = BitVec::new(0);
    for i in 0..50 {
        a.push(i % 2 == 0);
    }
    let mut b: BitVec = BitVec::new(0);
    for i in 0..70 {
        b.push(i % 3 == 0);
    }
    let b_boxed: BitVec<Box<[usize]>> = b.clone().into();

    let mut expected = a.clone();
    for i in 0..70 {
        expected.push(b[i]);
    }

    a.append(&b_boxed);
    assert_eq!(a, expected);
}

#[test]
fn test_reserve() {
    let mut b: BitVec = BitVec::new(0);
    b.reserve(100);
    assert!(b.capacity() >= 100);
    assert_eq!(b.len(), 0);

    let mut b: BitVec = BitVec::new(50);
    b.reserve(100);
    assert!(b.capacity() >= 150);

    let mut b: BitVec = BitVec::new(0);
    b.reserve_exact(100);
    assert!(b.capacity() >= 100);
    assert_eq!(b.len(), 0);

    let mut b: BitVec = BitVec::new(50);
    b.reserve_exact(100);
    assert!(b.capacity() >= 150);
}

/// Test get_value / get_value_unchecked / append_value with a given word type.
macro_rules! test_value_ops_word_type {
    ($W:ty) => {{
        let bpw = <$W>::BITS as usize;

        // width == 0 returns 0
        {
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            bv.push(true);
            assert_eq!(bv.get_value(0, 0), (0 as $W));
        }

        // width == W::BITS, word-aligned (bit_index == 0)
        {
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            let val: $W = !(0 as $W) / 3; // alternating bits
            bv.append_value(val, bpw);
            assert_eq!(bv.len(), bpw);
            assert_eq!(bv.get_value(0, bpw), val);
            assert_eq!(unsafe { bv.get_value_unchecked(0, bpw) }, val);
        }

        // width == W::BITS, not word-aligned (bit_index > 0, spans two words)
        {
            for offset in 1..bpw {
                let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
                // Prepend `offset` zero bits
                for _ in 0..offset {
                    bv.push(false);
                }
                let val: $W = !(0 as $W) / 5; // known pattern
                bv.append_value(val, bpw);
                assert_eq!(
                    bv.get_value(offset, bpw),
                    val,
                    "width={bpw}, offset={offset}"
                );
                assert_eq!(
                    unsafe { bv.get_value_unchecked(offset, bpw) },
                    val,
                    "width={bpw}, offset={offset}"
                );
            }
        }

        // Single-word case: bit_index + width <= W::BITS
        {
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            // Write a known full word, then read sub-fields from it
            let val: $W = !(0 as $W) / 3;
            bv.append_value(val, bpw);

            for width in 1..bpw {
                for pos in 0..=(bpw - width) {
                    let mask = if width == bpw {
                        !(0 as $W)
                    } else {
                        ((1 as $W) << width) - (1 as $W)
                    };
                    let expected = (val >> pos) & mask;
                    assert_eq!(
                        bv.get_value(pos, width),
                        expected,
                        "single-word: pos={pos}, width={width}"
                    );
                }
            }
        }

        // Two-word (spanning) case: bit_index + width > W::BITS
        {
            let val0: $W = !(0 as $W) / 3;
            let val1: $W = !(0 as $W) / 5;
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            bv.append_value(val0, bpw);
            bv.append_value(val1, bpw);

            for width in 2..=bpw {
                // Only positions where the read spans a word boundary
                let lo = bpw - width + 1;
                let hi = bpw - 1;
                for pos in lo..=hi {
                    let mask = if width == bpw {
                        !(0 as $W)
                    } else {
                        ((1 as $W) << width) - (1 as $W)
                    };
                    let expected = ((val0 >> pos) | (val1 << (bpw - pos))) & mask;
                    assert_eq!(
                        bv.get_value(pos, width),
                        expected,
                        "spanning: pos={pos}, width={width}"
                    );
                }
            }
        }

        // append_value: width == 0 is a no-op
        {
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            bv.append_value(!(0 as $W), 0);
            assert_eq!(bv.len(), 0);
        }

        // append_value: round-trip for every width at various offsets
        {
            for width in 1..=bpw {
                for offset in 0..bpw {
                    let val: $W = if width == bpw {
                        !(0 as $W)
                    } else {
                        (((1 as $W) << width) - (1 as $W)) & (!(0 as $W) / 7)
                    };
                    let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
                    // Prepend `offset` bits to misalign
                    for _ in 0..offset {
                        bv.push(true);
                    }
                    bv.append_value(val, width);
                    assert_eq!(
                        bv.get_value(offset, width),
                        val,
                        "append round-trip: width={width}, offset={offset}"
                    );
                }
            }
        }

        // append_value masks excess bits in `value`
        {
            for width in 1..bpw {
                let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
                bv.append_value(!(0 as $W), width); // all bits set, but only `width` should survive
                let mask = ((1 as $W) << width) - (1 as $W);
                assert_eq!(bv.get_value(0, width), mask, "masking: width={width}");
            }
            // width == W::BITS: all bits survive
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            bv.append_value(!(0 as $W), bpw);
            assert_eq!(bv.get_value(0, bpw), !(0 as $W));
        }

        // Multiple appends at sequential positions
        {
            let mut bv: BitVec<Vec<$W>> = BitVec::new(0);
            let width = bpw / 2;
            let mask = ((1 as $W) << width) - (1 as $W);
            let n = 20;
            let vals: Vec<$W> = (0..n).map(|i| (i as $W).wrapping_mul(17) & mask).collect();
            for &v in &vals {
                bv.append_value(v, width);
            }
            for (i, &v) in vals.iter().enumerate() {
                assert_eq!(
                    bv.get_value(i * width, width),
                    v,
                    "sequential: i={i}, width={width}"
                );
            }
        }
    }};
}

#[test]
fn test_value_ops_u8() {
    test_value_ops_word_type!(u8);
}

#[test]
fn test_value_ops_u16() {
    test_value_ops_word_type!(u16);
}

#[test]
fn test_value_ops_u32() {
    test_value_ops_word_type!(u32);
}

#[test]
fn test_value_ops_u64() {
    test_value_ops_word_type!(u64);
}

#[test]
fn test_value_ops_usize() {
    test_value_ops_word_type!(usize);
}
