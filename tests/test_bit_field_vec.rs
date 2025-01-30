/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use common_traits::AsBytes;
use common_traits::Atomic;
use common_traits::AtomicUnsignedInt;
use common_traits::CastableFrom;
use common_traits::CastableInto;
use common_traits::IntoAtomic;
use core::sync::atomic::Ordering;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::AtomicUsize;
use sux::prelude::*;

#[test]
fn test() {
    test_param::<u8>();
    test_param::<u16>();
    test_param::<u32>();
    test_param::<u64>();
    test_param::<u128>();
    test_param::<usize>();
}

#[test]
fn test_atomic() {
    test_atomic_param::<u8>();
    test_atomic_param::<u16>();
    test_atomic_param::<u32>();
    test_atomic_param::<u64>();
    test_atomic_param::<usize>();
}

#[test]
fn test_bit_field_vec_apply() {
    test_bit_field_vec_apply_param::<u8>();
    test_bit_field_vec_apply_param::<u16>();
    test_bit_field_vec_apply_param::<u32>();
    test_bit_field_vec_apply_param::<u64>();
}

fn test_bit_field_vec_apply_param<W: Word + CastableInto<u64> + CastableFrom<u64>>() {
    for bit_width in 0..W::BITS {
        let n = 100;
        let u = W::ONE << bit_width.saturating_sub(1).min(60);
        let mut rng = SmallRng::seed_from_u64(0);

        let mut cp = BitFieldVec::<W>::new(bit_width, n);
        for _ in 0..10 {
            let values = (0..n)
                .map(|_| rng.random_range(0..u.cast()).cast())
                .collect::<Vec<W>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                cp.set(i, values[i]);
            }

            let new_values = (0..n)
                .map(|_| rng.random_range(0..u.cast()).cast())
                .collect::<Vec<W>>();

            // Test that apply_in_place happens in the right order
            let mut i = 0;
            cp.apply_in_place(|v| {
                assert_eq!(v, values[i]);
                let res = new_values[i];
                i += 1;
                res
            });

            for (i, c) in cp.iter().enumerate() {
                assert_eq!(c, new_values[i], "idx: {}", i);
            }
        }
    }
}

fn test_param<W: Word + CastableInto<u64> + CastableFrom<u64>>() {
    for bit_width in 0..W::BITS {
        let n = 100;
        let u = W::ONE << bit_width.saturating_sub(1).min(60);
        let mut rng = SmallRng::seed_from_u64(0);

        let mut v = BitFieldVec::<W>::new(bit_width, n);
        assert_eq!(v.bit_width(), bit_width);
        assert_eq!(
            v.mask(),
            if bit_width == 0 {
                W::ZERO
            } else {
                (W::ONE << bit_width) - W::ONE
            }
        );
        for _ in 0..10 {
            let values = (0..n)
                .map(|_| rng.random_range(0..u.cast()).cast())
                .collect::<Vec<W>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                v.set(i, values[i]);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(v.get(i), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(v.get(i), values[i]);
            }

            for from in 0..v.len() {
                let mut iter = v.into_unchecked_iter_from(from);
                for v in &values[from..] {
                    unsafe {
                        assert_eq!(iter.next_unchecked(), *v);
                    }
                }
            }

            for from in 0..v.len() {
                let mut iter = v.into_rev_unchecked_iter_from(from);
                for v in values[..from].iter().rev() {
                    unsafe {
                        assert_eq!(iter.next_unchecked(), *v);
                    }
                }
            }

            for from in 0..v.len() {
                for (i, v) in v.iter_from(from).enumerate() {
                    assert_eq!(v, values[i + from]);
                }
            }

            let (b, w, l) = v.clone().into_raw_parts();
            assert_eq!(unsafe { BitFieldVec::<W>::from_raw_parts(b, w, l) }, v);
        }
    }
}

fn test_atomic_param<W: Word + IntoAtomic + CastableInto<u64> + CastableFrom<u64>>()
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    use sux::traits::bit_field_slice::AtomicBitFieldSlice;

    for bit_width in 0..W::BITS {
        let n: usize = 100;
        let u: u64 = 1 << bit_width;
        let mut rng = SmallRng::seed_from_u64(0);

        let v = AtomicBitFieldVec::<W>::new(bit_width, n);
        assert_eq!(v.bit_width(), bit_width);
        assert_eq!(
            v.mask(),
            if bit_width == 0 {
                W::ZERO
            } else {
                (W::ONE << bit_width) - W::ONE
            }
        );
        for _ in 0..10 {
            let values: Vec<W> = (0..n)
                .map(|_| rng.random_range(0..u).cast())
                .collect::<Vec<_>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                v.set_atomic(i, values[i], Ordering::Relaxed);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(v.get_atomic(i, Ordering::Relaxed), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(v.get_atomic(i, Ordering::Relaxed), values[i]);
            }
        }

        let w: BitFieldVec<W> = v.into();
        let x = w.clone();
        let y: AtomicBitFieldVec<W> = x.into();
        let z: AtomicBitFieldVec<W> = w.into();

        let (b, w, l) = z.into_raw_parts();
        let z = unsafe { AtomicBitFieldVec::<W>::from_raw_parts(b, w, l) };
        for i in 0..n {
            assert_eq!(
                z.get_atomic(i, Ordering::Relaxed),
                y.get_atomic(i, Ordering::Relaxed),
            );
        }
    }
}

#[test]
fn test_clear() {
    let mut b = BitFieldVec::<usize, _>::new(50, 10);
    for i in 0..10 {
        b.set(i, i);
    }
    b.clear();
    assert_eq!(b.len(), 0);
}

#[test]
fn test_usize() {
    use sux::traits::bit_field_slice::BitFieldSlice;
    use sux::traits::bit_field_slice::BitFieldSliceMut;

    const BITS: usize = core::mem::size_of::<usize>() * 8;
    let mut c = BitFieldVec::<usize>::new(BITS, 4);
    c.set(0, -1_isize as usize);
    c.set(1, 1234567);
    c.set(2, 0);
    c.set(3, -1_isize as usize);
    assert_eq!(c.get(0), -1_isize as usize);
    assert_eq!(c.get(1), 1234567);
    assert_eq!(c.get(2), 0);
    assert_eq!(c.get(3), -1_isize as usize);
}

#[test]
fn test_bit_width_zero() {
    use sux::traits::bit_field_slice::BitFieldSlice;

    let c = BitFieldVec::<usize>::new(0, 1000);
    for i in 0..c.len() {
        assert_eq!(c.get(i), 0);
    }
}

#[test]
fn test_from_slice() {
    use sux::traits::bit_field_slice::BitFieldSlice;
    use sux::traits::bit_field_slice::BitFieldSliceMut;

    let mut c = BitFieldVec::new(12, 1000);
    for i in 0..c.len() {
        c.set(i, i)
    }

    let s = BitFieldVec::<usize>::from_slice(&c).unwrap();
    for i in 0..c.len() {
        assert_eq!({ s.get(i) }, c.get(i));
    }
    let s = BitFieldVec::<u16>::from_slice(&c).unwrap();
    for i in 0..c.len() {
        assert_eq!(s.get(i) as usize, c.get(i));
    }
    assert!(BitFieldVec::<u8>::from_slice(&c).is_err())
}

#[test]
fn test_push() {
    use sux::traits::bit_field_slice::BitFieldSlice;

    let mut c = BitFieldVec::new(12, 0);
    for i in 0..1000 {
        c.push(i);
    }
    for i in 0..1000 {
        assert_eq!(c.get(i), i);
    }
}

#[test]
fn test_resize() {
    use sux::traits::bit_field_slice::BitFieldSlice;

    let mut c = BitFieldVec::new(12, 0);
    c.resize(100, 2_usize);
    for i in 0..100 {
        assert_eq!(c.get(i), 2);
    }
    c.resize(50, 0);
    for i in 0..50 {
        assert_eq!(c.get(i), 2);
    }
    assert_eq!(c.len(), 50);
}

#[test]
fn test_pop() {
    use sux::traits::bit_field_slice::BitFieldSlice;

    let mut c = BitFieldVec::new(12, 0);
    for i in 0..1000 {
        c.push(i);
    }
    for i in (500..1000).rev() {
        assert_eq!(c.pop(), Some(i));
    }
    for i in 0..500 {
        assert_eq!(c.get(i), i);
    }
    for i in (0..500).rev() {
        assert_eq!(c.pop(), Some(i));
    }
    assert_eq!(c.pop(), None);
    assert_eq!(c.pop(), None);
}

#[test]
fn test_unaligned() {
    for bit_width in [50, 56, 57, 58, 60, 64] {
        let mut c = BitFieldVec::new(bit_width, 0);
        for i in 0_usize..10 {
            c.push(i);
        }
        for i in 0_usize..10 {
            assert_eq!(c.get_unaligned(i), i);
        }
    }
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_59() {
    let c = BitFieldVec::<usize, _>::new(59, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_61() {
    let c = BitFieldVec::<usize, _>::new(59, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_62() {
    let c = BitFieldVec::<usize, _>::new(59, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_63() {
    let c = BitFieldVec::<usize, _>::new(59, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[test]
fn test_get_addr() {
    let c = BitFieldVec::<usize, _>::new(3, 100);
    let begin_addr = c.addr_of(0) as usize;
    assert_eq!(c.addr_of(50) as usize - begin_addr, 16);

    let c = BitFieldVec::<u16, _>::new(3, 100);
    let begin_addr = c.addr_of(0) as usize;
    assert_eq!(c.addr_of(50) as usize - begin_addr, 18);
}

#[test]
fn test_eq() {
    let mut b = BitFieldVec::<usize>::new(3, 10);
    let c = BitFieldVec::<usize>::new(3, 9);
    assert_ne!(b, c);
    let mut c = BitFieldVec::<usize>::new(3, 10);
    assert_eq!(b, c);

    b.push(3);
    assert_ne!(b, c);
    c.push(3);
    assert_eq!(b, c);

    for i in 0..64 {
        b.push(i % 4);
        c.push(i % 4);
        assert_eq!(b, c);
    }

    let c: BitFieldVec<usize, Box<[usize]>> = c.into();
    assert_eq!(b, c);
    let (bits, w, l) = c.into_raw_parts();
    let d = unsafe { BitFieldVec::from_raw_parts(bits.as_ref(), w, l) };
    assert_eq!(b, d);
}

#[test]
fn test_reset() {
    let mut b = BitFieldVec::<usize, _>::new(50, 10);
    for i in 0..10 {
        b.set(i, i);
    }
    b.reset();
    for w in &b {
        assert_eq!(w, 0);
    }
}

#[test]
fn test_atomic_reset() {
    let mut b = AtomicBitFieldVec::<usize, _>::new(50, 10);
    for i in 0..10 {
        b.set_atomic(i, 1, Ordering::Relaxed);
    }
    b.reset_atomic(Ordering::Relaxed);
    for i in 0..10 {
        assert_eq!(b.get_atomic(i, Ordering::Relaxed), 0);
    }
}

#[test]
fn test_set_len() {
    let mut b = BitFieldVec::<usize, _>::new(50, 10);
    unsafe {
        b.set_len(5);
    }
    assert_eq!(b.len(), 5);
}

#[test]
fn test_from() {
    // Vec to atomic vec
    let mut b = BitFieldVec::<usize, Vec<usize>>::new(50, 10);
    for i in 0..10 {
        b.set(i, i);
    }
    let b: AtomicBitFieldVec<usize, Vec<AtomicUsize>> = b.into();
    let b: BitFieldVec<usize, Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i);
    }

    // Boxed slice to atomic boxed slice
    let bits = vec![0; 10].into_boxed_slice();
    let mut b = unsafe { BitFieldVec::<usize, Box<[usize]>>::from_raw_parts(bits, 50, 10) };
    for i in 0..10 {
        b.set(i, i);
    }
    let b: AtomicBitFieldVec<usize, Box<[AtomicUsize]>> = b.into();
    let b: BitFieldVec<usize, Box<[usize]>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i);
    }

    // Reference to atomic reference
    let bits = vec![0; 10].into_boxed_slice();
    let mut b = unsafe { BitFieldVec::<usize, Box<[usize]>>::from_raw_parts(bits, 50, 10) };
    for i in 0..10 {
        b.set(i, i);
    }
    let (bits, w, l) = b.into_raw_parts();
    let b = unsafe { BitFieldVec::<usize, &[usize]>::from_raw_parts(bits.as_ref(), w, l) };
    let b: AtomicBitFieldVec<usize, &[AtomicUsize]> = b.into();
    let b: BitFieldVec<usize, &[usize]> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i);
    }

    // Mutable reference to mutable reference
    let mut bits = vec![0; 10].into_boxed_slice();
    let mut b =
        unsafe { BitFieldVec::<usize, &mut [usize]>::from_raw_parts(bits.as_mut(), 50, 10) };
    for i in 0..10 {
        b.set(i, i);
    }
    let b: AtomicBitFieldVec<usize, &mut [AtomicUsize]> = b.into();
    let b: BitFieldVec<usize, &mut [usize]> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i);
    }

    // Vec to boxed slice
    let mut b = BitFieldVec::<usize, Vec<usize>>::new(50, 10);
    for i in 0..10 {
        b.set(i, i);
    }
    let b: BitFieldVec<usize, Box<[usize]>> = b.into();
    let b: BitFieldVec<usize, Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.get(i), i);
    }
}

#[test]
fn test_macro() {
    let b = bit_field_vec![5];
    assert_eq!(b.len(), 0);
    assert_eq!(b.bit_width(), 5);

    // 10 values of bit width 6, all set to 3
    let b = bit_field_vec![6; 10; 3];
    assert_eq!(b.len(), 10);
    assert_eq!(b.bit_width(), 6);
    assert!(b.iter().all(|x| x == 3));

    // List of values
    let b = bit_field_vec![10; 4, 500, 2, 0, 1];
    assert_eq!(b.len(), 5);
    assert_eq!(b.bit_width(), 10);
    assert_eq!(b.get(0), 4);
    assert_eq!(b.get(1), 500);
    assert_eq!(b.get(2), 2);
    assert_eq!(b.get(3), 0);
    assert_eq!(b.get(4), 1);
}

#[test]
fn test_slice() {
    let mut b = BitFieldVec::<u64>::new(6, 50);

    assert_eq!(b.as_slice(), vec![0; 5]);

    b.set(2, 1);

    assert_eq!(b.as_slice(), vec![4096, 0, 0, 0, 0]);

    let mut_slice = b.as_mut_slice();
    mut_slice[2] = 1;

    assert_eq!(b.as_slice(), vec![4096, 0, 1, 0, 0]);
    assert_eq!(b.get(21), 4);
}

fn atomic_slice_eq<T: Atomic>(actual: &[T], expected: &[T])
where
    <T as Atomic>::NonAtomicType: PartialEq,
    <T as Atomic>::NonAtomicType: std::fmt::Debug,
{
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(
            actual.load(Ordering::Relaxed),
            expected.load(Ordering::Relaxed)
        );
    }
}

#[test]
fn test_slice_atomic() {
    let b = AtomicBitFieldVec::<u64>::new(6, 50);
    let mut v = Vec::new();
    for _ in 0..5 {
        v.push(std::sync::atomic::AtomicU64::new(0));
    }

    atomic_slice_eq(b.as_slice(), v.as_slice());

    b.set_atomic(2, 1, Ordering::Relaxed);
    v[0].store(4096, Ordering::Relaxed);

    atomic_slice_eq(b.as_slice(), v.as_slice());

    b.as_slice()[2].store(1, Ordering::Relaxed);
    v[2].store(1, Ordering::Relaxed);

    atomic_slice_eq(b.as_slice(), v.as_slice());
    assert_eq!(b.get_atomic(21, Ordering::Relaxed), 4);
}
