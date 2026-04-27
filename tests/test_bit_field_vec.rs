/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use atomic_primitive::PrimitiveAtomicUnsigned;
use atomic_primitive::{Atomic, AtomicPrimitive, PrimitiveAtomic};
use bit_field_slice::*;
use core::sync::atomic::Ordering;
use num_primitive::{PrimitiveNumber, PrimitiveNumberAs};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use sux::prelude::*;
use sux::traits::Word;

use value_traits::slices::{SliceByValue, SliceByValueMut};

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

fn test_bit_field_vec_apply_param<W: Word + PrimitiveNumberAs<u64>>()
where
    u64: PrimitiveNumberAs<W>,
{
    for bit_width in 0..W::BITS as usize {
        let n = 100;
        let u = W::ONE << (bit_width.saturating_sub(1).min(60) as u32);
        let mut rng = SmallRng::seed_from_u64(0);

        let mut cp = BitFieldVec::<Vec<W>>::new(bit_width, n);
        for _ in 0..10 {
            let values = (0..n)
                .map(|_| rng.random_range(0u64..u.as_to()).as_to())
                .collect::<Vec<W>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                cp.set_value(i, values[i]);
            }

            let new_values = (0..n)
                .map(|_| rng.random_range(0u64..u.as_to()).as_to())
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

fn test_param<W: Word + PrimitiveNumberAs<u64>>()
where
    u64: PrimitiveNumberAs<W>,
{
    for bit_width in 0..W::BITS as usize {
        let n = 100;
        let u = W::ONE << (bit_width.saturating_sub(1).min(60) as u32);
        let mut rng = SmallRng::seed_from_u64(0);

        let mut v = BitFieldVec::<Vec<W>>::new(bit_width, n);
        assert_eq!(v.bit_width(), bit_width);
        assert_eq!(
            v.mask(),
            if bit_width == 0 {
                W::ZERO
            } else {
                (W::ONE << bit_width as u32) - W::ONE
            }
        );
        for _ in 0..10 {
            let values = (0..n)
                .map(|_| rng.random_range(0u64..u.as_to()).as_to())
                .collect::<Vec<W>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                v.set_value(i, values[i]);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(v.index_value(i), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(v.index_value(i), values[i]);
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
                let mut iter = v.into_unchecked_iter_back_from(from);
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
            assert_eq!(unsafe { BitFieldVec::<Vec<W>>::from_raw_parts(b, w, l) }, v);
        }
    }
}

fn test_atomic_param<W: Word + AtomicPrimitive + PrimitiveNumberAs<u64>>()
where
    u64: PrimitiveNumberAs<W>,
    Atomic<W>: PrimitiveAtomicUnsigned,
{
    use sux::traits::bit_field_slice::{AtomicBitFieldSlice, AtomicBitWidth};

    for bit_width in 0..W::BITS as usize {
        let n: usize = 100;
        let u: u64 = 1 << bit_width;
        let mut rng = SmallRng::seed_from_u64(0);

        let v = AtomicBitFieldVec::<Vec<W::Atomic>>::new(bit_width, n);
        assert_eq!(v.atomic_bit_width(), bit_width);
        assert_eq!(
            v.mask(),
            if bit_width == 0 {
                W::ZERO
            } else {
                (W::ONE << bit_width as u32) - W::ONE
            }
        );
        for _ in 0..10 {
            let values: Vec<W> = (0..n)
                .map(|_| rng.random_range(0..u).as_to())
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

        let w: BitFieldVec<Vec<W>> = v.into();
        let x = w.clone();
        let y: AtomicBitFieldVec<Vec<W::Atomic>> = x.into();
        let z: AtomicBitFieldVec<Vec<W::Atomic>> = w.into();

        let (b, w, l) = z.into_raw_parts();
        let z = unsafe { AtomicBitFieldVec::<Vec<W::Atomic>>::from_raw_parts(b, w, l) };
        for i in 0..n {
            assert_eq!(
                z.get_atomic(i, Ordering::Relaxed),
                y.get_atomic(i, Ordering::Relaxed),
            );
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_clear() {
    let mut b = BitFieldVec::<Vec<usize>>::new(50, 10);
    for i in 0..10 {
        b.set_value(i, i);
    }
    b.clear();
    assert_eq!(b.len(), 0);
}

#[test]
fn test_usize() {
    const BITS: usize = core::mem::size_of::<usize>() * 8;
    let mut c = BitFieldVec::<Vec<usize>>::new(BITS, 4);
    c.set_value(0, -1_isize as usize);
    c.set_value(1, 1234567);
    c.set_value(2, 0);
    c.set_value(3, -1_isize as usize);
    assert_eq!(c.index_value(0), -1_isize as usize);
    assert_eq!(c.index_value(1), 1234567);
    assert_eq!(c.index_value(2), 0);
    assert_eq!(c.index_value(3), -1_isize as usize);
}

#[test]
fn test_bit_width_zero() {
    let c = BitFieldVec::<Vec<usize>>::new(0, 1000);
    for i in 0..c.len() {
        assert_eq!(c.index_value(i), 0);
    }
}

#[test]
fn test_from_slice() -> Result<()> {
    let mut c = BitFieldVec::<Vec<usize>>::new(12, 1000);
    for i in 0..c.len() {
        c.set_value(i, i)
    }

    let s = BitFieldVec::<Vec<usize>>::from_slice(&c)?;
    for i in 0..c.len() {
        assert_eq!({ s.index_value(i) }, c.index_value(i));
    }
    let s = BitFieldVec::<Vec<u16>>::from_slice(&c)?;
    for i in 0..c.len() {
        assert_eq!(s.index_value(i) as usize, c.index_value(i));
    }
    assert!(BitFieldVec::<Vec<u8>>::from_slice(&c).is_err());
    Ok(())
}

#[test]
fn test_push() {
    let mut c = BitFieldVec::<Vec<usize>>::new(12, 0);
    for i in 0..1000 {
        c.push(i);
    }
    for i in 0..1000 {
        assert_eq!(c.index_value(i), i);
    }
}

#[test]
fn test_resize() {
    let mut c = BitFieldVec::<Vec<usize>>::new(12, 0);
    c.resize(100, 2_usize);
    for i in 0..100 {
        assert_eq!(c.index_value(i), 2);
    }
    c.resize(50, 0);
    for i in 0..50 {
        assert_eq!(c.index_value(i), 2);
    }
    assert_eq!(c.len(), 50);
}

#[test]
fn test_pop() {
    let mut c = BitFieldVec::<Vec<usize>>::new(12, 0);
    for i in 0..1000 {
        c.push(i);
    }
    for i in (500..1000).rev() {
        assert_eq!(c.pop(), Some(i));
    }
    for i in 0..500 {
        assert_eq!(c.index_value(i), i);
    }
    for i in (0..500).rev() {
        assert_eq!(c.pop(), Some(i));
    }
    assert_eq!(c.pop(), None);
    assert_eq!(c.pop(), None);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_unaligned() {
    for bit_width in [50, 56, 57, 58, 60, 64] {
        let mut c = BitFieldVec::<Box<[usize]>>::new_padded(bit_width, 100);
        for i in 0..10 {
            c.set_value(i, i);
        }
        for i in 0_usize..10 {
            assert_eq!(c.get_unaligned(i), i);
        }
    }
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_unchecked_59() {
    let c = BitFieldVec::<Vec<usize>>::new(59, 1);
    assert_eq!(unsafe { c.get_unaligned_unchecked(0) }, 0);
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_unchecked_61() {
    let c = BitFieldVec::<Vec<usize>>::new(61, 1);
    assert_eq!(unsafe { c.get_unaligned_unchecked(0) }, 0);
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_unchecked_62() {
    let c = BitFieldVec::<Vec<usize>>::new(62, 1);
    assert_eq!(unsafe { c.get_unaligned_unchecked(0) }, 0);
}

#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_unchecked_63() {
    let c = BitFieldVec::<Vec<usize>>::new(63, 1);
    assert_eq!(unsafe { c.get_unaligned_unchecked(0) }, 0);
}

#[cfg(target_pointer_width = "64")]
#[cfg(debug_assertions)]
#[should_panic]
#[test]
fn test_unaligned_unchecked_no_padding() {
    let c = BitFieldVec::<Vec<usize>>::new(17, 2);
    assert_eq!(unsafe { c.get_unaligned_unchecked(1) }, 0);
}

#[should_panic]
#[test]
fn test_unaligned_59() {
    let c = BitFieldVec::<Vec<usize>>::new(59, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[should_panic]
#[test]
fn test_unaligned_61() {
    let c = BitFieldVec::<Vec<usize>>::new(61, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[should_panic]
#[test]
fn test_unaligned_62() {
    let c = BitFieldVec::<Vec<usize>>::new(62, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[should_panic]
#[test]
fn test_unaligned_63() {
    let c = BitFieldVec::<Vec<usize>>::new(63, 1);
    assert_eq!(c.get_unaligned(0), 0);
}

#[cfg(target_pointer_width = "64")]
#[should_panic]
#[test]
fn test_unaligned_no_padding() {
    let c = BitFieldVec::<Vec<usize>>::new(17, 2);
    assert_eq!(c.get_unaligned(1), 0);
}

#[test]
fn test_get_addr() {
    let c = BitFieldVec::<Vec<usize>>::new(3, 100);
    let begin_addr = c.addr_of(0) as usize;
    assert_eq!(c.addr_of(50) as usize - begin_addr, 16);

    let c = BitFieldVec::<Vec<u16>>::new(3, 100);
    let begin_addr = c.addr_of(0) as usize;
    assert_eq!(c.addr_of(50) as usize - begin_addr, 18);
}

#[test]
fn test_eq() {
    let mut b = BitFieldVec::<Vec<usize>>::new(3, 10);
    let c = BitFieldVec::<Vec<usize>>::new(3, 9);
    assert_ne!(b, c);
    let mut c = BitFieldVec::<Vec<usize>>::new(3, 10);
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

    let c: BitFieldVec<Box<[usize]>> = c.into();
    assert_eq!(b, c);
    let (bits, w, l) = c.into_raw_parts();
    let d = unsafe { BitFieldVec::from_raw_parts(bits.as_ref(), w, l) };
    assert_eq!(b, d);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_reset() {
    let mut b = BitFieldVec::<Vec<usize>>::new(50, 10);
    for i in 0..10 {
        b.set_value(i, i);
    }
    b.reset();
    for w in &b {
        assert_eq!(w, 0);
    }
    for i in 0..10 {
        b.set_value(i, i);
    }
    #[cfg(feature = "rayon")]
    {
        b.par_reset();
        for w in &b {
            assert_eq!(w, 0);
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_atomic_reset() {
    use std::sync::atomic::AtomicUsize;

    let mut b = AtomicBitFieldVec::<Vec<AtomicUsize>>::new(50, 10);
    for i in 0..10 {
        b.set_atomic(i, 1, Ordering::Relaxed);
    }
    b.reset_atomic(Ordering::Relaxed);
    for i in 0..10 {
        assert_eq!(b.get_atomic(i, Ordering::Relaxed), 0);
    }
    for i in 0..10 {
        b.set_atomic(i, 1, Ordering::Relaxed);
    }
    #[cfg(feature = "rayon")]
    {
        b.par_reset_atomic(Ordering::Relaxed);
        for i in 0..10 {
            assert_eq!(b.get_atomic(i, Ordering::Relaxed), 0);
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_set_len() {
    let mut b = BitFieldVec::<Vec<usize>>::new(50, 10);
    unsafe {
        b.set_len(5);
    }
    assert_eq!(b.len(), 5);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_from() {
    use std::sync::atomic::AtomicUsize;
    // Vec to atomic vec
    let mut b = BitFieldVec::<Vec<usize>>::new(50, 10);
    for i in 0..10 {
        b.set_value(i, i);
    }
    let b: AtomicBitFieldVec<Vec<AtomicUsize>> = b.into();
    let b: BitFieldVec<Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.index_value(i), i);
    }

    // Boxed slice to atomic boxed slice
    let mut b: BitFieldVec<Box<[usize]>> = BitFieldVec::<Vec<usize>>::new(50, 10).into();
    for i in 0..10 {
        b.set_value(i, i);
    }
    let b: AtomicBitFieldVec<Box<[AtomicUsize]>> = b.into();
    let b: BitFieldVec<Box<[usize]>> = b.into();
    for i in 0..10 {
        assert_eq!(b.index_value(i), i);
    }

    // Mutable reference to mutable reference
    let mut b: BitFieldVec<Box<[usize]>> = BitFieldVec::<Vec<usize>>::new(50, 10).into();
    for i in 0..10 {
        b.set_value(i, i);
    }
    let b_mut: BitFieldVec<&mut [usize]> = (&mut b).into();
    if let Result::<AtomicBitFieldVec<&mut [AtomicUsize]>, _>::Ok(b) = b_mut.try_into() {
        let b: BitFieldVec<&mut [usize]> = b.into();
        for i in 0..10 {
            assert_eq!(b.index_value(i), i);
        }
    }

    // Vec to boxed slice
    let mut b = BitFieldVec::<Vec<usize>>::new(50, 10);
    for i in 0..10 {
        b.set_value(i, i);
    }
    let b: BitFieldVec<Box<[usize]>> = b.into();
    let b: BitFieldVec<Vec<usize>> = b.into();
    for i in 0..10 {
        assert_eq!(b.index_value(i), i);
    }

    // Commuting refs
    let mut b: BitFieldVec<Box<[usize]>> = b.into();
    {
        let b: BitFieldVec<&[usize]> = (&b).into();
        for i in 0..10 {
            assert_eq!(b.index_value(i), i);
        }
    }

    {
        let b: BitFieldVec<&mut [usize]> = (&mut b).into();
        for i in 0..10 {
            assert_eq!(b.index_value(i), i);
        }
    }

    // Commuting refs
    let mut b = <AtomicBitFieldVec>::new(50, 10);
    for i in 0..10 {
        b.set_atomic(i, i, Ordering::Relaxed);
    }
    {
        let b: AtomicBitFieldVec<&[AtomicUsize]> = (&b).into();
        for i in 0..10 {
            assert_eq!(b.get_atomic(i, Ordering::Relaxed), i);
        }
    }

    {
        let b: AtomicBitFieldVec<&mut [AtomicUsize]> = (&mut b).into();
        for i in 0..10 {
            assert_eq!(b.get_atomic(i, Ordering::Relaxed), i);
        }
    }
}

#[test]
fn test_macro() {
    let b = bit_field_vec![5];
    assert_eq!(b.len(), 0);
    assert_eq!(b.bit_width(), 5);

    // 10 values of bit width 6, all set to 3
    let b = bit_field_vec![6 => 3; 10];
    assert_eq!(b.len(), 10);
    assert_eq!(b.bit_width(), 6);
    assert!(b.iter().all(|x| x == 3));

    // List of values
    let b = bit_field_vec![10; 4, 500, 2, 0, 1];
    assert_eq!(b.len(), 5);
    assert_eq!(b.bit_width(), 10);
    assert_eq!(b.index_value(0), 4);
    assert_eq!(b.index_value(1), 500);
    assert_eq!(b.index_value(2), 2);
    assert_eq!(b.index_value(3), 0);
    assert_eq!(b.index_value(4), 1);
}

#[test]
fn test_slice() {
    let mut b = BitFieldVec::<Vec<u64>>::new(6, 50);

    assert_eq!(b.as_slice(), vec![0; 5]);

    b.set_value(2, 1);

    assert_eq!(b.as_slice(), vec![4096, 0, 0, 0, 0]);

    let mut_slice = b.as_mut_slice();
    mut_slice[2] = 1;

    assert_eq!(b.as_slice(), vec![4096, 0, 1, 0, 0]);
    assert_eq!(b.index_value(21), 4);
}

fn atomic_slice_eq<T: PrimitiveAtomic>(actual: &[T], expected: &[T])
where
    T::Value: PartialEq + std::fmt::Debug,
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
    let b = AtomicBitFieldVec::<Vec<std::sync::atomic::AtomicU64>>::new(6, 50);
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

#[test]
fn test_try_chunks_mut() {
    let mut b = bit_field_vec![7 => 1; 10];
    assert!(b.try_chunks_mut(9).is_err());
    assert_eq!(
        b.try_chunks_mut(10).unwrap().next().unwrap(),
        bit_field_vec![7 => 1; 10]
    );
    assert_eq!(
        b.try_chunks_mut(11).unwrap().next().unwrap(),
        bit_field_vec![7 => 1; 10]
    );

    let mut b = bit_field_vec![16 => 1; 10];
    assert!(b.try_chunks_mut(3).is_err());
    let mut iter = b.try_chunks_mut(4).unwrap();
    assert_eq!(iter.next().unwrap(), bit_field_vec![16 => 1; 4]);
    assert_eq!(iter.next().unwrap(), bit_field_vec![16 => 1; 4]);
    assert_eq!(iter.next().unwrap(), bit_field_vec![16 => 1; 2]);
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_from() {
    let b = bit_field_vec![7; 0, 1, 2, 3, 4, 5];
    for i in 0..b.len() {
        let mut iter = b.iter_from(i);
        for j in i..b.len() {
            assert_eq!(iter.next(), Some(j));
        }
    }
}

#[test]
fn test_iter_double_ended() {
    let b = bit_field_vec![7; 0, 1, 2, 3, 4, 5];

    // next_back only
    let mut iter = b.into_iter();
    for j in (0..b.len()).rev() {
        assert_eq!(iter.next_back(), Some(j));
    }
    assert_eq!(iter.next_back(), None);

    // next and next_back interleaved
    let mut iter = b.into_iter();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(5));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next_back(), Some(4));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next_back(), Some(3));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);

    // single-element vector
    let b = bit_field_vec![7; 42];
    let mut iter = b.into_iter();
    assert_eq!(iter.next_back(), Some(42));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);

    // empty vector
    let b = BitFieldVec::<Vec<usize>>::new(7, 0);
    let mut iter = b.into_iter();
    assert_eq!(iter.next_back(), None);

    // rev()
    let b = bit_field_vec![7; 0, 1, 2, 3, 4, 5];
    let rev: Vec<_> = b.into_iter().rev().collect();
    assert_eq!(rev, vec![5, 4, 3, 2, 1, 0]);

    // various bit widths
    for bit_width in 1..=usize::BITS as usize {
        let n = 10usize;
        let mask = if bit_width == usize::BITS as usize {
            usize::MAX
        } else {
            (1usize << bit_width) - 1
        };
        let mut bfv = BitFieldVec::<Vec<usize>>::new(bit_width, n);
        for i in 0..n {
            bfv.set_value(i, i & mask);
        }
        let rev: Vec<_> = bfv.into_iter().rev().collect();
        let expected: Vec<_> = (0..n).rev().map(|i| i & mask).collect();
        assert_eq!(rev, expected, "Failed for bit_width={bit_width}");
    }
}

#[cfg(feature = "epserde")]
#[test]
fn test_epserde() {
    use epserde::prelude::Aligned64;
    use epserde::utils::AlignedCursor;

    macro_rules! test_epserde_word {
        ($W:ty) => {{
            let n = 100;
            for bit_width in 0..=<$W>::BITS as usize {
                let mut bfv = BitFieldVec::<Vec<$W>>::new(bit_width, n);
                let mask: $W = if bit_width == 0 {
                    0
                } else {
                    <$W>::MAX >> (<$W>::BITS as usize - bit_width)
                };
                for i in 0..n {
                    bfv.set_value(i, (i as $W) & mask);
                }

                let mut cursor = <AlignedCursor<Aligned64>>::new();
                unsafe {
                    use epserde::ser::Serialize;
                    bfv.serialize(&mut cursor).expect("Could not serialize");
                }

                let len = cursor.len();
                cursor.set_position(0);
                let bfv2 = unsafe {
                    use epserde::deser::Deserialize;
                    <BitFieldVec<Vec<$W>>>::read_mem(&mut cursor, len)
                        .expect("Could not deserialize")
                };
                let bfv2 = bfv2.uncase();

                assert_eq!(bfv.len(), bfv2.len());
                assert_eq!(bfv.bit_width(), bfv2.bit_width());
                for i in 0..n {
                    assert_eq!(
                        bfv.get_value(i),
                        bfv2.get_value(i),
                        "Mismatch at index {i}, bit_width={bit_width}, word={}",
                        stringify!($W)
                    );
                }
            }
        }};
    }

    test_epserde_word!(u8);
    test_epserde_word!(u16);
    test_epserde_word!(u32);
    test_epserde_word!(u64);

    // Also test Box<[W]> backend (the deserialization target type)
    let mut bfv = BitFieldVec::<Vec<u32>>::new(12, 50);
    for i in 0..50 {
        bfv.set_value(i, (i as u32) & 0xFFF);
    }
    let bfv: BitFieldVec<Box<[u32]>> = bfv.into();

    let mut cursor = <AlignedCursor<Aligned64>>::new();
    unsafe {
        use epserde::ser::Serialize;
        bfv.serialize(&mut cursor).expect("Could not serialize");
    }

    let len = cursor.len();
    cursor.set_position(0);
    let bfv2 = unsafe {
        use epserde::deser::Deserialize;
        <BitFieldVec<Box<[u32]>>>::read_mem(&mut cursor, len).expect("Could not deserialize")
    };
    let bfv2 = bfv2.uncase();

    assert_eq!(bfv.len(), bfv2.len());
    for i in 0..50 {
        assert_eq!(bfv.get_value(i), bfv2.get_value(i));
    }
}
