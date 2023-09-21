/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use core::sync::atomic::Ordering;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::AtomicUsize;
use sux::prelude::CompactArray;
use sux::prelude::*;

#[test]
fn test_compact_array() {
    for bit_width in 0..64 {
        let n = 100;
        let u = 1 << bit_width;
        let mut rng = SmallRng::seed_from_u64(0);

        let mut cp = CompactArray::new(bit_width, n);
        for _ in 0..10 {
            let values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                cp.set(i, values[i]);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(cp.get(i), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(cp.get(i), values[i]);
            }

            for from in 0..cp.len() {
                let mut iter = cp.iter_val_from_unchecked(from);
                for v in &values[from..] {
                    unsafe {
                        assert_eq!(iter.next_unchecked(), *v);
                    }
                }
            }

            for from in 0..cp.len() {
                for (i, v) in cp.iter_val_from(from).enumerate() {
                    assert_eq!(v, values[i + from]);
                }
            }
        }
        // convert to atomic
        let cp: CompactArray<Vec<AtomicUsize>> = cp.into();
        for _ in 0..10 {
            let values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                cp.set(i, values[i], Ordering::Relaxed);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(cp.get(i, Ordering::Relaxed), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(cp.get(i, Ordering::Relaxed), values[i]);
            }
        }
    }
}

#[test]
fn test_compact_array_usize() {
    const BITS: usize = core::mem::size_of::<usize>() * 8;
    let mut c = CompactArray::new(BITS, 4);
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
fn test_width_zero() {
    let c = CompactArray::new(0, 1000);
    for i in 0..c.len() {
        assert_eq!(c.get(i), 0);
    }
}
