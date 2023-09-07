use core::sync::atomic::{AtomicU64, Ordering};
use epserde::*;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use sux::prelude::*;

#[test]
fn test_bitmap() {
    let n = 50;
    let n2 = 100;
    let u = 1000;

    let mut rng = SmallRng::seed_from_u64(0);

    let mut bm = BitMap::new(u);

    for _ in 0..10 {
        let mut values = (0..u).collect::<Vec<_>>();
        let (indices, _) = values.partial_shuffle(&mut rng, n2);

        for i in indices[..n].iter().copied() {
            bm.set(i, 1);
        }

        for i in 0..u {
            assert_eq!(bm.get(i) != 0, indices[..n].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i, 1);
        }

        for i in 0..u {
            assert_eq!(bm.get(i) != 0, indices.contains(&i));
        }

        for i in indices[..n].iter().copied() {
            bm.set(i, 0);
        }

        for i in 0..u {
            assert_eq!(bm.get(i) != 0, indices[n..].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i, 0);
        }

        for i in 0..u {
            assert_eq!(bm.get(i), 0);
        }
    }

    let bm: BitMap<Vec<AtomicU64>> = bm.into();
    for _ in 0..10 {
        let mut values = (0..u).collect::<Vec<_>>();
        let (indices, _) = values.partial_shuffle(&mut rng, n2);

        for i in indices[..n].iter().copied() {
            bm.set_atomic(i, 1, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(
                bm.get_atomic(i, Ordering::Relaxed) != 0,
                indices[..n].contains(&i)
            );
        }

        for i in indices[n..].iter().copied() {
            bm.set_atomic(i, 1, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(
                bm.get_atomic(i, Ordering::Relaxed) != 0,
                indices.contains(&i)
            );
        }

        for i in indices[..n].iter().copied() {
            bm.set_atomic(i, 0, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(
                bm.get_atomic(i, Ordering::Relaxed) != 0,
                indices[n..].contains(&i)
            );
        }

        for i in indices[n..].iter().copied() {
            bm.set_atomic(i, 0, Ordering::Relaxed);
        }

        for i in 0..u {
            assert_eq!(bm.get_atomic(i, Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn test_epsserde() {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut b = BitMap::new(200);
    for i in 0..200 {
        b.set(i, rng.next_u64() % 2);
    }

    let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file).unwrap());
    b.serialize(&mut file).unwrap();
    drop(file);

    let c = epserde::map::<BitMap<Vec<u64>>>(&tmp_file, epserde::Flags::empty()).unwrap();

    for i in 0..200 {
        assert_eq!(b.get(i), c.get(i));
    }

    let mut b = CountingBitmap::new(200);
    for i in 0..200 {
        b.set(i, rng.next_u64() % 2);
    }

    let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file).unwrap());
    b.serialize(&mut file).unwrap();
    drop(file);

    let c = epserde::map::<CountingBitmap<Vec<u64>, usize>>(&tmp_file, epserde::Flags::empty())
        .unwrap();

    for i in 0..200 {
        assert_eq!(b.get(i), c.get(i));
    }
}
