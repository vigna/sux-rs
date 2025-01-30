/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]
use std::iter::zip;

use anyhow::Result;
use epserde::prelude::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::prelude::*;

#[test]
#[cfg(feature = "rayon")]
fn test_elias_fano_concurrent() -> Result<()> {
    use rayon::prelude::*;
    use sux::dict::elias_fano::EliasFanoConcurrentBuilder;
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let efb = EliasFanoConcurrentBuilder::new(n, u);
        // push concurrently the values
        values
            .par_iter()
            .enumerate()
            .for_each(|(index, value)| unsafe { efb.set(index, *value) });
        // Finish the creation of elias-fano
        let _ef = efb.build();
    }
    Ok(())
}

#[test]
fn test_elias_fano() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(n, u);
        // push the values
        for value in values.iter() {
            efb.push(*value);
        }
        // Finish the creation of elias-fano
        let ef = efb.build();

        // Add an index on ones (will make it into an IndexedSeq)
        let ef = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _>::new) };
        // Add an index on zeros (will make it into an IndexedDict + Succ + Pred)
        let ef = unsafe { ef.map_high_bits(SelectZeroAdaptConst::<_, _>::new) };

        for v in 0..u {
            let res = values.binary_search(&v);
            let contains = res.is_ok();
            assert_eq!(ef.contains(v), contains);

            if contains {
                let i = res.unwrap();
                assert_eq!(ef.get(i), v);
                assert_eq!(ef.get(ef.index_of(v).unwrap()), v);
            } else {
                assert_eq!(ef.index_of(v), None);
            }
        }

        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!({ ef.get(i) }, *v);
        }

        let mut iterator = ef.iter().enumerate();
        while let Some((i, v)) = iterator.next() {
            assert_eq!(v, values[i]);
            assert_eq!(iterator.len(), ef.len() - i - 1);
        }

        for from in 0..ef.len() {
            let mut iterator = ef.iter_from(from).enumerate();
            while let Some((i, v)) = iterator.next() {
                assert_eq!(v, values[i + from]);
                assert_eq!(iterator.len(), ef.len() - i - from - 1);
            }
        }

        let last = values.last().unwrap();
        let mut lower_bound = 0;
        for (i, v) in values.iter().enumerate() {
            if lower_bound > *v {
                continue;
            }
            loop {
                assert!(ef.succ(lower_bound).unwrap() == (i, *v));
                lower_bound += 1;
                if lower_bound > values[i] {
                    break;
                }
            }
        }

        assert_eq!(None, ef.succ(last + 1));

        let mut lower_bound = 0;
        for (i, v) in values.iter().enumerate() {
            if lower_bound >= *v {
                continue;
            }
            loop {
                assert!(ef.succ_strict(lower_bound).unwrap() == (i, *v));
                lower_bound += 1;
                if lower_bound >= values[i] {
                    break;
                }
            }
        }

        assert_eq!(None, ef.succ_strict(last));

        let first = *values.first().unwrap();
        let mut upper_bound = first + 1;
        for (i, v) in values.iter().enumerate() {
            if upper_bound <= *v {
                continue;
            }
            // Skip repeated items as we return the index of the last one
            if i + 1 < values.len() && values[i] == values[i + 1] {
                continue;
            }
            loop {
                assert!(ef.pred_strict(upper_bound).unwrap() == (i, *v));
                upper_bound += 1;
                if i + 1 == values.len() || upper_bound > values[i + 1] {
                    break;
                }
            }
        }

        for upper_bound in 0..first + 1 {
            assert_eq!(None, ef.pred_strict(upper_bound));
        }

        let first = *values.first().unwrap();
        let mut upper_bound = first;
        for (i, v) in values.iter().enumerate() {
            if upper_bound < *v {
                continue;
            }
            // Skip repeated items as we return the index of the last one
            if i + 1 < values.len() && values[i] == values[i + 1] {
                continue;
            }
            loop {
                assert!(ef.pred(upper_bound).unwrap() == (i, *v));
                upper_bound += 1;
                if i + 1 == values.len() || upper_bound >= values[i + 1] {
                    break;
                }
            }
        }

        for upper_bound in 0..first {
            assert_eq!(None, ef.pred(upper_bound));
        }
    }

    Ok(())
}

#[test]
#[should_panic]
fn test_too_many_values() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(0);
    efb.push(1);
    efb.push(2);
}

#[test]
#[should_panic]
fn test_non_monotone() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(1);
    efb.push(0);
}

#[test]
#[should_panic]
fn test_too_large() {
    let mut efb = EliasFanoBuilder::new(2, 10);
    efb.push(11);
}

#[test]
#[should_panic]
fn test_from_non_monotone() {
    let _ef: EliasFano = vec![1, 0].into();
}

#[test]
fn test_extend() {
    let mut efb = EliasFanoBuilder::new(3, 10);
    let v = vec![0, 1, 2];
    efb.extend(v.clone());
    let ef = efb.build();
    zip(ef.iter(), v.iter()).for_each(|(a, b)| assert_eq!(a, *b));
}

#[test]
fn test_epserde() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(100, 1000), (100, 100), (1000, 100)] {
        let mut values = (0..n).map(|_| rng.random_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(n, u);
        // push the values
        for value in values.iter() {
            efb.push(*value);
        }
        // Finish the creation of elias-fano
        let ef = unsafe { efb.build().map_high_bits(SelectAdaptConst::<_, _>::new) };

        let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
        let schema = ef.serialize_with_schema(&mut file)?;
        drop(file);
        println!("{}", schema.to_csv());

        let c = <EliasFano<
            SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>>,
            BitFieldVec<usize, Box<[usize]>>,
        >>::mmap(&tmp_file, epserde::deser::Flags::empty())?;

        for i in 0..n {
            assert_eq!(ef.get(i), c.get(i));
        }
    }
    Ok(())
}

#[test]
fn test_convenience_methods() {
    let mut efb = EliasFanoBuilder::new(10, 100);
    for i in 0..10 {
        efb.push(i * 10);
    }
    let ef = efb.build_with_seq();
    for i in 0..10 {
        assert_eq!(ef.get(i), i * 10);
    }

    let mut efb = EliasFanoBuilder::new(10, 100);
    for i in 0..10 {
        efb.push(i * 10);
    }
    let ef = efb.build_with_dict();
    for i in 0..10 {
        assert_eq!(unsafe { ef.succ_unchecked::<false>(i * 10) }, (i, i * 10));
    }

    let mut efb = EliasFanoBuilder::new(10, 100);
    for i in 0..10 {
        efb.push(i * 10);
    }
    let ef = efb.build_with_seq_and_dict();
    for i in 0..10 {
        assert_eq!(ef.get(i), i * 10);
        assert_eq!(ef.succ(i * 10).unwrap(), (i, i * 10));
    }
}
