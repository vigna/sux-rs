/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]
use anyhow::Result;
use epserde::prelude::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::Ordering;
use sux::prelude::BitFieldVec;
use sux::prelude::*;

#[test]
#[cfg(feature = "rayon")]
fn test_elias_fano_concurrent() -> Result<()> {
    use rayon::prelude::*;
    use sux::dict::elias_fano::EliasFanoConcurrentBuilder;
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let efb = EliasFanoConcurrentBuilder::new(n, u);
        // push concurrently the values
        values
            .par_iter()
            .enumerate()
            .for_each(|(index, value)| unsafe { efb.set(index, *value, Ordering::SeqCst) });
        // Finish the creation of elias-fano
        let _ef = efb.build();
    }
    Ok(())
}

#[test]
fn test_elias_fano() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(10, 1000), (100, 1000), (100, 100), (1000, 100), (1000, 10)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(n, u);
        // push the values
        for value in values.iter() {
            efb.push(*value)?;
        }
        // Finish the creation of elias-fano
        let ef = efb.build();
        println!("{:?}", ef);
        // do a slow select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), *v);
            assert_eq!({ ef.get(i) }, *v);
        }
        // Add the ones indices
        let ef: EliasFano<QuantumIndex> = ef.convert_to().unwrap();

        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), *v);
        }

        // Add the indices
        let ef: sux::dict::elias_fano::EliasFano<QuantumZeroIndex<QuantumIndex>> =
            ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!({ ef.get(i) }, *v);
        }
        println!("{:?}", ef);

        let mut iterator = ef.into_iter().enumerate();
        while let Some((i, v)) = iterator.next() {
            assert_eq!(v, values[i]);
            assert_eq!(iterator.len(), ef.len() - i - 1);
        }

        for from in 0..ef.len() {
            let mut iterator = ef.into_iter_from(from).enumerate();
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
                assert!(ef.succ(&lower_bound).unwrap() == (i, *v));
                lower_bound += 1;
                if lower_bound > values[i] {
                    break;
                }
            }
        }

        assert_eq!(None, ef.succ(&(last + 1)));
    }

    Ok(())
}

#[test]
fn test_epserde() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (n, u) in [(100, 1000), (100, 100), (1000, 100)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(n, u);
        // push the values
        for value in values.iter() {
            efb.push(*value)?;
        }
        // Finish the creation of elias-fano
        let ef: EliasFano = efb.build();
        // Add the ones indices
        let ef: EliasFano<QuantumIndex, BitFieldVec> = ef.convert_to().unwrap();

        let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
        let schema = ef.serialize_with_schema(&mut file)?;
        drop(file);
        println!("{}", schema.to_csv());

        let c = <EliasFano<QuantumIndex, BitFieldVec>>::mmap(
            &tmp_file,
            epserde::deser::Flags::empty(),
        )?;

        for i in 0..n {
            assert_eq!(ef.get(i), c.get(i));
        }
    }
    Ok(())
}
