/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use epserde::*;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::Ordering;
use sux::prelude::*;

#[test]
#[cfg(feature = "rayon")]
fn test_elias_fano_concurrent() -> Result<()> {
    use rayon::prelude::*;
    use sux::dict::elias_fano::EliasFanoAtomicBuilder;
    let mut rng = SmallRng::seed_from_u64(0);
    for (u, n) in [(1000, 100), (100, 100), (100, 1000)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let efb = EliasFanoAtomicBuilder::new(u, n);
        // push concurrently the values
        values
            .par_iter()
            .enumerate()
            .for_each(|(index, value)| unsafe { efb.set(index, *value, Ordering::SeqCst) });
        // Finish the creation of elias-fano
        let ef = efb.build();
        println!("{:?}", ef);
    }
    Ok(())
}

#[test]
fn test_elias_fano() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (u, n) in [(1000, 100), (100, 100), (100, 1000)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(u, n);
        // push the values
        for value in values.iter() {
            efb.push(*value)?;
        }
        // Finish the creation of elias-fano
        let ef = efb.build();
        println!("{:?}", ef);
        // do a slow select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!({ ef.get(i) }, *v);
        }
        // Add the ones indices
        let ef: EliasFano<
            SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        > = ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!({ ef.get(i) }, *v);
        }
        println!("{:?}", ef);

        // Add the indices
        let ef: sux::dict::elias_fano::EliasFano<
            SparseZeroIndex<SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        > = ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!({ ef.get(i) }, *v);
        }
        println!("{:?}", ef);
    }

    Ok(())
}

#[test]
fn test_epsserde() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(0);
    for (u, n) in [(1000, 100), (100, 100), (100, 1000)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(u, n);
        // push the values
        for value in values.iter() {
            efb.push(*value)?;
        }
        // Finish the creation of elias-fano
        let ef: DefaultEliasFano = efb.build();
        // Add the ones indices
        let ef: EliasFano<
            SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        > = ef.convert_to().unwrap();

        let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
        let schema = ef.serialize_with_schema(&mut file)?;
        drop(file);
        println!("{}", schema.to_csv());

        let c = <EliasFano<
            SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        >>::mmap(&tmp_file, epserde::Flags::empty())?;

        for i in 0..n {
            assert_eq!(ef.get(i as usize), c.get(i as usize));
        }
    }
    Ok(())
}
