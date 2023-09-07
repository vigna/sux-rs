use anyhow::Result;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::Ordering;
use sux::prelude::*;

#[test]
#[cfg(feature = "rayon")]
fn test_elias_fano_concurrent() -> Result<()> {
    use rayon::prelude::*;
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
            assert_eq!(ef.get(i) as u64, *v);
        }
        // Add the ones indices
        let ef: EliasFano<
            SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        > = ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!(ef.get(i) as u64, *v);
        }
        println!("{:?}", ef);

        // Add the indices
        let ef: EliasFano<
            SparseZeroIndex<SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        > = ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!(ef.get(i) as u64, *v);
        }
        println!("{:?}", ef);
    }

    Ok(())
}
