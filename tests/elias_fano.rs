use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::prelude::*;

#[test]
fn test_elias_fano() {
    let mut rng = SmallRng::seed_from_u64(0);
    for (u, n) in [(1000, 100), (100, 100), (100, 1000)] {
        let mut values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

        values.sort();

        // create the builder for the "in memory" elias-fano
        let mut efb = EliasFanoBuilder::new(u, n);
        // push the values
        for value in values.iter() {
            efb.push(*value).unwrap();
        }
        // Finish the creation of elias-fano
        let ef = efb.build();

        // do a slow select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!(ef.get(i).unwrap() as u64, *v);
        }
        // Add the ones indices
        let ef: EliasFano<SparseIndex<BitMap<Vec<u64>>, Vec<u64>, 8>, CompactArray<Vec<u64>>> =
            ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!(ef.get(i).unwrap() as u64, *v);
        }

        // Add the indices
        let ef: EliasFano<
            SparseZeroIndex<SparseIndex<BitMap<Vec<u64>>, Vec<u64>, 8>, Vec<u64>, 8>,
            CompactArray<Vec<u64>>,
        > = ef.convert_to().unwrap();
        // do a fast select
        for (i, v) in values.iter().enumerate() {
            assert_eq!(ef.select(i).unwrap() as u64, *v);
            assert_eq!(ef.get(i).unwrap() as u64, *v);
        }
    }
}
