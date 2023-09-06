use sux::prelude::*;

pub fn main() {
    // create the builder for the "in memory" elias-fano
    let mut efb = EliasFanoBuilder::new(100, 10);
    // push the values
    for value in 0..10 {
        efb.push(value).unwrap();
    }
    // Finish the creation of elias-fano
    let ef = efb.build();
    // do a slow select
    assert_eq!(ef.select(3).unwrap(), 2);
    assert_eq!(ef.get(3), 2);
    // Add the ones indices
    let ef: EliasFano<
        SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>,
        CompactArray<Vec<u64>>,
    > = ef.convert_to().unwrap();
    // do a fast select
    assert_eq!(ef.select(3).unwrap(), 2);
    assert_eq!(ef.get(3), 2);

    // Add the indices
    let ef: EliasFano<
        SparseZeroIndex<SparseIndex<CountingBitmap<Vec<u64>, usize>, Vec<u64>, 8>, Vec<u64>, 8>,
        CompactArray<Vec<u64>>,
    > = ef.convert_to().unwrap();
    // do a fast select
    assert_eq!(ef.select(3).unwrap(), 2);
    assert_eq!(ef.get(3), 2);
}
