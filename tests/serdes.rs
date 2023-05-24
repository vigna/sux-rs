use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::io::{Seek, Write};
use sux::prelude::*;

#[test]
fn test_serdes() {
    let u = 10_000;
    let n = 1_000;
    let mut rng = SmallRng::seed_from_u64(0);

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

    let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file).unwrap());
        ef.serialize(&mut file).unwrap();
    }

    let mut file = std::fs::File::open(tmp_file).unwrap();
    let file_len = file.seek(std::io::SeekFrom::End(0)).unwrap();
    let mmap = unsafe {
        mmap_rs::MmapOptions::new(file_len as _)
            .unwrap()
            .with_file(file, 0)
            .map()
            .unwrap()
    };

    let ef2 = <EliasFano<BitMap<&[u64]>, CompactArray<&[u64]>>>::deserialize(&mmap)
        .unwrap()
        .0;

    for (idx, value) in values.iter().enumerate() {
        assert_eq!(ef2.get(idx).unwrap(), *value);
    }
}
