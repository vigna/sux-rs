use anyhow::Result;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use sux::prelude::*;

#[test]
fn test_serdes() -> Result<()> {
    let u = 10_000;
    let n = 1_000;
    let mut rng = SmallRng::seed_from_u64(0);

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
    println!("{} {}", ef.mem_size(), ef.mem_used());
    println!("{}", EliasFanoBuilder::mem_upperbound(u, n) / 8);

    let tmp_file = std::env::temp_dir().join("test_serdes_ef.bin");
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
        ef.serialize(&mut file)?;
    }

    let file = std::fs::File::open(&tmp_file)?;
    let file_len = file.metadata()?.len();
    let mmap = unsafe {
        mmap_rs::MmapOptions::new(file_len as _)?
            .with_file(file, 0)
            .map()?
    };

    let ef = <EliasFano<BitMap<&[u64]>, CompactArray<&[u64]>>>::deserialize(&mmap)?.0;

    for (idx, value) in values.iter().enumerate() {
        assert_eq!(ef.get(idx), *value);
    }

    let ef = map::<_, EliasFano<BitMap<&[u64]>, CompactArray<&[u64]>>>(&tmp_file, &Flags::empty())?;

    for (idx, value) in values.iter().enumerate() {
        assert_eq!(ef.get(idx), *value);
    }

    let ef = map::<_, EliasFano<BitMap<&[u64]>, CompactArray<&[u64]>>>(
        &tmp_file,
        &Flags::TRANSPARENT_HUGE_PAGES,
    )?;

    for (idx, value) in values.iter().enumerate() {
        assert_eq!(ef.get(idx), *value);
    }

    let ef =
        load::<_, EliasFano<BitMap<&[u64]>, CompactArray<&[u64]>>>(&tmp_file, &Flags::empty())?;

    for (idx, value) in values.iter().enumerate() {
        assert_eq!(ef.get(idx), *value);
    }

    Ok(())
}
