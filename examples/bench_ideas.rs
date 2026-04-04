// Benchmark various construction ideas.
//
// cargo run --release --example bench_ideas -- --int 100000000
// cargo run --release --example bench_ideas -- --file uk-2007-05.urls

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::func::shard_edge::FuseLge3NoShards;
use sux::func::{VBuilder, VFunc};
use sux::utils::{FromCloneableIntoIterator, FromSlice};
use std::time::Instant;

type BFV = BitFieldVec<Box<[usize]>>;

fn timed(label: &str, f: impl FnOnce() -> Result<()>) -> Result<()> {
    let start = Instant::now();
    f()?;
    eprintln!("  {label}: {:?}", start.elapsed());
    Ok(())
}

fn bench_int(n: usize) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<usize> = (0..n).collect();

    eprintln!("=== {n} integer keys ===\n");

    timed("128-bit sigs, default (8 thr)", || {
        let _ = <VFunc<u64, BFV>>::try_new(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n, no_logging![],
        )?; Ok(())
    })?;

    timed("64-bit sigs, no shards", || {
        let _ = <VFunc<u64, BFV, [u64; 1], FuseLge3NoShards>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default(), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, offline", || {
        let _ = <VFunc<u64, BFV>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default().offline(true), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, 4 threads", || {
        let _ = <VFunc<u64, BFV>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default().max_num_threads(4), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, force high-mem", || {
        let _ = <VFunc<u64, BFV>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default().low_mem(false), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, parallel hash", || {
        let _ = <VFunc<u64, BFV>>::try_par_new_with_builder(
            &keys, &values, VBuilder::default(), no_logging![],
        )?; Ok(())
    })?;

    Ok(())
}

fn bench_strings(path: &str, max_n: usize) -> Result<()> {
    use std::io::{BufRead, BufReader};

    eprintln!("Reading keys from {path}...");
    let keys: Vec<String> = BufReader::new(std::fs::File::open(path)?)
        .lines().take(max_n).collect::<std::io::Result<_>>()?;
    let n = keys.len();
    let values: Vec<usize> = (0..n).collect();
    eprintln!("  {n} keys loaded\n");

    timed("128-bit sigs, default (8 thr)", || {
        let _ = <VFunc<str, BFV>>::try_new(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n, no_logging![],
        )?; Ok(())
    })?;

    timed("64-bit sigs, no shards", || {
        let _ = <VFunc<str, BFV, [u64; 1], FuseLge3NoShards>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default(), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, offline", || {
        let _ = <VFunc<str, BFV>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default().offline(true), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, 4 threads", || {
        let _ = <VFunc<str, BFV>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default().max_num_threads(4), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, force high-mem", || {
        let _ = <VFunc<str, BFV>>::try_new_with_builder(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n,
            VBuilder::default().low_mem(false), no_logging![],
        )?; Ok(())
    })?;

    timed("128-bit sigs, parallel hash", || {
        let _ = <VFunc<str, BFV>>::try_par_new_with_builder(
            &keys, &values, VBuilder::default(), no_logging![],
        )?; Ok(())
    })?;

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} --int <n> | --file <path> [n]", args[0]);
        std::process::exit(1);
    }
    match args[1].as_str() {
        "--int" => bench_int(args[2].parse()?)?,
        "--file" => {
            let n = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);
            bench_strings(&args[2], n)?;
        }
        _ => { eprintln!("Unknown flag"); std::process::exit(1); }
    }
    Ok(())
}
