// Benchmark construction with different thread counts.
//
// cargo run --release --example bench_threads -- --int 30000000
// cargo run --release --example bench_threads -- --file trec.terms 1000000

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::utils::{FromCloneableIntoIterator, FromSlice};
use std::time::Instant;

type BFV = BitFieldVec<Box<[usize]>>;

fn timed(f: impl FnOnce() -> Result<()>) -> Result<std::time::Duration> {
    let start = Instant::now();
    f()?;
    Ok(start.elapsed())
}

fn bench_int(n: usize) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<usize> = (0..n).collect();

    eprintln!("=== {n} integer keys ===\n");

    // Sequential baseline
    let t = timed(|| {
        let _ = <VFunc<u64, BFV>>::try_new(
            FromSlice::new(&keys),
            FromCloneableIntoIterator::from(0..n),
            n,
            no_logging![],
        )?;
        Ok(())
    })?;
    eprintln!("  Sequential (default 8 thr solve): {:?}", t);

    // Sequential with different solve thread counts
    for threads in [1, 2, 4, 8, 12, 16, 20] {
        let t = timed(|| {
            let _ = <VFunc<u64, BFV>>::try_new_with_builder(
                FromSlice::new(&keys),
                FromCloneableIntoIterator::from(0..n),
                n,
                VBuilder::default().max_num_threads(threads),
                no_logging![],
            )?;
            Ok(())
        })?;
        eprintln!("  Sequential hash, {threads:2} thr solve: {:?}", t);
    }

    eprintln!();

    // Parallel hash with different solve thread counts
    for threads in [1, 2, 4, 8, 12, 16, 20] {
        let t = timed(|| {
            let _ = <VFunc<u64, BFV>>::try_par_new_with_builder(
                &keys,
                &values,
                VBuilder::default().max_num_threads(threads),
                no_logging![],
            )?;
            Ok(())
        })?;
        eprintln!("  Parallel hash,    {threads:2} thr solve: {:?}", t);
    }

    Ok(())
}

fn bench_strings(path: &str, max_n: usize) -> Result<()> {
    use std::io::{BufRead, BufReader};

    eprintln!("Reading keys from {path}...");
    let keys: Vec<String> = BufReader::new(std::fs::File::open(path)?)
        .lines()
        .take(max_n)
        .collect::<std::io::Result<_>>()?;
    let n = keys.len();
    let values: Vec<usize> = (0..n).collect();
    eprintln!("  {n} keys\n");

    // Sequential baseline
    let t = timed(|| {
        let _ = <VFunc<str, BFV>>::try_new(
            FromSlice::new(&keys),
            FromCloneableIntoIterator::from(0..n),
            n,
            no_logging![],
        )?;
        Ok(())
    })?;
    eprintln!("  Sequential (default 8 thr solve): {:?}", t);

    for threads in [1, 2, 4, 8, 12, 16, 20] {
        let t = timed(|| {
            let _ = <VFunc<str, BFV>>::try_new_with_builder(
                FromSlice::new(&keys),
                FromCloneableIntoIterator::from(0..n),
                n,
                VBuilder::default().max_num_threads(threads),
                no_logging![],
            )?;
            Ok(())
        })?;
        eprintln!("  Sequential hash, {threads:2} thr solve: {:?}", t);
    }

    eprintln!();

    for threads in [1, 2, 4, 8, 12, 16, 20] {
        let t = timed(|| {
            let _ = <VFunc<str, BFV>>::try_par_new_with_builder(
                &keys,
                &values,
                VBuilder::default().max_num_threads(threads),
                no_logging![],
            )?;
            Ok(())
        })?;
        eprintln!("  Parallel hash,    {threads:2} thr solve: {:?}", t);
    }

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
        _ => {
            eprintln!("Unknown flag");
            std::process::exit(1);
        }
    }
    Ok(())
}
