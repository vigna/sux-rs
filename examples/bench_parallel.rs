// Benchmark parallel vs sequential VFunc construction.
//
// Run with:
//   cargo run --release --example bench_parallel -- --int 1000000
//   cargo run --release --example bench_parallel -- --int 100000000
//   cargo run --release --example bench_parallel -- --file trec.terms
//   cargo run --release --example bench_parallel -- --file uk-2007-05.urls
//   cargo run --release --example bench_parallel -- --file uk-2007-05.urls 1000000

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::utils::{FromCloneableIntoIterator, FromSlice};
use std::time::Instant;

const ROUNDS: usize = 3;

fn median(times: &mut Vec<std::time::Duration>) -> std::time::Duration {
    times.sort();
    times[times.len() / 2]
}

fn bench_int(n: usize) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<usize> = (0..n).collect();

    eprintln!("=== VFunc<u64> ({n} keys, {ROUNDS} rounds) ===");

    // Warm up
    let _ = <VFunc<u64, BitFieldVec<Box<[usize]>>>>::try_new(
        FromSlice::new(&keys),
        FromCloneableIntoIterator::from(0..n),
        n,
        no_logging![],
    );

    let mut seq_times = Vec::new();
    for _ in 0..ROUNDS {
        let start = Instant::now();
        let _ = <VFunc<u64, BitFieldVec<Box<[usize]>>>>::try_new(
            FromSlice::new(&keys),
            FromCloneableIntoIterator::from(0..n),
            n,
            no_logging![],
        )?;
        seq_times.push(start.elapsed());
    }

    let mut par_times = Vec::new();
    for _ in 0..ROUNDS {
        let start = Instant::now();
        let _ = <VFunc<u64, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
            &keys,
            &values,
            VBuilder::default(),
            no_logging![],
        )?;
        par_times.push(start.elapsed());
    }

    let seq = median(&mut seq_times);
    let par = median(&mut par_times);
    eprintln!("  sequential: {seq:?}");
    eprintln!("  parallel:   {par:?}");
    eprintln!(
        "  speedup:    {:.2}x",
        seq.as_secs_f64() / par.as_secs_f64()
    );

    Ok(())
}

fn bench_strings(path: &str, max_n: usize) -> Result<()> {
    use std::io::{BufRead, BufReader};

    eprintln!("Reading up to {max_n} keys from {path}...");
    let start = Instant::now();
    let keys: Vec<String> = BufReader::new(std::fs::File::open(path)?)
        .lines()
        .take(max_n)
        .collect::<std::io::Result<_>>()?;
    let n = keys.len();
    let values: Vec<usize> = (0..n).collect();
    eprintln!("  Read {n} keys in {:?}", start.elapsed());

    eprintln!("\n=== VFunc<str> ({n} keys, {ROUNDS} rounds) ===");

    // Warm up
    let _ = <VFunc<str, BitFieldVec<Box<[usize]>>>>::try_new(
        FromSlice::new(&keys),
        FromCloneableIntoIterator::from(0..n),
        n,
        no_logging![],
    );

    let mut seq_times = Vec::new();
    for _ in 0..ROUNDS {
        let start = Instant::now();
        let _ = <VFunc<str, BitFieldVec<Box<[usize]>>>>::try_new(
            FromSlice::new(&keys),
            FromCloneableIntoIterator::from(0..n),
            n,
            no_logging![],
        )?;
        seq_times.push(start.elapsed());
    }

    let mut par_times = Vec::new();
    for _ in 0..ROUNDS {
        let start = Instant::now();
        let _ = <VFunc<str, BitFieldVec<Box<[usize]>>>>::try_par_new_with_builder(
            &keys,
            &values,
            VBuilder::default(),
            no_logging![],
        )?;
        par_times.push(start.elapsed());
    }

    let seq = median(&mut seq_times);
    let par = median(&mut par_times);
    eprintln!("  sequential: {seq:?}");
    eprintln!("  parallel:   {par:?}");
    eprintln!(
        "  speedup:    {:.2}x",
        seq.as_secs_f64() / par.as_secs_f64()
    );

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  {} --int <n>", args[0]);
        eprintln!("  {} --file <path> [n]", args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "--int" => {
            let n: usize = args[2].parse()?;
            bench_int(n)?;
        }
        "--file" => {
            let path = &args[2];
            let n: usize = args
                .get(3)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(usize::MAX);
            bench_strings(path, n)?;
        }
        _ => {
            eprintln!("Unknown flag: {}", args[1]);
            std::process::exit(1);
        }
    }

    Ok(())
}
