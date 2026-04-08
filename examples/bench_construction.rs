// Profile construction phases for VFunc, VFilter, LcpMmphf, SignedFunc.
//
// Uses the progress logger's built-in timing to identify which phases
// dominate construction time.
//
// Run with:
//   cargo run --release --example bench_construction -- --int 1000000
//   cargo run --release --example bench_construction -- --int 100000000
//   cargo run --release --example bench_construction -- --file uk-2007-05.urls
//   cargo run --release --example bench_construction -- --file trec.terms
//   cargo run --release --example bench_construction -- --file uk-2007-05.urls 1000000

use anyhow::Result;
use dsi_progress_logger::ProgressLogger;
use sux::bits::BitFieldVec;
use sux::dict::VFilter;
use sux::func::lcp2_mmphf::Lcp2Mmphf;
use sux::func::lcp_mmphf::LcpMmphf;
use sux::func::shard_edge::FuseLge3Shards;
use sux::func::{LcpMmphfInt, SignedFunc, VFunc};
use sux::utils::{FromCloneableIntoIterator, FromSlice};
use std::time::Instant;

fn timed<R>(label: &str, f: impl FnOnce() -> Result<R>) -> Result<R> {
    let start = Instant::now();
    let r = f()?;
    eprintln!("  ** {label}: {:?}", start.elapsed());
    Ok(r)
}

type DefLcpMmphfStr = LcpMmphf<str, BitFieldVec<Box<[usize]>>, [u64; 2], FuseLge3Shards>;
type DefLcp2MmphfStr = Lcp2Mmphf<str, BitFieldVec<Box<[usize]>>, [u64; 2], FuseLge3Shards>;

fn bench_int(n: usize) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let mut pl = ProgressLogger::default();

    eprintln!("\n=== VFunc<u64> ({n} keys) ===");
    timed("VFunc total", || {
        <VFunc<u64, BitFieldVec<Box<[usize]>>>>::try_new(
            FromSlice::new(&keys),
            FromCloneableIntoIterator::from(0..n),
            n,
            &mut pl,
        )
    })?;

    eprintln!("\n=== VFilter<u64> ({n} keys, 8-bit) ===");
    timed("VFilter total", || {
        <VFilter<VFunc<u64, Box<[u8]>>>>::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

    eprintln!("\n=== LcpMmphfInt<u64> ({n} keys) ===");
    timed("LcpMmphfInt total", || {
        LcpMmphfInt::<u64>::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

    eprintln!("\n=== SignedFunc<VFunc<u64>> ({n} keys, u16 hashes) ===");
    timed("SignedFunc<VFunc> total", || {
        <SignedFunc<VFunc<u64, BitFieldVec<Box<[usize]>>>, Box<[u16]>>>::try_new(
            FromSlice::new(&keys),
            n,
            &mut pl,
        )
    })?;

    eprintln!("\n=== SignedFunc<LcpMmphfInt<u64>> ({n} keys, u64 hashes) ===");
    timed("SignedFunc<LcpMmphfInt> total", || {
        <SignedFunc<LcpMmphfInt<u64>, Box<[u64]>>>::try_new(
            FromSlice::new(&keys),
            n,
            &mut pl,
        )
    })?;

    Ok(())
}

fn bench_strings(path: &str, max_n: usize) -> Result<()> {
    use std::io::{BufRead, BufReader};

    eprintln!("Reading up to {max_n} keys from {path}...");
    let start = Instant::now();
    let mut keys: Vec<String> = BufReader::new(std::fs::File::open(path)?)
        .lines()
        .take(max_n)
        .collect::<std::io::Result<_>>()?;
    let n = keys.len();
    eprintln!("  Read {n} keys in {:?}", start.elapsed());

    let sorted = keys.windows(2).all(|w| w[0] < w[1]);
    if !sorted {
        eprintln!("  Sorting...");
        let start = Instant::now();
        keys.sort();
        let before = keys.len();
        keys.dedup();
        eprintln!(
            "  Sorted ({} dups removed) in {:?}",
            before - keys.len(),
            start.elapsed()
        );
    }
    let n = keys.len();

    let mut pl = ProgressLogger::default();

    eprintln!("\n=== VFunc<str> ({n} keys) ===");
    timed("VFunc total", || {
        <VFunc<str, BitFieldVec<Box<[usize]>>>>::try_new(
            FromSlice::new(&keys),
            FromCloneableIntoIterator::from(0..n),
            n,
            &mut pl,
        )
    })?;

    eprintln!("\n=== VFilter<str> ({n} keys, 8-bit) ===");
    timed("VFilter total", || {
        <VFilter<VFunc<str, Box<[u8]>>>>::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

    eprintln!("\n=== LcpMmphfStr ({n} keys) ===");
    timed("LcpMmphfStr total", || {
        DefLcpMmphfStr::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

    eprintln!("\n=== Lcp2MmphfStr ({n} keys) ===");
    timed("Lcp2MmphfStr total", || {
        DefLcp2MmphfStr::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

    eprintln!("\n=== SignedFunc<LcpMmphfStr> ({n} keys, u64 hashes) ===");
    timed("SignedFunc<LcpMmphfStr> total", || {
        <SignedFunc<DefLcpMmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

    eprintln!("\n=== SignedFunc<Lcp2MmphfStr> ({n} keys, u64 hashes) ===");
    timed("SignedFunc<Lcp2MmphfStr> total", || {
        <SignedFunc<DefLcp2MmphfStr, Box<[u64]>>>::try_new(FromSlice::new(&keys), n, &mut pl)
    })?;

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

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();

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
