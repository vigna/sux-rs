// Benchmark sequential vs parallel construction for all structures.
//
// cargo run --release --example bench_par -- --int 1000000
// cargo run --release --example bench_par -- --int 100000000
// cargo run --release --example bench_par -- --int 1000000000
// cargo run --release --example bench_par -- --file trec.terms
// cargo run --release --example bench_par -- --file uk-2007-05.urls

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::dict::VFilter;
use sux::func::shard_edge::FuseLge3Shards;
use sux::func::lcp_mmphf::{LcpMmphf, LcpMmphfInt};
use sux::func::lcp2_mmphf::{Lcp2Mmphf, Lcp2MmphfInt};
use sux::func::{SignedFunc, BitSignedFunc, VFunc};
use sux::utils::{FromCloneableIntoIterator, FromSlice};
use std::time::Instant;

type BFV = BitFieldVec<Box<[usize]>>;
type DefLcpStr = LcpMmphf<str, BFV, [u64; 2], FuseLge3Shards>;
type DefLcp2Str = Lcp2Mmphf<str, BFV, [u64; 2], FuseLge3Shards>;

fn timed(label: &str, f: impl FnOnce() -> Result<()>) -> Result<std::time::Duration> {
    let start = Instant::now();
    f()?;
    let d = start.elapsed();
    eprintln!("  {label}: {d:?}");
    Ok(d)
}

fn ratio(seq: std::time::Duration, par: std::time::Duration) {
    eprintln!(
        "    speedup: {:.2}x\n",
        seq.as_secs_f64() / par.as_secs_f64()
    );
}

fn bench_int(n: usize) -> Result<()> {
    let keys: Vec<u64> = (0..n as u64).collect();
    let values: Vec<usize> = (0..n).collect();

    eprintln!("=== {n} integer keys ===\n");

    // VFunc
    let seq = timed("VFunc seq", || {
        let _ = <VFunc<u64, BFV>>::try_new(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("VFunc par", || {
        let _ = <VFunc<u64, BFV>>::try_par_new(&keys, &values, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // VFilter
    let seq = timed("VFilter seq", || {
        let _ = <VFilter<VFunc<u64, Box<[u8]>>>>::try_new(
            FromSlice::new(&keys), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("VFilter par", || {
        let _ = <VFilter<VFunc<u64, Box<[u8]>>>>::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // LcpMmphfInt
    let seq = timed("LcpMmphfInt seq", || {
        let _ = LcpMmphfInt::<u64>::try_new(FromSlice::new(&keys), n, no_logging![])?; Ok(())
    })?;
    let par = timed("LcpMmphfInt par", || {
        let _ = LcpMmphfInt::<u64>::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // Lcp2MmphfInt
    let seq = timed("Lcp2MmphfInt seq", || {
        let _ = Lcp2MmphfInt::<u64>::try_new(FromSlice::new(&keys), n, no_logging![])?; Ok(())
    })?;
    let par = timed("Lcp2MmphfInt par", || {
        let _ = Lcp2MmphfInt::<u64>::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // SignedFunc<VFunc>
    let seq = timed("SignedFunc<VFunc> seq", || {
        let _ = <SignedFunc<VFunc<u64, BFV>, Box<[u16]>>>::try_new(
            FromSlice::new(&keys), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("SignedFunc<VFunc> par", || {
        let _ = <SignedFunc<VFunc<u64, BFV>, Box<[u16]>>>::try_par_new(
            &keys, no_logging![],
        )?; Ok(())
    })?;
    ratio(seq, par);

    // SignedFunc<LcpMmphfInt>
    let seq = timed("SignedFunc<LcpMmphfInt> seq", || {
        let _ = <SignedFunc<LcpMmphfInt<u64>, Box<[u64]>>>::try_new(
            FromSlice::new(&keys), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("SignedFunc<LcpMmphfInt> par", || {
        let _ = <SignedFunc<LcpMmphfInt<u64>, Box<[u64]>>>::try_par_new(
            &keys, no_logging![],
        )?; Ok(())
    })?;
    ratio(seq, par);

    Ok(())
}

fn bench_strings(path: &str, max_n: usize) -> Result<()> {
    use std::io::{BufRead, BufReader};

    eprintln!("Reading keys from {path}...");
    let mut keys: Vec<String> = BufReader::new(std::fs::File::open(path)?)
        .lines().take(max_n).collect::<std::io::Result<_>>()?;
    let sorted = keys.windows(2).all(|w| w[0] < w[1]);
    if !sorted {
        keys.sort();
        keys.dedup();
    }
    let n = keys.len();
    let values: Vec<usize> = (0..n).collect();
    eprintln!("  {n} keys\n");

    // VFunc
    let seq = timed("VFunc<str> seq", || {
        let _ = <VFunc<str, BFV>>::try_new(
            FromSlice::new(&keys), FromCloneableIntoIterator::from(0..n), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("VFunc<str> par", || {
        let _ = <VFunc<str, BFV>>::try_par_new(&keys, &values, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // VFilter
    let seq = timed("VFilter<str> seq", || {
        let _ = <VFilter<VFunc<str, Box<[u8]>>>>::try_new(
            FromSlice::new(&keys), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("VFilter<str> par", || {
        let _ = <VFilter<VFunc<str, Box<[u8]>>>>::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // LcpMmphfStr
    let seq = timed("LcpMmphfStr seq", || {
        let _ = DefLcpStr::try_new(FromSlice::new(&keys), n, no_logging![])?; Ok(())
    })?;
    let par = timed("LcpMmphfStr par", || {
        let _ = DefLcpStr::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // Lcp2MmphfStr
    let seq = timed("Lcp2MmphfStr seq", || {
        let _ = DefLcp2Str::try_new(FromSlice::new(&keys), n, no_logging![])?; Ok(())
    })?;
    let par = timed("Lcp2MmphfStr par", || {
        let _ = DefLcp2Str::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

    // SignedFunc<LcpMmphfStr>
    let seq = timed("SignedFunc<LcpMmphfStr> seq", || {
        let _ = <SignedFunc<DefLcpStr, Box<[u64]>>>::try_new(
            FromSlice::new(&keys), n, no_logging![],
        )?; Ok(())
    })?;
    let par = timed("SignedFunc<LcpMmphfStr> par", || {
        let _ = <SignedFunc<DefLcpStr, Box<[u64]>>>::try_par_new(&keys, no_logging![])?; Ok(())
    })?;
    ratio(seq, par);

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
