// Benchmark high-mem vs low-mem peeler at various key counts and thread counts.
// Run with: cargo run --release --example bench_peeler

use anyhow::Result;
use dsi_progress_logger::no_logging;
use sux::bits::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::utils::FromCloneableIntoIterator;
use std::time::Instant;

fn bench(n: usize, threads: usize, low_mem: Option<bool>) -> Result<std::time::Duration> {
    let mut builder = VBuilder::default()
        .expected_num_keys(n)
        .max_num_threads(threads);
    if let Some(lm) = low_mem {
        builder = builder.low_mem(lm);
    }

    let start = Instant::now();
    let _func = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
        FromCloneableIntoIterator::from(0..n),
        FromCloneableIntoIterator::from(0..n),
        n,
        builder,
        no_logging![],
    )?;
    Ok(start.elapsed())
}

fn main() -> Result<()> {
    let key_counts = [100_000, 1_000_000, 10_000_000, 100_000_000];
    let thread_counts = [1, 2, 4, 8];

    for &n in &key_counts {
        eprintln!("\n=== {n} keys ===");
        eprintln!(
            "{:>8}  {:>12}  {:>12}  {:>12}  {:>6}",
            "threads", "auto", "high-mem", "low-mem", "hi/lo"
        );

        for &threads in &thread_counts {
            // Warm up
            let _ = bench(n.min(100_000), threads, None);

            let auto = bench(n, threads, None)?;
            let high = bench(n, threads, Some(false))?;
            let low = bench(n, threads, Some(true))?;

            let ratio = high.as_secs_f64() / low.as_secs_f64();
            eprintln!(
                "{threads:>8}  {auto:>12.3?}  {high:>12.3?}  {low:>12.3?}  {ratio:>6.2}",
            );
        }
    }

    Ok(())
}
