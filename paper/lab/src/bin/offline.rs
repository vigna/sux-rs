//! Offline construction of PHast-R: builds a function online, offline, or
//! both, reporting construction times; in the latter case, it checks that
//! the two functions are identical (comparing their debug representations,
//! which contain all their data) and that queries agree. In offline mode keys are generated on the fly, and
//! never stored. Allocated memory is tracked by a counting allocator: the
//! peak is reported at the end, and with --trace the amount of allocated
//! memory is printed on standard error every 50 ms. With --log the progress
//! of the constructions is logged on standard error.
//!
//! Usage: offline [-n keys] [-m online|offline|both] [-v <S>:<log2 L>:<lambda>[:<log2 R>]] [--trace] [--log]

use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use lab::GxKey;
use mem_dbg::{MemSize, SizeFlags};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use sux::func::PHastR;
use sux::utils::FromCloneableIntoIterator;

/// A global allocator recording the allocated memory and its peak.
struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let c = CURRENT.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(c, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let c = CURRENT.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(c, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CURRENT.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if new > l.size() {
            let c = CURRENT.fetch_add(new - l.size(), Ordering::Relaxed) + new - l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        } else {
            CURRENT.fetch_sub(l.size() - new, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, long, default_value = "both")]
    mode: String,
    #[arg(short, long, default_value = "8:10:4.25")]
    variant: String,
    /// Prints the allocated memory every 50 ms on standard error​
    #[arg(long)]
    trace: bool,
    /// Logs the progress of the constructions on standard error​
    #[arg(long)]
    log: bool,
}

/// The key of index i.
fn key(i: u64) -> GxKey {
    GxKey(i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
}

fn main() {
    let a = Args::parse();
    if a.log {
        sux::init_env_logger().unwrap();
    }
    // The progress logger of the constructions, if any
    let pl = || a.log.then(ProgressLogger::default);
    if a.trace {
        let start = Instant::now();
        std::thread::spawn(move || {
            loop {
                eprintln!(
                    "MEM {:.2} {}",
                    start.elapsed().as_secs_f64(),
                    CURRENT.load(Ordering::Relaxed) >> 20
                );
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }
    let p: Vec<&str> = a.variant.split(':').collect();
    let (b, sbits, name) = lab::parse_config(&p);
    assert!(sbits <= 8, "byte seeds only");
    let n = a.n;
    let report = |what: &str, secs: f64, f: &PHastR<GxKey>| {
        println!(
            "{name} {what:8} n={n} {:.4} bits/key  build {:.1} ns/key",
            f.mem_size(SizeFlags::default()) as f64 * 8.0 / n as f64,
            secs * 1e9 / n as f64
        );
    };
    let offline = || {
        let keys = FromCloneableIntoIterator::new((0..n as u64).map(key));
        let start = Instant::now();
        let f: PHastR<GxKey> =
            PHastR::try_new_with_builder(keys, b.clone().offline(true), &mut pl()).unwrap();
        report("offline", start.elapsed().as_secs_f64(), &f);
        f
    };
    match a.mode.as_str() {
        "offline" => {
            offline();
        }
        "online" | "both" => {
            let keys: Vec<GxKey> = (0..n as u64).map(key).collect();
            let start = Instant::now();
            let on: PHastR<GxKey> =
                PHastR::try_par_new_with_builder(&keys, b.clone(), &mut pl()).unwrap();
            report("online", start.elapsed().as_secs_f64(), &on);
            if a.mode == "both" {
                let off = offline();
                assert!(
                    format!("{on:?}") == format!("{off:?}"),
                    "the functions differ"
                );
                for k in &keys {
                    assert_eq!(on.get(k), off.get(k));
                }
                println!("identical");
            }
        }
        m => panic!("unknown mode {m}"),
    }
    println!(
        "peak allocated memory {:.1} MiB",
        PEAK.load(Ordering::Relaxed) as f64 / (1 << 20) as f64
    );
}
