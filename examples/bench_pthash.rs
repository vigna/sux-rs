/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::{Context, Result};
use clap::Parser;
use dsi_progress_logger::*;
use pthash::{
    BuildConfiguration, DictionaryDictionary, Hashable, Minimal, MurmurHash2_128, MurmurHash2_64,
    PartitionedPhf, Phf,
};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use sux::{
    bit_field_vec,
    traits::{BitFieldSlice, BitFieldSliceMut},
};
use zstd::Decoder;

fn bench(n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
        eprintln!("{} ns/key", timings.last().unwrap());
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "Min: {} Median: {} Max: {} Average: {}",
        timings[0],
        timings[timings.len() / 2],
        timings.last().unwrap(),
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}

#[derive(Parser, Debug)]
#[command(about = "Benchmark PTHash + array with strings or 64-bit integers", long_about = None)]
struct Args {
    /// The maximum number of strings to read from the file, or the number of 64-bit keys.
    n: usize,
    /// The maximum number of samples to take when testing query speed.
    samples: usize,
    /// Use this number of threads.
    threads: usize,
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    /// The number of repetitions.
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
}

struct HashableU64(u64);
#[allow(dead_code)]
struct HashableStr(String);

impl Hashable for HashableU64 {
    type Bytes<'a>
        = [u8; 8]
    where
        Self: 'a;

    fn as_bytes(&self) -> Self::Bytes<'_> {
        self.0.to_ne_bytes()
    }
}

struct HashableVecu8<T: AsRef<[u8]>>(T);
impl<T: AsRef<[u8]>> Hashable for HashableVecu8<T> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn as_bytes(&self) -> Self::Bytes<'_> {
        self.0.as_ref()
    }
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();
    if args.sig64 {
        _main::<Minimal, MurmurHash2_64, DictionaryDictionary>(args)
    } else {
        _main::<Minimal, MurmurHash2_128, DictionaryDictionary>(args)
    }
}

fn _main<M: pthash::Minimality, H: pthash::Hasher, D: pthash::Encoder>(args: Args) -> Result<()> {
    let mut pl = concurrent_progress_logger![];
    let temp_dir = tempfile::TempDir::new()?;

    if let Some(filename) = args.filename {
        // Tuned by zack on the 2023-09-06 graph on a machine with two Intel Xeon Gold 6342 CPUs
        let mut config = BuildConfiguration::new(temp_dir.path().into());
        config.c = 5.;
        config.alpha = 0.94;
        config.num_partitions = 5_000_000_000_usize.div_ceil(10000000) as u64;
        config.num_threads = args.threads as _;

        pl.start(format!("Building MPH with parameters: {:?}", config));
        let mut func = PartitionedPhf::<M, H, D>::new();

        if args.zstd {
            func.par_build_in_internal_memory_from_bytes(
                || {
                    BufReader::new(Decoder::new(File::open(&filename).unwrap()).unwrap())
                        .lines()
                        .par_bridge()
                        .map_with(pl.clone(), |pl, r| {
                            pl.light_update();
                            HashableVecu8(r.unwrap())
                        })
                },
                &config,
            )
            .context("Failed to build MPH")?;
        } else {
            func.par_build_in_internal_memory_from_bytes(
                || {
                    BufReader::new(File::open(&filename).unwrap())
                        .lines()
                        .par_bridge()
                        .map_with(pl.clone(), |pl, r| {
                            pl.light_update();
                            HashableVecu8(r.unwrap())
                        })
                },
                &config,
            )
            .context("Failed to build MPH")?;
        }

        pl.info(format_args!("Construction completed"));

        pl.info(format_args!("Assigning values..."));

        let n = func.num_keys() as usize;

        let mut output = Vec::with_capacity(n);
        output.extend(0..n);

        let keys;
        if args.zstd {
            for (i, k) in BufReader::new(Decoder::new(File::open(&filename).unwrap()).unwrap())
                .lines()
                .enumerate()
            {
                output[func.hash(HashableVecu8(k.unwrap())) as usize] = i;
            }
            pl.done();

            keys = BufReader::new(Decoder::new(File::open(&filename).unwrap()).unwrap())
                .lines()
                .map(|r| r.unwrap())
                .take(100000000)
                .collect::<Vec<_>>();
        } else {
            for (i, k) in BufReader::new(File::open(&filename).unwrap())
                .lines()
                .enumerate()
            {
                output[func.hash(HashableVecu8(k.unwrap())) as usize] = i;
            }
            pl.done();

            keys = BufReader::new(File::open(&filename).unwrap())
                .lines()
                .map(|r| r.unwrap())
                .take(args.samples)
                .collect::<Vec<_>>();
        }

        pl.done_with_count(args.n);

        bench(n, args.repeats, || {
            for k in &keys {
                std::hint::black_box(output[func.hash(HashableVecu8(k)) as usize]);
            }
        });
    } else {
        // Tuned by zack on the 2023-09-06 graph on a machine with two Intel Xeon Gold 6342 CPUs
        let mut config = BuildConfiguration::new(temp_dir.path().into());
        config.c = 5.;
        config.alpha = 0.94;
        config.num_partitions = args.n.div_ceil(10000000) as u64;
        config.num_threads = args.threads as _;

        pl.start(format!("Building MPH with parameters: {:?}", config));

        let mut func = PartitionedPhf::<M, H, D>::new();

        func.par_build_in_internal_memory_from_bytes(
            || (0_u64..args.n as u64).into_par_iter().map(HashableU64),
            &config,
        )
        .context("Failed to build MPH")?;

        let mut output = bit_field_vec![(args.n - 1).ilog2() as usize + 1 => 0; args.n];
        for i in 0..args.n {
            output.set(func.hash(i.to_ne_bytes().as_slice()) as usize, i);
        }

        pl.done_with_count(args.n);

        bench(args.samples, args.repeats, || {
            let mut key: u64 = 0;
            for _ in 0..args.samples {
                key = key.wrapping_add(0x9e3779b97f4a7c15);
                std::hint::black_box(output.get(func.hash(key.to_ne_bytes().as_slice()) as usize));
            }
        });
    }
    Ok(())
}
