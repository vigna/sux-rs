/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::{Context, Result};
use clap::Parser;
use dsi_progress_logger::*;
use pthash::{
    BuildConfiguration, DictionaryDictionary, Hashable, Minimal, MurmurHash2_64, PartitionedPhf,
    Phf,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sux::{bit_field_vec, traits::BitFieldSlice};

#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    #[arg(short)]
    /// The maximum number strings to use from the file, or the number of 64-bit keys.
    n: usize,
    /// A name for the ε-serde serialized function with u64 keys.
    func: String,
}

struct HashableU64(u64);
impl Hashable for HashableU64 {
    type Bytes<'a>
        = [u8; 8]
    where
        Self: 'a;

    fn as_bytes(&self) -> Self::Bytes<'_> {
        self.0.to_ne_bytes()
    }
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    let mut pl = ProgressLogger::default();
    let temp_dir = tempfile::TempDir::new()?;
    // Tuned by zack on the 2023-09-06 graph on a machine with two Intel Xeon Gold 6342 CPUs
    let mut config = BuildConfiguration::new(temp_dir.path().into());
    config.c = 5.;
    config.alpha = 0.94;
    config.num_partitions = args.n.div_ceil(10000000) as u64;
    config.num_threads = num_cpus::get() as u64;

    pl.start(format!("Building MPH with parameters: {:?}", config));

    let mut func = PartitionedPhf::<Minimal, MurmurHash2_64, DictionaryDictionary>::new();

    func.par_build_in_internal_memory_from_bytes(
        || (0_u64..args.n as u64).into_par_iter().map(HashableU64),
        &config,
    )
    .context("Failed to build MPH")?;
    pl.done_with_count(args.n);

    let mut output = Vec::with_capacity(args.n);
    output.extend(0..args.n);

    let output = bit_field_vec![args.n.ilog2() as usize => 0; args.n];

    pl.start("Querying (independent)...");
    for i in 0..args.n {
        std::hint::black_box(output.get(func.hash(i.to_ne_bytes().as_slice()) as usize));
    }
    pl.done_with_count(args.n);

    let mut x = 0;

    pl.start("Querying (dependent)...");
    for i in 0..args.n {
        x = output.get(func.hash((i ^ (x & 1)).to_ne_bytes().as_slice()) as usize);
        std::hint::black_box(
            (),
        );
    }
    pl.done_with_count(args.n);

    Ok(())
}
