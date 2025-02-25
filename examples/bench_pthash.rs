/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

 #![allow(clippy::collapsible_else_if)]
use std::{fs::File, io::BufReader};

use anyhow::{Context, Result};
use bytelines::ByteLines;
use clap::Parser;
use dsi_progress_logger::*;
use pthash::{
    BuildConfiguration, DictionaryDictionary, Hashable, Minimal, MurmurHash2_128, MurmurHash2_64,
    PartitionedPhf, Phf,
};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use sux::{bit_field_vec, traits::BitFieldSlice};
use zstd::Decoder;
#[derive(Parser, Debug)]
#[command(about = "Benchmark VFunc with strings or 64-bit integers", long_about = None)]
struct Args {
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.
    filename: Option<String>,
    #[arg(short)]
    /// The filename containing the keys is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// The number of 64-bit keys.
    n: Option<usize>,
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

    let mut pl = concurrent_progress_logger![];
    let temp_dir = tempfile::TempDir::new()?;

    if let Some(filename) = args.filename {
        // Tuned by zack on the 2023-09-06 graph on a machine with two Intel Xeon Gold 6342 CPUs
        let mut config = BuildConfiguration::new(temp_dir.path().into());
        config.c = 5.;
        config.alpha = 0.94;
        config.num_partitions = 5_000_000_000_usize.div_ceil(10000000) as u64;
        config.num_threads = num_cpus::get() as u64;

        pl.start(format!("Building MPH with parameters: {:?}", config));

        let mut func = PartitionedPhf::<Minimal, MurmurHash2_128, DictionaryDictionary>::new();

        if args.zstd {
            func.par_build_in_internal_memory_from_bytes(
                || {
                    ByteLines::new(BufReader::new(
                        Decoder::new(File::open(&filename).unwrap()).unwrap(),
                    ))
                    .into_iter()
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
                    ByteLines::new(BufReader::new(File::open(&filename).unwrap()))
                        .into_iter()
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

        let mut keys;
        if args.zstd {
            for (i, k) in ByteLines::new(BufReader::new(
                Decoder::new(File::open(&filename).unwrap()).unwrap(),
            ))
            .into_iter()
            .enumerate()
            {
                output[func.hash(HashableVecu8(k.unwrap())) as usize] = i;
            }

            keys = ByteLines::new(BufReader::new(
                Decoder::new(File::open(&filename).unwrap()).unwrap(),
            ))
            .into_iter()
            .map(|r| r.unwrap())
            .take(100000000)
            .collect::<Vec<_>>();
        } else {
            for (i, k) in ByteLines::new(BufReader::new(File::open(&filename).unwrap()))
                .into_iter()
                .enumerate()
            {
                output[func.hash(HashableVecu8(k.unwrap())) as usize] = i;
            }

            keys = ByteLines::new(BufReader::new(File::open(&filename).unwrap()))
                .into_iter()
                .map(|r| r.unwrap())
                .take(100000000)
                .collect::<Vec<_>>();
        }

        pl.done();

        pl.start("Querying (independent)...");
        for k in &keys {
            std::hint::black_box(output[func.hash(HashableVecu8(k)) as usize]);
        }
        pl.done_with_count(keys.len());

        let mut result = 0;
        pl.start("Querying (dependent)...");
        for k in &mut keys {
            k[0] ^= (result & 1) as u8;
            result = std::hint::black_box(output[func.hash(HashableVecu8(k)) as usize]);
        }
        pl.done_with_count(keys.len());
    } else if let Some(n) = args.n {
        // Tuned by zack on the 2023-09-06 graph on a machine with two Intel Xeon Gold 6342 CPUs
        let mut config = BuildConfiguration::new(temp_dir.path().into());
        config.c = 5.;
        config.alpha = 0.94;
        config.num_partitions = n.div_ceil(10000000) as u64;
        config.num_threads = num_cpus::get() as u64;

        pl.start(format!("Building MPH with parameters: {:?}", config));

        let mut func = PartitionedPhf::<Minimal, MurmurHash2_64, DictionaryDictionary>::new();

        func.par_build_in_internal_memory_from_bytes(
            || (0_u64..n as u64).into_par_iter().map(HashableU64),
            &config,
        )
        .context("Failed to build MPH")?;
        pl.done_with_count(n);

        let mut output = Vec::with_capacity(n);
        output.extend(0..n);

        let output = bit_field_vec![n.ilog2() as usize => 0; n];

        pl.start("Querying (independent)...");
        for i in 0..n {
            std::hint::black_box(output.get(func.hash(i.to_ne_bytes().as_slice()) as usize));
        }
        pl.done_with_count(n);

        let mut x = 0;

        pl.start("Querying (dependent)...");
        for i in 0..n {
            x = output.get(func.hash((i ^ (x & 1)).to_ne_bytes().as_slice()) as usize);
            std::hint::black_box(());
        }
        pl.done_with_count(n);
    }
    Ok(())
}
