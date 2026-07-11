/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Shared CLI utilities for binaries and example benchmarks.

use std::fmt::Display;
use std::time::Duration;

use crate::bits::BitFieldVec;
use crate::func::VBuilder;
use crate::func::shard_edge::ShardEdge;

/// Reads up to `n` lines from `filename` using a [`DekoBufLineLender`] into a
/// single concatenated [`String`] buffer, returning the buffer together with an
/// `offsets` vector such that line `i` is `&buffer[offsets[i]..offsets[i+1]]`.
///
/// [`DekoBufLineLender`]: crate::utils::DekoBufLineLender
#[cfg(feature = "deko")]
pub fn read_concat_lines(filename: &str, n: usize) -> anyhow::Result<(String, Vec<usize>)> {
    use lender::FallibleLender;

    let mut buffer = String::new();
    // Cap the initial capacity hint: the default parallel path passes
    // n = usize::MAX (read until EOF), which would abort with a capacity
    // overflow. The Vec grows naturally past the cap.
    let mut offsets: Vec<usize> = Vec::with_capacity(n.min(1 << 20));
    offsets.push(0);
    let mut lender = crate::utils::DekoBufLineLender::from_path(filename)?;
    let mut count = 0usize;
    while let Some(line) = lender.next()? {
        if count == n {
            break;
        }
        buffer.push_str(line);
        offsets.push(buffer.len());
        count += 1;
    }
    Ok((buffer, offsets))
}

/// Builds a `Vec<&str>` of slices into `buffer` using the offsets
/// produced by [`read_concat_lines`].
pub fn str_slice_from_offsets<'a>(buffer: &'a str, offsets: &[usize]) -> Vec<&'a str> {
    offsets.windows(2).map(|w| &buffer[w[0]..w[1]]).collect()
}

/// Reservoir-samples up to `n` lines from `filename` using a
/// [`DekoBufLineLender`]. Each line in the file has equal probability of
/// appearing in the result. The result is shuffled so that the strings are not
/// in file order.
///
/// [`DekoBufLineLender`]: crate::utils::DekoBufLineLender
#[cfg(feature = "deko")]
pub fn reservoir_sample(filename: &str, n: usize, seed: u64) -> anyhow::Result<Vec<String>> {
    use lender::FallibleLender;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut reservoir: Vec<String> = Vec::with_capacity(n);
    let mut lender = crate::utils::DekoBufLineLender::from_path(filename)?;
    let mut i = 0usize;
    while let Some(line) = lender.next()? {
        if i < n {
            reservoir.push(line.to_owned());
        } else {
            let j = rng.random_range(0..=i);
            if j < n {
                reservoir[j] = line.to_owned();
            }
        }
        i += 1;
    }
    reservoir.shuffle(&mut rng);
    Ok(reservoir)
}

/// Packs `n` strings into a contiguous byte buffer by cycling through
/// `queries`.
///
/// Returns the buffer and offsets such that query `i` is `&packed[offsets[i] as
/// usize .. offsets[i+1] as usize]`.
///
/// If `queries` has fewer than `n` elements, the queries are repeated
/// in a round-robin fashion so that exactly `n` entries are produced.
pub fn pack_strings(queries: &[String], n: usize) -> (Vec<u8>, Vec<usize>) {
    assert!(!queries.is_empty(), "no queries to pack");
    let mut packed = Vec::new();
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0usize);
    for i in 0..n {
        packed.extend_from_slice(queries[i % queries.len()].as_bytes());
        offsets.push(packed.len());
    }
    (packed, offsets)
}

/// Hash types for signed functions.​
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum HashTypes {
    U8,
    U16,
    U32,
    U64,
}

impl Display for HashTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashTypes::U8 => write!(f, "u8"),
            HashTypes::U16 => write!(f, "u16"),
            HashTypes::U32 => write!(f, "u32"),
            HashTypes::U64 => write!(f, "u64"),
        }
    }
}

/// VBuilder options shared by all function-building CLIs.​
#[derive(clap::Args, Debug)]
pub struct BuilderArgs {
    /// Use this number of threads.​
    #[arg(short, long)]
    pub threads: Option<usize>,
    /// Use disk-based buckets to reduce memory usage at construction time.
    /// Requires --sequential (-s).​
    #[arg(short, long, requires = "sequential")]
    pub offline: bool,
    /// Sort shards and check for duplicate signatures.​
    #[arg(short, long)]
    pub check_dups: bool,
    /// A 64-bit seed for the pseudorandom number generator.​
    #[arg(long)]
    pub seed: Option<u64>,
    /// The target relative space overhead due to sharding.​
    #[arg(long, default_value_t = 0.001)]
    pub eps: f64,
    /// Always use the low-mem peel-by-signature algorithm (slightly slower).​
    #[arg(long)]
    pub low_mem: bool,
    /// Always use the high-mem peel-by-signature algorithm (slightly faster).​
    #[arg(long, conflicts_with = "low_mem")]
    pub high_mem: bool,
}

impl BuilderArgs {
    /// Applies these options to a [`VBuilder`].
    pub fn configure<D: Send + Sync, S, E: ShardEdge<S, 3>>(
        &self,
        builder: VBuilder<D, S, E>,
    ) -> VBuilder<D, S, E> {
        let mut builder = builder
            .offline(self.offline)
            .check_dups(self.check_dups)
            .eps(self.eps);
        if let Some(seed) = self.seed {
            builder = builder.seed(seed);
        }
        if let Some(threads) = self.threads {
            builder = builder.max_num_threads(threads);
        }
        if self.low_mem {
            builder = builder.low_mem(true);
        }
        if self.high_mem {
            builder = builder.low_mem(false);
        }
        builder
    }

    /// Creates and configures a default [`VBuilder`] with [`BitFieldVec`]
    /// storage.
    pub fn to_builder(&self) -> VBuilder<BitFieldVec<Box<[usize]>>> {
        self.configure(VBuilder::default())
    }
}

/// Shard/edge type used during construction.
#[derive(clap::ValueEnum, Copy, Clone, Debug, Default)]
pub enum ShardEdgeType {
    /// Fuse 3-hypergraphs with lazy Gaussian elimination and ε-cost sharding.​
    #[default]
    FuseLge3Shards,
    /// Fuse 3-hypergraphs with ε-cost sharding, no lazy Gaussian elimination.​
    Fuse3Shards,
    /// Fuse 3-hypergraphs without sharding or lazy Gaussian elimination and 64-bit signatures.​
    Fuse3NoShards64,
    /// Fuse 3-hypergraphs without sharding or lazy Gaussian elimination and 128-bit signatures.​
    Fuse3NoShards128,
    /// MWHC 3-hypergraphs with ε-cost sharding.​
    #[cfg(feature = "mwhc")]
    Mwhc3,
    /// MWHC 3-hypergraphs without ε-cost sharding.​
    #[cfg(feature = "mwhc")]
    Mwhc3NoShards,
    /// Fuse 3-hypergraphs with lazy Gaussian elimination, ε-cost sharding, and full
    /// signatures.​
    FuseLge3FullSigs,
}

/// Shared CLI flags that select the *type* of shard/edge used during
/// construction.
///
/// These flags live here (rather than in [`BuilderArgs`]) because they don't
/// configure a [`VBuilder`] instance: they select its type parameters.
///
/// Binaries that dispatch on shard/edge type (`vfunc`, `vfilter`, `comp_vfunc`)
/// flatten this struct next to [`BuilderArgs`]; binaries that pin their
/// shard/edge type (`lcp_mmphf`) don't.
#[derive(clap::Args, Debug)]
pub struct ShardingArgs {
    /// Shard/edge type.​
    #[arg(long, short = 'E', value_enum, default_value_t)]
    pub shard_edge: ShardEdgeType,
}

fn parse_duration(value: &str) -> anyhow::Result<Duration> {
    anyhow::ensure!(!value.is_empty(), "empty duration string");
    let mut duration = Duration::from_secs(0);
    let mut acc = String::new();
    for c in value.chars() {
        if c.is_ascii_digit() {
            acc.push(c);
        } else if c.is_whitespace() {
            continue;
        } else {
            let dur: u64 = acc.parse()?;
            match c {
                's' => duration += Duration::from_secs(dur),
                'm' => duration += Duration::from_secs(dur * 60),
                'h' => duration += Duration::from_secs(dur * 3600),
                'd' => duration += Duration::from_secs(dur * 86400),
                _ => anyhow::bail!("invalid duration suffix: {c}"),
            }
            acc.clear();
        }
    }
    if !acc.is_empty() {
        duration += Duration::from_millis(acc.parse()?);
    }
    Ok(duration)
}

/// Shared CLI argument for the progress-logger log interval.
#[derive(clap::Args, Debug, Clone)]
pub struct LogIntervalArg {
    /// How often to log progress. Supported suffixes: "s" (seconds),
    /// "m" (minutes), "h" (hours), "d" (days). A bare number is
    /// interpreted as milliseconds. Examples: "10s", "1m30s", "500".​
    #[arg(long, value_parser = parse_duration, default_value = "10s")]
    pub log_interval: Duration,
}
