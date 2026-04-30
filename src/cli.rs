/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Shared CLI types for the `vfunc` and `lcp_mmphf` binaries.

use std::fmt::Display;
use std::time::Duration;

use crate::bits::BitFieldVec;
use crate::func::VBuilder;
use crate::func::shard_edge::ShardEdge;

/// Reads up to `n` lines from `filename` (optionally compressed) into a
/// single concatenated [`String`] buffer, returning the buffer together
/// with an `offsets` vector such that line `i` is
/// `&buffer[offsets[i] .. offsets[i+1]]`.
///
/// Used by the parallel in-memory construction path of the CLI
/// utilities: callers build a `Vec<&str>` of slices into the buffer via
/// [`str_slice_from_offsets`], giving cache-friendly access during
/// signature hashing — one big allocation plus fixed-size `&str`
/// references, instead of `n` independent heap allocations as with a
/// `Vec<String>`.
#[cfg(feature = "deko")]
pub fn read_lines_concatenated(filename: &str, n: usize) -> anyhow::Result<(String, Vec<usize>)> {
    use lender::FallibleLender;

    let mut buffer = String::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(n.saturating_add(1));
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
/// produced by [`read_lines_concatenated`].
#[inline]
pub fn str_slice_from_offsets<'a>(buffer: &'a str, offsets: &[usize]) -> Vec<&'a str> {
    offsets.windows(2).map(|w| &buffer[w[0]..w[1]]).collect()
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
