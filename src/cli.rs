/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Shared CLI types for the `vfunc` and `lcp_mmphf` binaries.

use std::fmt::Display;

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

/// Shared CLI flags that select the *type* of shard-edge used during
/// construction.
///
/// These flags live here (rather than in [`BuilderArgs`]) because they
/// don't configure a [`VBuilder`] instance — they select its type
/// parameters. Binaries that dispatch on shard-edge type (`vfunc`,
/// `vfilter`, `comp_vfunc`) flatten this struct next to [`BuilderArgs`];
/// binaries that pin their shard-edge type (`lcp_mmphf`) don't.
#[derive(clap::Args, Debug)]
pub struct ShardingArgs {
    /// Use 64-bit signatures.​
    #[arg(long, requires = "no_shards")]
    pub sig64: bool,
    /// Do not use sharding.​
    #[arg(long)]
    pub no_shards: bool,
}
