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
use crate::traits::{BitFieldSlice, Word};
use crate::utils::BinSafe;

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
    /// Use disk-based buckets to reduce memory usage at construction time.​
    #[arg(short, long)]
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
    pub fn configure<
        W: Word + BinSafe,
        D: BitFieldSlice<Value = W> + Send + Sync,
        S,
        E: ShardEdge<S, 3>,
    >(
        &self,
        builder: VBuilder<W, D, S, E>,
    ) -> VBuilder<W, D, S, E> {
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
    pub fn to_builder(&self) -> VBuilder<usize, BitFieldVec<Box<[usize]>>> {
        self.configure(VBuilder::default())
    }
}
