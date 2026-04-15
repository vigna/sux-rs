/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use rdst::RadixKey;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sux::bits::BitVec;
use sux::cli::{BuilderArgs, ShardingArgs, read_lines_concatenated, str_slice_from_offsets};
use sux::func::codec::Huffman;
use sux::func::shard_edge::{FuseLge3NoShards, FuseLge3Shards, ShardEdge};
use sux::func::{CompVFunc, VBuilder};
use sux::init_env_logger;
use sux::utils::lenders::FromSlice;
use sux::utils::{DekoBufLineLender, FromCloneableIntoIterator, Sig, SigVal, ToSig};

#[derive(Parser, Debug)]
#[command(
    about = "Creates a CompVFunc mapping each key to a (compressed) integer value and serializes it with ε-serde.",
    long_about = None,
    next_line_help = true,
    max_term_width = 100,
)]
#[clap(group(
    ArgGroup::new("input")
        .required(true)
        .multiple(true)
        .args(&["filename", "n"]),
))]
struct Args {
    /// The number of keys; if no filename is provided, use the 64-bit keys
    /// [0 . . n).​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line (at most N keys will
    /// be read); it can be compressed with any format supported by
    /// the deko crate.​
    #[arg(short, long)]
    filename: Option<String>,
    /// A file containing the values, one ASCII decimal integer per
    /// line. Compulsory. The number of values must match the number
    /// of keys.​
    #[arg(short, long)]
    values: String,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Use the single-threaded *sequential* construction path that
    /// streams keys through a lender instead of materialising them
    /// in memory. Slower for large builds (sig-store population is
    /// the bottleneck) but useful when keys don't fit in RAM. The
    /// default is the parallel, in-memory path.​
    #[arg(short, long)]
    sequential: bool,
    /// Cap on the number of distinct codeword lengths in the Huffman
    /// decoding table. Rare symbols beyond the cap are diverted to the
    /// escape codeword and stored as literals.​
    #[arg(long, default_value_t = 20)]
    huffman_max_length: usize,
    /// Cumulative-entropy fraction beyond which infrequent symbols are
    /// diverted to the escape codeword. Truncates the Huffman table
    /// once the kept symbols cover this fraction of the total bit
    /// budget.​
    #[arg(long, default_value_t = 0.9)]
    huffman_entropy_threshold: f64,
    #[clap(flatten)]
    builder: BuilderArgs,
    #[clap(flatten)]
    sharding: ShardingArgs,
}

fn read_values(path: &str) -> Result<Vec<u64>> {
    let file = File::open(path).with_context(|| format!("open values file {path}"))?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: u64 = trimmed.parse().with_context(|| {
            format!(
                "values file {path}, line {}: cannot parse `{trimmed}`",
                lineno + 1
            )
        })?;
        values.push(v);
    }
    Ok(values)
}

/// Store the built function to `out`, if provided. Extracted so the
/// four branches below each call it once at their tail.
macro_rules! maybe_store {
    ($func:expr, $out:expr) => {
        if let Some(out) = $out {
            unsafe { $func.store(&out)? };
        }
    };
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    if args.sharding.no_shards {
        if args.sharding.sig64 {
            main_with_types::<[u64; 1], FuseLge3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], FuseLge3NoShards>(args)
        }
    } else {
        main_with_types::<[u64; 2], FuseLge3Shards>(args)
    }
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, u64>: RadixKey,
    CompVFunc<str, u64, BitVec<Box<[usize]>>, S, E>: Serialize,
    CompVFunc<usize, u64, BitVec<Box<[usize]>>, S, E>: Serialize,
{
    let vbuilder: VBuilder<BitVec<Box<[usize]>>, S, E> =
        args.builder.configure(VBuilder::default());
    let huffman = Huffman::length_limited(args.huffman_max_length, args.huffman_entropy_threshold);

    let values = read_values(&args.values)?;
    let n_values = values.len();

    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    if let Some(filename) = &args.filename {
        let n = n_values;
        if args.sequential {
            // Sequential: stream keys through the lender, no
            // materialisation.
            let keys = DekoBufLineLender::from_path(filename)?.take(n);
            let func = <CompVFunc<str, u64, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                n,
                huffman,
                vbuilder,
                &mut pl,
            )?;
            maybe_store!(func, args.func);
        } else {
            // Parallel: read all keys into a single concatenated
            // buffer, then build a `Vec<&str>` of slices into it.
            // This gives cache-friendly access during sig hashing
            // (one big allocation, fixed-size &str references) vs
            // `Vec<String>` which would do `n` independent heap
            // allocations.
            let (buffer, offsets) = read_lines_concatenated(filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            if keys.len() != n {
                bail!("key count mismatch: read {} keys, expected {n}", keys.len());
            }
            let func = <CompVFunc<str, u64, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, huffman, vbuilder, &mut pl,
            )?;
            maybe_store!(func, args.func);
        }
    } else {
        let n = args.n.unwrap();
        if n != n_values {
            bail!("n={n} but the values file has {n_values} entries");
        }
        if args.sequential {
            // Sequential: wrap `0..n` as a lender so we avoid
            // materialising `n * 8` bytes of keys.
            let keys = FromCloneableIntoIterator::from(0_usize..n);
            let func = <CompVFunc<usize, u64, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                n,
                huffman,
                vbuilder,
                &mut pl,
            )?;
            maybe_store!(func, args.func);
        } else {
            // Parallel: materialise keys as `Vec<usize>`. Costs
            // `n * 8` bytes of memory but lets the sig-hashing
            // phase run on all cores.
            let keys: Vec<usize> = (0..n).collect();
            let func =
                <CompVFunc<usize, u64, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                    &keys, &values, huffman, vbuilder, &mut pl,
                )?;
            maybe_store!(func, args.func);
        }
    }

    Ok(())
}
