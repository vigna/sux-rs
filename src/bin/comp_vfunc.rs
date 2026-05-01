/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use mem_dbg::{FlatType, MemSize};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rdst::RadixKey;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sux::bits::BitVec;
use sux::cli::{BuilderArgs, read_lines_concatenated, str_slice_from_offsets};
use sux::func::codec::HuffmanConf;
use sux::func::shard_edge::{Fuse3NoShards, Fuse3Shards, ShardEdge};
use sux::func::{CompVFunc, VBuilder};
use sux::init_env_logger;
use sux::traits::TryIntoUnaligned;
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
        .args(["filename", "n"]),
))]
#[clap(group(
    ArgGroup::new("value_source")
        .required(true)
        .args(["values", "geometric", "zipf", "uniform"]),
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
    /// line. The number of values must match the number of keys.​
    #[arg(short, long)]
    values: Option<String>,
    /// Generate values with a geometric distribution (trailing zeros
    /// of a random u64).​
    #[arg(long)]
    geometric: bool,
    /// Generate values with a Zipf distribution (exponent 1) over
    /// [0 . . K).​
    #[arg(long)]
    zipf: Option<usize>,
    /// Generate values uniformly distributed in [0 . . K).​
    #[arg(long)]
    uniform: Option<usize>,
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Hashes keys sequentially without loading them in RAM.​
    #[arg(short, long)]
    sequential: bool,
    /// Cap on the number of distinct codeword lengths in the Huffman decoding
    /// table, which is the main factor in decoding speed. Rare symbols beyond
    /// the cap are diverted to the escape codeword and stored as literals.​
    #[arg(long, default_value_t = 20)]
    huffman_max_table_len: usize,
    /// Cumulative-entropy fraction beyond which infrequent symbols are
    /// diverted to the escape codeword. Truncates the Huffman table
    /// once the kept symbols cover this fraction of the total bit
    /// budget.​
    #[arg(long, default_value_t = 0.9)]
    huffman_entropy_threshold: f64,
    #[clap(flatten)]
    builder: BuilderArgs,
    /// Shard/edge type (CompVFunc does not support LGE variants).​
    #[arg(long, short = 'E', value_enum, default_value_t = sux::cli::ShardEdgeType::Fuse3Shards)]
    shard_edge: sux::cli::ShardEdgeType,
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

fn read_values(path: &str) -> Result<Vec<usize>> {
    let file = File::open(path).with_context(|| format!("open values file {path}"))?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: usize = trimmed.parse().with_context(|| {
            format!(
                "values file {path}, line {}: cannot parse `{trimmed}`",
                lineno + 1
            )
        })?;
        values.push(v);
    }
    Ok(values)
}

/// Optionally convert to unaligned form and store the built function
/// to `out`, if provided. Extracted so the four branches below each
/// call it once at their tail.
macro_rules! maybe_store {
    ($func:expr, $out:expr, $unaligned:expr) => {
        if let Some(out) = $out {
            if $unaligned {
                unsafe { $func.try_into_unaligned().unwrap().store(&out)? };
            } else {
                unsafe { $func.store(&out)? };
            }
        }
    };
}

fn main() -> Result<()> {
    use sux::cli::ShardEdgeType;
    init_env_logger()?;

    let args = Args::parse();
    match args.shard_edge {
        ShardEdgeType::Fuse3NoShards64 => main_with_types::<[u64; 1], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3NoShards128 => main_with_types::<[u64; 2], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3Shards => main_with_types::<[u64; 2], Fuse3Shards>(args),
        _ => {
            bail!("comp_vfunc only supports --edge fuse, fuse-no-shards-64, and fuse-no-shards-128")
        }
    }
}

fn zipf_cdf(s: f64, n: usize) -> Vec<f64> {
    let h: f64 = (1..=n).map(|i| 1.0 / (i as f64).powf(s)).sum();
    let mut cdf = vec![0.0; n];
    let mut acc = 0.0;
    for i in 1..=n {
        acc += 1.0 / (i as f64).powf(s) / h;
        cdf[i - 1] = acc;
    }
    cdf
}

fn sample_zipf(cdf: &[f64], rng: &mut SmallRng) -> usize {
    let u: f64 = (rng.random::<u64>() >> 11) as f64 / ((1u64 << 53) as f64);
    cdf.partition_point(|&p| p < u)
}

fn generate_synthetic_values(args: &Args, n: usize) -> Result<Vec<usize>> {
    let mut rng = SmallRng::seed_from_u64(0);
    if args.geometric {
        Ok((0..n)
            .map(|_| rng.random::<u64>().trailing_zeros() as usize)
            .collect())
    } else if let Some(k) = args.zipf {
        let cdf = zipf_cdf(1.0, k);
        Ok((0..n).map(|_| sample_zipf(&cdf, &mut rng)).collect())
    } else if let Some(k) = args.uniform {
        Ok((0..n)
            .map(|_| (rng.random::<u64>() % k as u64) as usize)
            .collect())
    } else {
        bail!("one of --values, --geometric, --zipf, or --uniform is required");
    }
}

fn count_keys(filename: &str) -> Result<usize> {
    let mut lender = DekoBufLineLender::from_path(filename)?;
    let mut count = 0;
    while lender.next()?.is_some() {
        count += 1;
    }
    Ok(count)
}

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    CompVFunc<str, BitVec<Box<[usize]>>, S, E>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    CompVFunc<usize, BitVec<Box<[usize]>>, S, E>:
        Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let vbuilder: VBuilder<BitVec<Box<[usize]>>, S, E> =
        args.builder.configure(VBuilder::default());
    let huffman =
        HuffmanConf::length_limited(args.huffman_max_table_len, args.huffman_entropy_threshold);

    let mut pl = ProgressLogger::default();
    pl.log_interval(args.log.log_interval);

    if let Some(filename) = &args.filename {
        let values: Vec<usize> = if let Some(ref path) = args.values {
            read_values(path)?
        } else {
            let n = match args.n {
                Some(n) => n,
                None => count_keys(filename)?,
            };
            generate_synthetic_values(&args, n)?
        };
        let n = values.len();
        if args.sequential {
            let keys = DekoBufLineLender::from_path(filename)?.take(n);
            let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                huffman,
                vbuilder,
                &mut pl,
            )?;
            maybe_store!(func, args.func, args.unaligned);
        } else {
            let (buffer, offsets) = read_lines_concatenated(filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            if keys.len() != n {
                bail!("key count mismatch: read {} keys, expected {n}", keys.len());
            }
            let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, huffman, vbuilder, &mut pl,
            )?;
            maybe_store!(func, args.func, args.unaligned);
        }
    } else {
        let n = args.n.unwrap();
        let values: Vec<usize> = if let Some(ref path) = args.values {
            let v = read_values(path)?;
            ensure!(v.len() == n, "n={n} but values file has {} values", v.len());
            v
        } else {
            generate_synthetic_values(&args, n)?
        };
        if args.sequential {
            let keys = FromCloneableIntoIterator::from(0_usize..n);
            let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                huffman,
                vbuilder,
                &mut pl,
            )?;
            maybe_store!(func, args.func, args.unaligned);
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, huffman, vbuilder, &mut pl,
            )?;
            maybe_store!(func, args.func, args.unaligned);
        }
    }

    Ok(())
}
