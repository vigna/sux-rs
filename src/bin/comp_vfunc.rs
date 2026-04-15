/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Builds a [`CompVFunc`](sux::func::CompVFunc) and serializes it with
//! ε-serde.
//!
//! Mirrors the `vfunc` binary's parameter surface, minus the two-step
//! variant (CompVFunc is already the "compressed" two-step analog). A
//! compulsory `--values` file supplies the per-key values as ASCII
//! decimal integers, one per line. Internal output type is `u64`
//! (which on 64-bit targets is identical to `usize`); see the
//! `--values` documentation for the ASCII integer format.
//!
//! # Key-loading strategy
//!
//! By default the CLI runs the *parallel* construction path, which
//! requires the keys to be available as a `&[B]` slice. For
//! integer-range mode (`--n`) we just collect `0..n` into a
//! `Vec<usize>`. For filename mode we read the entire keys file into
//! a single concatenated `String` buffer and build a `Vec<&str>`
//! whose elements point into that buffer — far more cache-friendly
//! than `Vec<String>` (one allocation plus fixed-size references,
//! instead of `n` independent heap allocations).
//!
//! The `--sequential` flag switches to the streaming lender path,
//! which hashes signatures on a single thread but avoids
//! materialising all keys in memory at once. This is slower but
//! useful when the key set doesn't fit in RAM.
//!
//! # Test-speed note
//!
//! For benchmarking build speed, the default (parallel, in-memory)
//! path is the intended one — it matches what the `bench_comp_vfunc`
//! and `profile_comp_vfunc` examples use.

#![allow(clippy::collapsible_else_if)]
use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser};
use epserde::ser::Serialize;
use lender::FallibleLender;
use rdst::RadixKey;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sux::bits::BitVec;
use sux::cli::{BuilderArgs, ShardingArgs};
use sux::func::codec::Huffman;
use sux::func::shard_edge::{FuseLge3NoShards, FuseLge3Shards, ShardEdge};
use sux::func::{CompVFunc, VBuilder};
use sux::init_env_logger;
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
    /// [0 . . n).​
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
    #[arg(long)]
    values: String,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Use the single-threaded *sequential* construction path that
    /// streams keys through a lender instead of materialising them
    /// in memory. Slower for large builds (sig-store population is
    /// the bottleneck) but useful when keys don't fit in RAM. The
    /// default is the parallel, in-memory path.​
    #[arg(long)]
    sequential: bool,
    /// Cap on the number of distinct codeword lengths in the Huffman
    /// decoding table. Rare symbols beyond the cap are diverted to the
    /// escape codeword and stored as literals. Unlimited by default.​
    #[arg(long)]
    huffman_max_length: Option<usize>,
    /// Cumulative-entropy fraction beyond which infrequent symbols are
    /// diverted to the escape codeword. Truncates the Huffman table
    /// once the kept symbols cover this fraction of the total bit
    /// budget. Default `1.0` (no truncation).​
    #[arg(long, default_value_t = 1.0)]
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

/// Reads up to `n` lines from `filename` (optionally compressed) into
/// a single concatenated [`String`] buffer, returning the buffer
/// together with an `offsets` vector such that line `i` is
/// `&buffer[offsets[i] .. offsets[i+1]]`. The caller builds a
/// `Vec<&str>` from this with [`str_slice_from_offsets`] to get a
/// slice the parallel constructor can consume.
fn read_lines_concatenated(filename: &str, n: usize) -> Result<(String, Vec<usize>)> {
    let mut buffer = String::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(n + 1);
    offsets.push(0);
    let mut lender = DekoBufLineLender::from_path(filename)?;
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

#[inline]
fn str_slice_from_offsets<'a>(buffer: &'a str, offsets: &[usize]) -> Vec<&'a str> {
    offsets.windows(2).map(|w| &buffer[w[0]..w[1]]).collect()
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
    CompVFunc<str, BitVec<Box<[usize]>>, S, E>: Serialize,
    CompVFunc<usize, BitVec<Box<[usize]>>, S, E>: Serialize,
{
    let vbuilder: VBuilder<BitVec<Box<[usize]>>, S, E> =
        args.builder.configure(VBuilder::default());
    let huffman = Huffman::length_limited(
        args.huffman_max_length.unwrap_or(usize::MAX),
        args.huffman_entropy_threshold,
    );

    let values = read_values(&args.values)?;
    let n_values = values.len();

    // We time only the constructor call — file I/O, key
    // materialisation, and value parsing are deliberately excluded.
    // In realistic usage keys are streamed, so those phases don't
    // reflect the interesting cost.
    let t_build = std::time::Instant::now();

    if let Some(filename) = &args.filename {
        let n = n_values;
        if args.sequential {
            // Sequential: stream keys through the lender, no
            // materialisation.
            let keys = DekoBufLineLender::from_path(filename)?.take(n);
            let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys, &values, n, huffman, vbuilder,
            )?;
            eprintln!(
                "comp_vfunc: construction in {:.3}s",
                t_build.elapsed().as_secs_f64()
            );
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
            let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, huffman, vbuilder,
            )?;
            eprintln!(
                "comp_vfunc: construction in {:.3}s",
                t_build.elapsed().as_secs_f64()
            );
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
            let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys, &values, n, huffman, vbuilder,
            )?;
            eprintln!(
                "comp_vfunc: construction in {:.3}s",
                t_build.elapsed().as_secs_f64()
            );
            maybe_store!(func, args.func);
        } else {
            // Parallel: materialise keys as `Vec<usize>`. Costs
            // `n * 8` bytes of memory but lets the sig-hashing
            // phase run on all cores.
            let keys: Vec<usize> = (0..n).collect();
            let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, huffman, vbuilder,
            )?;
            maybe_store!(func, args.func);
        }
    }

    Ok(())
}
