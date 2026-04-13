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

#![allow(clippy::collapsible_else_if)]
use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser};
use epserde::ser::Serialize;
use lender::FallibleLender;
use rdst::RadixKey;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sux::bits::{BitFieldVec, BitVec};
use sux::cli::{BuilderArgs, ShardingArgs};
use sux::func::codec::Huffman;
use sux::func::shard_edge::{FuseLge3NoShards, FuseLge3Shards, ShardEdge};
use sux::func::{CompVFunc, VBuilder};
use sux::init_env_logger;
use sux::utils::{
    DekoBufLineLender, FromCloneableIntoIterator, Sig, SigVal, ToSig,
};

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
    #[arg(long)]
    values: String,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
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
    let vbuilder: VBuilder<BitFieldVec<Box<[usize]>>, S, E> =
        args.builder.configure(VBuilder::default());
    let huffman = Huffman::new();

    let values = read_values(&args.values)?;
    let n_values = values.len();

    if let Some(filename) = &args.filename {
        // Stream keys directly from disk: `DekoBufLineLender` is a
        // fallible rewindable lender of `&str`, so keys never all
        // live in memory at once. The ε-serde-side `n` equals the
        // values count (the `--values` file is compulsory and must
        // match the key count).
        let n = n_values;
        let keys = DekoBufLineLender::from_path(filename)?.take(n);
        let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
            keys, &values, n, huffman, vbuilder,
        )?;
        if let Some(out) = args.func {
            unsafe { func.store(&out)? };
        }
    } else {
        let n = args.n.unwrap();
        if n != n_values {
            bail!("n={n} but the values file has {n_values} entries");
        }
        // Stream the integer range through a lender instead of
        // materializing a `Vec<usize>`.
        let keys = FromCloneableIntoIterator::from(0_usize..n);
        let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
            keys, &values, n, huffman, vbuilder,
        )?;
        if let Some(out) = args.func {
            unsafe { func.store(&out)? };
        }
    }

    Ok(())
}
