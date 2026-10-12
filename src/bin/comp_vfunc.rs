/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use anyhow::{Result, bail, ensure};
use clap::{ArgGroup, Parser};
use epserde::ser::Serialize;
use lender::FallibleLender;
use mem_dbg::{FlatType, MemSize};
use rdst::RadixKey;
use sux::bits::BitVec;
use sux::cli::{
    AnyLineLender, BuilderArgs, Decompression, KeyBuffer, ValueArgs, check_num_keys, count_lines,
    init_env_logger, store,
};
use sux::func::codec::HuffmanConf;
use sux::func::shard_edge::{Fuse3NoShards, Fuse3Shards, ShardEdge};
use sux::func::{CompVFunc, VBuilder};
use sux::traits::TryIntoUnaligned;
use sux::utils::{
    FromCloneableIntoIterator, FromIntoFallibleLenderFactory, FromSlice, Sig, SigVal, ToSig,
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
        .args(["filename", "n"]),
))]
struct Args {
    /// The number of keys: if a file of keys is specified, its first N keys
    /// are used, and it is an error if it contains fewer keys; otherwise, the
    /// 64-bit keys [0 . . N) are used.​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line.​
    #[arg(short, long)]
    filename: Option<String>,
    /// Decompress the file of keys using this format (by default, it is read
    /// as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
    #[clap(flatten)]
    values: ValueArgs,
    /// Save the structure in unaligned form (faster, if available).​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function (if not specified, the
    /// function is built but not saved).​
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
    /// Shard/edge type: CompVFunc does not support lazy Gaussian elimination,
    /// so only fuse3-shards (the default, unlike in the other binaries),
    /// fuse3-no-shards64, and fuse3-no-shards128 are supported.​
    #[arg(long, short = 'E', value_enum, default_value_t = sux::cli::ShardEdgeType::Fuse3Shards)]
    shard_edge: sux::cli::ShardEdgeType,
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

/// Validates the Huffman entropy threshold: a finite fraction in `[0.0, 1.0]`.
///
/// `RangeInclusive::contains` is false for NaN and infinities, so this rejects
/// those as well as out-of-range values.
fn validate_entropy_threshold(threshold: f64) -> Result<()> {
    ensure!(
        (0.0..=1.0).contains(&threshold),
        "--huffman-entropy-threshold must be a finite fraction in [0.0, 1.0], got {threshold}"
    );
    Ok(())
}

/// Validates arguments whose constraints clap's type system cannot enforce.
fn validate_args(args: &Args) -> Result<()> {
    validate_entropy_threshold(args.huffman_entropy_threshold)
}

fn main() -> Result<()> {
    use sux::cli::ShardEdgeType;
    init_env_logger()?;

    let args = Args::parse();
    validate_args(&args)?;
    match args.shard_edge {
        ShardEdgeType::Fuse3NoShards64 => main_with_types::<[u64; 1], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3NoShards128 => main_with_types::<[u64; 2], Fuse3NoShards>(args),
        ShardEdgeType::Fuse3Shards => main_with_types::<[u64; 2], Fuse3Shards>(args),
        _ => {
            bail!(
                "comp_vfunc only supports --shard-edge fuse3-shards, fuse3-no-shards64, and fuse3-no-shards128"
            )
        }
    }
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

    let mut pl = args.log.progress_logger();

    if let Some(filename) = &args.filename {
        let n = match args.n {
            // Generated values need the number of keys
            None if args.values.file.is_none() => Some(count_lines(filename, args.decompress)?),
            n => n,
        };
        let values = args.values.values(n)?;
        let n = values.len();
        args.values.check_keys(filename, args.decompress, n)?;
        if args.sequential {
            let keys = FromIntoFallibleLenderFactory::new(|| {
                AnyLineLender::from_path(filename, args.decompress).map(|keys| keys.take(n))
            })?;
            let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                huffman,
                vbuilder,
                &mut pl,
            )?;
            check_num_keys(func.len(), Some(n))?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let buffer = KeyBuffer::from_path(filename, Some(n), args.decompress)?;
            let func = <CompVFunc<str, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &buffer.keys(),
                &values,
                huffman,
                vbuilder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    } else {
        let n = args.n.unwrap();
        let values = args.values.values(Some(n))?;
        if args.sequential {
            let keys = FromCloneableIntoIterator::from(0_usize..n);
            let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                keys,
                FromSlice::new(&values),
                huffman,
                vbuilder,
                &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <CompVFunc<usize, BitVec<Box<[usize]>>, S, E>>::try_par_new_with_builder(
                &keys, &values, huffman, vbuilder, &mut pl,
            )?;
            unsafe { store(func, args.func.as_ref(), args.unaligned) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_entropy_threshold;

    #[test]
    fn entropy_threshold_is_validated() {
        for ok in [0.0, 0.5, 1.0] {
            assert!(
                validate_entropy_threshold(ok).is_ok(),
                "{ok} should be valid"
            );
        }
        for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                validate_entropy_threshold(bad).is_err(),
                "{bad} should be rejected"
            );
        }
    }
}
