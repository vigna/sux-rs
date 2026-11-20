/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use anyhow::{Result, bail};
use clap::{ArgGroup, Parser};
use lender::FallibleLender;
use sux::bits::BitFieldVec;
use sux::cli::{
    AnyLineLender, Decompression, KeyBuffer, PHastRArgs, check_num_keys, init_env_logger, store,
};
use sux::func::{PHastR, PHastRBuilder, PHunc};
use sux::utils::FromCloneableIntoIterator;

#[derive(Parser, Debug)]
#[command(about = "Creates a PHunc mapping each key to its index in the input (or, with --bits, to a pseudorandom value) and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .multiple(true)
                .args(["filename", "n"]),
))]
struct Args {
    /// The number of keys: if a file of keys is specified, its first N keys
    /// are used, and it is an error if it contains fewer keys; otherwise, the
    /// 64-bit keys [0 . . N) are used.​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line.​
    #[arg(short, long)]
    filename: Option<String>,
    /// Decompress the file of keys using this format (by default, it is read
    /// as is).​
    #[arg(long, value_enum)]
    decompress: Option<Decompression>,
    /// Save the structure in unaligned form (faster).​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function (if not specified, the
    /// function is built but not saved).​
    func: Option<String>,
    /// Hashes keys sequentially without loading them in RAM.​
    #[arg(short, long)]
    sequential: bool,
    /// Map each key to a pseudorandom value of this number of bits (a
    /// function of the index of the key in the input), in place of the
    /// index: 8, 16, 32, and 64 bits use a boxed slice, other widths a
    /// bit-field vector.​
    #[arg(short, long)]
    bits: Option<usize>,
    #[clap(flatten)]
    phastr: PHastRArgs,
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

/// Returns a pseudorandom 64-bit value associated with an index (a
/// SplitMix64 step).
fn hash(index: usize) -> u64 {
    let mut z = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Builds and stores a function with seeds stored in `$seeds` and values
/// stored in `$values`; `$value` maps the index of a key in the input to its
/// value.
macro_rules! build {
    ($args:expr, $builder:expr, $seeds:ty, $values:ty, $value:expr) => {{
        let args: &Args = $args;
        let builder: PHastRBuilder = $builder;
        let value = $value;
        let mut pl = args.log.progress_logger();

        if let Some(filename) = &args.filename {
            if args.sequential {
                let func = <PHunc<PHastR<str, $seeds>, $values>>::try_new_with_builder(
                    AnyLineLender::from_path(filename, args.decompress)?
                        .take(args.n.unwrap_or(usize::MAX)),
                    FromCloneableIntoIterator::from((0_usize..).map(value)),
                    builder,
                    &mut pl,
                )?;
                check_num_keys(func.len(), args.n)?;
                unsafe { store(func, args.func.as_ref(), args.unaligned) }
            } else {
                let buffer = KeyBuffer::from_path(filename, args.n, args.decompress)?;
                let keys = buffer.keys();
                let values: Vec<_> = (0..keys.len()).map(value).collect();
                let func = <PHunc<PHastR<str, $seeds>, $values>>::try_par_new_with_builder(
                    &keys, &values, builder, &mut pl,
                )?;
                unsafe { store(func, args.func.as_ref(), args.unaligned) }
            }
        } else {
            let n = args.n.unwrap();
            if args.sequential {
                let func = <PHunc<PHastR<usize, $seeds>, $values>>::try_new_with_builder(
                    FromCloneableIntoIterator::from(0_usize..n),
                    FromCloneableIntoIterator::from((0_usize..).map(value)),
                    builder,
                    &mut pl,
                )?;
                unsafe { store(func, args.func.as_ref(), args.unaligned) }
            } else {
                let keys: Vec<usize> = (0..n).collect();
                let values: Vec<_> = (0..n).map(value).collect();
                let func = <PHunc<PHastR<usize, $seeds>, $values>>::try_par_new_with_builder(
                    &keys, &values, builder, &mut pl,
                )?;
                unsafe { store(func, args.func.as_ref(), args.unaligned) }
            }
        }
    }};
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    let builder = args.phastr.builder(args.n);

    // The default number of bits of a seed is at most 8, so seeds are
    // stored in bytes unless more bits are requested
    macro_rules! build_with_values {
        ($values:ty, $value:expr) => {
            if args.phastr.wide_seeds() {
                build!(
                    &args,
                    builder.clone(),
                    BitFieldVec<Box<[usize]>>,
                    $values,
                    $value
                )
            } else {
                build!(&args, builder.clone(), Box<[u8]>, $values, $value)
            }
        };
    }
    let build = || -> Result<()> {
        match args.bits {
            None => build_with_values!(BitFieldVec<Box<[usize]>>, |i: usize| i),
            Some(8) => build_with_values!(Box<[u8]>, |i: usize| hash(i) as u8),
            Some(16) => build_with_values!(Box<[u16]>, |i: usize| hash(i) as u16),
            Some(32) => build_with_values!(Box<[u32]>, |i: usize| hash(i) as u32),
            Some(64) => build_with_values!(Box<[u64]>, hash),
            Some(bits) if (1..usize::BITS as usize).contains(&bits) => {
                // The bit width is that of the largest value, so it is the
                // requested one, unless there are very few keys
                let mask = (1_usize << bits) - 1;
                build_with_values!(BitFieldVec<Box<[usize]>>, move |i: usize| hash(i) as usize
                    & mask)
            }
            Some(bits) => bail!("Unsupported number of bits: {bits}"),
        }
    };
    args.phastr.install(build)
}
