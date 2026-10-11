/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use anyhow::{Result, bail};
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use sux::bits::BitFieldVec;
use sux::cli::{read_concat_lines, str_slice_from_offsets};
use sux::func::phast_r::SeedStoreBuild;
use sux::func::{PHastR, PHastRBuilder};
use sux::init_env_logger;
use sux::traits::TryIntoUnaligned;
use sux::utils::{DekoBufLineLender, FromCloneableIntoIterator};

#[derive(Parser, Debug)]
#[command(about = "Creates a PHastR minimal perfect hash function on a set of keys and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .multiple(true)
                .args(["filename", "n"]),
))]
struct Args {
    /// The number of keys; if no filename is provided, use the 64-bit keys
    /// [0 . . n).​
    #[arg(short, long)]
    n: Option<usize>,
    /// A file containing UTF-8 keys, one per line (at most N keys will be read); it can be compressed with any format supported by the deko crate.​
    #[arg(short, long)]
    filename: Option<String>,
    /// Save the structure in unaligned form (faster).​
    #[arg(long, short)]
    unaligned: bool,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Hashes keys sequentially without loading them in RAM.​
    #[arg(short, long)]
    sequential: bool,
    /// Keep the hashes of the keys in temporary files in the default
    /// temporary directory (TMPDIR), which needs about sixteen bytes of free
    /// space per key. Requires --sequential (-s).​
    #[arg(short, long, requires = "sequential")]
    offline: bool,
    /// Use this number of threads.​
    #[arg(short, long)]
    threads: Option<usize>,
    /// The number of bits of a seed [default: 8]; seeds of more than 8 bits
    /// are stored in a bit-field vector.​
    #[arg(long)]
    seed_bits: Option<u32>,
    /// The base-2 logarithm of the number of layouts [default: 2].​
    #[arg(long)]
    log2_layouts: Option<u32>,
    /// The base-2 logarithm of the slice length [default: 10].​
    #[arg(long)]
    log2_slice_len: Option<u32>,
    /// The expected number of keys per bucket [default: 4.25].​
    #[arg(long)]
    bucket_size: Option<f64>,
    /// A 64-bit seed for the hash function.​
    #[arg(long)]
    seed: Option<u64>,
    #[clap(flatten)]
    log: sux::cli::LogIntervalArg,
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    let mut builder = PHastRBuilder::default().offline(args.offline);
    if let Some(seed_bits) = args.seed_bits {
        builder = builder.seed_bits(seed_bits);
    }
    if let Some(log2_layouts) = args.log2_layouts {
        builder = builder.log2_layouts(log2_layouts);
    }
    if let Some(log2_slice_len) = args.log2_slice_len {
        builder = builder.log2_slice_len(log2_slice_len);
    }
    if let Some(bucket_size) = args.bucket_size {
        builder = builder.bucket_size(bucket_size);
    }
    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }

    // The default number of bits of a seed is at most 8, so seeds are
    // stored in bytes unless more bits are requested
    let build = || {
        if args.seed_bits.is_some_and(|b| b > 8) {
            main_with_types::<BitFieldVec<Box<[usize]>>>(&args, builder.clone())
        } else {
            main_with_types::<Box<[u8]>>(&args, builder.clone())
        }
    };
    match args.threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?
            .install(build),
        None => build(),
    }
}

/// Stores a function, if a name was provided, possibly in unaligned form.
fn store<F: Serialize + TryIntoUnaligned<Unaligned: Serialize>>(
    func: F,
    filename: &Option<String>,
    unaligned: bool,
) -> Result<()> {
    if let Some(filename) = filename {
        if unaligned {
            unsafe { func.try_into_unaligned()?.store(filename) }?;
        } else {
            unsafe { func.store(filename) }?;
        }
    }
    Ok(())
}

fn main_with_types<D: SeedStoreBuild + TryIntoUnaligned + Send + Sync>(
    args: &Args,
    builder: PHastRBuilder,
) -> Result<()>
where
    PHastR<str, D>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
    PHastR<usize, D>: Serialize + TryIntoUnaligned<Unaligned: Serialize>,
{
    let mut pl = ProgressLogger::default();
    pl.log_interval(args.log.log_interval);

    if let Some(filename) = &args.filename {
        let n = args.n.unwrap_or(usize::MAX);
        if args.sequential {
            let func = <PHastR<str, D>>::try_new_with_builder(
                DekoBufLineLender::from_path(filename)?.take(n),
                builder,
                &mut pl,
            )?;
            store(func, &args.func, args.unaligned)
        } else {
            // Read all keys into a single concatenated buffer, then build
            // a Vec<&str> of slices into it for cache-friendly access
            // during hashing
            let (buffer, offsets) = read_concat_lines(filename, n)?;
            let keys = str_slice_from_offsets(&buffer, &offsets);
            if let Some(n_hint) = args.n {
                if keys.len() != n_hint {
                    bail!(
                        "key count mismatch: read {} keys, expected {n_hint}",
                        keys.len()
                    );
                }
            }
            let func = <PHastR<str, D>>::try_par_new_with_builder(&keys, builder, &mut pl)?;
            store(func, &args.func, args.unaligned)
        }
    } else {
        let n = args.n.unwrap();
        if args.sequential {
            let func = <PHastR<usize, D>>::try_new_with_builder(
                FromCloneableIntoIterator::new(0..n),
                builder,
                &mut pl,
            )?;
            store(func, &args.func, args.unaligned)
        } else {
            let keys: Vec<usize> = (0..n).collect();
            let func = <PHastR<usize, D>>::try_par_new_with_builder(&keys, builder, &mut pl)?;
            store(func, &args.func, args.unaligned)
        }
    }
}
