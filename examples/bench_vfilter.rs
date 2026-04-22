/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::Parser;
use epserde::prelude::*;
use fallible_iterator::FallibleIterator;
use lender::*;
use sux::{
    bits::{BitFieldVec, BitFieldVecU},
    dict::VFilter,
    func::{shard_edge::*, *},
    traits::{BitFieldSlice, Word},
    utils::{BinSafe, LineLender, Sig, ToSig, ZstdLineLender},
};

#[cfg(target_pointer_width = "64")]
const INCR: usize = 0x9e3779b97f4a7c15;
#[cfg(not(target_pointer_width = "64"))]
const INCR: usize = 0x9e3779b9;

fn bench(n: usize, repeats: usize, mut f: impl FnMut()) {
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        f();
        timings.push(start.elapsed().as_nanos() as f64 / n as f64);
        eprintln!("{} ns/key", timings.last().unwrap());
    }
    timings.sort_unstable_by(|a, b| a.total_cmp(b));
    eprintln!(
        "Min: {} Median: {} Max: {} Average: {}",
        timings[0],
        timings[timings.len() / 2],
        timings.last().unwrap(),
        timings.iter().sum::<f64>() / timings.len() as f64
    );
}

#[derive(Parser, Debug)]
#[command(about = "Benchmarks VFilter with strings or 64-bit integers.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// The maximum number of strings to use from the file, or the number of 64-bit keys.​
    n: usize,
    /// A name for the ε-serde serialized filter.​
    func: String,
    /// The number of bits of the hashes used by the filter.​
    #[arg(short, long, default_value_t = 8)]
    bits: u32,
    #[arg(short = 'f', long)]
    /// A file containing UTF-8 keys, one per line. If not specified, the 64-bit keys [0..n) are used.​
    filename: Option<String>,
    /// The number of repetitions.​
    #[arg(short, long, default_value = "5")]
    repeats: usize,
    /// The input file is compressed with zstd.​
    #[arg(short, long)]
    zstd: bool,
    /// Use 64-bit signatures.​
    #[arg(long)]
    sig64: bool,
    /// Do not use sharding.​
    #[arg(long)]
    no_shards: bool,
    /// Use unaligned reads.​
    #[arg(long)]
    unaligned: bool,
    /// Use slower edge logic reducing the probability of duplicate arcs for big
    /// shards.​
    #[arg(long, conflicts_with_all = ["sig64", "no_shards"])]
    full_sigs: bool,
    /// Use 3-hypergraphs.​
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with_all = ["sig64", "full_sigs"])]
    mwhc: bool,
}

macro_rules! fuse {
    ($args: expr, $main: ident, $ty: ty) => {
        if $args.no_shards {
            if $args.sig64 {
                $main::<$ty, [u64; 1], FuseLge3NoShards>($args)
            } else {
                $main::<$ty, [u64; 2], FuseLge3NoShards>($args)
            }
        } else {
            if $args.full_sigs {
                $main::<$ty, [u64; 2], FuseLge3FullSigs>($args)
            } else {
                $main::<$ty, [u64; 2], FuseLge3Shards>($args)
            }
        }
    };
}

#[cfg(feature = "mwhc")]
macro_rules! mwhc {
    ($args: expr, $main: ident, $ty: ty) => {
        if $args.no_shards {
            $main::<$ty, [u64; 2], Mwhc3NoShards>($args)
        } else {
            $main::<$ty, [u64; 2], Mwhc3Shards>($args)
        }
    };
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    #[cfg(feature = "mwhc")]
    if args.mwhc {
        return match args.bits {
            8 => mwhc!(args, main_with_types_boxed_slice, u8),
            16 => mwhc!(args, main_with_types_boxed_slice, u16),
            32 => mwhc!(args, main_with_types_boxed_slice, u32),
            64 => mwhc!(args, main_with_types_boxed_slice, u64),
            _ => mwhc!(args, main_with_types_bit_field_vec, u64),
        };
    }

    match args.bits {
        8 => fuse!(args, main_with_types_boxed_slice, u8),
        16 => fuse!(args, main_with_types_boxed_slice, u16),
        32 => fuse!(args, main_with_types_boxed_slice, u32),
        64 => fuse!(args, main_with_types_boxed_slice, u64),
        _ => fuse!(args, main_with_types_bit_field_vec, u64),
    }
}

fn main_with_types_boxed_slice<
    W: Word + BinSafe + TypeHash + AlignHash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    u64: num_primitive::PrimitiveNumberAs<W>,
    Box<[W]>: BitFieldSlice<Value = W>,
    VFilter<VFunc<usize, Box<[W]>, S, E>>: Deserialize,
    VFilter<VFunc<str, Box<[W]>, S, E>>: Deserialize,
{
    if args.unaligned {
        panic!("Unaligned reads are not supported for backend Box<[W]>");
    }

    if let Some(filename) = args.filename {
        let keys: Vec<_> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        };

        let filter = unsafe { VFilter::<VFunc<str, Box<[W]>, S, E>>::load_full(&args.func) }?;

        bench(args.n, args.repeats, || {
            for key in &keys {
                std::hint::black_box(filter.contains(key.as_str()));
            }
        });
    } else {
        // No filename
        let filter = unsafe { VFilter::<VFunc<usize, Box<[W]>, S, E>>::load_full(&args.func) }?;

        bench(args.n, args.repeats, || {
            let mut key: usize = 0;
            for _ in 0..args.n {
                key = key.wrapping_add(INCR);
                std::hint::black_box(filter.contains(key));
            }
        });
    }
    Ok(())
}

fn main_with_types_bit_field_vec<
    W: Word + BinSafe + TypeHash + AlignHash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    args: Args,
) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    u64: num_primitive::PrimitiveNumberAs<W>,
    VFilter<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>: Deserialize,
    VFilter<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>: Deserialize,
    VFilter<VFunc<usize, BitFieldVecU<Box<[W]>>, S, E>>: Deserialize,
    VFilter<VFunc<str, BitFieldVecU<Box<[W]>>, S, E>>: Deserialize,
{
    if let Some(filename) = args.filename {
        let keys: Vec<_> = if args.zstd {
            ZstdLineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        } else {
            LineLender::from_path(filename)?
                .map_into_iter(|x| Ok(x.to_owned()))
                .take(args.n)
                .collect()?
        };

        if args.unaligned {
            let filter = unsafe {
                VFilter::<VFunc<str, BitFieldVecU<Box<[W]>>, S, E>>::load_full(&args.func)
            }?;
            bench(args.n, args.repeats, || {
                for key in &keys {
                    std::hint::black_box(filter.contains(key.as_str()));
                }
            });
        } else {
            let filter = unsafe {
                VFilter::<VFunc<str, BitFieldVec<Box<[W]>>, S, E>>::load_full(&args.func)
            }?;
            bench(args.n, args.repeats, || {
                for key in &keys {
                    std::hint::black_box(filter.contains(key.as_str()));
                }
            });
        }
    } else {
        if args.unaligned {
            let filter = unsafe {
                VFilter::<VFunc<usize, BitFieldVecU<Box<[W]>>, S, E>>::load_full(&args.func)
            }?;
            bench(args.n, args.repeats, || {
                let mut key: usize = 0;
                for _ in 0..args.n {
                    key = key.wrapping_add(INCR);
                    std::hint::black_box(filter.contains(key));
                }
            });
        } else {
            let filter = unsafe {
                VFilter::<VFunc<usize, BitFieldVec<Box<[W]>>, S, E>>::load_full(&args.func)
            }?;
            bench(args.n, args.repeats, || {
                let mut key: usize = 0;
                for _ in 0..args.n {
                    key = key.wrapping_add(INCR);
                    std::hint::black_box(filter.contains(key));
                }
            });
        }
    }
    Ok(())
}
