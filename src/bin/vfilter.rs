/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
#![allow(clippy::collapsible_else_if)]
use anyhow::Result;
use clap::{ArgGroup, Parser};
use common_traits::CastableFrom;
use dsi_progress_logger::*;
use epserde::ser::{Serialize, SerializeInner};
use epserde::traits::{AlignHash, TypeHash, ZeroCopy};
use lender::Lender;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::func::*;
use sux::prelude::VBuilder;
use sux::traits::{BitFieldSlice, BitFieldSliceMut, Word};
use sux::utils::{FromIntoIterator, LineLender, Sig, SigVal, ToSig, ZstdLineLender};

#[derive(Parser, Debug)]
#[command(about = "Generates a VFilter and serializes it with ε-serde", long_about = None)]
#[clap(group(
            ArgGroup::new("input")
                .required(true)
                .args(&["filename", "n"]),
))]
struct Args {
    /// The number of keys. If no filename is provided, use the 64-bit keys
    /// [0..n).
    n: usize,
    /// An optional name for the ε-serde serialized function.
    func: Option<String>,
    /// The number of bits of the signatures.
    #[arg(short, long, default_value_t = 8)]
    bits: u32,
    #[arg(short, long)]
    /// A file containing UTF-8 keys, one per line. At most N keys will be read.
    filename: Option<String>,
    /// Use this number of threads.
    #[arg(short, long)]
    threads: Option<usize>,
    /// Whether the file is compressed with zstd.
    #[arg(short, long)]
    zstd: bool,
    /// Use disk-based buckets to reduce memory usage at construction time.
    #[arg(short, long)]
    offline: bool,
    /// Sort shards and check for duplicate signatures.
    #[arg(short, long)]
    check_dups: bool,
    /// A 64-bit seed for the pseudorandom number generator.
    #[arg(long)]
    seed: Option<u64>,
    /// Use 64-bit signatures.
    #[arg(long)]
    sig64: bool,
    /// Do not use sharding.
    #[arg(long)]
    no_shards: bool,
    /// Use 3-hypergraph.
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with = "sig64")]
    mwhc: bool,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    #[cfg(feature = "mwhc")]
    if args.mwhc {
        return match args.bits {
            8 => {
                if args.no_shards {
                    main_with_types::<u8, [u64; 2], Mwhc3NoShards>(args)
                } else {
                    main_with_types::<u8, [u64; 2], Mwhc3Shards>(args)
                }
            }
            16 => {
                if args.no_shards {
                    main_with_types::<u16, [u64; 2], Mwhc3NoShards>(args)
                } else {
                    main_with_types::<u16, [u64; 2], Mwhc3Shards>(args)
                }
            }
            32 => {
                if args.no_shards {
                    main_with_types::<u32, [u64; 2], Mwhc3NoShards>(args)
                } else {
                    main_with_types::<u32, [u64; 2], Mwhc3Shards>(args)
                }
            }
            _ => unimplemented!("{}-bit signatures", args.bits),
        };
    }

    match args.bits {
        8 => {
            if args.no_shards {
                if args.sig64 {
                    main_with_types::<u8, [u64; 1], Fuse3NoShards>(args)
                } else {
                    main_with_types::<u8, [u64; 2], Fuse3NoShards>(args)
                }
            } else {
                main_with_types::<u8, [u64; 2], Fuse3Shards>(args)
            }
        }
        16 => {
            if args.no_shards {
                if args.sig64 {
                    main_with_types::<u16, [u64; 1], Fuse3NoShards>(args)
                } else {
                    main_with_types::<u16, [u64; 2], Fuse3NoShards>(args)
                }
            } else {
                main_with_types::<u16, [u64; 2], Fuse3Shards>(args)
            }
        }
        32 => {
            if args.no_shards {
                if args.sig64 {
                    main_with_types::<u32, [u64; 1], Fuse3NoShards>(args)
                } else {
                    main_with_types::<u32, [u64; 2], Fuse3NoShards>(args)
                }
            } else {
                main_with_types::<u32, [u64; 2], Fuse3Shards>(args)
            }
        }
        _ => unimplemented!("{}-bit signatures", args.bits),
    }
}

fn set_builder<W: ZeroCopy + Word, D: BitFieldSlice<W> + Send + Sync, S, E: ShardEdge<S, 3>>(
    builder: VBuilder<W, D, S, E>,
    args: &Args,
) -> VBuilder<W, D, S, E> {
    let mut builder = builder
        .offline(args.offline)
        .check_dups(args.check_dups)
        .expected_num_keys(args.n);
    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }
    if let Some(threads) = args.threads {
        builder = builder.max_num_threads(threads);
    }
    builder
}

fn main_with_types<W: Word + ZeroCopy + Send + Sync, S: Sig + Send + Sync, E: ShardEdge<S, 3>>(
    args: Args,
) -> Result<()>
where
    W: CastableFrom<u64> + SerializeInner + TypeHash + AlignHash,
    <W as SerializeInner>::SerType: Word + ZeroCopy,
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, ()>: RadixKey,
    Box<[W]>: BitFieldSlice<W> + BitFieldSliceMut<W>,
    for<'a> <Box<[W]> as BitFieldSliceMut<W>>::ChunksMut<'a>: Send,
    for<'a> <<Box<[W]> as BitFieldSliceMut<W>>::ChunksMut<'a> as Iterator>::Item: Send,
    <E as SerializeInner>::SerType: ShardEdge<S, 3>, // Wierd
    VFunc<usize, usize, BitFieldVec, S, E>: Serialize,
    VFunc<str, usize, BitFieldVec, S, E>: Serialize,
    VFunc<usize, W, Box<[W]>, S, E>: Serialize,
    VFunc<str, W, Box<[W]>, S, E>: Serialize + TypeHash, // TODO: this is weird
    VFilter<W, VFunc<usize, W, Box<[W]>, S, E>>: Serialize,
    VFilter<W, VFunc<usize, W, Box<[W]>, S, <E as SerializeInner>::SerType>>: TypeHash,
    VFilter<
        <W as SerializeInner>::SerType,
        VFunc<str, W, Box<[W]>, S, <E as SerializeInner>::SerType>,
    >: TypeHash + AlignHash, // Weird
{
    let mut pl = ProgressLogger::default();
    let n = args.n;

    if let Some(ref filename) = &args.filename {
        let builder = set_builder(VBuilder::<W, Box<[W]>, S, E>::default(), &args);
        let filter = if args.zstd {
            builder.try_build_filter(ZstdLineLender::from_path(filename)?.take(n), &mut pl)?
        } else {
            let t = LineLender::from_path(filename)?.take(n);
            builder.try_build_filter(t, &mut pl)?
        };
        if let Some(func) = args.func {
            filter.store(func)?;
        }
    } else {
        let builder = set_builder(VBuilder::<W, Box<[W]>, S, E>::default(), &args);
        let filter = builder.try_build_filter(FromIntoIterator::from(0_usize..n), &mut pl)?;
        if let Some(func) = args.func {
            filter.store(func)?;
        }
    }

    Ok(())
}
