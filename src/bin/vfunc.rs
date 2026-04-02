/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::collapsible_else_if)]
use std::ops::{BitXor, BitXorAssign};

use anyhow::Result;
use clap::{ArgGroup, Parser};
use dsi_progress_logger::*;
use epserde::ser::Serialize;
use lender::FallibleLender;
use rdst::RadixKey;
use sux::bits::BitFieldVec;
use sux::cli::{BuilderArgs, HashTypes};
use sux::dict::SignedVFunc;
use sux::func::vfunc2::VFunc2;
use sux::func::{shard_edge::*, *};
use sux::init_env_logger;
use sux::prelude::VBuilder;
use sux::utils::{
    DekoBufLineLender, EmptyVal, FromCloneableIntoIterator, FromSlice, Sig, SigVal, ToSig,
};

#[derive(Parser, Debug)]
#[command(about = "Creates a (possibly signed) VFunc mapping each input to its rank and serializes it with ε-serde.", long_about = None, next_line_help = true, max_term_width = 100)]
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
    /// A file containing UTF-8 keys, one per line (at most N keys will be read); it can be compressed with any format supported by the deko crate.​
    #[arg(short, long)]
    filename: Option<String>,
    /// A name for the ε-serde serialized function.​
    func: Option<String>,
    /// Use the two-step variant (less space for skewed distributions, slightly slower queries).​
    #[arg(long, conflicts_with = "hash_type")]
    two_step: bool,
    /// Sign the function using hashes of this type.​
    #[arg(long)]
    hash_type: Option<HashTypes>,
    /// Use 64-bit signatures.​
    #[arg(long, requires = "no_shards")]
    sig64: bool,
    /// Do not use sharding.​
    #[arg(long)]
    no_shards: bool,
    /// Use slower edge logic reducing the probability of duplicate arcs for big
    /// shards.​
    #[arg(long, conflicts_with_all = ["sig64", "no_shards"])]
    full_sigs: bool,
    /// Use 3-hypergraphs.​
    #[cfg(feature = "mwhc")]
    #[arg(long, conflicts_with_all = ["sig64", "full_sigs"])]
    mwhc: bool,
    #[clap(flatten)]
    builder: BuilderArgs,
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    if args.two_step {
        return main_two_step(args);
    }

    #[cfg(feature = "mwhc")]
    if args.mwhc {
        return if args.no_shards {
            main_with_types::<[u64; 2], Mwhc3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], Mwhc3Shards>(args)
        };
    }

    if args.no_shards {
        if args.sig64 {
            main_with_types::<[u64; 1], FuseLge3NoShards>(args)
        } else {
            main_with_types::<[u64; 2], FuseLge3NoShards>(args)
        }
    } else {
        if args.full_sigs {
            main_with_types::<[u64; 2], FuseLge3FullSigs>(args)
        } else {
            main_with_types::<[u64; 2], FuseLge3Shards>(args)
        }
    }
}

macro_rules! filename_save_sign(
    ($h: ty, $builder:expr, $filename: expr, $func: expr, $n: expr, $pl: expr) => {{
        use sux::utils::ShardStore;
        use num_primitive::PrimitiveNumber;
        use value_traits::slices::SliceByValueMut;

        let (func, mut store) = $builder.try_build_func_and_store(
            DekoBufLineLender::from_path($filename)?.take($n),
            FromCloneableIntoIterator::from(0_usize..),
            BitFieldVec::new_unaligned,
            false,
            &mut $pl,
        )?;

        let num_keys = func.len();
        let mut hashes = vec![<$h>::MIN; num_keys].into_boxed_slice();

        $pl.item_name("hash");
        $pl.expected_updates(Some(num_keys));
        $pl.start("Storing hashes...");

        for shard in store.iter() {
            for sig_val in shard.iter() {
                let pos = sig_val.val;
                let hash = <$h>::as_from(func.remixed_hash_by_sig(sig_val.sig));
                hashes.set_value(pos, hash);
                $pl.light_update();
            }
        }

        $pl.done();

        let func = SignedVFunc::from_parts(func, hashes);
        if let Some(filename) = $func {
            unsafe { func.store(filename) }?;
        }
    }}
);

macro_rules! n_save_sign(
    ($h: ty, $builder:expr, $n: expr, $func: expr, $pl: expr) => {{
        let func: SignedVFunc<_, Box<[$h]>> = SignedVFunc::try_new_with_builder(
            FromCloneableIntoIterator::new(0_usize..$n),
            $n,
            $builder,
            &mut $pl,
        )?;
        if let Some(filename) = $func {
            unsafe { func.store(filename) }?;
        }
    }}
);

fn main_with_types<S: Sig + Send + Sync, E: ShardEdge<S, 3>>(args: Args) -> Result<()>
where
    str: ToSig<S>,
    usize: ToSig<S>,
    SigVal<S, usize>: RadixKey,
    SigVal<S, EmptyVal>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    SigVal<E::LocalSig, EmptyVal>: BitXor + BitXorAssign,
    VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>: Serialize,
    VFunc<str, BitFieldVec<Box<[usize]>>, S, E>: Serialize,
    VFunc<usize, Box<[u8]>, S, E>: Serialize,
    VFunc<str, Box<[u8]>, S, E>: Serialize,
    SignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u8]>>: Serialize,
    SignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u16]>>: Serialize,
    SignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u32]>>: Serialize,
    SignedVFunc<VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>, Box<[u64]>>: Serialize,
    SignedVFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u8]>>: Serialize,
    SignedVFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u16]>>: Serialize,
    SignedVFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u32]>>: Serialize,
    SignedVFunc<VFunc<str, BitFieldVec<Box<[usize]>>, S, E>, Box<[u64]>>: Serialize,
{
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    if let Some(filename) = &args.filename {
        let n = args.n.unwrap_or(usize::MAX);
        let mut builder = args
            .builder
            .configure(VBuilder::<BitFieldVec<Box<[usize]>>, S, E>::default());
        if let Some(n_hint) = args.n {
            builder = builder.expected_num_keys(n_hint);
        }
        match args.hash_type {
            None => {
                let (func, _) = builder.try_build_func_and_store(
                    DekoBufLineLender::from_path(filename)?.take(n),
                    FromCloneableIntoIterator::from(0_usize..),
                    BitFieldVec::<Box<[usize]>>::new_unaligned,
                    true,
                    &mut pl,
                )?;
                if let Some(filename) = args.func {
                    unsafe { func.store(filename) }?;
                }
            }
            Some(HashTypes::U8) => {
                filename_save_sign!(u8, builder, filename, args.func, n, pl)
            }
            Some(HashTypes::U16) => {
                filename_save_sign!(u16, builder, filename, args.func, n, pl)
            }
            Some(HashTypes::U32) => {
                filename_save_sign!(u32, builder, filename, args.func, n, pl)
            }
            Some(HashTypes::U64) => {
                filename_save_sign!(u64, builder, filename, args.func, n, pl)
            }
        }
    } else {
        let n = args.n.unwrap();
        let builder = args
            .builder
            .configure(VBuilder::<BitFieldVec<Box<[usize]>>, S, E>::default());
        match args.hash_type {
            None => {
                let func = <VFunc<usize, BitFieldVec<Box<[usize]>>, S, E>>::try_new_with_builder(
                    FromCloneableIntoIterator::from(0_usize..n),
                    FromCloneableIntoIterator::from(0_usize..),
                    n,
                    builder,
                    &mut pl,
                )?;
                if let Some(filename) = args.func {
                    unsafe { func.store(filename) }?;
                }
            }
            Some(HashTypes::U8) => {
                n_save_sign!(u8, builder, n, args.func, pl)
            }
            Some(HashTypes::U16) => {
                n_save_sign!(u16, builder, n, args.func, pl)
            }
            Some(HashTypes::U32) => {
                n_save_sign!(u32, builder, n, args.func, pl)
            }
            Some(HashTypes::U64) => {
                n_save_sign!(u64, builder, n, args.func, pl)
            }
        }
    }

    Ok(())
}

fn main_two_step(args: Args) -> Result<()> {
    #[cfg(not(feature = "no_logging"))]
    let mut pl = ProgressLogger::default();
    #[cfg(feature = "no_logging")]
    let mut pl = Option::<ConcurrentWrapper<ProgressLogger>>::None;

    let builder = args.builder.to_builder();

    if let Some(filename) = &args.filename {
        let n = if let Some(n) = args.n {
            n
        } else {
            pl.info(format_args!("Counting keys..."));
            let mut lender = DekoBufLineLender::from_path(filename)?;
            let mut count = 0usize;
            while FallibleLender::next(&mut lender)?.is_some() {
                count += 1;
            }
            pl.info(format_args!("Found {count} keys"));
            count
        };
        let func: VFunc2<str, BitFieldVec<Box<[usize]>>> = VFunc2::try_new_with_builder(
            DekoBufLineLender::from_path(filename)?.take(n),
            FromCloneableIntoIterator::from(0_usize..),
            n,
            builder,
            &mut pl,
        )?;
        if let Some(filename) = args.func {
            unsafe { func.store(filename) }?;
        }
    } else {
        let n = args.n.unwrap();
        let keys: Vec<usize> = (0..n).collect();
        let vals: Vec<usize> = (0..n).collect();
        let func: VFunc2<usize, BitFieldVec<Box<[usize]>>> = VFunc2::try_new_with_builder(
            FromSlice::new(&keys),
            FromSlice::new(&vals),
            n,
            builder,
            &mut pl,
        )?;
        if let Some(filename) = args.func {
            unsafe { func.store(filename) }?;
        }
    }

    Ok(())
}
