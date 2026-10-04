//! Space breakdown (first level, remapping, further levels) of PHast+ with
//! wrapping (reference implementation) and of PHast-R with wrapping, on the
//! same keys.
//!
//! Usage: parts [-n keys] [-m multiplier] [-l lambda] [-d depth]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use mem_dbg::{DbgFlags, MemDbg};
use ph::GetSize;
use ph::phast::{DefaultCompressedArray, Function2, Generic, GenericCore, ShiftOnlyWrapped};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::Bits8;
use sux::func::{PHastR, PHastRBuilder};

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 5.0)]
    l: f64,
    #[arg(short, default_value_t = 0)]
    d: u32,
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    let bits = |bytes: usize| bytes as f64 * 8.0 / a.n as f64;
    let params = Generic::new(Bits8, (a.l * 100.0).round() as u16);
    let f: Function2<GenericCore, Bits8, ShiftOnlyWrapped<3>, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_p_hash_sc(&keys, &params, BuildGxHash, ShiftOnlyWrapped::<3>);
    let (l0, remap, further) = f.component_sizes();
    println!(
        "ph w3:    total {:.4}  first level {:.4}  remapping {:.4}  further levels {:.4}",
        bits(f.size_bytes()),
        bits(l0),
        bits(remap),
        bits(further)
    );
    // SAFETY: GxKey is a transparent wrapper around u64
    let gkeys: &[GxKey] = unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    let g: PHastR<GxKey> = PHastRBuilder::default()
        .wrap(3)
        .bucket_size(a.l)
        .repair_depth(a.d)
        .repair_candidates(if a.d == 0 { 0 } else { 16 })
        .try_build(gkeys, no_logging![])
        .unwrap();
    g.mem_dbg(DbgFlags::default() | DbgFlags::PERCENTAGE)
        .unwrap();
}
