//! Space breakdown (first level, remapping, further levels) of PHast+ with
//! wrapping (reference implementation) and of PHast-R, on the same keys,
//! with the empirical entropy of the seeds of the first level of PHast+.
//!
//! Usage: parts [-n keys] [-l lambda of PHast+] [-r lambda of PHast-R]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use mem_dbg::{DbgFlags, MemDbg};
use ph::GetSize;
use ph::phast::{
    Conf, DefaultCompressedArray, Function2, GenericCore, ShiftOnlyWrapped, ShiftWrappedCore,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::Bits8;
use sux::func::{PHastR, PHastRBuilder};

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 5.0)]
    l: f64,
    /// Expected bucket size of PHast-R.
    #[arg(short, default_value_t = 4.5)]
    r: f64,
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    let bits = |bytes: usize| bytes as f64 * 8.0 / a.n as f64;
    let conf = Conf::generic_with_hash(Bits8, (a.l * 100.0).round() as u32, BuildGxHash);
    let f: Function2<GenericCore, Bits8, ShiftWrappedCore<3>, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_conf_sc(&keys, conf, ShiftOnlyWrapped::<3>);
    let (l0, remap, further) = f.component_sizes();
    {
        let conf = *f.level0_conf();
        let mut cnt = vec![0usize; 256];
        let nb = ph::phast::Core::buckets_num(&conf);
        for b in 0..nb {
            cnt[f.level0_seed(b) as usize] += 1;
        }
        let h: f64 = cnt
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / nb as f64;
                -p * p.log2()
            })
            .sum();
        let mut sorted = cnt.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let mut acc = 0.0;
        eprint!(
            "PH ENTROPY {h:.4} bits (zero seeds {:.3}%) ",
            100.0 * cnt[0] as f64 / nb as f64
        );
        for (i, &c) in sorted.iter().enumerate() {
            acc += c as f64 / nb as f64;
            if [15, 31, 63, 127, 191].contains(&i) {
                eprint!("top{}={:.3} ", i + 1, acc);
            }
        }
        eprintln!();
    }
    println!(
        "ph w3:    total {:.4}  first level {:.4}  remapping {:.4}  further levels {:.4}",
        bits(f.size_bytes()),
        bits(l0),
        bits(remap),
        bits(further)
    );
    // SAFETY: GxKey is a transparent wrapper around u64
    let gkeys: &[GxKey] = unsafe { std::slice::from_raw_parts(keys.as_ptr().cast(), keys.len()) };
    let g: PHastR<GxKey> = PHastR::try_par_new_with_builder(
        gkeys,
        PHastRBuilder::default().bucket_size(a.r),
        no_logging![],
    )
    .unwrap();
    g.mem_dbg(DbgFlags::default() | DbgFlags::PERCENTAGE)
        .unwrap();
}
