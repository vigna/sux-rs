use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::dict::EliasFanoConcurrentBuilder;
use sux::prelude::*;
use sux::traits::{IndexedSeq, Pred, PredUnchecked, Succ, SuccUnchecked, TryIntoUnaligned};

/// Number of pregenerated queries (must be a power of 2 for masking).
const NUM_QUERIES: usize = 1 << 20;
const QUERY_MASK: usize = NUM_QUERIES - 1;

/// (n, l) pairs: element count (power of 2) and desired number of lower bits.
/// The upper bound u = 2^l * n is chosen so that l = floor(log₂(u/n)).
const CONFIGS: &[(usize, usize)] = &[
    (1 << 20, 2),
    (1 << 20, 4),
    (1 << 20, 8),
    (1 << 20, 16),
];

fn n_label(n: usize) -> &'static str {
    if n == 1 << 20 { "1M" } else { "1G" }
}

/// The high bits of the structure under test.
type High = SelectZeroAdaptConst<
    SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>,
    Box<[usize]>,
    12,
    3,
>;

type EfAligned = EliasFano<u64, High>;

/// The rkyv-archived counterpart of [`EfAligned`].
#[cfg(feature = "rkyv")]
type ArchivedEfAligned = sux::dict::elias_fano::ArchivedEliasFano<u64, High, BitFieldVec<Box<[u64]>>>;

/// Build an Elias–Fano structure with `n` elements and `l` lower bits.
/// Returns the structure and the first/last values in the monotone sequence.
fn build_ef(n: usize, l: usize) -> (EfAligned, u64, u64) {
    let u = (1u64 << l) * n as u64;
    let mut rng = SmallRng::seed_from_u64(0);
    let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
    values.sort_unstable();

    let first = values[0];
    let last = values[n - 1];

    let mut builder = EliasFanoBuilder::new(n, u);
    for &v in &values {
        builder.push(v);
    }
    drop(values);

    let ef = unsafe {
        builder
            .build()
            .map_high_bits(|h| SelectZeroAdaptConst::new(SelectAdaptConst::new(h)))
    };
    (ef, first, last)
}

/// Generate `NUM_QUERIES` random indices in [0, n) using bit masking.
fn gen_indices(n: usize) -> Vec<usize> {
    let mask = n - 1; // n is a power of 2
    let mut rng = SmallRng::seed_from_u64(1);
    (0..NUM_QUERIES)
        .map(|_| rng.random::<u64>() as usize & mask)
        .collect()
}

/// Generate `NUM_QUERIES` random values suitable for succ/pred queries.
/// All values are in [first, u/2), well within the valid range.
fn gen_values(n: usize, l: usize, first: u64) -> Vec<u64> {
    let u = (1u64 << l) * n as u64;
    let mask = (u >> 1) - 1;
    let mut rng = SmallRng::seed_from_u64(2);
    (0..NUM_QUERIES)
        .map(|_| (rng.random::<u64>() & mask).max(first))
        .collect()
}

/// Benchmarks a single operation on a single representation of the structure.
///
/// Each iteration performs one operation, cycling through the pregenerated
/// query array using a counter and masking.
fn bench_arm<E, Q: Copy, R>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    param: &str,
    queries: &[Q],
    ef: &E,
    op: impl Fn(&E, Q) -> R,
) {
    group.bench_function(BenchmarkId::new(name, param), |b| {
        let mut ctr = 0usize;
        b.iter(|| {
            let q = queries[ctr & QUERY_MASK];
            ctr = ctr.wrapping_add(1);
            black_box(op(ef, q))
        })
    });
}

// Serialized images of the structures under test.
//
// Building a configuration is expensive (the largest one sorts eight gigabytes
// of values), and so is writing it out, so each configuration is serialized
// once and the images are reused by every benchmark. The native structure is
// still rebuilt by each benchmark, so at most one configuration is resident at
// a time.

#[cfg(any(all(feature = "epserde", feature = "mmap"), feature = "rkyv"))]
mod images {
    use super::EfAligned;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    /// The paths of the serialized images of one configuration.
    pub struct Images {
        /// The ε-serde image.
        #[cfg(all(feature = "epserde", feature = "mmap"))]
        pub eps: PathBuf,
        /// The rkyv image.
        #[cfg(feature = "rkyv")]
        pub rkyv: PathBuf,
    }

    /// Returns the images of the given configuration, writing them out on the
    /// first call.
    ///
    /// The images are leaked so that they can be borrowed by every benchmark;
    /// they are just paths, and the temporary directory containing the files
    /// is deleted when the process exits.
    pub fn images(n: usize, l: usize, ef: &EfAligned) -> &'static Images {
        static DIR: OnceLock<tempfile::TempDir> = OnceLock::new();
        static CACHE: OnceLock<Mutex<BTreeMap<(usize, usize), &'static Images>>> = OnceLock::new();

        let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
        let mut cache = cache.lock().unwrap();
        if let Some(images) = cache.get(&(n, l)) {
            return images;
        }

        let dir = DIR.get_or_init(|| tempfile::tempdir().expect("cannot create a temporary dir"));
        let images = Box::leak(Box::new(Images {
            #[cfg(all(feature = "epserde", feature = "mmap"))]
            eps: {
                let path = dir.path().join(format!("ef-{n}-{l}.eps"));
                unsafe { epserde::ser::Serialize::store(ef, &path) }
                    .expect("cannot write the ε-serde image");
                path
            },
            #[cfg(feature = "rkyv")]
            rkyv: {
                use std::io::Write;
                let path = dir.path().join(format!("ef-{n}-{l}.rkyv"));
                let file = std::fs::File::create(&path).expect("cannot create the rkyv image");
                let writer = rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(
                    ef,
                    rkyv::ser::writer::IoWriter::new(std::io::BufWriter::new(file)),
                )
                .expect("cannot write the rkyv image");
                writer
                    .into_inner()
                    .flush()
                    .expect("cannot flush the rkyv image");
                path
            },
        }));

        // `ef` is unused if neither format is enabled, which cannot happen
        // here, but silences the compiler in exotic feature combinations.
        let _ = ef;

        cache.insert((n, l), images);
        images
    }
}

/// Memory-maps a file for rkyv zero-copy access.
///
/// The mapping is page-aligned, which satisfies the alignment required by the
/// archived structure.
#[cfg(feature = "rkyv")]
fn mmap_file(path: &std::path::Path) -> mmap_rs::Mmap {
    let file = std::fs::File::open(path).expect("cannot open the rkyv image");
    let len = file.metadata().expect("cannot stat the rkyv image").len() as usize;
    unsafe {
        mmap_rs::MmapOptions::new(len)
            .expect("cannot set up the mapping")
            .with_file(&file, 0)
            .map()
            .expect("cannot map the rkyv image")
    }
}

/// Benchmarks one operation on all available representations of the structure.
///
/// The `index` variant generates index queries, the `value` variant generates
/// value queries. The four representations are the in-memory structure
/// (`aligned`), its unaligned variant (`unaligned`), the ε-copy deserialized
/// ε-serde image (`eps`), and the zero-copy rkyv archive (`rkyv`), the last
/// two being read from a memory-mapped file.
macro_rules! bench_ef {
    ($queries:ident, $fn_name:ident, $group_name:expr, |$ef:ident, $q:ident| $op:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            for &(n, l) in CONFIGS {
                let (ef, first, _) = build_ef(n, l);
                let queries = $queries(n, l, first);
                let param = format!("{}/l={}", n_label(n), l);
                let _ = first;

                bench_arm(&mut group, "aligned", &param, &queries, &ef, |$ef, $q| $op);

                #[cfg(any(all(feature = "epserde", feature = "mmap"), feature = "rkyv"))]
                let images = images::images(n, l, &ef);

                #[cfg(all(feature = "epserde", feature = "mmap"))]
                {
                    let case = unsafe {
                        <EfAligned as epserde::deser::Deserialize>::load_mmap(
                            &images.eps,
                            epserde::deser::Flags::empty(),
                        )
                    }
                    .expect("cannot map the ε-serde image");
                    bench_arm(
                        &mut group,
                        "eps",
                        &param,
                        &queries,
                        case.uncase(),
                        |$ef, $q| $op,
                    );
                }

                #[cfg(feature = "rkyv")]
                {
                    let map = mmap_file(&images.rkyv);
                    // SAFETY: the image was written by serializing an `EfAligned`.
                    let archived = unsafe { rkyv::access_unchecked::<ArchivedEfAligned>(&map) };
                    bench_arm(&mut group, "rkyv", &param, &queries, archived, |$ef, $q| $op);
                }

                let ef = ef.try_into_unaligned().unwrap();
                bench_arm(
                    &mut group,
                    "unaligned",
                    &param,
                    &queries,
                    &ef,
                    |$ef, $q| $op,
                );
            }
            group.finish();
        }
    };
}

/// Adapts [`gen_indices`] to the signature expected by [`bench_ef`].
fn indices(n: usize, _l: usize, _first: u64) -> Vec<usize> {
    gen_indices(n)
}

/// Adapts [`gen_values`] to the signature expected by [`bench_ef`].
fn values(n: usize, l: usize, first: u64) -> Vec<u64> {
    gen_values(n, l, first)
}

bench_ef!(
    indices,
    bench_get_unchecked,
    "ef_get_unchecked",
    |ef, i| unsafe { IndexedSeq::get_unchecked(ef, i) }
);

bench_ef!(indices, bench_get, "ef_get", |ef, i| IndexedSeq::get(ef, i));

bench_ef!(
    values,
    bench_succ_unchecked,
    "ef_succ_unchecked",
    |ef, v| unsafe { SuccUnchecked::succ_unchecked::<false>(ef, v) }
);

bench_ef!(values, bench_succ, "ef_succ", |ef, v| Succ::succ(ef, v));

bench_ef!(
    values,
    bench_pred_unchecked,
    "ef_pred_unchecked",
    |ef, v| unsafe { PredUnchecked::pred_unchecked::<false>(ef, v) }
);

bench_ef!(values, bench_pred, "ef_pred", |ef, v| Pred::pred(ef, v));

bench_ef!(
    values,
    bench_rank_unchecked,
    "ef_rank_unchecked",
    |ef, v| unsafe { PredUnchecked::rank_unchecked(ef, v) }
);

bench_ef!(values, bench_rank, "ef_rank", |ef, v| Pred::rank(ef, v));

fn bench_build_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_build_seq");
    for &(n, l) in CONFIGS {
        let u = (1u64 << l) * n as u64;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
        values.sort_unstable();
        let param = format!("{}/l={}", n_label(n), l);

        group.bench_function(BenchmarkId::new("push", &param), |b| {
            b.iter(|| {
                let mut builder = EliasFanoBuilder::new(n, u);
                for &v in &values {
                    builder.push(v);
                }
                black_box(builder.build());
            })
        });
    }
    group.finish();
}

fn bench_build_concurrent(c: &mut Criterion) {
    let thread_counts = [4, 8, 16];
    let mut group = c.benchmark_group("ef_build_conc");
    for &(n, l) in CONFIGS {
        let u = (1u64 << l) * n as u64;
        let mut rng = SmallRng::seed_from_u64(0);
        let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..u)).collect();
        values.sort_unstable();

        for num_threads in thread_counts {
            let param = format!("{}/l={}/t={}", n_label(n), l, num_threads);
            let chunk_size = n.div_ceil(num_threads);
            let chunks: Vec<(usize, &[u64])> = values
                .chunks(chunk_size)
                .enumerate()
                .map(|(i, chunk)| (i * chunk_size, chunk))
                .collect();

            group.bench_function(BenchmarkId::new("set", &param), |b| {
                b.iter(|| {
                    let efcb = EliasFanoConcurrentBuilder::new(n, u);
                    std::thread::scope(|s| {
                        for &(start, chunk) in &chunks {
                            let efcb = &efcb;
                            s.spawn(move || {
                                for (j, &v) in chunk.iter().enumerate() {
                                    unsafe { efcb.set(start + j, v) };
                                }
                            });
                        }
                    });
                    black_box(efcb.build());
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get_unchecked,
    bench_get,
    bench_succ_unchecked,
    bench_succ,
    bench_pred_unchecked,
    bench_pred,
    bench_rank_unchecked,
    bench_rank,
    bench_build_sequential,
    bench_build_concurrent,
);
criterion_main!(benches);
