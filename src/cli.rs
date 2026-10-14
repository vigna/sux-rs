/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Shared CLI utilities for binaries and example benchmarks.
//!
//! This module is enabled by the `cli` feature, and contains:
//!
//! - argument structs that binaries flatten into their parsers:
//!   [`BuilderArgs`] (the configuration of a [`VBuilder`]), [`ShardingArgs`]
//!   (the shard/edge type), [`ValueArgs`] (the values of a function), and
//!   [`LogIntervalArg`] (the log interval of progress loggers);
//! - readers of input files: [`KeyBuffer`] reads keys in a single buffer for
//!   parallel constructors, [`count_lines`] counts lines, and
//!   [`reservoir_sample`] samples lines; input files are decompressed by an
//!   [`AnyLineLender`] in the format specified by the `--decompress` option
//!   of the binaries, a [`Decompression`];
//! - [`store`] and [`store_as_is`], which store structures with ε-serde,
//!   logging their type, so that users know which type to load;
//! - helpers for the example benchmarks: [`bench`](fn@bench), which times
//!   repeated runs, [`pack_strings`], [`gen_sorted_keys`], and [`INCR`];
//! - [`init_env_logger`], which sets up logging.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dsi_progress_logger::{ProgressLog, ProgressLogger};
use epserde::ser::Serialize;
use lender::FallibleLender;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::bits::BitFieldVec;
use crate::func::VBuilder;
use crate::func::shard_edge::ShardEdge;
use crate::traits::TryIntoUnaligned;
pub use crate::utils::lenders::{AnyLineLender, Decompression};

/// Initializes the `env_logger` logger with a custom format including
/// timestamps with elapsed time since initialization.
///
/// The default level is `info`; use the `RUST_LOG` environment variable to
/// change it.
///
/// # Panics
///
/// Logging panics if the time elapsed since initialization cannot be
/// represented by a [`jiff::Span`], which in practice never happens.
pub fn init_env_logger() -> Result<()> {
    use jiff::{
        SpanRound,
        fmt::friendly::{Designator, Spacing, SpanPrinter},
    };
    use std::io::Write;

    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));

    let start = std::time::Instant::now();
    let printer = SpanPrinter::new()
        .spacing(Spacing::None)
        .designator(Designator::Compact);
    let span_round = SpanRound::new()
        .largest(jiff::Unit::Day)
        .smallest(jiff::Unit::Millisecond)
        .days_are_24_hours();

    builder.format(move |buf, record| {
        let Ok(ts) = jiff::Timestamp::try_from(std::time::SystemTime::now()) else {
            return Err(std::io::Error::other("Failed to get timestamp"));
        };
        let style = buf.default_level_style(record.level());
        let elapsed = start.elapsed();
        let span = jiff::Span::new()
            .seconds(elapsed.as_secs() as i64)
            .milliseconds(elapsed.subsec_millis() as i64);
        let span = span.round(span_round).expect("Failed to round span");
        writeln!(
            buf,
            "{} {} {style}{}{style:#} [{:?}] {} - {}",
            ts.strftime("%F %T%.3f"),
            printer.span_to_string(&span),
            record.level(),
            std::thread::current().id(),
            record.target(),
            record.args()
        )
    });
    builder.try_init()?;
    Ok(())
}

/// Keys read from a file into a single buffer.
///
/// Parallel constructors need all keys in memory: concatenating them in a
/// single buffer avoids an allocation per key, and makes access to the keys
/// cache-friendly when computing their hashes.
#[derive(Debug, Clone)]
pub struct KeyBuffer {
    buffer: String,
    /// Key `i` is `&buffer[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
}

impl KeyBuffer {
    /// Reads keys, one per line, from a file, decompressed in the given
    /// format, or as it is if `decompression` is [`None`].
    ///
    /// If `n` is [`Some`], the first `n` keys are read, and it is an error
    /// if the file contains fewer keys; otherwise, all keys are read.
    pub fn from_path(
        path: impl AsRef<Path>,
        n: Option<usize>,
        decompression: Option<Decompression>,
    ) -> Result<Self> {
        let limit = n.unwrap_or(usize::MAX);
        let mut buffer = String::new();
        // Cap the initial capacity, as limit might be usize::MAX: the vector
        // grows past the cap as needed
        let mut offsets = Vec::with_capacity(limit.saturating_add(1).min(1 << 20));
        offsets.push(0);
        let mut lender = AnyLineLender::from_path(path, decompression)?;
        // Do not read past the limit, as the next line might be invalid
        while offsets.len() <= limit {
            let Some(line) = lender.next()? else {
                break;
            };
            buffer.push_str(line);
            offsets.push(buffer.len());
        }
        let keys = Self { buffer, offsets };
        check_num_keys(keys.len(), n)?;
        Ok(keys)
    }

    /// Returns the number of keys.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns whether there are no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the keys, as slices of the buffer.
    pub fn keys(&self) -> Vec<&str> {
        self.offsets
            .windows(2)
            .map(|w| &self.buffer[w[0]..w[1]])
            .collect()
    }
}

/// Checks that the number of keys read from a file is equal to `n`, if `n`
/// is [`Some`].
///
/// Binaries reading keys sequentially use only the first `n` keys of a file,
/// and call this function after construction to check that the file
/// contained at least `n` keys.
pub fn check_num_keys(num_keys: usize, n: Option<usize>) -> Result<()> {
    if let Some(n) = n {
        ensure!(
            num_keys == n,
            "key count mismatch: read {num_keys} keys, expected {n}"
        );
    }
    Ok(())
}

/// Returns the number of lines of a file, decompressed in the given format,
/// or read as it is if `decompression` is [`None`].
pub fn count_lines(path: impl AsRef<Path>, decompression: Option<Decompression>) -> Result<usize> {
    let mut lender = AnyLineLender::from_path(path, decompression)?;
    let mut count = 0;
    while lender.next()?.is_some() {
        count += 1;
    }
    Ok(count)
}

/// Reservoir-samples up to `n` lines from `filename` using an
/// [`AnyLineLender`]. Each line in the file has equal probability of
/// appearing in the result. The result is shuffled so that the strings are not
/// in file order.
pub fn reservoir_sample(
    filename: &str,
    n: usize,
    seed: u64,
    decompression: Option<Decompression>,
) -> Result<Vec<String>> {
    use rand::seq::SliceRandom;

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut reservoir: Vec<String> = Vec::with_capacity(n);
    let mut lender = AnyLineLender::from_path(filename, decompression)?;
    let mut i = 0usize;
    while let Some(line) = lender.next()? {
        if i < n {
            reservoir.push(line.to_owned());
        } else {
            let j = rng.random_range(0..=i);
            if j < n {
                reservoir[j] = line.to_owned();
            }
        }
        i += 1;
    }
    reservoir.shuffle(&mut rng);
    Ok(reservoir)
}

/// Packs `n` strings into a contiguous byte buffer by cycling through
/// `queries`.
///
/// Returns the buffer and offsets such that query `i` is `&packed[offsets[i] as
/// usize .. offsets[i+1] as usize]`.
///
/// If `queries` has fewer than `n` elements, the queries are repeated
/// in a round-robin fashion so that exactly `n` entries are produced.
///
/// # Panics
///
/// Panics if `queries` is empty.
pub fn pack_strings(queries: &[String], n: usize) -> (Vec<u8>, Vec<usize>) {
    assert!(!queries.is_empty(), "no queries to pack");
    let mut packed = Vec::new();
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0usize);
    for i in 0..n {
        packed.extend_from_slice(queries[i % queries.len()].as_bytes());
        offsets.push(packed.len());
    }
    (packed, offsets)
}

/// Stores a structure with ε-serde, possibly in unaligned form, if a path is
/// provided.
///
/// The full name of the type of the stored structure is logged, so that
/// users know which type to use to load it. If `unaligned` is true, the
/// structure is converted with [`TryIntoUnaligned::try_into_unaligned`]
/// before storing it.
///
/// # Safety
///
/// See [`store_as_is`].
pub unsafe fn store<F: Serialize + TryIntoUnaligned<Unaligned: Serialize>>(
    structure: F,
    path: Option<impl AsRef<Path>>,
    unaligned: bool,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if unaligned {
        // We log the type using the Unaligned alias, which is shorter than the
        // unaligned type and is what users write to load the structure
        let name = format!("Unaligned<{}>", short_type_name::<F>());
        unsafe { store_named(&structure.try_into_unaligned()?, path, &name) }
    } else {
        unsafe { store_named(&structure, path, &short_type_name::<F>()) }
    }
}

/// Stores a structure with ε-serde as it is, if a path is provided.
///
/// The full name of the type of the stored structure is logged, so that
/// users know which type to use to load it. Use this function for structures
/// that have no unaligned form; otherwise, use [`store`].
///
/// # Safety
///
/// Serialization with ε-serde writes padding bytes, which are
/// uninitialized: the caller must ensure that the padding bytes of the
/// stored structure are never read (see [`Serialize`]).
pub unsafe fn store_as_is<F: Serialize>(
    structure: &F,
    path: Option<impl AsRef<Path>>,
) -> Result<()> {
    match path {
        Some(path) => unsafe { store_named(structure, path, &short_type_name::<F>()) },
        None => Ok(()),
    }
}

/// Stores a structure with ε-serde, logging the given type name.
///
/// # Safety
///
/// See [`store_as_is`].
unsafe fn store_named<F: Serialize>(
    structure: &F,
    path: impl AsRef<Path>,
    type_name: &str,
) -> Result<()> {
    let path = path.as_ref();
    unsafe { structure.store(path) }.with_context(|| format!("cannot store {}", path.display()))?;
    log::info!("Stored structure of type {type_name} in {}", path.display());
    Ok(())
}

/// Returns the name of a type without module paths (e.g., `VFunc<str,
/// BitFieldVec<Box<[usize]>>>`), so that it can be used with the
/// [prelude](crate::prelude).
fn short_type_name<T: ?Sized>() -> String {
    let name = core::any::type_name::<T>();
    let mut result = String::with_capacity(name.len());
    let mut segments = name.split("::").peekable();
    while let Some(segment) = segments.next() {
        if segments.peek().is_some() {
            // Drop the trailing module name, keeping what precedes it
            let keep = segment
                .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
                .map_or(0, |i| i + 1);
            result.push_str(&segment[..keep]);
        } else {
            result.push_str(segment);
        }
    }
    result
}

/// Hash types for signed functions.​
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum HashTypes {
    /// 8-bit hashes.​
    U8,
    /// 16-bit hashes.​
    U16,
    /// 32-bit hashes.​
    U32,
    /// 64-bit hashes.​
    U64,
}

/// Options configuring a [`VBuilder`], shared by the binaries building
/// structures based on it.​
#[derive(clap::Args, Debug)]
pub struct BuilderArgs {
    /// Use at most this number of threads [default: min(16, available
    /// cores)].​
    #[arg(short, long)]
    pub threads: Option<usize>,
    /// Use disk-based buckets to reduce memory usage at construction time.
    /// Requires --sequential (-s).​
    #[arg(short, long, requires = "sequential")]
    pub offline: bool,
    /// The directory in which --offline (-o) creates the directory of its
    /// buckets [default: TMPDIR].​
    #[arg(long, requires = "offline")]
    pub temp_dir: Option<std::path::PathBuf>,
    /// Sort shards and check for duplicate signatures.​
    #[arg(short, long)]
    pub check_dups: bool,
    /// A 64-bit seed for the pseudorandom number generator.​
    #[arg(long)]
    pub seed: Option<u64>,
    /// The target relative space overhead due to sharding.​
    #[arg(long, default_value_t = 0.001)]
    pub eps: f64,
    /// The maximum probability of a construction retry caused by duplicate
    /// hyperedges.​
    #[arg(long, default_value_t = 0.05)]
    pub retry_prob: f64,
    /// Always use the low-mem peel-by-signature algorithm (slightly slower).​
    #[arg(long)]
    pub low_mem: bool,
    /// Always use the high-mem peel-by-signature algorithm (slightly faster).​
    #[arg(long, conflicts_with = "low_mem")]
    pub high_mem: bool,
}

impl BuilderArgs {
    /// Applies these options to a [`VBuilder`].
    pub fn configure<D: Send + Sync, S, E: ShardEdge<S, 3>>(
        &self,
        builder: VBuilder<D, S, E>,
    ) -> VBuilder<D, S, E> {
        let mut builder = builder
            .offline(self.offline)
            .check_dups(self.check_dups)
            .eps(self.eps)
            .retry_prob(self.retry_prob);
        if let Some(seed) = self.seed {
            builder = builder.seed(seed);
        }
        if let Some(temp_dir) = &self.temp_dir {
            builder = builder.temp_dir(temp_dir);
        }
        if let Some(threads) = self.threads {
            builder = builder.max_num_threads(threads);
        }
        if self.low_mem {
            builder = builder.low_mem(true);
        }
        if self.high_mem {
            builder = builder.low_mem(false);
        }
        builder
    }

    /// Creates and configures a default [`VBuilder`] with [`BitFieldVec`]
    /// storage.
    pub fn to_builder(&self) -> VBuilder<BitFieldVec<Box<[usize]>>> {
        self.configure(VBuilder::default())
    }
}

/// Shard/edge type used during construction.
#[derive(clap::ValueEnum, Copy, Clone, Debug, Default)]
pub enum ShardEdgeType {
    /// Fuse 3-hypergraphs with lazy Gaussian elimination and ε-cost sharding.​
    #[default]
    FuseLge3Shards,
    /// Fuse 3-hypergraphs with ε-cost sharding, no lazy Gaussian elimination.​
    Fuse3Shards,
    /// Fuse 3-hypergraphs without sharding or lazy Gaussian elimination and 64-bit signatures.​
    Fuse3NoShards64,
    /// Fuse 3-hypergraphs without sharding or lazy Gaussian elimination and 128-bit signatures.​
    Fuse3NoShards128,
    /// MWHC 3-hypergraphs with ε-cost sharding.​
    #[cfg(feature = "mwhc")]
    Mwhc3,
    /// MWHC 3-hypergraphs without ε-cost sharding.​
    #[cfg(feature = "mwhc")]
    Mwhc3NoShards,
    /// Fuse 3-hypergraphs with lazy Gaussian elimination, ε-cost sharding, and full
    /// signatures.​
    FuseLge3FullSigs,
}

/// Shared CLI flags that select the *type* of shard/edge used during
/// construction.
///
/// These flags live here (rather than in [`BuilderArgs`]) because they don't
/// configure a [`VBuilder`] instance: they select its type parameters.
///
/// Binaries that dispatch on shard/edge type (`vfunc`, `vfunc2`, `vfilter`)
/// flatten this struct next to [`BuilderArgs`]; `comp_vfunc` has its own
/// option, as it supports only some types and has a different default;
/// binaries that pin their shard/edge type (`lcp_mmphf`) have no option.
#[derive(clap::Args, Debug)]
pub struct ShardingArgs {
    /// Shard/edge type.​
    #[arg(long, short = 'E', value_enum, default_value_t)]
    pub shard_edge: ShardEdgeType,
}

/// Options specifying the values of a function, read from a file or
/// generated pseudorandomly.​
#[derive(clap::Args, Debug)]
#[command(group(
    clap::ArgGroup::new("value_source")
        .required(true)
        .args(["values", "geometric", "zipf", "uniform"]),
))]
pub struct ValueArgs {
    /// A file containing the values, one ASCII decimal integer per line. If
    /// N is specified, the file must contain exactly N values; otherwise, N
    /// is the number of values. Additional keys in the file of keys are
    /// ignored, unless --check is specified.​
    #[arg(short = 'v', long = "values", id = "values", value_name = "VALUES")]
    pub file: Option<String>,
    /// Generate values with a geometric distribution (trailing zeros
    /// of a random u64).​
    #[arg(long)]
    pub geometric: bool,
    /// Generate values with a Zipf distribution (exponent 1) over
    /// [0 . . K).​
    #[arg(long)]
    pub zipf: Option<NonZeroUsize>,
    /// Generate values uniformly distributed in [0 . . K).​
    #[arg(long)]
    pub uniform: Option<NonZeroUsize>,
    /// With an explicit --values file, verify that the key count matches the
    /// value count. This makes an extra pass over the key file, so it is
    /// off by default.​
    #[arg(long, requires = "values", conflicts_with_all = ["geometric", "zipf", "uniform"])]
    pub check: bool,
}

impl ValueArgs {
    /// Returns the values, read from the file of values or generated.
    ///
    /// `n` is the number of keys, if known: if values are read from a file
    /// and `n` is [`Some`], it is an error if the file does not contain
    /// exactly `n` values; if values are generated, `n` must be [`Some`].
    pub fn values(&self, n: Option<usize>) -> Result<Vec<usize>> {
        if let Some(path) = &self.file {
            let values = read_values(path)?;
            if let Some(n) = n {
                ensure!(
                    values.len() == n,
                    "n={n} but values file has {} values",
                    values.len()
                );
            }
            return Ok(values);
        }
        let Some(n) = n else {
            bail!("the number of keys is necessary to generate values");
        };
        let mut rng = SmallRng::seed_from_u64(0);
        Ok(if self.geometric {
            (0..n)
                .map(|_| rng.random::<u64>().trailing_zeros() as usize)
                .collect()
        } else if let Some(k) = self.zipf {
            let cdf = zipf_cdf(1.0, k.get());
            (0..n).map(|_| sample_zipf(&cdf, &mut rng)).collect()
        } else if let Some(k) = self.uniform {
            (0..n).map(|_| rng.random_range(0..k.get())).collect()
        } else {
            bail!("one of --values, --geometric, --zipf, or --uniform is required");
        })
    }

    /// If `--check` was specified, checks that a file of keys contains
    /// exactly `n` keys.
    pub fn check_keys(
        &self,
        path: impl AsRef<Path>,
        decompression: Option<Decompression>,
        n: usize,
    ) -> Result<()> {
        if self.check {
            // Without this check, keys past the n-th one would be silently
            // ignored. It costs an extra pass over the file of keys.
            let key_count = count_lines(path, decompression)?;
            ensure!(key_count == n, "key count {key_count} != value count {n}");
        }
        Ok(())
    }
}

/// Reads values, one ASCII decimal integer per line, skipping empty lines.
fn read_values(path: &str) -> Result<Vec<usize>> {
    use std::io::BufRead;

    let file = std::fs::File::open(path).with_context(|| format!("open values file {path}"))?;
    let reader = std::io::BufReader::new(file);
    let mut values = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: usize = trimmed.parse().with_context(|| {
            format!(
                "values file {path}, line {}: cannot parse `{trimmed}`",
                lineno + 1
            )
        })?;
        values.push(v);
    }
    Ok(values)
}

/// Returns the cumulative distribution function of a Zipf distribution with
/// exponent `s` over [0 . . `n`).
fn zipf_cdf(s: f64, n: usize) -> Vec<f64> {
    let h: f64 = (1..=n).map(|i| 1.0 / (i as f64).powf(s)).sum();
    let mut cdf = vec![0.0; n];
    let mut acc = 0.0;
    for i in 1..=n {
        acc += 1.0 / (i as f64).powf(s) / h;
        cdf[i - 1] = acc;
    }
    cdf
}

/// Samples a distribution given its cumulative distribution function.
fn sample_zipf(cdf: &[f64], rng: &mut SmallRng) -> usize {
    let u: f64 = (rng.random::<u64>() >> 11) as f64 / ((1u64 << 53) as f64);
    // The clamp keeps the sample in range when accumulated floating-point
    // error leaves the last CDF entry slightly below one and u falls above it.
    cdf.partition_point(|&p| p < u).min(cdf.len() - 1)
}

fn parse_duration(value: &str) -> Result<Duration> {
    ensure!(!value.is_empty(), "empty duration string");
    let mut duration = Duration::from_secs(0);
    let mut acc = String::new();
    for c in value.chars() {
        if c.is_ascii_digit() {
            acc.push(c);
        } else if c.is_whitespace() {
            continue;
        } else {
            let dur: u64 = acc.parse()?;
            let scale = match c {
                's' => 1,
                'm' => 60,
                'h' => 3600,
                'd' => 86400,
                _ => bail!("invalid duration suffix: {c}"),
            };
            let seconds = dur
                .checked_mul(scale)
                .ok_or_else(|| anyhow::anyhow!("duration component overflows u64 seconds"))?;
            duration = duration
                .checked_add(Duration::from_secs(seconds))
                .ok_or_else(|| anyhow::anyhow!("duration exceeds the supported range"))?;
            acc.clear();
        }
    }
    if !acc.is_empty() {
        duration = duration
            .checked_add(Duration::from_millis(acc.parse()?))
            .ok_or_else(|| anyhow::anyhow!("duration exceeds the supported range"))?;
    }
    Ok(duration)
}

/// Shared CLI argument for the progress-logger log interval.
#[derive(clap::Args, Debug, Clone)]
pub struct LogIntervalArg {
    /// How often to log progress. Supported suffixes: "s" (seconds),
    /// "m" (minutes), "h" (hours), "d" (days). A bare number is
    /// interpreted as milliseconds. Examples: "10s", "1m30s", "500".​
    #[arg(long, value_parser = parse_duration, default_value = "10s")]
    pub log_interval: Duration,
}

impl LogIntervalArg {
    /// Returns a [`ProgressLogger`] logging at this interval.
    pub fn progress_logger(&self) -> ProgressLogger {
        let mut pl = ProgressLogger::default();
        pl.log_interval(self.log_interval);
        pl
    }
}

/// The increment used by the example benchmarks to generate integer
/// queries.
///
/// Queries are the multiples of this odd constant (2⁶⁴/φ, or 2³²/φ on 32-bit
/// platforms), which are distinct and spread evenly.
#[cfg(target_pointer_width = "64")]
pub const INCR: usize = 0x9e3779b97f4a7c15;
/// The increment used by the example benchmarks to generate integer
/// queries.
///
/// Queries are the multiples of this odd constant (2⁶⁴/φ, or 2³²/φ on 32-bit
/// platforms), which are distinct and spread evenly.
#[cfg(not(target_pointer_width = "64"))]
pub const INCR: usize = 0x9e3779b9;

/// Runs `repeats` times a benchmark performing `n` operations, printing on
/// standard error the time per operation of each run and, at the end, the
/// minimum, median, maximum, and average time.
///
/// # Panics
///
/// Panics if `repeats` is zero.
pub fn bench(n: usize, repeats: usize, mut f: impl FnMut()) {
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

/// Returns `k` distinct pseudorandom 64-bit keys in increasing order.
///
/// The keys are generated by a [`SmallRng`] seeded with `seed`, so on a
/// given platform the same arguments yield the same keys: the `lcp_mmphf`
/// binary uses this function with seed zero to generate keys, and the
/// `bench_lcp_mmphf` example uses it to generate them again.
pub fn gen_sorted_keys(k: usize, seed: u64) -> Vec<u64> {
    // Draw additional values to account for the expected number of
    // collisions, k² / 2⁶⁵
    let extra = (((k as u128) * (k as u128)) >> 65) as usize;
    let draw = k + 2 * extra;
    let mut rng = SmallRng::seed_from_u64(seed);
    loop {
        let mut keys: Vec<u64> = (0..draw).map(|_| rng.random()).collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() >= k {
            keys.truncate(k);
            return keys;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_type_name_drops_module_paths() {
        assert_eq!(
            short_type_name::<crate::func::VFunc<str, BitFieldVec<Box<[usize]>>>>(),
            "VFunc<str, BitFieldVec<Box<[usize]>>>"
        );
        assert_eq!(short_type_name::<[u64; 2]>(), "[u64; 2]");
    }

    #[test]
    fn duration_overflow_is_an_error() {
        assert!(parse_duration(&format!("{}d", u64::MAX)).is_err());
        assert!(parse_duration(&format!("{}s1s", u64::MAX)).is_err());
    }

    #[test]
    fn key_buffer_does_not_read_past_limit() -> Result<()> {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(b"good\n\xff\n")?;
        let keys = KeyBuffer::from_path(file.path(), Some(1), None)?;
        assert_eq!(keys.keys(), ["good"]);

        let mut invalid = tempfile::NamedTempFile::new()?;
        invalid.write_all(b"\xff\n")?;
        let keys = KeyBuffer::from_path(invalid.path(), Some(0), None)?;
        assert!(keys.is_empty());
        Ok(())
    }

    #[test]
    fn key_buffer_checks_the_number_of_keys() -> Result<()> {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(b"a\nb\nc\n")?;
        assert_eq!(
            KeyBuffer::from_path(file.path(), None, None)?.keys(),
            ["a", "b", "c"]
        );
        assert_eq!(
            KeyBuffer::from_path(file.path(), Some(2), None)?.keys(),
            ["a", "b"]
        );
        assert!(KeyBuffer::from_path(file.path(), Some(4), None).is_err());
        assert_eq!(count_lines(file.path(), None)?, 3);
        Ok(())
    }
}
