/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The construction of [`PHastR`] from a lender (see
//! [`PHastR::try_new_with_builder`]).
//!
//! Keys are read once: for each key we store a *record* containing its
//! signature for the first level and its second hash (see [`level_sig`]),
//! from which we can compute its signature for every level. The records of
//! the keys bumped from a level are stored for the following level. Records
//! are kept in memory or, in [offline mode], in temporary files.
//!
//! The records of a level are partitioned by the upper bits of their
//! signature for the level. Since the bucket of a key is a nondecreasing
//! function of its signature, a partition contains the keys of a range of
//! consecutive buckets, except that the first and the last bucket of the
//! range can have keys in the adjacent partitions: sweeps read partitions in
//! increasing order, passing the keys of the last bucket of a partition to
//! the following one. In a file, each partition is a sequence of blocks.
//!
//! The function built is identical to that built by
//! [`PHastR::try_par_new_with_builder`] with the same builder, provided that
//! the current [rayon] pool has the same number of threads: the seed chosen
//! for a bucket depends on the multiset of the signatures of its keys, but
//! not on their order, and sweeps are split into the same chunks. The only
//! exception is the detection of duplicate keys, which uses the records (see
//! [`has_duplicates`]).
//!
//! [offline mode]: PHastRBuilder::offline
//! [rayon]: https://docs.rs/rayon

use super::builder::*;
use super::sigs::*;
use super::sweep::*;
use super::*;
use crate::dict::elias_fano::EliasFanoBuilder;
use anyhow::bail;
use dsi_progress_logger::no_logging;
use std::fs::File;
use std::io::{self, Write};
use std::sync::Mutex;
use std::time::Instant;
use zerocopy::IntoBytes;

/// The signature of a key for the first level and its second hash.
type Record = [u64; 2];

/// The base-2 logarithm of the maximum number of partitions of a store,
/// which is the number of partitions of the first level (whose number of
/// keys is not known in advance): in offline mode, with blocks of [`BLOCK`]
/// records, the buffers of a writer take at most 256 MiB, and partitions are
/// small enough to be loaded quickly by a sweep up to about 10¹¹ keys.
#[cfg(not(test))]
const MAX_LOG2_PARTITIONS: u32 = 14;
#[cfg(test)]
const MAX_LOG2_PARTITIONS: u32 = 5;

/// The number of records of a block.
#[cfg(not(test))]
const BLOCK: usize = 1024;
#[cfg(test)]
const BLOCK: usize = 16;

/// Returns the base-2 logarithm of the number of partitions of the store of
/// a level after the first one with `k` keys.
///
/// With *P* partitions, in offline mode each thread writing the store uses
/// *P* blocks of sixteen bytes per record, and each thread sweeping the
/// level buffers a few partitions (about four, with *k* / *P* signatures of
/// eight bytes each): the total is minimized when *P* is about
/// √(2*k* / [`BLOCK`]).
fn log2_partitions(k: usize) -> u32 {
    ((2.0 * k as f64 / BLOCK as f64).sqrt().ceil() as usize)
        .next_power_of_two()
        .ilog2()
        .min(MAX_LOG2_PARTITIONS)
}

/// The signature of a record for a level: its first component for the
/// first level, and the value of [`level_sig`] with the salt of the level
/// for the following ones.
#[derive(Debug, Clone, Copy)]
struct SigFn(Option<u64>);

impl SigFn {
    /// Returns the signature of a record.
    #[inline(always)]
    const fn sig(self, r: &Record) -> u64 {
        match self.0 {
            None => r[0],
            Some(salt) => level_sig(r[0], r[1], salt),
        }
    }
}

/// Returns the partition of a signature, given the base-2 logarithm of the
/// number of partitions.
#[inline(always)]
const fn partition(sig: u64, log2_partitions: u32) -> usize {
    // Two shifts, so that a single partition needs no special case
    (sig >> 1 >> (63 - log2_partitions)) as usize
}

/// Reads exactly `buf.len()` bytes of `file` starting at `offset`, without
/// changing the current position (so that several threads can read at the
/// same time).
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let (mut buf, mut offset) = (buf, offset);
        while !buf.is_empty() {
            match file.seek_read(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// The index of the blocks of the partitions of a [`Store`] kept in a file.
///
/// Each block occupies [`BLOCK`] records of the file, so blocks are
/// identified by their index in the file. All blocks are full, except for
/// the last one written by each writer for each partition.
struct Blocks {
    /// For each partition, its full blocks.
    full: Vec<Vec<u32>>,
    /// For each partition, its blocks that are not full, with their length.
    partial: Vec<Vec<(u32, u32)>>,
    /// The number of blocks of the file.
    len: u32,
}

/// The storage of the records of a [`Store`].
enum Storage {
    /// A temporary file, containing the blocks of all partitions.
    File(File, Blocks),
    /// A vector for each partition.
    Memory(Vec<Vec<Record>>),
}

/// The records of the keys of a level, partitioned by the upper bits of
/// their signature for the level.
struct Store {
    /// The signature of a record for the level.
    sig: SigFn,
    /// The base-2 logarithm of the number of partitions.
    log2_partitions: u32,
    /// The number of records.
    len: usize,
    storage: Storage,
}

impl Store {
    /// Returns the number of partitions.
    fn num_partitions(&self) -> usize {
        1 << self.log2_partitions
    }

    /// Returns the number of records of partition `p`.
    fn partition_len(&self, p: usize) -> usize {
        match &self.storage {
            Storage::File(_, blocks) => {
                blocks.full[p].len() * BLOCK
                    + blocks.partial[p]
                        .iter()
                        .map(|&(_, len)| len as usize)
                        .sum::<usize>()
            }
            Storage::Memory(partitions) => partitions[p].len(),
        }
    }

    /// Calls `f` on the records of partition `p`, using `buf` as a buffer.
    fn for_each(
        &self,
        p: usize,
        buf: &mut Vec<Record>,
        mut f: impl FnMut(&Record) -> io::Result<()>,
    ) -> io::Result<()> {
        match &self.storage {
            Storage::File(file, blocks) => {
                let full = blocks.full[p].iter().map(|&b| (b, BLOCK as u32));
                for (b, len) in full.chain(blocks.partial[p].iter().copied()) {
                    buf.resize(len as usize, [0; 2]);
                    let pos = b as u64 * (BLOCK * size_of::<Record>()) as u64;
                    read_at(file, buf.as_mut_bytes(), pos)?;
                    for r in buf.iter() {
                        f(r)?;
                    }
                }
            }
            Storage::Memory(partitions) => {
                for r in &partitions[p] {
                    f(r)?;
                }
            }
        }
        Ok(())
    }
}

/// Writes records to a new [`Store`]: several threads can write at the same
/// time, each using its own [`Buffers`].
struct StoreWriter {
    sig: SigFn,
    log2_partitions: u32,
    storage: Mutex<Storage>,
}

impl StoreWriter {
    /// Creates a writer for a store with the given signature function and
    /// number of partitions, kept in an anonymous temporary file if
    /// `offline` is true, and in memory otherwise.
    fn new(sig: SigFn, log2_partitions: u32, offline: bool) -> io::Result<Self> {
        let partitions = 1 << log2_partitions;
        Ok(Self {
            sig,
            log2_partitions,
            storage: Mutex::new(if offline {
                Storage::File(
                    tempfile::tempfile()?,
                    Blocks {
                        full: vec![vec![]; partitions],
                        partial: vec![vec![]; partitions],
                        len: 0,
                    },
                )
            } else {
                Storage::Memory(vec![vec![]; partitions])
            }),
        })
    }

    /// Returns a set of buffers writing to this store.
    fn buffers(&self) -> Buffers<'_> {
        let partitions = 1 << self.log2_partitions;
        let offline = matches!(*self.storage.lock().unwrap(), Storage::File(..));
        Buffers {
            writer: self,
            offline,
            // A single zeroed allocation, so that pages are allocated only
            // when used, and returned to the system when dropped
            blocks: if offline {
                vec![[0; 2]; partitions * BLOCK]
            } else {
                vec![]
            },
            lens: if offline { vec![0; partitions] } else { vec![] },
            parts: if offline {
                vec![]
            } else {
                vec![vec![]; partitions]
            },
            len: 0,
        }
    }

    /// Returns the store, given the number of records written to it.
    fn finish(self, len: usize) -> Store {
        let mut storage = self.storage.into_inner().unwrap();
        if let Storage::Memory(partitions) = &mut storage {
            partitions.iter_mut().for_each(Vec::shrink_to_fit);
        }
        Store {
            sig: self.sig,
            log2_partitions: self.log2_partitions,
            len,
            storage,
        }
    }
}

/// The buffers of a thread writing to a [`StoreWriter`]: in offline mode, a
/// block for each partition, which is appended to the file when full;
/// otherwise, the records of each partition, which are moved to the store
/// at the end.
struct Buffers<'a> {
    writer: &'a StoreWriter,
    /// Whether the store is kept in a file.
    offline: bool,
    /// In offline mode, a block for each partition.
    blocks: Vec<Record>,
    /// In offline mode, the number of records in the block of each
    /// partition.
    lens: Vec<u32>,
    /// Otherwise, the records of each partition.
    parts: Vec<Vec<Record>>,
    /// The number of records pushed.
    len: usize,
}

impl Buffers<'_> {
    /// Adds a record to the store.
    #[inline(always)]
    fn push(&mut self, r: Record) -> io::Result<()> {
        let p = partition(self.writer.sig.sig(&r), self.writer.log2_partitions);
        self.len += 1;
        if !self.offline {
            self.parts[p].push(r);
            return Ok(());
        }
        let len = &mut self.lens[p];
        self.blocks[p * BLOCK + *len as usize] = r;
        *len += 1;
        if *len as usize == BLOCK {
            self.flush(p)?;
        }
        Ok(())
    }

    /// Appends the block of partition `p` to the file (if it is not full,
    /// the rest of the block contains garbage).
    fn flush(&mut self, p: usize) -> io::Result<()> {
        let len = self.lens[p];
        let mut storage = self.writer.storage.lock().unwrap();
        let Storage::File(file, blocks) = &mut *storage else {
            unreachable!("blocks are written only to files")
        };
        if blocks.len == u32::MAX {
            return Err(io::Error::other("too many blocks"));
        }
        file.write_all(self.blocks[p * BLOCK..][..BLOCK].as_bytes())?;
        if len as usize == BLOCK {
            blocks.full[p].push(blocks.len);
        } else {
            blocks.partial[p].push((blocks.len, len));
        }
        blocks.len += 1;
        self.lens[p] = 0;
        Ok(())
    }

    /// Moves the remaining records to the store, and returns the number of
    /// records pushed.
    fn finish(mut self) -> io::Result<usize> {
        if self.offline {
            for p in 0..self.lens.len() {
                if self.lens[p] != 0 {
                    self.flush(p)?;
                }
            }
        } else {
            let mut storage = self.writer.storage.lock().unwrap();
            let Storage::Memory(partitions) = &mut *storage else {
                unreachable!("records are moved only to memory")
            };
            for (partition, part) in partitions.iter_mut().zip(&mut self.parts) {
                if partition.is_empty() {
                    std::mem::swap(partition, part);
                } else {
                    partition.append(part);
                }
            }
        }
        Ok(self.len)
    }
}

/// The parts of a level whose records are in a [`Store`]: each part is made
/// of consecutive partitions, so that it contains at least 2 · [`WINDOW`]
/// buckets (unless there is a single part), and part *p* contains the
/// buckets from the bucket of the first signature of its first partition
/// (included) to that of the first signature of the following part
/// (excluded).
///
/// The keys of the first bucket of a part can thus have their signature in
/// the previous part: they are passed by the loader of the previous part, if
/// it was loaded just before, or found by reading it.
struct Partitioned<'a> {
    /// The records of the level.
    store: &'a Store,
    /// The base-2 logarithm of the number of parts.
    log2_parts: u32,
    /// The first bucket of each part, followed by the number of buckets.
    first: Vec<usize>,
    /// The number of buckets.
    buckets: usize,
    /// The base-2 logarithm of the minimum number of buckets of a part.
    log2_min_buckets: u32,
    /// The first error happened while reading the store, if any.
    error: Mutex<Option<io::Error>>,
}

impl<'a> Partitioned<'a> {
    /// Creates the parts of a level with the given geometry whose records
    /// are in `store`.
    fn new(store: &'a Store, g: &Geometry) -> Self {
        let mut log2_parts = store.log2_partitions;
        while log2_parts > 0 && g.buckets >> log2_parts < 2 * WINDOW {
            log2_parts -= 1;
        }
        let first: Vec<usize> = (0..1usize << log2_parts)
            .map(|p| {
                if p == 0 {
                    0
                } else {
                    g.bucket((p as u64) << (64 - log2_parts))
                }
            })
            .chain(std::iter::once(g.buckets))
            .collect();
        let min_buckets = first.windows(2).map(|w| w[1] - w[0]).min().unwrap();
        Self {
            store,
            log2_parts,
            first,
            buckets: g.buckets,
            log2_min_buckets: min_buckets.ilog2(),
            error: Mutex::new(None),
        }
    }

    /// Calls `f` on the signatures of the keys whose record is in partition
    /// `q` of the store, recording the first error.
    fn read(&self, q: usize, records: &mut Vec<Record>, mut f: impl FnMut(u64)) {
        let sig = self.store.sig;
        if let Err(e) = self.store.for_each(q, records, |r| {
            f(sig.sig(r));
            Ok(())
        }) {
            self.error.lock().unwrap().get_or_insert(e);
        }
    }
}

/// The state of a loader of the parts of a [`Partitioned`] level.
#[derive(Default)]
struct Loader {
    /// The part following the last loaded one.
    next: Option<usize>,
    /// The signatures of the keys of the first bucket of part `next` found
    /// in the partitions of the last loaded part.
    carry: Vec<u64>,
    /// The signatures to distribute into buckets.
    sigs: Vec<u64>,
    /// A buffer for reading records.
    records: Vec<Record>,
}

impl PartSource for Partitioned<'_> {
    type Loader = Loader;

    fn log2_min_buckets(&self) -> u32 {
        self.log2_min_buckets
    }

    #[inline(always)]
    fn part(&self, b: usize) -> usize {
        // The largest p such that the first bucket of part p, that is,
        // ⌊p · buckets / 2^log2_parts⌋, is at most b (in 64-bit arithmetic,
        // as the shift could overflow on 32-bit platforms)
        (((((b + 1) as u64) << self.log2_parts) - 1) / self.buckets as u64) as usize
    }

    #[inline(always)]
    fn first_bucket(&self, p: usize) -> usize {
        self.first[p]
    }

    #[inline(always)]
    fn buckets(&self, p: usize, _g: &Geometry) -> usize {
        self.first[p + 1] - self.first[p]
    }

    fn load(
        &self,
        loader: &mut Loader,
        p: usize,
        g: &Geometry,
        range: std::ops::Range<usize>,
        sigs: &mut Vec<u64>,
        begin: &mut Vec<usize>,
        temp: &mut Vec<u64>,
    ) {
        let (first, end) = (self.first[p], self.first[p + 1]);
        // The base-2 logarithm of the number of partitions of a part
        let c = self.store.log2_partitions - self.log2_parts;
        let Loader {
            next,
            carry,
            sigs: buf,
            records,
        } = loader;
        buf.clear();
        // The keys of the first bucket whose signature is in the previous
        // part
        if p > 0 && range.contains(&first) {
            if *next == Some(p) {
                buf.extend_from_slice(carry);
            } else {
                for q in (p - 1) << c..p << c {
                    self.read(q, records, |h| {
                        if g.bucket(h) == first {
                            buf.push(h);
                        }
                    });
                }
            }
        }
        // The keys of the last bucket of a part (the first of the following
        // one) can have their signature in the following part
        carry.clear();
        for q in p << c..(p + 1) << c {
            self.read(q, records, |h| {
                let b = g.bucket(h);
                if b == end {
                    carry.push(h);
                } else if range.contains(&b) {
                    buf.push(h);
                }
            });
        }
        *next = Some(p + 1);
        distribute(&[buf.as_slice()], g, range, sigs, begin, temp);
    }
}

/// The records of the keys of a level after the first one.
enum Keys {
    /// The records of a level that can bump keys.
    Stored(Store),
    /// The records of the last level.
    Last(Vec<Record>),
}

/// Assigns seeds to the buckets of the level of index `index` whose records
/// are in a store, logging its progress (see [`sweep_bumping_level`]).
fn sweep_store(
    store: &Store,
    g: &Geometry,
    weights: &[i64; 7],
    index: usize,
    pl: &mut impl ProgressLog,
) -> Result<Level> {
    let parts = Partitioned::new(store, g);
    let out = sweep_bumping_level(&parts, g, weights, index, store.len, pl);
    if let Some(e) = parts.error.into_inner().unwrap() {
        return Err(e.into());
    }
    Ok(out)
}

/// Applies `f` to disjoint ranges of the partitions of a store, one for each
/// thread, possibly in parallel, and returns the results in order.
fn by_thread<T: Send>(
    store: &Store,
    f: impl Fn(std::ops::Range<usize>) -> io::Result<T> + Sync,
) -> io::Result<Vec<T>> {
    let (np, threads) = (
        store.num_partitions(),
        num_threads().min(store.num_partitions()),
    );
    let ranges: Vec<_> = (0..threads)
        .map(|t| t * np / threads..(t + 1) * np / threads)
        .collect();
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        ranges.into_par_iter().map(&f).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        ranges.into_iter().map(f).collect()
    }
}

/// Returns the records of the `k` keys whose bucket has no seed, given the
/// salt of the following level: in a new store (kept in a file if `offline`
/// is true), or in a vector, if the following level is the last one. The
/// progress of the pass, in records read, is logged on a concurrent logger
/// obtained from `pl`.
fn collect_bumped(
    store: &Store,
    g: &Geometry,
    seeds: &[u16],
    k: usize,
    salt: u64,
    offline: bool,
    pl: &mut impl ProgressLog,
) -> Result<Keys> {
    let sig = store.sig;
    let mut cpl = pl.concurrent();
    cpl.item_name("key");
    cpl.expected_updates(store.len);
    cpl.start(format!(
        "Collecting the {k} keys of buckets without a seed..."
    ));
    let keys = if k <= LAST_LEVEL_THRESHOLD {
        let parts = by_thread(store, |range| {
            let (mut out, mut records, mut pl) = (vec![], vec![], cpl.clone());
            for p in range {
                store.for_each(p, &mut records, |r| {
                    if seeds[g.bucket(sig.sig(r))] == 0 {
                        out.push(*r);
                    }
                    Ok(())
                })?;
                pl.update_with_count(store.partition_len(p));
            }
            Ok(out)
        })?;
        let records = parts.concat();
        debug_assert_eq!(records.len(), k);
        Keys::Last(records)
    } else {
        let writer = StoreWriter::new(SigFn(Some(salt)), log2_partitions(k), offline)?;
        let lens = by_thread(store, |range| {
            let (mut buffers, mut records, mut pl) = (writer.buffers(), vec![], cpl.clone());
            for p in range {
                store.for_each(p, &mut records, |r| {
                    if seeds[g.bucket(sig.sig(r))] == 0 {
                        buffers.push(*r)?;
                    }
                    Ok(())
                })?;
                pl.update_with_count(store.partition_len(p));
            }
            buffers.finish()
        })?;
        let len = lens.iter().sum();
        debug_assert_eq!(len, k);
        Keys::Stored(writer.finish(len))
    };
    cpl.done();
    Ok(keys)
}

/// Returns whether two records of a store are equal.
///
/// The construction from a slice considers two keys duplicates if they have
/// the same signature for a level and the same hash with a third seed; here,
/// as keys are no longer available, we consider two keys duplicates if they
/// have the same signature for the first level and the same second hash.
/// The two criteria give different results only if there are two distinct
/// keys with the same 128 bits of hashes; in this case, the construction
/// from a slice would fail anyway (such keys would collide at every level),
/// unless their hashes with the third seed coincide, too.
///
/// The progress of the check, in records, is logged on a concurrent logger
/// obtained from `pl`.
fn has_duplicates(store: &Store, pl: &mut impl ProgressLog) -> Result<bool> {
    let mut cpl = pl.concurrent();
    cpl.item_name("key");
    cpl.expected_updates(store.len);
    cpl.start("Checking for duplicate keys...");
    // Equal records have the same signature, and thus the same partition
    let found = by_thread(store, |range| {
        let (mut all, mut records, mut pl) = (vec![], vec![], cpl.clone());
        for p in range {
            all.clear();
            store.for_each(p, &mut records, |r| {
                all.push(*r);
                Ok(())
            })?;
            all.sort_unstable();
            if all.windows(2).any(|w| w[0] == w[1]) {
                return Ok(true);
            }
            pl.update_with_count(all.len());
        }
        Ok(false)
    })?;
    cpl.done();
    Ok(found.into_iter().any(|x| x))
}

/// Returns whether two records in a slice are equal (see
/// [`has_duplicates`]).
fn has_duplicates_in_memory(records: &[Record]) -> bool {
    let mut all = records.to_vec();
    all.sort_unstable();
    all.windows(2).any(|w| w[0] == w[1])
}

impl PHastRBuilder {
    /// Builds a function on the keys returned by a lender, which is read
    /// just once (see [`PHastR::try_new_with_builder`]).
    ///
    /// Records take sixteen bytes per key, in memory or, in offline mode, on
    /// disk. In offline mode, besides buffers whose size does not depend on
    /// the number of keys, construction needs in memory about 0.7 bytes per
    /// key with the default parameters while building the first level (the
    /// seeds of its buckets, as 16-bit values, and its set of used slots),
    /// and then little more than the function itself.
    pub(super) fn try_build_from_lender<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStoreBuild,
        B: ?Sized + Borrow<K>,
    >(
        &self,
        keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        pl: &mut impl ProgressLog,
    ) -> Result<PHastR<K, D>> {
        self.check_params::<D>()?;
        let start = Instant::now();
        self.log_params(pl);
        let (num_keys, params0, seeds0, levels, remap) =
            self.build_levels_from_lender::<K, D, B>(keys, pl)?;
        let func = self.assemble(num_keys, params0, seeds0, levels, remap);
        log_completion(start, num_keys, pl);
        Ok(func)
    }

    /// Builds the levels on the keys returned by a lender (see
    /// [`build_levels`]), returning the number of keys,
    /// the parameters and the seeds of the first level, the following
    /// levels, and the remapping sequence.
    ///
    /// To save memory, the seeds of the first level are stored as soon as
    /// the level is built, its free slots are kept in an Elias–Fano
    /// sequence, and the remapping sequence is built at the end from the
    /// sets of used slots of the following levels, which are small.
    ///
    /// [`build_levels`]: Self::build_levels
    #[allow(clippy::type_complexity)]
    fn build_levels_from_lender<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStoreBuild,
        B: ?Sized + Borrow<K>,
    >(
        &self,
        mut keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        pl: &mut impl ProgressLog,
    ) -> Result<(usize, LevelParams, D, Vec<(LevelParams, Vec<u16>)>, Remap)> {
        let weights = |g: &Geometry| self.priority_weights(g);

        pl.item_name("key");
        pl.expected_updates(None);
        pl.start(format!(
            "Computing and storing 128-bit signatures sequentially {} using seed 0x{:016x}...",
            if self.offline {
                format!("on disk ({})", std::env::temp_dir().display())
            } else {
                "in RAM".to_owned()
            },
            self.seed
        ));
        let start = Instant::now();
        let writer = StoreWriter::new(SigFn(None), MAX_LOG2_PARTITIONS, self.offline)?;
        let mut buffers = writer.buffers();
        while let Some(key) = keys.next()? {
            let key: &K = key.borrow();
            buffers.push([
                K::to_sig(key, self.seed)[0],
                K::to_sig(key, self.seed ^ SECOND_HASH)[0],
            ])?;
            pl.light_update();
        }
        let n = buffers.finish()?;
        let store = writer.finish(n);
        pl.done();
        log_signatures(start, n, pl);

        if n == 0 {
            // A single empty level, as in build_levels
            let geom = self.geometry(0, 1, self.bucket_size);
            let seeds0 = D::from_seeds(&[0], self.seed_bits);
            let efb = EliasFanoBuilder::new(0, 1);
            return Ok((0, geom.level(), seeds0, vec![], remap(efb)));
        }

        // The first level
        let geom = self.geometry(n, n, self.bucket_size);
        let Level { seeds, occupied } = sweep_store(&store, &geom, &weights(&geom), 0, pl)?;
        let num_holes = n - occupied
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();
        let mut efb = EliasFanoBuilder::new(num_holes, n);
        for (w, &word) in occupied.iter().enumerate() {
            let mut free = !word;
            if n - w * 64 < 64 {
                free &= (1u64 << (n - w * 64)) - 1;
            }
            while free != 0 {
                efb.push(w * 64 + free.trailing_zeros() as usize);
                free &= free - 1;
            }
        }
        let holes = efb.build();
        drop(occupied);
        pl.info(format_args!(
            "Level 0: {} keys, {} bumped ({:.3}%)",
            n,
            num_holes,
            100.0 * num_holes as f64 / n as f64
        ));
        let mut cur = collect_bumped(
            &store,
            &geom,
            &seeds,
            num_holes,
            level_salt(self.seed, 1, 0),
            self.offline,
            pl,
        )?;
        drop(store);
        let params0 = geom.level();
        let seeds0 = D::from_seeds(&seeds, self.seed_bits);
        drop(seeds);

        // The following levels, as in build_levels, keeping their sets of
        // used slots and their output ranges
        let mut levels: Vec<(LevelParams, Vec<u16>)> = vec![];
        let mut used: Vec<(Vec<u64>, usize)> = vec![];
        let mut offset = 0;
        loop {
            // The index of the level
            let index = levels.len() + 1;
            let (mut level, seeds, occupied, next, k, m) = match cur {
                Keys::Stored(store) => {
                    let k = store.len;
                    let salt = level_salt(self.seed, index, 0);
                    let geom = self.geometry(k, k, self.bucket_size);
                    if has_duplicates(&store, pl)? {
                        bail!("Duplicate keys");
                    }
                    let out = sweep_store(&store, &geom, &weights(&geom), index, pl)?;
                    let used: usize = out.occupied.iter().map(|w| w.count_ones() as usize).sum();
                    let next = collect_bumped(
                        &store,
                        &geom,
                        &out.seeds,
                        k - used,
                        level_salt(self.seed, index + 1, 0),
                        self.offline,
                        pl,
                    )?;
                    let mut level = geom.level();
                    level.salt = salt;
                    (level, out.seeds, out.occupied, next, k, geom.m)
                }
                Keys::Last(records) => {
                    if records.is_empty() {
                        break;
                    }
                    let k = records.len();
                    // The last level does not bump: we enlarge the range and
                    // change the salt until we succeed
                    let mut attempt = 0u64;
                    loop {
                        let m = k + k / 4 + 16 + (attempt as usize / 8) * (k / 8 + 8);
                        let geom = self.geometry(k, m, self.bucket_size.min(3.0));
                        let salt = level_salt(self.seed, index, attempt);
                        if attempt == 0 && has_duplicates_in_memory(&records) {
                            bail!("Duplicate keys");
                        }
                        let sigs =
                            group(k, |i| level_sig(records[i][0], records[i][1], salt), &geom);
                        if let Some(out) =
                            sweep_level(&sigs, &geom, &weights(&geom), false, no_logging![])
                        {
                            let mut level = geom.level();
                            level.salt = salt;
                            break (
                                level,
                                out.seeds,
                                out.occupied,
                                Keys::Last(vec![]),
                                k,
                                geom.m,
                            );
                        }
                        attempt += 1;
                        if attempt > 1000 {
                            bail!("Could not build the last level");
                        }
                    }
                }
            };

            let num_bumped = match &next {
                Keys::Stored(store) => store.len,
                Keys::Last(records) => records.len(),
            };
            pl.info(format_args!(
                "Level {}: {} keys, {} bumped ({:.3}%)",
                index,
                k,
                num_bumped,
                100.0 * num_bumped as f64 / k as f64
            ));

            level.offset = offset as u64;
            offset += m;
            levels.push((level, seeds));
            used.push((occupied, m));
            cur = next;
        }

        // The remapping sequence, as in build_levels: the used slots of the
        // following levels are mapped, in order, to the free slots of the
        // first level, and their free slots to the last free slot assigned
        let mut efb = EliasFanoBuilder::new(offset, n);
        let mut holes = holes.iter();
        let mut last_hole = 0;
        for (occupied, m) in &used {
            for p in 0..*m {
                if is_occupied(occupied, p) {
                    last_hole = holes
                        .next()
                        .expect("there is a free slot for each bumped key");
                }
                efb.push(last_hole);
            }
        }
        debug_assert!(holes.next().is_none());
        Ok((n, params0, seeds0, levels, remap(efb)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::CollidingKey;
    use super::*;
    use crate::utils::FromSlice;
    use dsi_progress_logger::{ProgressLogger, no_logging};

    /// Asserts that two functions are identical.
    fn assert_same<K: ?Sized, D: SeedStore>(a: &PHastR<K, D>, b: &PHastR<K, D>) {
        let fields = |f: &PHastR<K, D>| {
            format!(
                "{:?}",
                (
                    f.seed,
                    f.num_keys,
                    f.pattern_shift,
                    f.default_shifts,
                    f.fast_scale,
                    f.params0,
                    &f.params,
                )
            )
        };
        assert_eq!(fields(a), fields(b));
        for i in 0..a.params0.buckets as usize {
            // SAFETY: i is smaller than the number of buckets
            unsafe { assert_eq!(a.seeds0.get_seed(i), b.seeds0.get_seed(i), "seed {i}") };
        }
        let seeds: u64 = a.params.iter().map(|p| p.buckets).sum();
        for i in 0..seeds as usize {
            // SAFETY: i is smaller than the number of seeds
            unsafe { assert_eq!(a.seeds.get_seed(i), b.seeds.get_seed(i), "seed {i}") };
        }
        assert_eq!(SliceByValue::len(&a.remap), SliceByValue::len(&b.remap));
        for i in 0..SliceByValue::len(&a.remap) {
            assert_eq!(
                SliceByValue::index_value(&a.remap, i),
                SliceByValue::index_value(&b.remap, i),
                "entry {i}"
            );
        }
    }

    /// Builds a function from a slice and from a lender, in memory and
    /// offline, checking that the functions are identical and correct.
    fn check<K: ?Sized + ToSig<[u64; 1]> + Sync, D: SeedStoreBuild>(
        keys: &[&K],
        builder: PHastRBuilder,
    ) -> Result<()> {
        let par = <PHastR<K, D>>::try_par_new_with_builder(keys, builder.clone(), no_logging![])?;
        for offline in [false, true] {
            let phf = <PHastR<K, D>>::try_new_with_builder(
                FromSlice::new(keys),
                builder.clone().offline(offline),
                no_logging![],
            )?;
            assert_same(&phf, &par);
        }
        let mut seen = vec![false; keys.len()];
        for &key in keys {
            let v = par.get(key);
            assert!(!seen[v], "duplicate output {v}");
            seen[v] = true;
        }
        Ok(())
    }

    fn check_u64<D: SeedStoreBuild>(n: usize, builder: PHastRBuilder) -> Result<()> {
        let keys: Vec<u64> = (0..n as u64).collect();
        let refs: Vec<&u64> = keys.iter().collect();
        check::<u64, D>(&refs, builder)
    }

    #[test]
    fn test_sizes() -> Result<()> {
        for n in [
            0, 1, 2, 3, 10, 100, 1000, 4096, 4097, 5000, 10_000, 100_000, 1_000_000,
        ] {
            check_u64::<Box<[u8]>>(n, PHastRBuilder::default())?;
        }
        Ok(())
    }

    #[test]
    fn test_parameters() -> Result<()> {
        for n in [1000, 100_000] {
            check_u64::<BitFieldVec<Box<[usize]>>>(n, PHastRBuilder::default().seed_bits(10))?;
            for log2_patterns in 0..=3 {
                check_u64::<Box<[u8]>>(
                    n,
                    PHastRBuilder::default()
                        .log2_patterns(log2_patterns)
                        .log2_slice_len(8),
                )?;
            }
        }
        check_u64::<BitFieldVec<Box<[usize]>>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        )?;
        check_u64::<Box<[u16]>>(
            300_000,
            PHastRBuilder::default()
                .seed_bits(11)
                .log2_slice_len(12)
                .bucket_size(6.75),
        )?;
        check_u64::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(16))?;
        check_u64::<Box<[u8]>>(
            100_000,
            PHastRBuilder::default()
                .seed_bits(4)
                .log2_slice_len(8)
                .bucket_size(2.0),
        )?;
        check_u64::<Box<[u8]>>(300_000, PHastRBuilder::default().log2_slice_len(6))?;
        check_u64::<Box<[u16]>>(
            20_000,
            PHastRBuilder::default()
                .seed_bits(16)
                .log2_slice_len(16)
                .bucket_size(70.0),
        )?;
        check_u64::<Box<[u8]>>(100_000, PHastRBuilder::default().bucket_size(70.0))?;
        check_u64::<Box<[u8]>>(100_000, PHastRBuilder::default().seed(42))
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_threads() -> Result<()> {
        // Chunks and gaps depend on the number of threads
        for threads in [1, 2, 3, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?;
            pool.install(|| {
                check_u64::<Box<[u8]>>(2_000_000, PHastRBuilder::default())?;
                check_u64::<BitFieldVec<Box<[usize]>>>(
                    1_000_000,
                    PHastRBuilder::default()
                        .seed_bits(10)
                        .log2_slice_len(11)
                        .bucket_size(6.0),
                )
            })?;
        }
        Ok(())
    }

    #[test]
    fn test_strings() -> Result<()> {
        let keys: Vec<String> = (0..200_000).map(|i| format!("key{i}")).collect();
        let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let par = <PHastR<str>>::try_par_new(&refs, no_logging![])?;
        // Lines from a reader
        for offline in [false, true] {
            let phf = <PHastR<str>>::try_new_with_builder(
                crate::utils::LineLender::new(std::io::Cursor::new(keys.join("\n"))),
                PHastRBuilder::default().offline(offline),
                no_logging![],
            )?;
            assert_same(&phf, &par);
        }
        Ok(())
    }

    #[test]
    fn test_signature_collisions() -> Result<()> {
        for n in [5_000u64, 300_000] {
            let keys: Vec<CollidingKey> = (0..n).map(CollidingKey).collect();
            let refs: Vec<&CollidingKey> = keys.iter().collect();
            check::<CollidingKey, Box<[u8]>>(&refs, PHastRBuilder::default())?;
        }
        Ok(())
    }

    #[test]
    fn test_duplicates() {
        // Duplicates detected at a level kept in a vector and at a level
        // kept in a store
        for offline in [false, true] {
            for n in [1000u64, 100_000] {
                let mut keys: Vec<u64> = (0..n).collect();
                keys.push(n / 2);
                assert!(
                    <PHastR<u64>>::try_new_with_builder(
                        FromSlice::new(&keys),
                        PHastRBuilder::default().offline(offline),
                        no_logging![],
                    )
                    .is_err()
                );
            }
            let mut keys: Vec<CollidingKey> = (0..100_000).map(CollidingKey).collect();
            keys.push(CollidingKey(50_000));
            assert!(
                <PHastR<CollidingKey>>::try_new_with_builder(
                    FromSlice::new(&keys),
                    PHastRBuilder::default().offline(offline),
                    no_logging![],
                )
                .is_err()
            );
        }
    }

    #[test]
    fn test_logging() -> Result<()> {
        // With half a million keys the second level can bump keys, so all
        // phases are logged; logging does not change the function
        let keys: Vec<u64> = (0..500_000).collect();
        let builder = PHastRBuilder::default();
        let quiet = <PHastR<u64>>::try_par_new_with_builder(&keys, builder.clone(), no_logging![])?;
        let mut pl = ProgressLogger::default();
        let par = <PHastR<u64>>::try_par_new_with_builder(&keys, builder.clone(), &mut pl)?;
        assert_same(&par, &quiet);
        for offline in [false, true] {
            let phf = <PHastR<u64>>::try_new_with_builder(
                FromSlice::new(&keys),
                builder.clone().offline(offline),
                &mut pl,
            )?;
            assert_same(&phf, &quiet);
        }
        Ok(())
    }

    #[test]
    fn test_parts() -> Result<()> {
        // Parts are consistent with buckets, and their records are those of
        // their buckets
        for offline in [false, true] {
            for n in [1, 100, 3000, 100_000, 1_000_000] {
                let g = PHastRBuilder::default().geometry(n, n, 4.5);
                let writer = StoreWriter::new(SigFn(None), MAX_LOG2_PARTITIONS, offline)?;
                let mut buffers = writer.buffers();
                let sig = |i: usize| mix(i as u64 + 1, 0x9E37_79B9_7F4A_7C15);
                for i in 0..n {
                    buffers.push([sig(i), i as u64])?;
                }
                let len = buffers.finish()?;
                let store = writer.finish(len);
                let parts = Partitioned::new(&store, &g);
                assert_eq!(parts.first[0], 0);
                for p in 0..parts.first.len() - 1 {
                    assert!(parts.buckets(p, &g) >= 1 << parts.log2_min_buckets);
                    for b in parts.first[p]..parts.first[p + 1] {
                        assert_eq!(parts.part(b), p, "bucket {b}");
                    }
                }
                // Loading all parts in order, and then each part on its own
                let mut all = vec![];
                let (mut sigs, mut begin, mut temp) = (vec![], vec![], vec![]);
                let mut loader = Loader::default();
                for p in 0..parts.first.len() - 1 {
                    let range = parts.first[p]..parts.first[p + 1];
                    parts.load(
                        &mut loader,
                        p,
                        &g,
                        range.clone(),
                        &mut sigs,
                        &mut begin,
                        &mut temp,
                    );
                    let mut alone = Loader::default();
                    let (mut s, mut bg, mut t) = (vec![], vec![], vec![]);
                    parts.load(&mut alone, p, &g, range.clone(), &mut s, &mut bg, &mut t);
                    assert_eq!(bg, begin);
                    for (i, w) in begin.windows(2).enumerate() {
                        let mut x = sigs[w[0]..w[1]].to_vec();
                        let mut y = s[w[0]..w[1]].to_vec();
                        x.sort_unstable();
                        y.sort_unstable();
                        assert_eq!(y, x);
                        for &h in &x {
                            assert_eq!(g.bucket(h), range.start + i);
                        }
                    }
                    all.extend_from_slice(&sigs[..begin[begin.len() - 1]]);
                }
                assert!(parts.error.into_inner()?.is_none());
                let mut expected: Vec<u64> = (0..n).map(sig).collect();
                expected.sort_unstable();
                all.sort_unstable();
                assert_eq!(all, expected);
            }
        }
        Ok(())
    }
}
