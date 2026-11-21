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
//! range can have keys in the adjacent partitions. Sweeps load *parts* made
//! of consecutive partitions (see [`Partitioned`]) in increasing order,
//! passing the keys of the last bucket of a part to the following one. On
//! disk, each partition is a sequence of blocks in a temporary file (see
//! [`LOG2_FILES`]).
//!
//! The function built is identical to that built by
//! [`PHastR::try_par_new_with_builder`] with the same builder, provided that
//! the current [rayon] pool has the same number of threads, or that the
//! builder sets the [number of sweep chunks]: the seed chosen for a bucket
//! depends on the multiset of the signatures of its keys, but not on their
//! order, and sweeps are split into the same chunks. The only exception is
//! the detection of duplicate keys, which uses the records (see
//! [`has_duplicates`]).
//!
//! A construction can also keep a value for each key, next to its record
//! in the store of the first level (see [`Item`]), and keep that store until
//! the function is built (see [`StoredValues`]): since the output of a key
//! depends just on its record, the values can then be written at the outputs
//! of their keys in parallel, without reading the keys again. This is how
//! [`PHunc`] is built from lenders.
//!
//! [offline mode]: PHastRBuilder::offline
//! [rayon]: https://docs.rs/rayon
//! [number of sweep chunks]: PHastRBuilder::sweep_chunks
//! [`PHunc`]: crate::func::PHunc

use super::builder::*;
use super::sigs::*;
use super::sweep::*;
use super::*;
use crate::dict::elias_fano::{EliasFano, EliasFanoBuilder};
use crate::utils::BinSafe;
use anyhow::{Context, bail};
use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use zerocopy::IntoBytes;

/// The signature of a key for the first level and its second hash.
type Record = [u64; 2];

/// An item of a [`Store`]: the record of a key, possibly with other data
/// (see [`WithValue`]).
trait Item: BinSafe {
    /// Returns the record of the key.
    fn record(&self) -> Record;
}

impl Item for Record {
    #[inline(always)]
    fn record(&self) -> Record {
        *self
    }
}

/// The record of a key of the first level, followed by the value of the key
/// (see [`StoredValues`]).
///
/// The structure is packed, so it has no padding whatever the type of the
/// value, and its fields must be read by copy.
#[cfg(feature = "rayon")]
#[derive(zerocopy::FromBytes, zerocopy::IntoBytes)]
#[repr(C, packed)]
struct WithValue<V> {
    record: Record,
    value: V,
}

#[cfg(feature = "rayon")]
impl<V: Copy> Clone for WithValue<V> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "rayon")]
impl<V: Copy> Copy for WithValue<V> {}

#[cfg(feature = "rayon")]
impl<V: BinSafe> Item for WithValue<V> {
    #[inline(always)]
    fn record(&self) -> Record {
        self.record
    }
}

/// Returns a vector of `len` zeroed items of a [`Store`] in a single zeroed
/// allocation, so that pages are allocated only when used, and returned to
/// the system when dropped (`vec!` does that just for some types).
fn zeroed<T: Item>(len: usize) -> Vec<T> {
    T::new_vec_zeroed(len).expect("Could not allocate the buffers of a store")
}

/// Returns the bytes of a slice of items of a [`Store`].
fn bytes<T: Item>(items: &[T]) -> &[u8] {
    // SAFETY: the items have no padding, as they implement IntoBytes, and no
    // interior mutability, as they are Copy
    let (pre, bytes, post) = unsafe { items.align_to::<u8>() };
    debug_assert!(pre.is_empty() && post.is_empty());
    bytes
}

/// The base-2 logarithm of the maximum number of partitions of a store,
/// which is the number of partitions of the first level (whose number of
/// keys is not known in advance): in offline mode, with blocks of [`BLOCK`]
/// records, the buffers of a writer take 256 MiB (more, if the items of the
/// store contain also a value), and partitions are small enough to be loaded
/// quickly by a sweep up to about 10¹¹ keys.
#[cfg(not(test))]
const MAX_LOG2_PARTITIONS: u32 = 14;
#[cfg(test)]
const MAX_LOG2_PARTITIONS: u32 = 5;

/// The initial number of records of the blocks of a writer.
#[cfg(not(test))]
const BLOCK: usize = 1024;
#[cfg(test)]
const BLOCK: usize = 16;

/// The maximum number of records of the blocks of a writer.
#[cfg(not(test))]
const MAX_BLOCK: usize = 1 << 16;
#[cfg(test)]
const MAX_BLOCK: usize = 64;

/// A writer doubles the size of its blocks (up to [`MAX_BLOCK`] records)
/// when it has written on average this number of blocks for each
/// partition: on huge key sets blocks become large, so that reading a
/// partition from a disk requires few seeks, whereas the memory used by the
/// buffers, a block for each partition, is at most a sixteenth of a byte
/// per record written, besides the initial blocks (in proportion more, if
/// the items of the store contain also a value).
#[cfg(not(test))]
const GROWTH: usize = 512;
#[cfg(test)]
const GROWTH: usize = 2;

/// Returns whether a writer that has written `len` records, using blocks of
/// `block` records for each of `partitions` partitions, has written on
/// average at least `growth` blocks for each partition (see [`GROWTH`]).
///
/// We divide, rather than multiply, as the product of the last three
/// arguments overflows on 32-bit platforms.
#[inline(always)]
const fn grows(len: usize, partitions: usize, block: usize, growth: usize) -> bool {
    len / (partitions * block) >= growth
}

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

/// The base-2 logarithm of the maximum number of temporary files of a
/// [`Store`] kept on disk.
///
/// Each file contains the blocks of a range of consecutive partitions. Since
/// a read moves the current position of its file, threads reading the same
/// file take turns; however, threads sweeping different chunks of a level,
/// or scanning different ranges of partitions, read different files.
///
/// Taking turns is not a bottleneck: with 96 threads, 16 and 64 files yield
/// the same construction time, both when the files are in the page cache
/// and when they are read from disk, where the device is the bound. On
/// volumes with a limit on operations per second, such as Amazon EBS, the
/// bound is the number of reads, which is high up to about 10¹⁰ keys, as
/// blocks are still small (see [`GROWTH`]). More files would also need
/// more file descriptors: two stores are open at the same time, and the
/// default limit on macOS is 256.
#[cfg(not(test))]
const LOG2_FILES: u32 = 4;
#[cfg(test)]
const LOG2_FILES: u32 = 2;

/// The table of the blocks of the partitions of a [`Store`] kept on disk.
///
/// Blocks have different lengths (see [`GROWTH`]), and are identified by
/// their position in the file of their partition and their length, both in
/// records.
struct Blocks {
    /// For each partition, the position and the length of its blocks.
    blocks: Vec<Vec<(u64, u32)>>,
    /// The number of records of each file.
    lens: Vec<u64>,
    /// The file of a partition is given by its index shifted right by this
    /// amount.
    shift: u32,
}

/// The storage of the items of a [`Store`].
enum Storage<T = Record> {
    /// Temporary files, each containing the blocks of a range of
    /// consecutive partitions (see [`LOG2_FILES`]).
    Files(Vec<Mutex<File>>, Blocks),
    /// A vector for each partition.
    Memory(Vec<Vec<T>>),
}

/// The records of the keys of a level, partitioned by the upper bits of
/// their signature for the level.
///
/// The items of the store of the first level can contain, besides the
/// record of a key, its value (see [`Item`]).
struct Store<T = Record> {
    /// The signature of a record for the level.
    sig: SigFn,
    /// The base-2 logarithm of the number of partitions.
    log2_partitions: u32,
    /// The number of items.
    len: usize,
    storage: Storage<T>,
}

impl<T: Item> Store<T> {
    /// Returns the number of partitions.
    const fn num_partitions(&self) -> usize {
        1 << self.log2_partitions
    }

    /// Returns the number of items of partition `p`.
    fn partition_len(&self, p: usize) -> usize {
        match &self.storage {
            Storage::Files(_, blocks) => blocks.blocks[p]
                .iter()
                .map(|&(_, len)| len as usize)
                .sum::<usize>(),
            Storage::Memory(partitions) => partitions[p].len(),
        }
    }

    /// Calls `f` on the items of partition `p`, in the order in which they
    /// were written by each thread, using `buf` as a buffer.
    fn for_each(
        &self,
        p: usize,
        buf: &mut Vec<T>,
        mut f: impl FnMut(&T) -> io::Result<()>,
    ) -> io::Result<()> {
        match &self.storage {
            Storage::Files(files, blocks) => {
                for &(pos, len) in &blocks.blocks[p] {
                    buf.resize(len as usize, T::new_zeroed());
                    {
                        let mut file = files[p >> blocks.shift].lock().unwrap();
                        file.seek(io::SeekFrom::Start(pos * size_of::<T>() as u64))?;
                        file.read_exact(buf.as_mut_bytes())?;
                    }
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

/// Writes items to a new [`Store`]: several threads can write at the same
/// time, each using its own [`Buffers`].
struct StoreWriter<T = Record> {
    sig: SigFn,
    log2_partitions: u32,
    storage: Mutex<Storage<T>>,
}

impl<T: Item> StoreWriter<T> {
    /// Creates a writer for a store with the given signature function and
    /// number of partitions, kept in anonymous temporary files in `dir`, if
    /// it is not `None`, and in memory otherwise.
    fn new(sig: SigFn, log2_partitions: u32, dir: Option<&Path>) -> io::Result<Self> {
        let partitions = 1 << log2_partitions;
        Ok(Self {
            sig,
            log2_partitions,
            storage: Mutex::new(if let Some(dir) = dir {
                let log2_files = log2_partitions.min(LOG2_FILES);
                let files = (0..1 << log2_files)
                    .map(|_| {
                        tempfile::tempfile_in(dir).map(Mutex::new).map_err(|e| {
                            io::Error::new(
                                e.kind(),
                                format!(
                                    "Could not create a temporary file in {}: {e}",
                                    dir.display()
                                ),
                            )
                        })
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                Storage::Files(
                    files,
                    Blocks {
                        blocks: vec![vec![]; partitions],
                        lens: vec![0; 1 << log2_files],
                        shift: log2_partitions - log2_files,
                    },
                )
            } else {
                Storage::Memory(vec![vec![]; partitions])
            }),
        })
    }

    /// Returns a set of buffers writing to this store.
    fn buffers(&self) -> Buffers<'_, T> {
        let partitions = 1 << self.log2_partitions;
        let offline = matches!(*self.storage.lock().unwrap(), Storage::Files(..));
        Buffers {
            writer: self,
            offline,
            // A single zeroed allocation (see zeroed)
            blocks: if offline {
                zeroed(partitions * BLOCK)
            } else {
                vec![]
            },
            lens: if offline { vec![0; partitions] } else { vec![] },
            block: BLOCK,
            parts: if offline {
                vec![]
            } else {
                vec![vec![]; partitions]
            },
            len: 0,
        }
    }

    /// Returns the store, given the number of items written to it.
    fn finish(self, len: usize) -> Store<T> {
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
/// block for each partition, which is appended to its file when full (the
/// length of the blocks grows with the number of items written, see
/// [`GROWTH`]); otherwise, the items of each partition, which are moved to
/// the store at the end.
struct Buffers<'a, T = Record> {
    writer: &'a StoreWriter<T>,
    /// Whether the store is kept on disk.
    offline: bool,
    /// In offline mode, a block for each partition.
    blocks: Vec<T>,
    /// In offline mode, the number of items in the block of each partition.
    lens: Vec<u32>,
    /// In offline mode, the length of the blocks, in items.
    block: usize,
    /// Otherwise, the items of each partition.
    parts: Vec<Vec<T>>,
    /// The number of items pushed.
    len: usize,
}

impl<T: Item> Buffers<'_, T> {
    /// Adds an item to the store.
    #[inline(always)]
    fn push(&mut self, item: T) -> io::Result<()> {
        let p = partition(
            self.writer.sig.sig(&item.record()),
            self.writer.log2_partitions,
        );
        if self.len == MAX_KEYS {
            return Err(io::Error::other("Too many keys for this platform"));
        }
        self.len += 1;
        if !self.offline {
            self.parts[p].push(item);
            return Ok(());
        }
        let len = &mut self.lens[p];
        self.blocks[p * self.block + *len as usize] = item;
        *len += 1;
        if *len as usize == self.block {
            self.flush(p)?;
            if self.block < MAX_BLOCK && grows(self.len, self.lens.len(), self.block, GROWTH) {
                self.grow()?;
            }
        }
        Ok(())
    }

    /// Appends the items in the block of partition `p` to its file.
    fn flush(&mut self, p: usize) -> io::Result<()> {
        let len = self.lens[p];
        let mut storage = self.writer.storage.lock().unwrap();
        let Storage::Files(files, blocks) = &mut *storage else {
            unreachable!("blocks are written only to files")
        };
        let f = p >> blocks.shift;
        files[f]
            .get_mut()
            .unwrap()
            .write_all(bytes(&self.blocks[p * self.block..][..len as usize]))?;
        blocks.blocks[p].push((blocks.lens[f], len));
        blocks.lens[f] += len as u64;
        self.lens[p] = 0;
        Ok(())
    }

    /// Doubles the length of the blocks, after flushing the blocks that are
    /// not empty.
    fn grow(&mut self) -> io::Result<()> {
        for p in 0..self.lens.len() {
            if self.lens[p] != 0 {
                self.flush(p)?;
            }
        }
        self.block *= 2;
        // The old buffers are freed first; the new ones are a zeroed
        // allocation (see zeroed)
        self.blocks = vec![];
        self.blocks = zeroed(self.lens.len() * self.block);
        Ok(())
    }

    /// Moves the remaining items to the store, and returns the number of
    /// items pushed.
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
                unreachable!("items are moved only to memory")
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
struct Partitioned<'a, T = Record> {
    /// The records of the level.
    store: &'a Store<T>,
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

impl<'a, T: Item> Partitioned<'a, T> {
    /// Creates the parts of a level with the given geometry whose records
    /// are in `store`.
    fn new(store: &'a Store<T>, g: &Geometry) -> Self {
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
    fn read(&self, q: usize, items: &mut Vec<T>, mut f: impl FnMut(u64)) {
        let sig = self.store.sig;
        if let Err(e) = self.store.for_each(q, items, |item| {
            f(sig.sig(&item.record()));
            Ok(())
        }) {
            self.error.lock().unwrap().get_or_insert(e);
        }
    }
}

/// The state of a loader of the parts of a [`Partitioned`] level.
struct Loader<T = Record> {
    /// The part following the last loaded one.
    next: Option<usize>,
    /// The signatures of the keys of the first bucket of part `next` found
    /// in the partitions of the last loaded part.
    carry: Vec<u64>,
    /// The signatures to distribute into buckets.
    sigs: Vec<u64>,
    /// A buffer for reading the items of the store.
    records: Vec<T>,
}

// Not derived, as the items need not have a default value
impl<T> Default for Loader<T> {
    fn default() -> Self {
        Self {
            next: None,
            carry: vec![],
            sigs: vec![],
            records: vec![],
        }
    }
}

impl<T: Item> PartSource for Partitioned<'_, T> {
    type Loader = Loader<T>;

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
    fn buckets(&self, p: usize) -> Range<usize> {
        self.first[p]..self.first[p + 1]
    }

    fn load(
        &self,
        loader: &mut Loader<T>,
        p: usize,
        g: &Geometry,
        range: Range<usize>,
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

impl Keys {
    /// Returns the number of keys.
    fn len(&self) -> usize {
        match self {
            Keys::Stored(store) => store.len,
            Keys::Last(records) => records.len(),
        }
    }
}

/// Assigns seeds to the buckets of the level of index `index` whose records
/// are in a store, logging its progress (see [`sweep_bumping_level`]).
fn sweep_store<T: SweepSeed>(
    store: &Store<impl Item>,
    g: &Geometry,
    weights: &[i64; 7],
    index: usize,
    pl: &mut impl ProgressLog,
) -> Result<SweptLevel<T>> {
    let parts = Partitioned::new(store, g);
    let level = sweep_bumping_level(&parts, g, weights, index, store.len, pl);
    if let Some(e) = parts.error.into_inner().unwrap() {
        return Err(e.into());
    }
    Ok(level)
}

/// Applies `f` to disjoint ranges of the partitions of a store, one for each
/// thread, possibly in parallel, and returns the results in order.
fn by_thread<T: Send>(
    store: &Store<impl Item>,
    f: impl Fn(Range<usize>) -> io::Result<T> + Sync,
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

/// Calls `f` on the records of the keys of the bumped buckets of a level
/// whose partition is in `range`, given the seeds of the buckets, updating
/// `pl` with the number of records read.
fn for_each_bumped<T: SweepSeed>(
    store: &Store<impl Item>,
    g: &Geometry,
    seeds: &[T],
    range: Range<usize>,
    mut pl: impl ProgressLog,
    mut f: impl FnMut(&Record) -> io::Result<()>,
) -> io::Result<()> {
    let sig = store.sig;
    let mut items = vec![];
    for p in range {
        store.for_each(p, &mut items, |item| {
            let r = item.record();
            if seeds[g.bucket(sig.sig(&r))].seed() == 0 {
                f(&r)?;
            }
            Ok(())
        })?;
        pl.update_with_count(store.partition_len(p));
    }
    Ok(())
}

/// Returns the records of the `k` keys of the bumped buckets, given the
/// salt of the following level: in a new store (kept on disk in `dir`, if
/// it is not `None`), or in a vector, if the following level is the last
/// one. The progress of the pass, in records read, is logged on a concurrent
/// logger obtained from `pl`.
fn collect_bumped<T: SweepSeed>(
    store: &Store<impl Item>,
    g: &Geometry,
    seeds: &[T],
    k: usize,
    salt: u64,
    dir: Option<&Path>,
    pl: &mut impl ProgressLog,
) -> Result<Keys> {
    let mut cpl = pl.concurrent();
    cpl.item_name("key");
    cpl.expected_updates(store.len);
    cpl.start_with(format_args!(
        "Collecting the {k} keys of buckets without a seed..."
    ));
    let keys = if k <= LAST_LEVEL_THRESHOLD {
        let parts = by_thread(store, |range| {
            let mut out = vec![];
            for_each_bumped(store, g, seeds, range, cpl.clone(), |r| {
                out.push(*r);
                Ok(())
            })?;
            Ok(out)
        })?;
        let records = parts.concat();
        debug_assert_eq!(records.len(), k);
        Keys::Last(records)
    } else {
        let writer = StoreWriter::new(SigFn(Some(salt)), log2_partitions(k), dir)?;
        let lens = by_thread(store, |range| {
            let mut buffers = writer.buffers();
            for_each_bumped(store, g, seeds, range, cpl.clone(), |r| buffers.push(*r))?;
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
/// the same signature for a level and the same hash with a third seed (see
/// [`Signatures::has_duplicates`]); here,
/// as keys are no longer available, we consider two keys duplicates if they
/// have the same signature for the first level and the same second hash.
/// The two criteria give different results only with negligible
/// probability: if two distinct keys have the same 128 bits of hashes, the
/// construction from a slice fails anyway (such keys collide at every
/// level), albeit with a different error if their hashes with the third
/// seed differ; conversely, if two distinct keys have the same signature for
/// a level and the same hash with the third seed (which happens with
/// probability about 2⁻¹²⁸), the construction from a slice reports
/// duplicates, whereas this construction succeeds.
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
            if contains_duplicates(&mut all) {
                return Ok(true);
            }
            pl.update_with_count(all.len());
        }
        Ok(false)
    })?;
    cpl.done();
    Ok(found.into_iter().any(|x| x))
}

impl PHastRBuilder {
    /// Builds a function on the keys returned by a lender, which is read
    /// just once (see [`PHastR::try_new_with_builder`]).
    ///
    /// Records take sixteen bytes per key, in memory or, in offline mode, on
    /// disk. In offline mode, besides buffers whose size depends little on
    /// the number of keys, construction needs in memory about 0.36 bytes per
    /// key with the default parameters while building the first level (the
    /// seeds of its buckets, as bytes when they have at most eight bits, and
    /// its set of used slots), and then little more than the function
    /// itself.
    pub(super) fn try_build_from_lender<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStoreBuild + MemSize + FlatType,
        B: ?Sized + Borrow<K>,
    >(
        &self,
        keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        pl: &mut impl ProgressLog,
    ) -> Result<PHastR<K, D>> {
        self.try_build(pl, |pl| self.build_levels_from_lender(keys, pl))
    }

    /// Sweeps the first level of a construction from a lender, with `n`
    /// keys whose records are in `store`, and collects the records of the
    /// keys of its bumped buckets; returns its seeds, its free slots, and
    /// the keys of the following level.
    fn first_level<T: SweepSeed>(
        &self,
        store: &Store<impl Item>,
        geom: &Geometry,
        weights: &[i64; 7],
        n: usize,
        pl: &mut impl ProgressLog,
    ) -> Result<(Vec<T>, EliasFano<usize>, Keys)> {
        let SweptLevel { seeds, occupied } =
            sweep_store(store, geom, weights, 0, pl).context("Could not sweep level 0")?;
        let num_holes = n - occupied.num_used();
        let mut efb = EliasFanoBuilder::new(num_holes, n);
        for_each_hole(0, &occupied, n, |p| efb.push(p));
        let holes = efb.build();
        drop(occupied);
        log_level(0, n, num_holes, pl);
        let cur = collect_bumped(
            store,
            geom,
            &seeds,
            num_holes,
            level_salt(self.seed, 1, 0),
            self.offline_dir().as_deref(),
            pl,
        )
        .context("Could not collect the keys bumped from level 0")?;
        Ok((seeds, holes, cur))
    }

    /// Starts logging the pass over the keys of a construction from a
    /// lender; `what` describes what is computed and stored.
    fn start_pass(&self, what: &str, dir: Option<&Path>, pl: &mut impl ProgressLog) {
        pl.item_name("key");
        pl.expected_updates(self.expected_num_keys);
        match dir {
            Some(dir) => pl.start_with(format_args!(
                "Computing and storing {what} sequentially on disk ({}) using seed 0x{:016x}...",
                dir.display(),
                self.seed
            )),
            None => pl.start_with(format_args!(
                "Computing and storing {what} sequentially in RAM using seed 0x{:016x}...",
                self.seed
            )),
        }
    }

    /// Builds a function on the keys returned by a lender (see
    /// [`build_levels`]).
    ///
    /// [`build_levels`]: Self::build_levels
    fn build_levels_from_lender<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStoreBuild,
        B: ?Sized + Borrow<K>,
    >(
        &self,
        mut keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        pl: &mut impl ProgressLog,
    ) -> Result<PHastR<K, D>> {
        let dir = self.offline_dir();
        self.start_pass("128-bit signatures", dir.as_deref(), pl);
        let start = Instant::now();
        let writer = StoreWriter::new(SigFn(None), MAX_LOG2_PARTITIONS, dir.as_deref())?;
        let mut buffers = writer.buffers();
        while let Some(key) = keys.next()? {
            let key: &K = key.borrow();
            buffers
                .push([
                    K::to_sig(key, self.seed)[0],
                    K::to_sig(key, self.seed ^ SECOND_HASH)[0],
                ])
                .context("Could not store the hashes of the keys")?;
            pl.light_update();
        }
        let n = buffers
            .finish()
            .context("Could not store the hashes of the keys")?;
        let store = writer.finish(n);
        pl.done();
        log_completion("Computation of signatures from inputs", start, n, pl);
        Ok(self.build_levels_from_store(store, n, false, pl)?.0)
    }

    /// Builds a function on the keys returned by a lender, as
    /// [`try_build_from_lender`] does, keeping a value for each key: returns
    /// the function and, unless there are no keys, the values, from which
    /// they can be written at the outputs of their keys (see
    /// [`StoredValues::for_each`]).
    ///
    /// The two lenders are read just once; the lender of the values can
    /// return more values than there are keys (the extra values are ignored),
    /// as in the constructors of [`VFunc`], but not fewer. The function
    /// `inspect` is called on each value used, in order.
    ///
    /// The construction stores each value next to the record of its key, in
    /// memory or, in offline mode, on disk, and it keeps the records and the
    /// values until the returned [`StoredValues`] is dropped.
    ///
    /// [`try_build_from_lender`]: Self::try_build_from_lender
    /// [`VFunc`]: crate::func::VFunc
    #[cfg(feature = "rayon")]
    pub(crate) fn try_build_from_lender_with_values<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStoreBuild + MemSize + FlatType,
        B: ?Sized + Borrow<K>,
        V: BinSafe,
    >(
        &self,
        mut keys: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        mut values: impl FallibleLender<Error: Error + Send + Sync + 'static>
        + for<'lend> FallibleLending<'lend, Lend = &'lend V>,
        mut inspect: impl FnMut(&V),
        pl: &mut impl ProgressLog,
    ) -> Result<(PHastR<K, D>, Option<StoredValues<V>>)> {
        let mut stored = None;
        let func = self.try_build(pl, |pl| {
            let dir = self.offline_dir();
            self.start_pass("128-bit signatures and values", dir.as_deref(), pl);
            let start = Instant::now();
            let writer =
                StoreWriter::<WithValue<V>>::new(SigFn(None), MAX_LOG2_PARTITIONS, dir.as_deref())?;
            let mut buffers = writer.buffers();
            let mut num_keys = 0;
            while let Some(key) = keys.next()? {
                num_keys += 1;
                let value =
                    values
                        .next()?
                        .ok_or(crate::func::BuildError::MismatchedKeysAndValues {
                            num_keys,
                            num_values: num_keys - 1,
                        })?;
                let key: &K = key.borrow();
                buffers
                    .push(WithValue {
                        record: [
                            K::to_sig(key, self.seed)[0],
                            K::to_sig(key, self.seed ^ SECOND_HASH)[0],
                        ],
                        value: *value,
                    })
                    .context("Could not store the hashes and the values of the keys")?;
                inspect(value);
                pl.light_update();
            }
            let n = buffers
                .finish()
                .context("Could not store the hashes and the values of the keys")?;
            let store = writer.finish(n);
            pl.done();
            log_completion("Computation of signatures from inputs", start, n, pl);
            let (func, store) = self.build_levels_from_store(store, n, true, pl)?;
            stored = store.map(|store| StoredValues { store });
            Ok(func)
        })?;
        Ok((func, stored))
    }

    /// Builds a function on `n` keys, given the store of their records;
    /// returns also the store if `keep` is true and there are keys.
    ///
    /// To save memory, the seeds of the first level are stored as soon as
    /// the level is built, its free slots are kept in an Elias–Fano
    /// sequence, and the remapping sequence is built at the end from the
    /// sets of used slots of the following levels, which are small. Unless
    /// `keep` is true, the store is dropped as soon as the keys of the
    /// second level have been collected.
    fn build_levels_from_store<K: ?Sized, D: SeedStoreBuild, I: Item>(
        &self,
        store: Store<I>,
        n: usize,
        keep: bool,
        pl: &mut impl ProgressLog,
    ) -> Result<(PHastR<K, D>, Option<Store<I>>)> {
        let weights = self.priority_weights();
        let dir = self.offline_dir();

        if n == 0 {
            return Ok((self.empty(), None));
        }

        // The first level: with at most eight bits per seed, its sweep keeps
        // a byte per bucket, which storage of bytes takes without copying
        let geom = self.geometry(n, n, self.bucket_size);
        let (seeds0, holes, mut cur) = if self.seed_bits <= 8 {
            let (seeds, holes, cur) = self.first_level::<u8>(&store, &geom, &weights, n, pl)?;
            (D::from_byte_seeds(seeds, self.seed_bits), holes, cur)
        } else {
            let (seeds, holes, cur) = self.first_level::<u16>(&store, &geom, &weights, n, pl)?;
            (D::from_seeds(&seeds, self.seed_bits), holes, cur)
        };
        // Dropped here, unless the caller needs it
        let store = keep.then_some(store);

        // The following levels, as in build_levels
        let mut following = FollowingLevels::default();
        loop {
            let index = following.next_index();
            let k = cur.len();
            let next = match cur {
                Keys::Stored(store) => {
                    let salt = level_salt(self.seed, index, 0);
                    let geom = self.geometry(k, k, self.bucket_size);
                    // Equal keys are bumped from the first level, so they
                    // are all in the second one
                    if index == 1
                        && has_duplicates(&store, pl)
                            .context("Could not check for duplicate keys")?
                    {
                        bail!("Duplicate keys");
                    }
                    let level = sweep_store(&store, &geom, &weights, index, pl)
                        .with_context(|| format!("Could not sweep level {index}"))?;
                    let num_bumped = k - level.occupied.num_used();
                    // Before collecting the bumped keys, which would be all
                    // keys
                    check_progress(index, k, num_bumped, self.bucket_size)?;
                    let next = collect_bumped(
                        &store,
                        &geom,
                        &level.seeds,
                        num_bumped,
                        level_salt(self.seed, index + 1, 0),
                        dir.as_deref(),
                        pl,
                    )
                    .with_context(|| {
                        format!("Could not collect the keys bumped from level {index}")
                    })?;
                    following.push(&geom, salt, level);
                    next
                }
                Keys::Last(records) => {
                    if records.is_empty() {
                        break;
                    }
                    self.last_level(
                        &mut following,
                        k,
                        &weights,
                        |i, salt| level_sig(records[i][0], records[i][1], salt),
                        |_, _| contains_duplicates(&mut records.clone()),
                    )?;
                    Keys::Last(vec![])
                }
            };
            log_level(index, k, next.len(), pl);
            cur = next;
        }

        Ok((
            self.assemble(n, geom.level(), seeds0, holes.iter(), following),
            store,
        ))
    }
}

/// The store of the first level of a function built from a lender, whose
/// items contain, besides the record of a key, its value (see
/// [`PHastRBuilder::try_build_from_lender_with_values`]).
#[cfg(feature = "rayon")]
pub(crate) struct StoredValues<V> {
    store: Store<WithValue<V>>,
}

#[cfg(feature = "rayon")]
impl<V: BinSafe> StoredValues<V> {
    /// Calls `f` on the output and on the value of each key of `func`,
    /// which must be the function built with these values, using a thread
    /// for each range of partitions (see [`by_thread`]); the outputs of the
    /// keys are distinct. The progress of the pass, in keys, is logged on a
    /// concurrent logger obtained from `pl`.
    pub(crate) fn for_each<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStore + Sync,
        P: AsRef<[LevelParams]> + Sync,
        R: SliceByValue<Value = usize> + Sync,
    >(
        &self,
        func: &PHastR<K, D, P, R>,
        f: impl Fn(usize, V) + Sync,
        pl: &mut impl ProgressLog,
    ) -> Result<()> {
        let mut cpl = pl.concurrent();
        cpl.item_name("key");
        cpl.expected_updates(self.store.len);
        cpl.start("Storing the values of the keys...");
        by_thread(&self.store, |range| {
            self.visit(func, range, &mut |i, v| f(i, v), &mut cpl.clone())
        })
        .context("Could not read the records and the values of the keys")?;
        cpl.done();
        Ok(())
    }

    /// Calls `f` on the output and on the value of each key of `func`, as
    /// [`for_each`] does, but using only the current thread, so that `f`
    /// can modify its state.
    ///
    /// [`for_each`]: Self::for_each
    pub(crate) fn for_each_seq<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStore,
        P: AsRef<[LevelParams]>,
        R: SliceByValue<Value = usize>,
    >(
        &self,
        func: &PHastR<K, D, P, R>,
        mut f: impl FnMut(usize, V),
        pl: &mut impl ProgressLog,
    ) -> Result<()> {
        pl.item_name("key");
        pl.expected_updates(self.store.len);
        pl.start("Storing the values of the keys...");
        self.visit(func, 0..self.store.num_partitions(), &mut f, pl)
            .context("Could not read the records and the values of the keys")?;
        pl.done();
        Ok(())
    }

    /// Calls `f` on the output and on the value of each key of `func` whose
    /// record is in a partition in `range`, updating `pl` with the number of
    /// keys.
    fn visit<
        K: ?Sized + ToSig<[u64; 1]>,
        D: SeedStore,
        P: AsRef<[LevelParams]>,
        R: SliceByValue<Value = usize>,
    >(
        &self,
        func: &PHastR<K, D, P, R>,
        range: Range<usize>,
        f: &mut impl FnMut(usize, V),
        pl: &mut impl ProgressLog,
    ) -> io::Result<()> {
        let mut items = vec![];
        for p in range {
            self.store.for_each(p, &mut items, |item| {
                let r = item.record;
                f(func.get_by_hashes(r[0], r[1]), item.value);
                Ok(())
            })?;
            pl.update_with_count(self.store.partition_len(p));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitFieldVec;
    use crate::utils::FromSlice;
    #[cfg(feature = "slow_tests")]
    use dsi_progress_logger::ProgressLogger;
    use dsi_progress_logger::no_logs;

    /// A key whose 64-bit signature with seed 0 collides with that of another
    /// key for the first 2 · `COLLIDING` keys (pairs 2*i*, 2*i* + 1); other
    /// seeds give independent signatures.
    #[derive(Clone, Copy)]
    struct CollidingKey(u64);
    const COLLIDING: u64 = 2000;

    impl ToSig<[u64; 1]> for CollidingKey {
        fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
            let k = key.borrow().0;
            if seed == 0 && k < 2 * COLLIDING {
                <u64 as ToSig<[u64; 1]>>::to_sig(k / 2, 0)
            } else {
                <u64 as ToSig<[u64; 1]>>::to_sig(k, seed)
            }
        }
    }

    /// Asserts that two functions are identical.
    fn assert_same<K: ?Sized, D: SeedStore>(a: &PHastR<K, D>, b: &PHastR<K, D>) {
        let fields = |f: &PHastR<K, D>| {
            format!(
                "{:?}",
                (
                    f.seed,
                    f.num_keys,
                    f.layout_shift,
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
    fn check<K: ?Sized + ToSig<[u64; 1]> + Sync, D: SeedStoreBuild + MemSize + FlatType>(
        keys: &[&K],
        builder: PHastRBuilder,
    ) -> Result<()> {
        let par = <PHastR<K, D>>::try_par_new_with_builder(keys, builder.clone(), no_logs![])?;
        for offline in [false, true] {
            let phf = <PHastR<K, D>>::try_new_with_builder(
                FromSlice::new(keys),
                builder.clone().offline(offline),
                no_logs![],
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

    fn check_u64<D: SeedStoreBuild + MemSize + FlatType>(
        n: usize,
        builder: PHastRBuilder,
    ) -> Result<()> {
        let keys: Vec<u64> = (0..n as u64).collect();
        let refs: Vec<&u64> = keys.iter().collect();
        check::<u64, D>(&refs, builder)
    }

    #[test]
    fn test_grows() {
        // With the constants of the writer of the first level, the test
        // gives the same result as exact arithmetic, even on 32-bit
        // platforms, where the product of its arguments overflows
        let (partitions, growth) = (1usize << 14, 512usize);
        for block in [1usize << 10, 1 << 16] {
            let threshold = partitions as u128 * block as u128 * growth as u128;
            for len in [0, 1 << 20, usize::MAX / 2, usize::MAX] {
                assert_eq!(
                    grows(len, partitions, block, growth),
                    len as u128 >= threshold
                );
            }
            if let Ok(len) = usize::try_from(threshold) {
                assert!(!grows(len - 1, partitions, block, growth));
                assert!(grows(len, partitions, block, growth));
            }
        }
    }

    #[test]
    fn test_equivalence() -> Result<()> {
        // The constructions from a slice and from a lender, in memory and
        // offline, give the same function: with the constants used in tests,
        // 10⁵ keys are enough for several partitions and for blocks of
        // maximum size. The tests of the functions themselves are in
        // tests/test_phastr.rs.
        for n in [0, 1, 100, 4096, 4097, 100_000] {
            check_u64::<Box<[u8]>>(n, PHastRBuilder::default())?;
        }
        // Rings of four words, a single layout with short slices, and
        // large buckets
        check_u64::<BitFieldVec<Box<[usize]>>>(
            100_000,
            PHastRBuilder::default()
                .seed_bits(10)
                .log2_slice_len(11)
                .bucket_size(6.0),
        )?;
        check_u64::<Box<[u8]>>(
            100_000,
            PHastRBuilder::default().log2_layouts(0).log2_slice_len(8),
        )?;
        check_u64::<Box<[u8]>>(100_000, PHastRBuilder::default().bucket_size(64.0))?;
        // Colliding signatures, with only the last level and with more
        // levels after the first one
        for n in [5_000u64, 300_000] {
            let keys: Vec<CollidingKey> = (0..n).map(CollidingKey).collect();
            let refs: Vec<&CollidingKey> = keys.iter().collect();
            check::<CollidingKey, Box<[u8]>>(&refs, PHastRBuilder::default())?;
        }
        Ok(())
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_sweep_chunks() -> Result<()> {
        // With a fixed number of sweep chunks, the function does not depend
        // on the size of the pool; by default, the number of chunks is the
        // number of threads. The key set is large enough for three chunks.
        let n = 400_000;
        let fixed = PHastRBuilder::default().sweep_chunks(3);
        assert_eq!(fixed.geometry(n, n, fixed.bucket_size).sweep_chunks(), 3);
        let keys: Vec<u64> = (0..n as u64).collect();
        let refs: Vec<&u64> = keys.iter().collect();
        let in_pool = |threads: usize, f: &(dyn Fn() -> Result<PHastR<u64>> + Sync)| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?
                .install(f)
        };
        let reference = in_pool(3, &|| {
            <PHastR<u64>>::try_par_new_with_builder(&refs, PHastRBuilder::default(), no_logs![])
        })?;
        for threads in [1, 2, 3, 4] {
            let par = in_pool(threads, &|| {
                <PHastR<u64>>::try_par_new_with_builder(&refs, fixed.clone(), no_logs![])
            })?;
            assert_same(&par, &reference);
            for offline in [false, true] {
                let phf = in_pool(threads, &|| {
                    <PHastR<u64>>::try_new_with_builder(
                        FromSlice::new(&refs),
                        fixed.clone().offline(offline),
                        no_logs![],
                    )
                })?;
                assert_same(&phf, &reference);
            }
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
                        no_logs![],
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
                    no_logs![],
                )
                .is_err()
            );
        }
    }

    // The following tests repeat the comparison of test_equivalence on
    // more sizes, parameters, thread pools, and types of keys

    #[cfg(feature = "slow_tests")]
    #[test]
    fn test_sizes() -> Result<()> {
        for n in [
            0, 1, 2, 3, 10, 100, 1000, 4096, 4097, 5000, 10_000, 100_000, 1_000_000,
        ] {
            check_u64::<Box<[u8]>>(n, PHastRBuilder::default())?;
        }
        Ok(())
    }

    #[cfg(feature = "slow_tests")]
    #[test]
    fn test_parameters() -> Result<()> {
        for n in [1000, 100_000] {
            check_u64::<BitFieldVec<Box<[usize]>>>(n, PHastRBuilder::default().seed_bits(10))?;
            for log2_layouts in 0..=3 {
                check_u64::<Box<[u8]>>(
                    n,
                    PHastRBuilder::default()
                        .log2_layouts(log2_layouts)
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
        check_u64::<BitFieldVec<Box<[usize]>>>(
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
        check_u64::<BitFieldVec<Box<[usize]>>>(
            20_000,
            PHastRBuilder::default()
                .seed_bits(16)
                .log2_slice_len(16)
                .bucket_size(64.0),
        )?;
        check_u64::<Box<[u8]>>(100_000, PHastRBuilder::default().bucket_size(64.0))?;
        check_u64::<Box<[u8]>>(100_000, PHastRBuilder::default().seed(42))
    }

    #[cfg(feature = "slow_tests")]
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

    #[cfg(feature = "slow_tests")]
    #[test]
    fn test_strings() -> Result<()> {
        let keys: Vec<String> = (0..200_000).map(|i| format!("key{i}")).collect();
        let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let par = <PHastR<str>>::try_par_new(&refs, no_logs![])?;
        // Lines from a reader
        for offline in [false, true] {
            let phf = <PHastR<str>>::try_new_with_builder(
                crate::utils::LineLender::new(std::io::Cursor::new(keys.join("\n"))),
                PHastRBuilder::default().offline(offline),
                no_logs![],
            )?;
            assert_same(&phf, &par);
        }
        Ok(())
    }

    #[cfg(feature = "slow_tests")]
    #[test]
    fn test_logging() -> Result<()> {
        // With a million keys the second level is not the last one (even
        // with one thread), so all phases are logged; logging does not
        // change the function
        let keys: Vec<u64> = (0..1_000_000).collect();
        let builder = PHastRBuilder::default();
        let quiet = <PHastR<u64>>::try_par_new_with_builder(&keys, builder.clone(), no_logs![])?;
        // Without a logger, log macros do not format their arguments
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Info)
            .try_init();
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
                let dir = offline.then(tempfile::env::temp_dir);
                let writer = StoreWriter::new(SigFn(None), MAX_LOG2_PARTITIONS, dir.as_deref())?;
                let mut buffers = writer.buffers();
                let sig = |i: usize| folded_multiply(i as u64 + 1, 0x9E37_79B9_7F4A_7C15);
                for i in 0..n {
                    buffers.push([sig(i), i as u64])?;
                }
                let len = buffers.finish()?;
                let store = writer.finish(len);
                let parts = Partitioned::new(&store, &g);
                assert_eq!(parts.first[0], 0);
                for p in 0..parts.first.len() - 1 {
                    assert!(parts.buckets(p).len() >= 1 << parts.log2_min_buckets);
                    for b in parts.buckets(p) {
                        assert_eq!(parts.part(b), p, "bucket {b}");
                    }
                }
                // Loading all parts in order, and then each part on its own
                let mut all = vec![];
                let (mut sigs, mut begin, mut temp) = (vec![], vec![], vec![]);
                let mut loader = Loader::default();
                for p in 0..parts.first.len() - 1 {
                    let range = parts.buckets(p);
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
