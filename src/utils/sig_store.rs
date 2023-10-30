/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Fast sorting and grouping of signatures and values.

A *signature* is a pair of 64-bit integers, and a *value* is a generic type
implementing [`epserde::traits::ZeroCopy`].
A [`SigStore`] accepts signatures and values in any order; then,
when you call [`SigStore::into_chunk_store`] you can specify the number of high bits
to use for grouping signatures into chunks.

The trait [`ToSig`] provides a standard way to generate signatures for a [`SigStore`].

*/

use anyhow::Result;
use epserde::traits::ZeroCopy;
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use std::iter;
use std::{
    collections::VecDeque,
    fmt::{Display, Formatter},
    fs::File,
    io::*,
    marker::PhantomData,
};

use crate::prelude::spooky_short;

/**

Trait for types that must be turned into a signature.
We provide implementations for all primitive types and strings
by turning them into slice of bytes and then hashing them with
[crate::utils::spooky::spooky_short], using the given seed.

*/

pub trait ToSig {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2];
}

impl ToSig for String {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key, seed);
        [spooky[0], spooky[1]]
    }
}

impl ToSig for &String {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key, seed);
        [spooky[0], spooky[1]]
    }
}

impl ToSig for str {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key, seed);
        [spooky[0], spooky[1]]
    }
}

impl ToSig for &str {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key, seed);
        [spooky[0], spooky[1]]
    }
}

macro_rules! to_sig_prim {
    ($($ty:ty),*) => {$(
        impl ToSig for $ty {
            fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
                let spooky = spooky_short(&key.to_ne_bytes(), seed);
                [spooky[0], spooky[1]]
            }
        }
    )*};
}

to_sig_prim!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

macro_rules! to_sig_slice {
    ($($ty:ty),*) => {$(
        impl ToSig for &[$ty] {
            fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
                // Alignemnt to u8 never fails or leave trailing/leading bytes
                let spooky = spooky_short(unsafe {key.align_to::<u8>().1 }, seed);
                [spooky[0], spooky[1]]
            }
        }
    )*};
}

to_sig_slice!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/**

This structure is used to accumulate key signatures (i.e., random-looking
hashes associated to keys) and associated values,
grouping them in different disk buffers by the high bits of the hash.
Along the way, it keeps track of the number of signatures with the same
`max_chunk_high_bits` high bits.

The implementation exploits the fact that signatures are randomly distributed,
and thus bucket sorting is very effective: at construction time you specify
the number of high bits to use for bucket sorting (say, 8), and when you
[push](`SigStore::push`) keys they will be stored in different disk buffers
(in this case, 256) depending on their high bits. The buffers will be stored
in a directory created by [`tempfile::TempDir`].

After all key signatures and values have been accumulated, you must
call [`SigStore::into_chunk_store`] to flush the buffers and obtain a
[`ChunkStore`].

When you call [`SigStore::into_chunk_store`] you can specify the number of high bits
to use for grouping signatures into chunks, and the necessary buffer splitting or merging
will be handled automatically by the resulting [`ChunkStore`].

*/
pub struct SigStore<T> {
    /// Number of keys added so far.
    num_keys: usize,
    /// The number of high bits used for bucket sorting (i.e., the number of files).
    buckets_high_bits: u32,
    /// The maximum number of high bits used for defining chunks in the call to
    /// [`SigStore::into_chunk_store`]. Chunk sizes will be computed incrementally
    /// for chunks defined by this number of high bits.
    max_chunk_high_bits: u32,
    /// A mask for the lowest `buckets_high_bits` bits.
    buckets_mask: u64,
    //A mask for the lowest `max_chunk_high_bits` bits.
    max_chunk_mask: u64,
    /// The writers associated to the buckets.
    writers: VecDeque<BufWriter<File>>,
    /// The number of keys in each bucket.
    buf_sizes: Vec<usize>,
    /// The number of keys with the same `max_chunk_high_bits` high bits.
    chunk_sizes: Vec<usize>,
    _marker: PhantomData<T>,
}

/**

An immutable container for the signatures and values accumulated by a [`SigStore`].

A [`ChunkStore`]
that can iterate on chunks.

, and it sorts them in a way that allows to group them into *chunks*
using their highest bits.



Each iterator returned by [`ChunkStore::next`] is owned
and can be scanned independently.

*/
#[derive(Debug)]
pub struct ChunkStore<T> {
    /// The number of high bits used for bucket sorting (i.e., the number of files).
    bucket_high_bits: u32,
    /// The number of high bits defining a chunk.
    chunk_high_bits: u32,
    /// The files associated to the buckets.
    files: Vec<File>,
    /// The number of keys in each bucket.
    buf_sizes: Vec<usize>,
    /// The number of keys in each chunk.
    chunk_sizes: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> ChunkStore<T> {
    /// Return the chunk sizes.
    pub fn chunk_sizes(&self) -> &Vec<usize> {
        &self.chunk_sizes
    }

    pub fn iter(&mut self) -> Result<ChunkIterator<'_, T>> {
        // Move all file pointers to the start
        for file in &mut self.files {
            file.seek(SeekFrom::Start(0))?;
        }

        Ok(ChunkIterator {
            store: self,
            next_file: 0,
            next_chunk: 0,
            next_chunk_start: 0,
            chunks: VecDeque::from(vec![]),
            _marker: PhantomData,
        })
    }
}

#[derive(Debug)]
pub struct ChunkIterator<'a, T> {
    store: &'a mut ChunkStore<T>,
    /// The next file to examine.
    next_file: usize,
    /// The next chunk to return.
    next_chunk: usize,
    /// The position where the next chunk to return start, in case buckets must be split.
    next_chunk_start: usize,
    /// The currently loaded buffer.
    chunks: VecDeque<Vec<([u64; 2], T)>>,
    _marker: PhantomData<T>,
}

impl<'a, T: ToOwned + ZeroCopy + Copy + Clone + Send + Sync> Iterator for ChunkIterator<'a, T> {
    type Item = (usize, Vec<([u64; 2], T)>);

    fn next(&mut self) -> Option<Self::Item> {
        let store = &mut self.store;
        if store.bucket_high_bits >= store.chunk_high_bits {
            // We need to aggregate some buckets to get a chunk
            if self.next_file >= store.files.len() {
                return None;
            }

            let to_aggr = 1 << (store.bucket_high_bits - store.chunk_high_bits);

            let len = store.chunk_sizes[self.next_chunk];
            let mut chunk = Vec::<([u64; 2], T)>::with_capacity(len);

            // SAFETY: we just allocated this vector so it is safe to set the length.
            // read_exact guarantees that the vector will be filled with data.
            #[allow(clippy::uninit_vec)]
            unsafe {
                chunk.set_len(len);
            }

            {
                let (pre, mut buf, post) = unsafe { chunk.align_to_mut::<u8>() };
                assert!(pre.is_empty());
                assert!(post.is_empty());
                for i in self.next_file..self.next_file + to_aggr {
                    let mut reader = &store.files[i];
                    let bytes = store.buf_sizes[i] * core::mem::size_of::<([u64; 2], T)>();
                    reader.read_exact(&mut buf[..bytes]).unwrap();
                    buf = &mut buf[bytes..];
                }
            }

            chunk.par_sort_unstable_by_key(|x| x.0);

            if chunk.par_windows(2).any(|w| w[0].0 == w[1].0) {
                return Some((usize::MAX, chunk));
            }

            let res = (self.next_chunk, chunk);
            self.next_file += to_aggr;
            self.next_chunk += 1;
            return Some(res);
        } else {
            if self.chunks.is_empty() {
                if self.next_file == store.files.len() {
                    return None;
                }
                let split_into = 1 << (store.chunk_high_bits - store.bucket_high_bits);

                let offset = self.next_file * split_into;
                for chunk in offset..offset + split_into {
                    self.chunks
                        .push_back(Vec::with_capacity(store.chunk_sizes[chunk]));
                }

                let len = store.buf_sizes[self.next_file];
                let mut data = Vec::<([u64; 2], T)>::with_capacity(1);
                unsafe {
                    data.set_len(1);
                }
                let chunk_mask = (1 << store.chunk_high_bits) - 1;
                store.files[self.next_file]
                    .seek(SeekFrom::Start(0))
                    .unwrap();
                for i in 0..len {
                    {
                        let (pre, buf, after) = unsafe { data.align_to_mut::<u8>() };
                        assert!(pre.is_empty());
                        assert!(after.is_empty());

                        store.files[self.next_file].read_exact(buf).unwrap();
                        let chunk = (data[0].0[0].rotate_left(store.chunk_high_bits) as usize
                            & chunk_mask)
                            - offset;
                        self.chunks[chunk].push(data[0]);
                    }
                }
                self.next_file += 1;
                self.next_chunk = 0;
            }

            let mut chunk = self.chunks.pop_front().unwrap();
            chunk.par_sort_unstable_by_key(|x| x.0);

            if chunk.par_windows(2).any(|w| w[0].0 == w[1].0) {
                return Some((usize::MAX, chunk));
            }

            let res = (self.next_chunk, chunk);
            self.next_chunk += 1;
            return Some(res);
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DuplicateSigError {}

impl Display for DuplicateSigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Duplicate signature detected")
    }
}

impl std::error::Error for DuplicateSigError {}

fn write_binary<T: ZeroCopy>(
    writer: &mut impl Write,
    tuples: &[([u64; 2], T)],
) -> std::io::Result<()> {
    let (pre, buf, post) = unsafe { tuples.align_to::<u8>() };
    debug_assert!(pre.is_empty());
    debug_assert!(post.is_empty());
    writer.write_all(buf)
}

impl<T: ZeroCopy> SigStore<T> {
    /// Create a new store with 2<sup>`buckets_high_bits`</sup> buffers, keeping
    /// counts for chunks defined by `max_chunk_high_bits` high bits.
    pub fn new(buckets_high_bits: u32, max_chunk_high_bits: u32) -> Result<Self> {
        let temp_dir = tempfile::TempDir::new()?;
        let mut writers = VecDeque::new();
        for i in 0..1 << buckets_high_bits {
            let file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .open(temp_dir.path().join(format!("{}.tmp", i)))?;
            writers.push_back(BufWriter::new(file));
        }
        Ok(Self {
            num_keys: 0,
            buckets_high_bits,
            max_chunk_high_bits,
            buckets_mask: (1u64 << buckets_high_bits) - 1,
            max_chunk_mask: (1u64 << max_chunk_high_bits) - 1,
            writers,
            buf_sizes: vec![0; 1 << buckets_high_bits],
            chunk_sizes: vec![0; 1 << max_chunk_high_bits],
            _marker: PhantomData,
        })
    }

    /// Adds a pair of signatures and values to the store.
    pub fn push(&mut self, value: &([u64; 2], T)) -> std::io::Result<()> {
        self.num_keys += 1;
        // high_bits can be 0
        let buffer =
            ((value.0[0].rotate_left(self.buckets_high_bits)) & self.buckets_mask) as usize;
        let chunk =
            ((value.0[0].rotate_left(self.max_chunk_high_bits)) & self.max_chunk_mask) as usize;

        self.buf_sizes[buffer] += 1;
        self.chunk_sizes[chunk] += 1;

        write_binary(&mut self.writers[buffer], std::slice::from_ref(value))
    }

    /// Adds pairs of signature and value to the store.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = ([u64; 2], T)>) -> Result<()> {
        for value in iter {
            self.push(&value)?;
        }
        Ok(())
    }

    /// The number of keys added to the store so far.
    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    /// Flush the buffers and return a pair given by [`ChunkStore`] whose chunks are defined by
    /// the `chunk_high_bits` high bits of the signatures, and the sizes of the chunks.
    ///
    /// It must hold that
    /// `chunk_high_bits` is at most the `max_chunk_high_bits` value provided
    /// at construction time, or this method will panic.
    pub fn into_chunk_store(mut self, chunk_high_bits: u32) -> Result<(ChunkStore<T>, Vec<usize>)> {
        assert!(chunk_high_bits <= self.max_chunk_high_bits);
        let mut files = Vec::with_capacity(self.writers.len());

        // Flush all writers
        for _ in 0..1 << self.buckets_high_bits {
            let mut writer = self.writers.pop_front().unwrap();
            writer.flush()?;
            let mut file = writer.into_inner()?;
            file.seek(SeekFrom::Start(0))?;
            files.push(file);
        }

        let chunk_sizes = self
            .chunk_sizes
            .chunks(1 << (self.max_chunk_high_bits - chunk_high_bits))
            .map(|x| x.iter().sum())
            .collect::<Vec<_>>();
        let iter = Vec::from_iter(chunk_sizes.iter().copied());
        Ok((
            ChunkStore {
                bucket_high_bits: self.buckets_high_bits,
                chunk_high_bits,
                files,
                buf_sizes: self.buf_sizes,
                chunk_sizes,
                _marker: PhantomData,
            },
            iter,
        ))
    }
}

#[test]

fn test_sig_sorter() {
    use rand::prelude::*;
    for max_chunk_bits in [0, 2, 8, 10] {
        for buckets_high_bits in [0, 2, 8, 10] {
            for chunk_high_bits in [0, 2, 8, 10] {
                if chunk_high_bits > max_chunk_bits {
                    continue;
                }
                let mut sig_sorter = SigStore::new(buckets_high_bits, max_chunk_bits).unwrap();
                let mut rand = SmallRng::seed_from_u64(0);

                for _ in (0..10000).rev() {
                    sig_sorter
                        .push(&([rand.next_u64(), rand.next_u64()], rand.next_u64()))
                        .unwrap();
                }
                let (mut chunk_store, _) = sig_sorter.into_chunk_store(chunk_high_bits).unwrap();
                let mut count = 0;
                let mut iter = chunk_store.iter().unwrap();
                while let Some(chunk) = iter.next() {
                    count += 1;
                    for w in chunk.1.windows(2) {
                        assert!(
                            w[0].0[0] < w[1].0[0]
                                || w[0].0[0] == w[1].0[0] && w[0].0[1] < w[1].0[1]
                        );
                    }
                }
                assert_eq!(count, 1 << chunk_high_bits);
            }
        }
    }
}

#[test]
fn test_u8() {
    use rand::prelude::*;
    let mut sig_sorter = SigStore::new(2, 2).unwrap();
    let mut rand = SmallRng::seed_from_u64(0);
    for _ in (0..1000).rev() {
        sig_sorter
            .push(&([rand.next_u64(), rand.next_u64()], rand.next_u64() as u8))
            .unwrap();
    }
    let (mut chunk_store, _) = sig_sorter.into_chunk_store(2).unwrap();
    let mut count = 0;
    let mut iter = chunk_store.iter().unwrap();
    while let Some(chunk) = iter.next() {
        count += 1;
        for w in chunk.1.windows(2) {
            assert!(w[0].0[0] < w[1].0[0] || w[0].0[0] == w[1].0[0] && w[0].0[1] < w[1].0[1]);
        }
    }
    assert_eq!(count, 4);
}

#[test]

fn test_dup() {
    let mut sig_sorter = SigStore::new(0, 0).unwrap();
    sig_sorter.push(&([0, 0], 0)).unwrap();
    sig_sorter.push(&([0, 0], 0)).unwrap();
    let mut dup = false;
    let (mut chunk_store, _) = sig_sorter.into_chunk_store(0).unwrap();
    let mut iter = chunk_store.iter().unwrap();
    while let Some(chunk) = iter.next() {
        if chunk.0 == usize::MAX {
            dup = true;
            break;
        }
    }
    assert!(dup);
}
