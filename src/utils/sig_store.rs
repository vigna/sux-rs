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
use rayon::slice::ParallelSliceMut;
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

This structure is used to sort key signatures (i.e., randomly-looking
hashes associated to keys) and associated values in a fast way,
and to group them by the high bits of the hash. It accepts signatures and values
in any order, and it sorts them in a way that allows to group them into *chunks*
using their highest bits.

The implementation exploits the fact that signatures are randomly distributed,
and thus bucket sorting is very effective: at construction time you specify
the number of high bits to use for bucket sorting (say, 8), and when you
[push](`SigStore::push`) keys they will be stored in different disk buffers
(in this case, 256) depending on their high bits. The buffer will be stored
in a directory created by [`tempfile::TempDir`].

When you call [`SigStore::into_chunk_store`] you can specify the number of high bits
to use for grouping signatures into chunks, and the necessary buffer splitting or merging
will be handled automatically.

*/
pub struct SigStore<T> {
    /// Number of keys added so far.
    num_keys: usize,
    /// The number of high bits used for bucket sorting (i.e., the number of files).
    buckets_high_bits: u32,
    /// The maximum number of high bits used for defining chunks in the call to
    /// [`SigStore::into_chunk_store`].
    max_chunk_high_bits: u32,
    /// A mask for the lowest `buckets_high_bits` bits.
    buckets_mask: u64,
    //A mask for the lowest `max_chunk_high_bits` bits.
    max_chunk_mask: u64,
    /// The writers associated to the buckets.
    writers: VecDeque<BufWriter<File>>,
    /// The number of keys in each bucket.
    buf_sizes: VecDeque<usize>,
    /// The number of keys with the same `max_chunk_high_bits` high bits.
    counts: Vec<usize>,
    _marker: PhantomData<T>,
}

/**

An iterator on chunks returned by [`ChunkStore::next`].

The iterator owns a sublist of the buckets corresponding
to at least one chunk. All such iterators can be scanned
independently.

As the iterator progresses, buckets are sorted, checked for duplicates,
and turned into pairs given by a chunk index and an associated slice of
pairs of signatures and values.

If the index of a returned chunk is `usize::MAX`, then the chunk contains
a duplicate signature.

*/
#[derive(Debug)]
pub struct ChunkIterator<T> {
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
    /// The next chunk to return.
    next_chunk: usize,
    _marker: PhantomData<T>,
}

/**

The iterator on iterators on chunks returned by [`SigStore::into_chunk_store`].

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
    files: VecDeque<File>,
    /// The number of keys in each bucket.
    buf_sizes: VecDeque<usize>,
    /// The number of keys in each chunk.
    chunk_sizes: VecDeque<usize>,
    /// The next chunk to return.
    next_chunk: usize,
    _marker: PhantomData<T>,
}

impl<T> ChunkStore<T> {
    /// Return the chunk sizes.
    pub fn chunk_sizes(&self) -> &VecDeque<usize> {
        &self.chunk_sizes
    }
}
impl<T> Iterator for ChunkStore<T> {
    type Item = ChunkIterator<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.files.is_empty() {
            return None;
        }
        if self.bucket_high_bits >= self.chunk_high_bits {
            // We need to aggregate some buckets to check a chunk
            let to_aggr = 1 << (self.bucket_high_bits - self.chunk_high_bits);
            let mut files = vec![];
            let mut buf_sizes = vec![];

            for _ in 0..to_aggr {
                files.push(self.files.pop_front().unwrap());
                buf_sizes.push(self.buf_sizes.pop_front().unwrap());
            }

            let res = ChunkIterator {
                bucket_high_bits: self.bucket_high_bits,
                chunk_high_bits: self.chunk_high_bits,
                files,
                buf_sizes,
                chunk_sizes: vec![self.chunk_sizes.pop_front().unwrap()], // Just one chunk
                next_chunk: self.next_chunk,
                _marker: PhantomData,
            };
            self.next_chunk += 1;
            return Some(res);
        }

        let num_chunks = 1 << (self.chunk_high_bits - self.bucket_high_bits);
        let mut chunk_sizes = vec![];
        // We get a few chunks in a single bucket
        for _ in 0..num_chunks {
            chunk_sizes.push(self.chunk_sizes.pop_front().unwrap());
        }

        let res = ChunkIterator {
            bucket_high_bits: self.bucket_high_bits,
            chunk_high_bits: self.chunk_high_bits,
            files: vec![self.files.pop_front().unwrap()], // Just one bucket
            buf_sizes: vec![self.buf_sizes.pop_front().unwrap()],
            chunk_sizes,
            next_chunk: self.next_chunk,
            _marker: PhantomData,
        };
        self.next_chunk += num_chunks;
        Some(res)
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

impl<T: ZeroCopy + Send> Iterator for ChunkIterator<T> {
    type Item = (usize, Box<[([u64; 2], T)]>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.files.is_empty() {
            None
        } else {
            let len = self.chunk_sizes.remove(0);
            let mut data = Vec::<([u64; 2], T)>::with_capacity(len);

            // SAFETY: we just allocated this vector so it is safe to set the length.
            // read_exact guarantees that the vector will be filled with data.
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(len);
            }

            if self.bucket_high_bits >= self.chunk_high_bits {
                let to_aggr = 1 << (self.bucket_high_bits - self.chunk_high_bits);

                {
                    let (pre, mut buf, post) = unsafe { data.align_to_mut::<u8>() };
                    assert!(pre.is_empty());
                    assert!(post.is_empty());
                    for _ in 0..to_aggr {
                        let mut reader = self.files.remove(0);
                        let bytes =
                            self.buf_sizes.remove(0) * core::mem::size_of::<([u64; 2], T)>();
                        reader.read_exact(&mut buf[..bytes]).unwrap();
                        buf = &mut buf[bytes..];
                    }
                }

                data.par_sort_unstable_by_key(|x| x.0);

                for w in data.windows(2) {
                    if w[0].0 == w[1].0 {
                        return Some((usize::MAX, Box::new([])));
                    }
                }

                let res = Some((self.next_chunk, data.into_boxed_slice()));
                self.next_chunk += 1;
                res
            } else {
                {
                    let (pre, buf, after) = unsafe { data.align_to_mut::<u8>() };
                    assert!(pre.is_empty());
                    assert!(after.is_empty());
                    self.files[0].read_exact(buf).unwrap();
                }

                data.par_sort_unstable_by_key(|x| x.0);

                for w in data.windows(2) {
                    if w[0].0 == w[1].0 {
                        return Some((usize::MAX, Box::new([])));
                    }
                }

                let res = Some((self.next_chunk, data.into_boxed_slice()));
                self.next_chunk += 1;
                if self.next_chunk % (1 << (self.chunk_high_bits - self.bucket_high_bits)) == 0 {
                    self.files.remove(0);
                }
                res
            }
        }
    }
}

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
    /// Create a new store with 2<sup>`buf_high_bits`</sup> buffers, keeping
    /// counts for chunks defined by at most `max_chunk_high_bits` high bits.
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
            buf_sizes: VecDeque::from(vec![0; 1 << buckets_high_bits]),
            counts: vec![0; 1 << max_chunk_high_bits],
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
        self.counts[chunk] += 1;

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
        let mut files = VecDeque::new();

        for _ in 0..1 << self.buckets_high_bits {
            let mut writer = self.writers.pop_front().unwrap();
            writer.flush()?;
            let mut file = writer.into_inner()?;
            file.seek(SeekFrom::Start(0))?;
            files.push_back(file);
        }

        let chunk_sizes = self
            .counts
            .chunks(1 << (self.max_chunk_high_bits - chunk_high_bits))
            .map(|x| x.iter().sum())
            .collect::<VecDeque<_>>();
        let iter = Vec::from_iter(chunk_sizes.iter().copied());
        Ok((
            ChunkStore {
                bucket_high_bits: self.buckets_high_bits,
                chunk_high_bits,
                files,
                buf_sizes: self.buf_sizes,
                chunk_sizes,
                next_chunk: 0,
                _marker: PhantomData,
            },
            iter,
        ))
    }
}

#[test]

fn test_sig_sorter() {
    use rand::prelude::*;
    for max_chunk_bits in [4, 6] {
        for high_bits in [0, 2, 4] {
            for chunk_bits in [0, 2, 4] {
                let mut sig_sorter = SigStore::new(high_bits, max_chunk_bits).unwrap();
                let mut rand = SmallRng::seed_from_u64(0);
                for _ in (0..1000).rev() {
                    sig_sorter
                        .push(&([rand.next_u64(), rand.next_u64()], rand.next_u64()))
                        .unwrap();
                }
                let (sorted_sig, _) = sig_sorter.into_chunk_store(chunk_bits).unwrap();
                let mut count = 0;
                for chunks in sorted_sig {
                    for chunk in chunks {
                        count += 1;
                        for w in chunk.1.windows(2) {
                            assert!(
                                w[0].0[0] < w[1].0[0]
                                    || w[0].0[0] == w[1].0[0] && w[0].0[1] < w[1].0[1]
                            );
                        }
                    }
                }
                assert_eq!(count, 1 << chunk_bits);
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
    let (sorted_sig, _) = sig_sorter.into_chunk_store(2).unwrap();
    let mut count = 0;
    for chunks in sorted_sig {
        for chunk in chunks {
            count += 1;
            for w in chunk.1.windows(2) {
                assert!(w[0].0[0] < w[1].0[0] || w[0].0[0] == w[1].0[0] && w[0].0[1] < w[1].0[1]);
            }
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
    let (chunk_store, _) = sig_sorter.into_chunk_store(0).unwrap();
    for chunks in chunk_store {
        for chunk in chunks {
            if chunk.0 == usize::MAX {
                dup = true;
                break;
            }
        }
    }
    assert!(dup);
}
