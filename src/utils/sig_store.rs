/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Fast sorting and grouping of signatures and values.
//!
//! A *signature* is a pair of 64-bit integers, and a *value* is a generic type
//! implementing [`epserde::traits::ZeroCopy`].
//!
//! A [`SigStore`] acts as a builder for a [`ChunkStore`]: it accepts
//! signature/value pairs in any order, and when you call
//! [`SigStore::into_chunk_store`] it returns an immutable  [`ChunkStore`] that
//! can [iterate on chunks of pairs, where chunks are defined by the highest
//! bits of signatures](ChunkStore::iter).
//!
//! The trait [`ToSig`] provides a standard way to generate signatures for a
//! [`SigStore`].

use super::spooky_short;
use anyhow::Result;
use epserde::prelude::*;
use mem_dbg::{MemDbg, MemSize};
use rdst::RadixKey;
use std::borrow::Cow;
use std::{collections::VecDeque, fs::File, io::*, marker::PhantomData};

/// A signature and a value.

#[derive(Epserde, Debug, Clone, Copy, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
pub struct SigVal<T: ZeroCopy + 'static> {
    pub sig: [u64; 2],
    pub val: T,
}

impl<T: ZeroCopy + 'static> RadixKey for SigVal<T> {
    const LEVELS: usize = 16;

    fn get_level(&self, level: usize) -> u8 {
        (self.sig[1 - level / 8] >> ((level % 8) * 8)) as u8
    }
}

/// Trait for types that must be turned into a signature.
///
/// We provide implementations for all primitive types and strings by turning
/// them into slice of bytes and then hashing them with
/// [crate::utils::spooky::spooky_short], using the given seed.

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

/// Accumulates key signatures (i.e., random-looking hashes associated to keys)
/// and associated values, grouping them in different disk buffers by the high
/// bits of the hash. Along the way, it keeps track of the number of signatures
/// with the same `max_chunk_high_bits` high bits.
///
/// The implementation exploits the fact that signatures are randomly
/// distributed, and thus bucket sorting is very effective: at construction time
/// you specify the number of high bits to use for bucket sorting (say, 8), and
/// when you [push](`SigStore::push`) keys they will be stored in different disk
/// buffers (in this case, 256) depending on their high bits. The buffers will
/// be stored in a directory created by [`tempfile::TempDir`].
///
/// After all key signatures and values have been accumulated, you must call
/// [`SigStore::into_chunk_store`] to flush the buffers and obtain a
/// [`ChunkStore`]. [`SigStore::into_chunk_store`] takes the the number of high
/// bits to use for grouping signatures into chunks, and the necessary buffer
/// splitting or merging will be handled automatically by the resulting
/// [`ChunkStore`].
#[derive(Debug)]
pub struct SigStore<T> {
    /// Number of keys added so far.
    len: usize,
    /// The number of high bits used for bucket sorting (i.e., the number of files).
    buckets_high_bits: u32,
    /// The maximum number of high bits used for defining chunks in the call to
    /// [`SigStore::into_chunk_store`]. Chunk sizes will be computed incrementally
    /// for chunks defined by this number of high bits.
    max_chunk_high_bits: u32,
    /// A mask for the lowest `buckets_high_bits` bits.
    buckets_mask: u64,
    // A mask for the lowest `max_chunk_high_bits` bits.
    max_chunk_mask: u64,
    /// The writers associated to the buckets.
    writers: VecDeque<BufWriter<File>>,
    /// The number of keys in each bucket.
    bucket_sizes: Vec<usize>,
    /// The number of keys with the same `max_chunk_high_bits` high bits.
    chunk_sizes: Vec<usize>,
    _marker: PhantomData<T>,
}

/// An container for the signatures and values accumulated by a [`SigStore`],
/// with the ability to [enumerate them grouped in chunks](ChunkStore::iter).
#[derive(Debug)]
pub struct ChunkStore<T> {
    /// The number of high bits used for bucket sorting.
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

impl<T: ZeroCopy + 'static> ChunkStore<T> {
    /// Return the chunk sizes.
    pub fn chunk_sizes(&self) -> &Vec<usize> {
        &self.chunk_sizes
    }

    /// Return an iterator on chunks.
    ///
    /// This method can be called multiple times.
    pub fn iter(&mut self) -> Result<ChunkIterator<'_, T>> {
        Ok(ChunkIterator {
            store: self,
            next_file: 0,
            next_chunk: 0,
            chunks: VecDeque::from(vec![]),
            _marker: PhantomData,
        })
    }
}

/// Enumerate chunks in a [`ChunkStore`].
///
/// A [`ChunkIterator`] handles the mapping between buckets and chunks. If a
/// chunk is made by one or more buckets, it will aggregate them as necessary;
/// if a bucket contains several chunks, it will split the bucket into chunks.
/// In all cases, each chunk is sorted and tested for duplicates: if duplicates
/// are detected, a fake pair containing `usize::MAX` and an empty chunk will be
/// returned.
///
/// Note that a [`ChunkIterator`] returns an owned variant of [`Cow`]. The
/// reason for using [`Cow`] is easier interoperability with in-memory
/// construction methods, which usually return borrowed variants.
#[derive(Debug)]
pub struct ChunkIterator<'a, T: ZeroCopy + 'static> {
    store: &'a mut ChunkStore<T>,
    /// The next file to examine.
    next_file: usize,
    /// The index of the next chunk to return.
    next_chunk: usize,
    /// The remaining chunks to emit, if there are several chunks per bucket.
    chunks: VecDeque<Vec<SigVal<T>>>,
    _marker: PhantomData<T>,
}

impl<'a, T: ZeroCopy + Send + Sync + 'static> Iterator for ChunkIterator<'a, T> {
    type Item = (usize, Cow<'a, [SigVal<T>]>);

    fn next(&mut self) -> Option<Self::Item> {
        let store = &mut self.store;
        if store.bucket_high_bits >= store.chunk_high_bits {
            // We need to aggregate one or more buckets to get a chunk
            if self.next_file >= store.files.len() {
                return None;
            }

            let to_aggr = 1 << (store.bucket_high_bits - store.chunk_high_bits);

            let len = store.chunk_sizes[self.next_chunk];
            let mut chunk = Vec::<SigVal<T>>::with_capacity(len);

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
                    let bytes = store.buf_sizes[i] * core::mem::size_of::<SigVal<T>>();
                    reader.read_exact(&mut buf[..bytes]).unwrap();
                    buf = &mut buf[bytes..];
                }
            }

            let res = (self.next_chunk, Cow::Owned(chunk));
            self.next_file += to_aggr;
            self.next_chunk += 1;
            Some(res)
        } else {
            // We need to split buckets in several chunks
            if self.chunks.is_empty() {
                if self.next_file == store.files.len() {
                    return None;
                }

                let split_into = 1 << (store.chunk_high_bits - store.bucket_high_bits);

                // Index of the first chunk we are going to retrieve
                let chunk_offset = self.next_file * split_into;
                for chunk in chunk_offset..chunk_offset + split_into {
                    self.chunks
                        .push_back(Vec::with_capacity(store.chunk_sizes[chunk]));
                }

                let mut len = store.buf_sizes[self.next_file];
                let buf_size = 1024;
                let mut buffer = Vec::<SigVal<T>>::with_capacity(buf_size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    buffer.set_len(buf_size);
                }
                let chunk_mask = (1 << store.chunk_high_bits) - 1;
                store.files[self.next_file]
                    .seek(SeekFrom::Start(0))
                    .unwrap();

                while len > 0 {
                    let to_read = buf_size.min(len);
                    unsafe {
                        buffer.set_len(to_read);
                    }
                    let (pre, buf, after) = unsafe { buffer.align_to_mut::<u8>() };
                    debug_assert!(pre.is_empty());
                    debug_assert!(after.is_empty());

                    store.files[self.next_file].read_exact(buf).unwrap();

                    // We move each signature/value pair into its chunk
                    for &v in &buffer {
                        let chunk = (v.sig[0].rotate_left(store.chunk_high_bits) as usize
                            & chunk_mask)
                            - chunk_offset;
                        self.chunks[chunk].push(v);
                    }
                    len -= to_read;
                }

                self.next_file += 1;
            }

            let res = (
                self.next_chunk,
                Cow::Owned(self.chunks.pop_front().unwrap()),
            );
            self.next_chunk += 1;
            Some(res)
        }
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T: ZeroCopy + Send + Sync> ExactSizeIterator for ChunkIterator<'a, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.store.chunk_sizes.len() - self.next_chunk
    }
}

fn write_binary<T: ZeroCopy>(writer: &mut impl Write, tuples: &[SigVal<T>]) -> std::io::Result<()> {
    let (pre, buf, post) = unsafe { tuples.align_to::<u8>() };
    debug_assert!(pre.is_empty());
    debug_assert!(post.is_empty());
    writer.write_all(buf)
}

impl<T: ZeroCopy + 'static> SigStore<T> {
    /// Create a new store with 2<sup>`buckets_high_bits`</sup> buffers, keeping
    /// counts for chunks defined by at most `max_chunk_high_bits` high bits.
    pub fn new(buckets_high_bits: u32, max_chunk_high_bits: u32) -> Result<Self> {
        let temp_dir = tempfile::TempDir::new()?;
        let mut writers = VecDeque::new();
        for i in 0..1 << buckets_high_bits {
            let file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(temp_dir.path().join(format!("{}.tmp", i)))?;
            writers.push_back(BufWriter::new(file));
        }
        Ok(Self {
            len: 0,
            buckets_high_bits,
            max_chunk_high_bits,
            buckets_mask: (1u64 << buckets_high_bits) - 1,
            max_chunk_mask: (1u64 << max_chunk_high_bits) - 1,
            writers,
            bucket_sizes: vec![0; 1 << buckets_high_bits],
            chunk_sizes: vec![0; 1 << max_chunk_high_bits],
            _marker: PhantomData,
        })
    }

    /// Adds a signature/value pair to this store.
    pub fn push(&mut self, sig_val: &SigVal<T>) -> std::io::Result<()> {
        self.len += 1;
        // high_bits can be 0
        let buffer =
            ((sig_val.sig[0].rotate_left(self.buckets_high_bits)) & self.buckets_mask) as usize;
        let chunk =
            ((sig_val.sig[0].rotate_left(self.max_chunk_high_bits)) & self.max_chunk_mask) as usize;

        self.bucket_sizes[buffer] += 1;
        self.chunk_sizes[chunk] += 1;

        write_binary(&mut self.writers[buffer], std::slice::from_ref(sig_val))
    }

    /// Adds signature/value pairs to this store.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = SigVal<T>>) -> Result<()> {
        for sig_val in iter {
            self.push(&sig_val)?;
        }
        Ok(())
    }

    /// The number of signature/value pairs added to the store so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Flush the buffers and return a pair given by [`ChunkStore`] whose chunks are defined by
    /// the `chunk_high_bits` high bits of the signatures.
    ///
    /// It must hold that
    /// `chunk_high_bits` is at most the `max_chunk_high_bits` value provided
    /// at construction time, or this method will panic.
    pub fn into_chunk_store(mut self, chunk_high_bits: u32) -> Result<ChunkStore<T>> {
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

        // Aggregate chunk sizes as necessary
        let chunk_sizes = self
            .chunk_sizes
            .chunks(1 << (self.max_chunk_high_bits - chunk_high_bits))
            .map(|x| x.iter().sum())
            .collect::<Vec<_>>();
        Ok(ChunkStore {
            bucket_high_bits: self.buckets_high_bits,
            chunk_high_bits,
            files,
            buf_sizes: self.bucket_sizes,
            chunk_sizes,
            _marker: PhantomData,
        })
    }
}

#[test]

fn test_sig_sorter() {
    use rand::prelude::*;
    for max_chunk_bits in [0, 2, 8, 9] {
        for buckets_high_bits in [0, 2, 8, 9] {
            for chunk_high_bits in [0, 2, 8, 9] {
                if chunk_high_bits > max_chunk_bits {
                    continue;
                }
                let mut sig_sorter = SigStore::new(buckets_high_bits, max_chunk_bits).unwrap();
                let mut rand = SmallRng::seed_from_u64(0);

                for _ in (0..10000).rev() {
                    sig_sorter
                        .push(&SigVal {
                            sig: [rand.next_u64(), rand.next_u64()],
                            val: rand.next_u64(),
                        })
                        .unwrap();
                }
                let mut chunk_store = sig_sorter.into_chunk_store(chunk_high_bits).unwrap();
                let mut count = 0;
                let iter = chunk_store.iter().unwrap();
                for chunk in iter {
                    count += 1;
                    for &w in chunk.1.iter() {
                        assert_eq!(
                            chunk.0,
                            w.sig[0].rotate_left(chunk_high_bits) as usize
                                & ((1 << chunk_high_bits) - 1)
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
            .push(&SigVal {
                sig: [rand.next_u64(), rand.next_u64()],
                val: rand.next_u64() as u8,
            })
            .unwrap();
    }
    let mut chunk_store = sig_sorter.into_chunk_store(2).unwrap();
    let mut count = 0;
    let iter = chunk_store.iter().unwrap();
    for chunk in iter {
        count += 1;
        for &w in chunk.1.iter() {
            assert_eq!(chunk.0, w.sig[0].rotate_left(2) as usize & ((1 << 2) - 1));
        }
    }
    assert_eq!(count, 4);
}
