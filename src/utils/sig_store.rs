/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Fast sorting and grouping of signatures and values into shards.
//!
//! A *signature* is a pair of 64-bit integers, and a *value* is a generic type
//! implementing [`epserde::traits::ZeroCopy`].
//!
//! A [`SigStore`] acts as a builder for a [`ShardStore`]: it accepts
//! signature/value pairs in any order, and when you call
//! [`SigStore::into_shard_store`] it returns an immutable  [`ShardStore`] that
//! can [iterate on shards of pairs, where shards are defined by the highest
//! bits of signatures](ShardStore::iter).
//!
//! The trait [`ToSig`] provides a standard way to generate signatures for a
//! [`SigStore`].

use super::spooky_short;
use anyhow::Result;
use epserde::prelude::*;
use mem_dbg::{MemDbg, MemSize};

use rdst::RadixKey;
use std::{collections::VecDeque, fs::File, io::*, marker::PhantomData};
use xxhash_rust::xxh3;

/// A signature and a value.

#[derive(Epserde, Debug, Clone, Copy, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
pub struct SigVal<V: ZeroCopy + 'static> {
    pub sig: [u64; 2],
    pub val: V,
}

impl<V: ZeroCopy + 'static> RadixKey for SigVal<V> {
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
        let hash128 = xxh3::xxh3_128_with_seed(key.as_bytes(), seed);
        [(hash128 >> 64) as u64, hash128 as u64]
    }
}

impl ToSig for &String {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let hash128 = xxh3::xxh3_128_with_seed(key.as_bytes(), seed);
        [(hash128 >> 64) as u64, hash128 as u64]
    }
}

impl ToSig for str {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let hash128 = xxh3::xxh3_128_with_seed(key.as_bytes(), seed);
        [(hash128 >> 64) as u64, hash128 as u64]
    }
}

impl ToSig for &str {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let hash128 = xxh3::xxh3_128_with_seed(key.as_bytes(), seed);
        [(hash128 >> 64) as u64, hash128 as u64]
    }
}

macro_rules! to_sig_prim {
    ($($ty:ty),*) => {$(
        impl ToSig for $ty {
            fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
                let hash128 = xxh3::xxh3_128_with_seed(&key.to_ne_bytes(), seed);
                [(hash128 >> 64) as u64, hash128 as u64]
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
                let hash128 = xxh3::xxh3_128_with_seed(unsafe {key.align_to::<u8>().1 }, seed);
                [(hash128 >> 64) as u64, hash128 as u64]
            }
        }
    )*};
}

to_sig_slice!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// Accumulates key signatures (i.e., random-looking hashes associated to keys)
/// and associated values, grouping them in different disk buffers by the high
/// bits of the hash. Along the way, it keeps track of the number of signatures
/// with the same `max_shard_high_bits` high bits.
///
/// The implementation exploits the fact that signatures are randomly
/// distributed, and thus bucket sorting is very effective: at construction time
/// you specify the number of high bits to use for bucket sorting (say, 8), and
/// when you [push](`SigStore::push`) keys they will be stored in different disk
/// buffers (in this case, 256) depending on their high bits. The buffers will
/// be stored in a directory created by [`tempfile::TempDir`].
///
/// After all key signatures and values have been accumulated, you must call
/// [`SigStore::into_shard_store`] to flush the buffers and obtain a
/// [`ShardStore`]. [`SigStore::into_shard_store`] takes the the number of high
/// bits to use for grouping signatures into shards, and the necessary buffer
/// splitting or merging will be handled automatically by the resulting
/// [`ShardStore`].
#[derive(Debug)]
pub struct SigStore<V, B> {
    /// Number of keys added so far.
    len: usize,
    /// The number of high bits used for bucket sorting (i.e., the number of files).
    buckets_high_bits: u32,
    /// The maximum number of high bits used for defining shards in the call to
    /// [`SigStore::into_shard_store`]. Shard sizes will be computed incrementally
    /// for shards defined by this number of high bits.
    max_shard_high_bits: u32,
    /// A mask for the lowest `buckets_high_bits` bits.
    buckets_mask: u64,
    // A mask for the lowest `max_shard_high_bits` bits.
    max_shard_mask: u64,
    /// The writers associated to the buckets.
    buckets: VecDeque<B>,
    /// The number of keys in each bucket.
    bucket_sizes: Vec<usize>,
    /// The number of keys with the same `max_shard_high_bits` high bits.
    shard_sizes: Vec<usize>,
    _marker: PhantomData<V>,
}

/// An container for the signatures and values accumulated by a [`SigStore`],
/// with the ability to [enumerate them grouped in shards](ShardStore::iter).
#[derive(Debug)]
pub struct ShardStore<V, B> {
    /// The number of high bits used for bucket sorting.
    bucket_high_bits: u32,
    /// The number of high bits defining a shard.
    shard_high_bits: u32,
    /// The buckets (files or vectors).
    buckets: Vec<B>,
    /// The number of keys in each bucket.
    buf_sizes: Vec<usize>,
    /// The number of keys in each shard.
    shard_sizes: Vec<usize>,
    _marker: PhantomData<V>,
}

impl<V: ZeroCopy + 'static, B> ShardStore<V, B> {
    /// Return the shard sizes.
    pub fn shard_sizes(&self) -> &Vec<usize> {
        &self.shard_sizes
    }
}

impl<'a, V: ZeroCopy + Send + Sync + 'static, B> IntoIterator for &'a mut ShardStore<V, B>
where
    ShardIterator<'a, V, B>: Iterator<Item = Vec<SigVal<V>>>,
{
    type IntoIter = ShardIterator<'a, V, B>;
    type Item = Vec<SigVal<V>>;
    /// Return an iterator on shards.
    ///
    /// This method can be called multiple times.
    fn into_iter(self) -> ShardIterator<'a, V, B> {
        ShardIterator {
            store: self,
            next_bucket: 0,
            next_shard: 0,
            shards: VecDeque::from(vec![]),
            _marker: PhantomData,
        }
    }
}

/// Enumerate shards in a [`ShardStore`].
///
/// A [`ShardIterator`] handles the mapping between buckets and shards. If a
/// shard is made by one or more buckets, it will aggregate them as necessary;
/// if a bucket contains several shards, it will split the bucket into shards.
///
/// Note that a [`ShardIterator`] returns owned data.
#[derive(Debug)]
pub struct ShardIterator<'a, V: ZeroCopy + 'static, B> {
    store: &'a mut ShardStore<V, B>,
    /// The next bucket to examine.
    next_bucket: usize,
    /// The next shard to return.
    next_shard: usize,
    /// The remaining shards to emit, if there are several shards per bucket.
    shards: VecDeque<Vec<SigVal<V>>>,
    _marker: PhantomData<V>,
}

impl<V: ZeroCopy + Send + Sync + 'static> Iterator for ShardIterator<'_, V, BufReader<File>> {
    type Item = Vec<SigVal<V>>;

    fn next(&mut self) -> Option<Self::Item> {
        let store = &mut self.store;
        if store.bucket_high_bits >= store.shard_high_bits {
            // We need to aggregate one or more buckets to get a shard
            if self.next_bucket >= store.buckets.len() {
                return None;
            }

            let to_aggr = 1 << (store.bucket_high_bits - store.shard_high_bits);

            let len = store.shard_sizes[self.next_shard];
            let mut shard = Vec::<SigVal<V>>::with_capacity(len);

            // SAFETY: we just allocated this vector so it is safe to set the length.
            // read_exact guarantees that the vector will be filled with data.
            #[allow(clippy::uninit_vec)]
            unsafe {
                shard.set_len(len);
            }

            {
                let (pre, mut buf, post) = unsafe { shard.align_to_mut::<u8>() };
                assert!(pre.is_empty());
                assert!(post.is_empty());
                for i in self.next_bucket..self.next_bucket + to_aggr {
                    let bytes = store.buf_sizes[i] * core::mem::size_of::<SigVal<V>>();
                    store.buckets[i].read_exact(&mut buf[..bytes]).unwrap();
                    buf = &mut buf[bytes..];
                }
            }

            let res = shard;
            self.next_bucket += to_aggr;
            self.next_shard += 1;
            Some(res)
        } else {
            // We need to split buckets in several shards
            if self.shards.is_empty() {
                if self.next_bucket == store.buckets.len() {
                    return None;
                }

                let split_into = 1 << (store.shard_high_bits - store.bucket_high_bits);

                // Index of the first shard we are going to retrieve
                let shard_offset = self.next_bucket * split_into;
                for shard in shard_offset..shard_offset + split_into {
                    self.shards
                        .push_back(Vec::with_capacity(store.shard_sizes[shard]));
                }

                let mut len = store.buf_sizes[self.next_bucket];
                let buf_size = 1024;
                let mut buffer = Vec::<SigVal<V>>::with_capacity(buf_size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    buffer.set_len(buf_size);
                }
                let shard_mask = (1 << store.shard_high_bits) - 1;
                store.buckets[self.next_bucket]
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

                    store.buckets[self.next_bucket].read_exact(buf).unwrap();

                    // We move each signature/value pair into its shard
                    for &v in &buffer {
                        let shard = (v.sig[0].rotate_left(store.shard_high_bits) as usize
                            & shard_mask)
                            - shard_offset;
                        self.shards[shard].push(v);
                    }
                    len -= to_read;
                }

                self.next_bucket += 1;
            }

            self.next_shard += 1;
            Some(self.shards.pop_front().unwrap())
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<V: ZeroCopy + Send + Sync + 'static> Iterator for ShardIterator<'_, V, Vec<SigVal<V>>> {
    type Item = Vec<SigVal<V>>;

    fn next(&mut self) -> Option<Self::Item> {
        let store = &mut self.store;
        if store.bucket_high_bits == store.shard_high_bits {
            // Shards and buckets are the same
            if self.next_bucket >= store.buckets.len() {
                return None;
            }

            let res = &store.buckets[self.next_bucket];
            self.next_bucket += 1;
            self.next_shard += 1;
            Some(res.clone())
        } else if store.bucket_high_bits > store.shard_high_bits {
            // We need to aggregate one or more buckets to get a shard
            if self.next_bucket >= store.buckets.len() {
                return None;
            }

            let to_aggr = 1 << (store.bucket_high_bits - store.shard_high_bits);

            let len = store.shard_sizes[self.next_shard];
            let mut shard = Vec::<SigVal<V>>::with_capacity(len);

            for i in self.next_bucket..self.next_bucket + to_aggr {
                shard.extend(store.buckets[i].iter());
            }

            let res = shard;
            self.next_bucket += to_aggr;
            self.next_shard += 1;
            Some(res)
        } else {
            // We need to split buckets in several shards
            if self.shards.is_empty() {
                if self.next_bucket == store.buckets.len() {
                    return None;
                }

                let split_into = 1 << (store.shard_high_bits - store.bucket_high_bits);

                // Index of the first shard we are going to retrieve
                let shard_offset = self.next_bucket * split_into;
                for shard in shard_offset..shard_offset + split_into {
                    self.shards
                        .push_back(Vec::with_capacity(store.shard_sizes[shard]));
                }

                let shard_mask = (1 << store.shard_high_bits) - 1;
                // We move each signature/value pair into its shard
                for &v in &store.buckets[self.next_bucket] {
                    let shard = (v.sig[0].rotate_left(store.shard_high_bits) as usize & shard_mask)
                        - shard_offset;
                    self.shards[shard].push(v);
                }
                self.next_bucket += 1;
            }

            self.next_shard += 1;
            Some(self.shards.pop_front().unwrap())
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<V: ZeroCopy + Send + Sync> ExactSizeIterator for ShardIterator<'_, V, BufReader<File>> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.store.shard_sizes.len() - self.next_shard
    }
}

impl<V: ZeroCopy + Send + Sync> ExactSizeIterator for ShardIterator<'_, V, Vec<SigVal<V>>> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.store.shard_sizes.len() - self.next_shard
    }
}

fn write_binary<V: ZeroCopy>(writer: &mut impl Write, tuples: &[SigVal<V>]) -> std::io::Result<()> {
    let (pre, buf, post) = unsafe { tuples.align_to::<u8>() };
    debug_assert!(pre.is_empty());
    debug_assert!(post.is_empty());
    writer.write_all(buf)
}

impl<V: ZeroCopy + 'static> SigStore<V, BufWriter<File>> {
    /// Create a new store with 2<sup>`buckets_high_bits`</sup> buffers, keeping
    /// counts for shards defined by at most `max_shard_high_bits` high bits.
    pub fn new_offline(buckets_high_bits: u32, max_shard_high_bits: u32) -> Result<Self> {
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
            max_shard_high_bits,
            buckets_mask: (1u64 << buckets_high_bits) - 1,
            max_shard_mask: (1u64 << max_shard_high_bits) - 1,
            buckets: writers,
            bucket_sizes: vec![0; 1 << buckets_high_bits],
            shard_sizes: vec![0; 1 << max_shard_high_bits],
            _marker: PhantomData,
        })
    }
}

impl<V: ZeroCopy + 'static> SigStore<V, Vec<SigVal<V>>> {
    /// Create a new store with 2<sup>`buckets_high_bits`</sup> buffers, keeping
    /// counts for shards defined by at most `max_shard_high_bits` high bits.
    pub fn new_online(buckets_high_bits: u32, max_shard_high_bits: u32) -> Result<Self> {
        let mut writers = VecDeque::new();
        writers.resize_with(1 << buckets_high_bits, std::vec::Vec::new);

        Ok(Self {
            len: 0,
            buckets_high_bits,
            max_shard_high_bits,
            buckets_mask: (1u64 << buckets_high_bits) - 1,
            max_shard_mask: (1u64 << max_shard_high_bits) - 1,
            buckets: writers,
            bucket_sizes: vec![0; 1 << buckets_high_bits],
            shard_sizes: vec![0; 1 << max_shard_high_bits],
            _marker: PhantomData,
        })
    }
}

pub trait TryPush<V> {
    type Error: std::error::Error + Send + Sync + 'static;
    fn try_push(&mut self, sig_val: V) -> Result<(), Self::Error>;
}

impl<V: ZeroCopy + 'static> TryPush<SigVal<V>> for SigStore<V, BufWriter<File>> {
    type Error = std::io::Error;

    fn try_push(&mut self, sig_val: SigVal<V>) -> Result<(), Self::Error> {
        self.len += 1;
        // high_bits can be 0
        let buffer =
            ((sig_val.sig[0].rotate_left(self.buckets_high_bits)) & self.buckets_mask) as usize;
        let shard =
            ((sig_val.sig[0].rotate_left(self.max_shard_high_bits)) & self.max_shard_mask) as usize;

        self.bucket_sizes[buffer] += 1;
        self.shard_sizes[shard] += 1;

        write_binary(&mut self.buckets[buffer], std::slice::from_ref(&sig_val))
    }
}

impl<V: ZeroCopy + 'static> TryPush<SigVal<V>> for SigStore<V, Vec<SigVal<V>>> {
    type Error = std::convert::Infallible;

    fn try_push(&mut self, sig_val: SigVal<V>) -> Result<(), Self::Error> {
        self.len += 1;
        // high_bits can be 0
        let buffer =
            ((sig_val.sig[0].rotate_left(self.buckets_high_bits)) & self.buckets_mask) as usize;
        let shard =
            ((sig_val.sig[0].rotate_left(self.max_shard_high_bits)) & self.max_shard_mask) as usize;

        self.bucket_sizes[buffer] += 1;
        self.shard_sizes[shard] += 1;

        self.buckets[buffer].push(sig_val);
        Ok(())
    }
}

impl<V: ZeroCopy + 'static, B> SigStore<V, B> {
    /// The number of signature/value pairs added to the store so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub trait IntoShardStore<V: ZeroCopy + 'static> {
    type Reader: 'static;
    fn into_shard_store(self, shard_high_bits: u32) -> Result<ShardStore<V, Self::Reader>>;
}

impl<V: ZeroCopy + 'static> IntoShardStore<V> for SigStore<V, BufWriter<File>> {
    type Reader = BufReader<File>;

    /// Flush the buffers and return a pair given by [`ShardStore`] whose shards are defined by
    /// the `shard_high_bits` high bits of the signatures.
    ///
    /// It must hold that
    /// `shard_high_bits` is at most the `max_shard_high_bits` value provided
    /// at construction time, or this method will panic.
    fn into_shard_store(mut self, shard_high_bits: u32) -> Result<ShardStore<V, Self::Reader>> {
        assert!(shard_high_bits <= self.max_shard_high_bits);
        let mut files = Vec::with_capacity(self.buckets.len());

        // Flush all writers
        for _ in 0..1 << self.buckets_high_bits {
            let mut writer = self.buckets.pop_front().unwrap();
            writer.flush()?;
            let mut file = writer.into_inner()?;
            file.seek(SeekFrom::Start(0))?;
            files.push(BufReader::new(file));
        }

        // Aggregate shard sizes as necessary
        let shard_sizes = self
            .shard_sizes
            .chunks(1 << (self.max_shard_high_bits - shard_high_bits))
            .map(|x| x.iter().sum())
            .collect::<Vec<_>>();
        Ok(ShardStore {
            bucket_high_bits: self.buckets_high_bits,
            shard_high_bits,
            buckets: files,
            buf_sizes: self.bucket_sizes,
            shard_sizes,
            _marker: PhantomData,
        })
    }
}

impl<V: ZeroCopy + 'static> IntoShardStore<V> for SigStore<V, Vec<SigVal<V>>> {
    type Reader = Vec<SigVal<V>>;
    /// Flush the buffers and return a pair given by [`ShardStore`] whose shards are defined by
    /// the `shard_high_bits` high bits of the signatures.
    ///
    /// It must hold that
    /// `shard_high_bits` is at most the `max_shard_high_bits` value provided
    /// at construction time, or this method will panic.
    fn into_shard_store(self, shard_high_bits: u32) -> Result<ShardStore<V, Self::Reader>> {
        assert!(shard_high_bits <= self.max_shard_high_bits);
        let files = self
            .buckets
            .into_iter()
            .map(|mut x| {
                x.shrink_to_fit();
                x
            })
            .collect();
        // Aggregate shard sizes as necessary
        let shard_sizes = self
            .shard_sizes
            .chunks(1 << (self.max_shard_high_bits - shard_high_bits))
            .map(|x| x.iter().sum())
            .collect::<Vec<_>>();
        Ok(ShardStore {
            bucket_high_bits: self.buckets_high_bits,
            shard_high_bits,
            buckets: files,
            buf_sizes: self.bucket_sizes,
            shard_sizes,
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn _test_sig_store<B>(mut sig_store: SigStore<u64, B>) -> anyhow::Result<()>
    where
        SigStore<u64, B>: IntoShardStore<u64> + TryPush<SigVal<u64>>,
        for<'a> &'a mut ShardStore<u64, <SigStore<u64, B> as IntoShardStore<u64>>::Reader>:
            IntoIterator<Item = Vec<SigVal<u64>>>,
    {
        let mut rand = SmallRng::seed_from_u64(0);
        let shard_high_bits = sig_store.max_shard_high_bits;

        for _ in (0..10000).rev() {
            sig_store.try_push(SigVal {
                sig: [rand.random(), rand.random()],
                val: rand.random(),
            })?;
        }
        let mut shard_store = sig_store.into_shard_store(shard_high_bits).unwrap();
        let mut count = 0;
        let iter = shard_store.into_iter();
        for shard in iter {
            for &w in shard.iter() {
                assert_eq!(
                    count,
                    w.sig[0].rotate_left(shard_high_bits) as usize & ((1 << shard_high_bits) - 1)
                );
            }
            count += 1;
        }
        assert_eq!(count, 1 << shard_high_bits);
        Ok(())
    }

    #[test]

    fn test_sig_store() -> anyhow::Result<()> {
        for max_shard_bits in [0, 2, 8, 9] {
            for buckets_high_bits in [0, 2, 8, 9] {
                for shard_high_bits in [0, 2, 8, 9] {
                    if shard_high_bits > max_shard_bits {
                        continue;
                    }
                    _test_sig_store(SigStore::new_online(buckets_high_bits, max_shard_bits)?)?;
                    _test_sig_store(SigStore::new_offline(buckets_high_bits, max_shard_bits)?)?;
                }
            }
        }

        Ok(())
    }

    fn _test_u8<B>(mut sig_store: SigStore<u8, B>) -> anyhow::Result<()>
    where
        SigStore<u8, B>: IntoShardStore<u8> + TryPush<SigVal<u8>>,
        for<'a> &'a mut ShardStore<u8, <SigStore<u8, B> as IntoShardStore<u8>>::Reader>:
            IntoIterator<Item = Vec<SigVal<u8>>>,
    {
        let mut rand = SmallRng::seed_from_u64(0);
        for _ in (0..1000).rev() {
            sig_store.try_push(SigVal {
                sig: [rand.random(), rand.random()],
                val: rand.random(),
            })?;
        }
        let mut shard_store = sig_store.into_shard_store(2)?;
        let mut count = 0;

        let iter = shard_store.into_iter();
        for shard in iter {
            for &w in shard.iter() {
                assert_eq!(count, w.sig[0].rotate_left(2) as usize & ((1 << 2) - 1));
            }
            count += 1;
        }
        assert_eq!(count, 4);

        Ok(())
    }

    #[test]
    fn test_u8() -> anyhow::Result<()> {
        _test_u8(SigStore::new_online(2, 2)?)?;
        _test_u8(SigStore::new_offline(2, 2)?)?;
        Ok(())
    }
}
