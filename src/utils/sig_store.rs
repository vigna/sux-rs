/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Fast sorting and grouping of signatures and values into shards, online and
//! offline.
//!
//! Traits and types in this module make it possible to accumulate key
//! signatures (i.e., random-looking hashes associated to keys) and associated
//! values, grouping them in different buffers (on disk or in core memory) by
//! the high bits of the hash.
//!
//! A [`SigStore`] acts as a builder for a [`ShardStore`]: it accepts
//! [signature/value pairs](SigVal) in any order, and when you call
//! [`into_shard_store`](SigStore::into_shard_store) it returns an immutable
//! [`ShardStore`] that can iterate on shards of signature/value pairs.
//!
//! The implementation exploits the fact that signatures are randomly
//! distributed, and thus bucket sorting is very effective: at construction time
//! you specify the number of high bits to use for bucket sorting (say, 8), and
//! when you [push](`SigStore::try_push`) signature/value pairs they will be
//! stored in different buckets (in this case, 256) depending on their high
//! bits.
//!
//! An [online](new_online) [`SigStore`] keeps the signature/value pairs in
//! memory, while an [offline](new_offline) [`SigStore`] writes them to disk.
//! After all key signatures and values have been accumulated,
//! [`into_shard_store`](SigStore::into_shard_store) will return a
//! [`ShardStore`]. [`into_shard_store`](SigStore::into_shard_store) takes the
//! the number of high bits to use for grouping signatures into shards, and the
//! necessary bucket splitting or merging will be handled automatically, albeit
//! the most efficient scenario is the one in which the number of buckets is
//! equal to the number of shards.
//!
//! You can iterate over the shards in a [`ShardStore`] multiple times using the
//! method [`iter`](ShardStore::iter), or just once using the method
//! [`into_iter`](ShardStore::into_iter). In the latter case resources
//! (i.e., files or memory) will be released as soon as they are consumed.
//!
//! The trait [`ToSig`] provides a standard way to generate signatures for a
//! [`SigStore`]. Implementations are provided for signatures types `[u64;1]`
//! and `[u64; 2]`. In the first case, after a few billion keys you will start
//! to find duplicate signatures. In both cases, shards are defined by the
//! highest bits of the (first) signature.
//!
//! Both signatures and values must be [`ZeroCopy`] so that they might be
//! serialized and deserialized efficiently in the offline case, and they must
//! be [`Send`] and [`Sync`].

#![allow(clippy::comparison_chain)]
#![allow(clippy::type_complexity)]
use anyhow::Result;
use epserde::prelude::*;
use mem_dbg::{MemDbg, MemSize};

use rapidhash::RapidInlineHasher;
use rdst::RadixKey;
use std::{
    borrow::{Borrow, BorrowMut},
    collections::VecDeque,
    fs::File,
    hash::Hasher,
    io::*,
    marker::PhantomData,
    ops::{BitXor, BitXorAssign},
    sync::Arc,
};

/// A trait for types that can be used as signatures.
pub trait Sig: ZeroCopy + Default + PartialEq + Eq + std::fmt::Debug {
    /// Extracts high bits from  the signature.
    ///
    /// These bits are used to shard elements. Note that `high_bits` can be 0,
    /// but it is guaranteed to be less than 64.
    ///
    /// # Panics
    ///
    /// In debug mode this method should panic if `mask` is not equal to `(1 <<
    /// high_bits) - 1`.
    fn high_bits(&self, high_bits: u32, mask: u64) -> u64;

    /// Extracts a 64-bit signature.
    ///
    /// This method is useful to obtain a 64-bit signature independently of the
    /// specific type of signature.
    ///
    /// Note that this value could be equal to the signature itself. If you are
    /// reusing the same signature for different purposes, you should mix
    /// thoroughly this value (see, e.g., [`mix64`](crate::func::mix64)).
    fn sig_u64(&self) -> u64;
}

impl Sig for [u64; 2] {
    #[inline(always)]
    fn high_bits(&self, high_bits: u32, mask: u64) -> u64 {
        debug_assert!(mask == (1 << high_bits) - 1);
        self[0].rotate_left(high_bits) & mask
    }

    #[inline(always)]
    fn sig_u64(&self) -> u64 {
        self[0] ^ self[1]
    }
}

impl Sig for [u64; 1] {
    #[inline(always)]
    fn high_bits(&self, high_bits: u32, mask: u64) -> u64 {
        debug_assert!(mask == (1 << high_bits) - 1);
        self[0].rotate_left(high_bits) & mask
    }

    #[inline(always)]
    fn sig_u64(&self) -> u64 {
        self[0]
    }
}

/// A signature and a value.
#[derive(Epserde, Debug, Clone, Copy, Default, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
pub struct SigVal<S: ZeroCopy + Sig, V: ZeroCopy> {
    pub sig: S,
    pub val: V,
}

impl<V: ZeroCopy> RadixKey for SigVal<[u64; 2], V> {
    const LEVELS: usize = 16;

    fn get_level(&self, level: usize) -> u8 {
        (self.sig[1 - level / 8] >> ((level % 8) * 8)) as u8
    }
}

impl<V: ZeroCopy> RadixKey for SigVal<[u64; 1], V> {
    const LEVELS: usize = 8;

    fn get_level(&self, level: usize) -> u8 {
        (self.sig[0] >> ((level % 8) * 8)) as u8
    }
}

#[derive(Epserde, Debug, Clone, Copy, Default, MemDbg, MemSize)]
#[repr(C)]
#[zero_copy]
/// A newtype around `()` that is used to implement [`BitXor`] and
/// [`BitXorAssign`] as no-ops.
pub struct EmptyVal(());

impl BitXor for EmptyVal {
    type Output = EmptyVal;

    fn bitxor(self, _: EmptyVal) -> Self::Output {
        EmptyVal(())
    }
}

impl BitXorAssign for EmptyVal {
    fn bitxor_assign(&mut self, _: EmptyVal) {}
}

impl<V: ZeroCopy + BitXor> BitXor<SigVal<[u64; 1], V>> for SigVal<[u64; 1], V>
where
    V::Output: ZeroCopy,
{
    type Output = SigVal<[u64; 1], V::Output>;

    fn bitxor(self, rhs: SigVal<[u64; 1], V>) -> Self::Output {
        SigVal {
            sig: [self.sig[0].bitxor(rhs.sig[0])],
            val: self.val.bitxor(rhs.val),
        }
    }
}

impl<V: ZeroCopy + BitXor> BitXor<SigVal<[u64; 2], V>> for SigVal<[u64; 2], V>
where
    V::Output: ZeroCopy,
{
    type Output = SigVal<[u64; 2], V::Output>;

    fn bitxor(self, rhs: SigVal<[u64; 2], V>) -> Self::Output {
        SigVal {
            sig: [
                self.sig[0].bitxor(rhs.sig[0]),
                self.sig[1].bitxor(rhs.sig[1]),
            ],
            val: self.val.bitxor(rhs.val),
        }
    }
}

impl<V: ZeroCopy + BitXorAssign> BitXorAssign<SigVal<[u64; 1], V>> for SigVal<[u64; 1], V> {
    fn bitxor_assign(&mut self, rhs: SigVal<[u64; 1], V>) {
        self.sig[0] ^= rhs.sig[0];
        self.val ^= rhs.val;
    }
}

impl<V: ZeroCopy + BitXorAssign> BitXorAssign<SigVal<[u64; 2], V>> for SigVal<[u64; 2], V> {
    fn bitxor_assign(&mut self, rhs: SigVal<[u64; 2], V>) {
        self.sig[0] ^= rhs.sig[0];
        self.sig[1] ^= rhs.sig[1];
        self.val ^= rhs.val;
    }
}

/// Trait for types that can be turned into a signature.
///
/// We provide implementations for all primitive types, `str`, `String`, `&str`,
/// `&String`, and slices of primitive types, by turning them into byte and
/// calling [`rapidhash::RapidInlineHasher`].
///
/// Note that for efficiency reasons the implementations are not
/// endianness-independent.
pub trait ToSig<S> {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> S;
}

impl ToSig<[u64; 2]> for String {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        let bytes = key.borrow().as_bytes();
        let mut rapid0 = RapidInlineHasher::new(seed);
        let mut rapid1 = RapidInlineHasher::new(!seed);
        rapid0.write(bytes);
        rapid1.write(bytes);
        [rapid0.finish(), rapid1.finish()]
    }
}

impl ToSig<[u64; 1]> for String {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut rapid = RapidInlineHasher::new(seed);
        let bytes = key.borrow().as_bytes();
        rapid.write(bytes);
        [rapid.finish()]
    }
}

impl ToSig<[u64; 2]> for &String {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        let mut rapid0 = RapidInlineHasher::new(seed);
        let mut rapid1 = RapidInlineHasher::new(!seed);
        let bytes = key.borrow().as_bytes();
        rapid0.write(bytes);
        rapid1.write(bytes);
        [rapid0.finish(), rapid1.finish()]
    }
}

impl ToSig<[u64; 1]> for &String {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut rapid = RapidInlineHasher::new(seed);
        let bytes = key.borrow().as_bytes();
        rapid.write(bytes);
        [rapid.finish()]
    }
}

impl ToSig<[u64; 2]> for str {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        let mut rapid0 = RapidInlineHasher::new(seed);
        let mut rapid1 = RapidInlineHasher::new(!seed);
        let bytes = key.borrow().as_bytes();
        rapid0.write(bytes);
        rapid1.write(bytes);
        [rapid0.finish(), rapid1.finish()]
    }
}

impl ToSig<[u64; 1]> for str {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut rapid = RapidInlineHasher::new(seed);
        let bytes = key.borrow().as_bytes();
        rapid.write(bytes);
        [rapid.finish()]
    }
}

impl ToSig<[u64; 2]> for &str {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        let mut rapid0 = RapidInlineHasher::new(seed);
        let mut rapid1 = RapidInlineHasher::new(!seed);
        let bytes = key.borrow().as_bytes();
        rapid0.write(bytes);
        rapid1.write(bytes);
        [rapid0.finish(), rapid1.finish()]
    }
}

impl ToSig<[u64; 1]> for &str {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut rapid = RapidInlineHasher::new(seed);
        let bytes = key.borrow().as_bytes();
        rapid.write(bytes);
        [rapid.finish()]
    }
}

macro_rules! to_sig_prim {
    ($($ty:ty),*) => {$(
        impl ToSig<[u64; 2]> for $ty {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
                let bytes = key.borrow().to_ne_bytes();
                let mut rapid0 = RapidInlineHasher::new(seed);
                let mut rapid1 = RapidInlineHasher::new(!seed);
                rapid0.write(&bytes);
                rapid1.write(&bytes);
                [rapid0.finish(), rapid1.finish()]
                }
        }
        impl ToSig<[u64;1]> for $ty {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                let bytes = key.borrow().to_ne_bytes();
                let mut rapid = RapidInlineHasher::new(seed);
                rapid.write(&bytes);
                [rapid.finish()]
            }
        }
    )*};
}

to_sig_prim!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

macro_rules! to_sig_slice {
    ($($ty:ty),*) => {$(
        impl ToSig<[u64; 2]> for &[$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
                // Alignment to u8 never fails or leave trailing/leading bytes
                let bytes = unsafe {key.borrow().align_to::<u8>().1 };
                let mut rapid0 = RapidInlineHasher::new(seed);
                let mut rapid1 = RapidInlineHasher::new(!seed);
                rapid0.write(bytes);
                rapid1.write(bytes);
                [rapid0.finish(), rapid1.finish()]
            }
        }
        impl ToSig<[u64;1]> for &[$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                // Alignment to u8 never fails or leave trailing/leading bytes
                let bytes = unsafe {key.borrow().align_to::<u8>().1 };
                let mut rapid = RapidInlineHasher::new(seed);
                rapid.write(&bytes);
                [rapid.finish()]
            }
        }
    )*};
}

to_sig_slice!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// A signature store.
///
/// The purpose of this trait is that of avoiding clumsy `where` clauses when
/// passing around a signature store. There is only one implementation,
/// [`SigStoreImpl`], but it is implemented only for certain combinations of
/// type parameters. Having this trait greatly simplifies the type signatures.
pub trait SigStore<S: Sig + ZeroCopy, V: ZeroCopy> {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Tries to add a new signature/value pair to the store.
    fn try_push(&mut self, sig_val: SigVal<S, V>) -> Result<(), Self::Error>;

    type ShardStore: ShardStore<S, V> + Send + Sync;
    /// Turns this store into a [`ShardStore`] whose shards are defined by the
    /// `shard_high_bits` high bits of the signatures.
    ///
    /// # Panics
    ///
    /// It must hold that `shard_high_bits` is at most
    /// [`max_shard_high_bits`](SigStore::max_shard_high_bits) or this method
    /// will panic.
    fn into_shard_store(self, shard_high_bits: u32) -> Result<Self::ShardStore>;

    /// Returns the number of signature/value pairs added to the store so far.
    fn len(&self) -> usize;

    /// Returns true if no signature/value pairs have been added to the store.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum number of high bits whose count we keep track of.
    ///
    /// Sharding cannot happen with more bits than this.
    fn max_shard_high_bits(&self) -> u32;

    /// The temporary directory used by the store, if any.
    fn temp_dir(&self) -> Option<&tempfile::TempDir>;
}

/// An implementation of [`SigStore`] that accumulates signature/value pairs in
/// memory or on disk.
///
/// See the [module documentation](crate::utils::sig_store) for more
/// information.
#[derive(Debug)]
pub struct SigStoreImpl<S, V, B> {
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
    /// The temporary directory used by the store, if offline.
    temp_dir: Option<tempfile::TempDir>,
    _marker: PhantomData<(S, V)>,
}

/// Creates a new on-disk store with 2<sup>`buckets_high_bits`</sup> buckets,
/// keeping counts for shards defined by at most `max_shard_high_bits` high
/// bits.
///
/// The type `S` is the type of the signatures (usually `[u64;1]` or `[u64;
/// 2]`), while `V` is the type of the values. The store will be written to a
/// [temporary directory](https://doc.rust-lang.org/std/env/fn.temp_dir.html),
/// and the files will be deleted when the store is dropped.
pub fn new_offline<S: ZeroCopy + Sig, V: ZeroCopy>(
    buckets_high_bits: u32,
    max_shard_high_bits: u32,
    _expected_num_keys: Option<usize>,
) -> Result<SigStoreImpl<S, V, BufWriter<File>>> {
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

    Ok(SigStoreImpl {
        len: 0,
        buckets_high_bits,
        max_shard_high_bits,
        buckets_mask: (1u64 << buckets_high_bits) - 1,
        max_shard_mask: (1u64 << max_shard_high_bits) - 1,
        buckets: writers,
        bucket_sizes: vec![0; 1 << buckets_high_bits],
        shard_sizes: vec![0; 1 << max_shard_high_bits],
        temp_dir: Some(temp_dir),
        _marker: PhantomData,
    })
}

/// Creates a new in-memory store with 2<sup>`buckets_high_bits`</sup> buckets,
/// keeping counts for shards defined by at most `max_shard_high_bits` high
/// bits.
///
/// The type `S` is the type of the signatures (usually `[u64;1]` or `[u64;
/// 2]`), while `V` is the type of the values.
///
/// If `expected_num_keys` is `Some(n)`, the store will be preallocated to
/// contain 1.05 * `n` keys.
pub fn new_online<S: ZeroCopy + Sig, V: ZeroCopy>(
    buckets_high_bits: u32,
    max_shard_high_bits: u32,
    expected_num_keys: Option<usize>,
) -> Result<SigStoreImpl<S, V, Vec<SigVal<S, V>>>> {
    let mut writers = VecDeque::new();
    let initial_capacity = expected_num_keys
        .map(|n| (n.div_ceil(1 << buckets_high_bits) as f64 * 1.05) as usize)
        .unwrap_or(0);
    writers.resize_with(1 << buckets_high_bits, || {
        Vec::with_capacity(initial_capacity)
    });

    Ok(SigStoreImpl {
        len: 0,
        buckets_high_bits,
        max_shard_high_bits,
        buckets_mask: (1u64 << buckets_high_bits) - 1,
        max_shard_mask: (1u64 << max_shard_high_bits) - 1,
        buckets: writers,
        bucket_sizes: vec![0; 1 << buckets_high_bits],
        shard_sizes: vec![0; 1 << max_shard_high_bits],
        temp_dir: None,
        _marker: PhantomData,
    })
}

impl<S: ZeroCopy + Sig + Send + Sync, V: ZeroCopy + Send + Sync> SigStore<S, V>
    for SigStoreImpl<S, V, BufWriter<File>>
{
    type Error = std::io::Error;

    fn try_push(&mut self, sig_val: SigVal<S, V>) -> Result<(), Self::Error> {
        self.len += 1;
        // high_bits can be 0
        let buffer = sig_val
            .sig
            .high_bits(self.buckets_high_bits, self.buckets_mask) as usize;
        let shard = sig_val
            .sig
            .high_bits(self.max_shard_high_bits, self.max_shard_mask) as usize;

        self.bucket_sizes[buffer] += 1;
        self.shard_sizes[shard] += 1;

        write_binary(&mut self.buckets[buffer], std::slice::from_ref(&sig_val))
    }

    type ShardStore = ShardStoreImpl<S, V, BufReader<File>>;

    fn into_shard_store(mut self, shard_high_bits: u32) -> Result<Self::ShardStore> {
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
        Ok(ShardStoreImpl {
            bucket_high_bits: self.buckets_high_bits,
            shard_high_bits,
            buckets: files,
            buf_sizes: self.bucket_sizes,
            shard_sizes,
            _marker: PhantomData,
        })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn max_shard_high_bits(&self) -> u32 {
        self.max_shard_high_bits
    }

    fn temp_dir(&self) -> Option<&tempfile::TempDir> {
        self.temp_dir.as_ref()
    }
}

impl<S: ZeroCopy + Sig + Send + Sync, V: ZeroCopy + Send + Sync> SigStore<S, V>
    for SigStoreImpl<S, V, Vec<SigVal<S, V>>>
{
    type Error = std::convert::Infallible;

    fn try_push(&mut self, sig_val: SigVal<S, V>) -> Result<(), Self::Error> {
        self.len += 1;

        let buffer = sig_val
            .sig
            .high_bits(self.buckets_high_bits, self.buckets_mask) as usize;
        let shard = sig_val
            .sig
            .high_bits(self.max_shard_high_bits, self.max_shard_mask) as usize;

        self.bucket_sizes[buffer] += 1;
        self.shard_sizes[shard] += 1;

        self.buckets[buffer].push(sig_val);
        Ok(())
    }

    type ShardStore = ShardStoreImpl<S, V, Arc<Vec<SigVal<S, V>>>>;

    fn into_shard_store(self, shard_high_bits: u32) -> Result<Self::ShardStore> {
        assert!(shard_high_bits <= self.max_shard_high_bits);
        let files = self
            .buckets
            .into_iter()
            .map(|mut x| {
                x.shrink_to_fit();
                Arc::new(x)
            })
            .collect();
        // Aggregate shard sizes as necessary
        let shard_sizes = self
            .shard_sizes
            .chunks(1 << (self.max_shard_high_bits - shard_high_bits))
            .map(|x| x.iter().sum())
            .collect::<Vec<_>>();
        Ok(ShardStoreImpl {
            bucket_high_bits: self.buckets_high_bits,
            shard_high_bits,
            buckets: files,
            buf_sizes: self.bucket_sizes,
            shard_sizes,
            _marker: PhantomData,
        })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn max_shard_high_bits(&self) -> u32 {
        self.max_shard_high_bits
    }

    fn temp_dir(&self) -> Option<&tempfile::TempDir> {
        None
    }
}

/// A container for the signatures and values accumulated by a [`SigStore`],
/// with the ability to [enumerate them grouped in shards](ShardStore::iter).
///
/// Also in this case, the purpose of this trait is that of avoiding clumsy
/// `where` clauses when passing around a signature store. There is only one
/// implementation, [`ShardStoreImpl`], but it is implemented only for certain
/// combinations of type parameters. Having this trait greatly simplifies the
/// type signatures.
pub trait ShardStore<S: Sig + ZeroCopy, V: ZeroCopy> {
    type ShardIterator<'a>: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync
    where
        Self: 'a;

    type ShardIntoIterator: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync;

    /// Returns the shard sizes.
    fn shard_sizes(&self) -> &[usize];

    /// Returns an iterator on shards.
    ///
    /// This method can be called multiple times.
    fn iter(&mut self) -> Self::ShardIterator<'_>;

    /// Returns an iterator on shards, consuming self.
    ///
    /// This method can be called multiple times.
    fn into_iter(self) -> Self::ShardIntoIterator;

    /// Returns the number of signature/value pairs in the store.
    fn len(&self) -> usize {
        self.shard_sizes().iter().sum()
    }
}

/// An implementation of [`ShardStore`].
///
/// See the [module documentation](crate::utils::sig_store) for more
/// information.
#[derive(Debug)]
pub struct ShardStoreImpl<S, V, B> {
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
    _marker: PhantomData<(S, V)>,
}

impl<S: ZeroCopy + Sig + Send + Sync, V: ZeroCopy + Send + Sync, B: Send + Sync> ShardStore<S, V>
    for ShardStoreImpl<S, V, B>
where
    for<'a> ShardIterator<S, V, B, Self>: Iterator<Item = Arc<Vec<SigVal<S, V>>>>,
    for<'a> ShardIterator<S, V, B, &'a mut Self>: Iterator<Item = Arc<Vec<SigVal<S, V>>>>,
{
    type ShardIterator<'a>
        = ShardIterator<S, V, B, &'a mut Self>
    where
        B: 'a;

    type ShardIntoIterator = ShardIterator<S, V, B, Self>;

    fn shard_sizes(&self) -> &[usize] {
        &self.shard_sizes
    }

    fn iter(&mut self) -> ShardIterator<S, V, B, &'_ mut Self> {
        ShardIterator {
            store: self,
            borrowed: true,
            next_bucket: 0,
            next_shard: 0,
            shards: VecDeque::from(vec![]),
            _marker: PhantomData,
        }
    }

    fn into_iter<'a>(self) -> ShardIterator<S, V, B, Self> {
        ShardIterator {
            store: self,
            borrowed: false,
            next_bucket: 0,
            next_shard: 0,
            shards: VecDeque::from(vec![]),
            _marker: PhantomData,
        }
    }
}

/// An iterator on shards in a [`ShardStore`].
///
/// A [`ShardIterator`] handles the mapping between buckets and shards. If a
/// shard is made by one or more buckets, it will aggregate them as necessary;
/// if a bucket contains several shards, it will split the bucket into shards.
#[derive(Debug)]
pub struct ShardIterator<S: ZeroCopy + Sig, V: ZeroCopy, B, T: BorrowMut<ShardStoreImpl<S, V, B>>> {
    store: T,
    /// Whether the store is borrowed.
    borrowed: bool,
    /// The next bucket to examine.
    next_bucket: usize,
    /// The next shard to return.
    next_shard: usize,
    /// The remaining shards to emit, if there are several shards per bucket.
    shards: VecDeque<Vec<SigVal<S, V>>>,
    _marker: PhantomData<(B, V)>,
}

impl<
        S: ZeroCopy + Sig + Send + Sync,
        V: ZeroCopy + Send + Sync,
        T: BorrowMut<ShardStoreImpl<S, V, BufReader<File>>>,
    > Iterator for ShardIterator<S, V, BufReader<File>, T>
{
    type Item = Arc<Vec<SigVal<S, V>>>;
    fn next(&mut self) -> Option<Self::Item> {
        let store = self.store.borrow_mut();
        if store.bucket_high_bits >= store.shard_high_bits {
            // We need to aggregate one or more buckets to get a shard
            if self.next_bucket >= store.buckets.len() {
                return None;
            }

            let to_aggr = 1 << (store.bucket_high_bits - store.shard_high_bits);

            let len = store.shard_sizes[self.next_shard];
            let mut shard = Vec::<SigVal<S, V>>::with_capacity(len);

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
                    let bytes = store.buf_sizes[i] * core::mem::size_of::<SigVal<S, V>>();
                    store.buckets[i].seek(SeekFrom::Start(0)).unwrap();
                    store.buckets[i].read_exact(&mut buf[..bytes]).unwrap();
                    // TODO: find a way to truncate / delete files
                    // in the borrowed case
                    buf = &mut buf[bytes..];
                }
            }

            let res = shard;
            self.next_bucket += to_aggr;
            self.next_shard += 1;
            Some(Arc::new(res))
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
                let mut buffer = Vec::<SigVal<S, V>>::with_capacity(buf_size);
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
                        let shard = v.sig.high_bits(store.shard_high_bits, shard_mask) as usize
                            - shard_offset;
                        self.shards[shard].push(v);
                    }
                    len -= to_read;
                }

                self.next_bucket += 1;
            }

            self.next_shard += 1;
            Some(Arc::new(self.shards.pop_front().unwrap()))
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<
        S: ZeroCopy + Sig + Send + Sync,
        V: ZeroCopy + Send + Sync,
        T: BorrowMut<ShardStoreImpl<S, V, Arc<Vec<SigVal<S, V>>>>>,
    > Iterator for ShardIterator<S, V, Arc<Vec<SigVal<S, V>>>, T>
{
    type Item = Arc<Vec<SigVal<S, V>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let store = self.store.borrow_mut();
        if store.bucket_high_bits == store.shard_high_bits {
            // Shards and buckets are the same
            if self.next_bucket >= store.buckets.len() {
                return None;
            }

            let res = if self.borrowed {
                store.buckets[self.next_bucket].clone()
            } else {
                std::mem::take(&mut store.buckets[self.next_bucket])
            };

            self.next_bucket += 1;
            self.next_shard += 1;
            Some(res)
        } else if store.bucket_high_bits > store.shard_high_bits {
            // We need to aggregate one or more buckets to get a shard
            if self.next_bucket >= store.buckets.len() {
                return None;
            }

            let to_aggr = 1 << (store.bucket_high_bits - store.shard_high_bits);

            let len = store.shard_sizes[self.next_shard];
            let mut shard = Vec::with_capacity(len);

            for i in self.next_bucket..self.next_bucket + to_aggr {
                if self.borrowed {
                    shard.extend(store.buckets[i].iter());
                } else {
                    shard.extend(std::mem::take(&mut store.buckets[i]).iter());
                }
            }

            let res = shard;
            self.next_bucket += to_aggr;
            self.next_shard += 1;
            Some(Arc::new(res))
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
                for &v in store.buckets[self.next_bucket].iter() {
                    let shard =
                        v.sig.high_bits(store.shard_high_bits, shard_mask) as usize - shard_offset;
                    self.shards[shard].push(v);
                }
                if !self.borrowed {
                    drop(std::mem::take(&mut store.buckets[self.next_bucket]));
                }
                self.next_bucket += 1;
            }

            self.next_shard += 1;
            Some(Arc::new(self.shards.pop_front().unwrap()))
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<
        S: ZeroCopy + Sig + Send + Sync,
        V: ZeroCopy + Send + Sync,
        B: Send + Sync,
        T: BorrowMut<ShardStoreImpl<S, V, B>>,
    > ExactSizeIterator for ShardIterator<S, V, B, T>
where
    for<'a> ShardIterator<S, V, B, T>: Iterator,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.store.borrow().shard_sizes.len() - self.next_shard
    }
}

fn write_binary<S: ZeroCopy + Sig, V: ZeroCopy>(
    writer: &mut impl Write,
    tuples: &[SigVal<S, V>],
) -> std::io::Result<()> {
    let (pre, buf, post) = unsafe { tuples.align_to::<u8>() };
    debug_assert!(pre.is_empty());
    debug_assert!(post.is_empty());
    writer.write_all(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn _test_sig_store<S: ZeroCopy + Sig + Send + Sync>(
        mut sig_store: impl SigStore<S, u64>,
        get_rand_sig: fn(&mut SmallRng) -> S,
    ) -> anyhow::Result<()> {
        let mut rand = SmallRng::seed_from_u64(0);
        let shard_high_bits = sig_store.max_shard_high_bits();

        for _ in (0..10000).rev() {
            sig_store.try_push(SigVal {
                sig: get_rand_sig(&mut rand),
                val: rand.random(),
            })?;
        }
        let mut shard_store = sig_store.into_shard_store(shard_high_bits).unwrap();

        for _ in 0..2 {
            let mut count = 0;
            for shard in shard_store.iter() {
                for &w in shard.iter() {
                    assert_eq!(
                        count,
                        w.sig.high_bits(shard_high_bits, (1 << shard_high_bits) - 1)
                    );
                }
                count += 1;
            }
            assert_eq!(count, 1 << shard_high_bits);
        }

        let mut count = 0;
        for shard in shard_store.into_iter() {
            for &w in shard.iter() {
                assert_eq!(
                    count,
                    w.sig.high_bits(shard_high_bits, (1 << shard_high_bits) - 1)
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
                    _test_sig_store(
                        new_online(buckets_high_bits, max_shard_bits, None)?,
                        |rand| [rand.random(), rand.random()],
                    )?;
                    _test_sig_store(
                        new_offline(buckets_high_bits, max_shard_bits, None)?,
                        |rand| [rand.random(), rand.random()],
                    )?;
                }
            }
        }

        Ok(())
    }

    fn _test_u8<S: ZeroCopy + Sig>(
        mut sig_store: impl SigStore<S, u8>,
        get_rand_sig: fn(&mut SmallRng) -> S,
    ) -> anyhow::Result<()> {
        let mut rand = SmallRng::seed_from_u64(0);
        for _ in (0..1000).rev() {
            sig_store.try_push(SigVal {
                sig: get_rand_sig(&mut rand),
                val: rand.random(),
            })?;
        }
        let mut shard_store = sig_store.into_shard_store(2)?;
        let mut count = 0;

        for shard in shard_store.iter() {
            for &w in shard.iter() {
                assert_eq!(count, w.sig.high_bits(2, (1 << 2) - 1));
            }
            count += 1;
        }
        assert_eq!(count, 4);

        Ok(())
    }

    #[test]
    fn test_u8() -> anyhow::Result<()> {
        _test_u8(new_online(2, 2, None)?, |rand| {
            [rand.random(), rand.random()]
        })?;
        _test_u8(new_offline(2, 2, None)?, |rand| {
            [rand.random(), rand.random()]
        })?;
        Ok(())
    }
}
