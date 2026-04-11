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
//! [signature/value pairs] in any order, and when you call
//! [`into_shard_store`] it returns an immutable [`ShardStore`] that can
//! iterate on shards of signature/value pairs.
//!
//! The implementation exploits the fact that signatures are randomly
//! distributed, and thus bucket sorting is very effective: at construction time
//! you specify the number of high bits to use for bucket sorting (say, 8), and
//! when you [push] signature/value pairs they will be stored in different
//! buckets (in this case, 256) depending on their high bits.
//!
//! An [online] [`SigStore`] keeps the signature/value pairs in memory, while
//! an [offline] [`SigStore`] writes them to disk. After all key signatures
//! and values have been accumulated, [`into_shard_store`] will return a
//! [`ShardStore`]. [`into_shard_store`] takes the the number of high bits to
//! use for grouping signatures into shards, and the necessary bucket splitting
//! or merging will be handled automatically, albeit the most efficient
//! scenario is the one in which the number of buckets is equal to the number
//! of shards.
//!
//! You can iterate over the shards in a [`ShardStore`] multiple times using
//! the method [`iter`], or just once using the method [`drain`]. In the
//! latter case resources (i.e., files or memory) will be released as soon as
//! they are consumed.
//!
//! The trait [`ToSig`] provides a standard way to generate signatures for a
//! [`SigStore`]. Implementations are provided for signatures types `[u64;1]`
//! and `[u64; 2]`. In the first case, after a few billion keys you will start
//! to find duplicate signatures. In both cases, shards are defined by the
//! highest bits of the (first) signature.
//!
//! Both signatures and values must be [`BinSafe`] so that they might be
//! serialized and deserialized efficiently in the offline case, and they must
//! be [`Send`] and [`Sync`].
//!
//! [signature/value pairs]: SigVal
//! [`into_shard_store`]: SigStore::into_shard_store
//! [push]: SigStore::try_push
//! [online]: new_online
//! [offline]: new_offline
//! [`iter`]: ShardStore::iter
//! [`drain`]: ShardStore::drain

#![allow(clippy::comparison_chain)]
#![allow(clippy::type_complexity)]
use anyhow::Result;
use mem_dbg::{MemDbg, MemSize};

use rdst::RadixKey;
use std::{
    borrow::{Borrow, BorrowMut},
    collections::VecDeque,
    fs::File,
    io::*,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{BitXor, BitXorAssign},
    sync::Arc,
};
use xxhash_rust::xxh3;
use zerocopy::{FromBytes, IntoBytes};

/// Convenience trait for signatures and values of a [`SigStore`].
pub trait BinSafe: FromBytes + IntoBytes + Copy + Send + Sync + 'static {}
impl<T: FromBytes + IntoBytes + Copy + Send + Sync + 'static> BinSafe for T {}

/// A trait for types that can be used as signatures.
pub trait Sig: BinSafe + Default + PartialEq + Eq + std::fmt::Debug {
    /// Extracts high bits from the signature.
    ///
    /// These bits are used to shard elements. Note that `high_bits` can be 0,
    /// but it is guaranteed to be less than 64.
    ///
    /// # Panics
    ///
    /// In debug mode this method should panic if `mask` is not equal to `(1 <<
    /// high_bits) - 1`.
    fn high_bits(&self, high_bits: u32, mask: u64) -> u64;

    /// Constructs a signature from the state of an [`Xxh3`] streaming
    /// hasher.
    ///
    /// [`Xxh3`]: xxh3::Xxh3
    fn from_hasher(hasher: &xxh3::Xxh3) -> Self;
}

impl Sig for [u64; 2] {
    #[inline(always)]
    fn high_bits(&self, high_bits: u32, mask: u64) -> u64 {
        debug_assert!(mask == (1 << high_bits) - 1);
        self[0].rotate_left(high_bits) & mask
    }

    #[inline(always)]
    fn from_hasher(hasher: &xxh3::Xxh3) -> Self {
        let h = hasher.digest128();
        [(h >> 64) as u64, h as u64]
    }
}

impl Sig for [u64; 1] {
    #[inline(always)]
    fn high_bits(&self, high_bits: u32, mask: u64) -> u64 {
        debug_assert!(mask == (1 << high_bits) - 1);
        self[0].rotate_left(high_bits) & mask
    }

    #[inline(always)]
    fn from_hasher(hasher: &xxh3::Xxh3) -> Self {
        [hasher.digest()]
    }
}

/// A signature and a value.
#[derive(Debug, Clone, Copy, Default, MemSize, MemDbg)]
pub struct SigVal<S: BinSafe + Sig, V: BinSafe> {
    pub sig: S,
    pub val: V,
}

impl<V: BinSafe> RadixKey for SigVal<[u64; 2], V> {
    const LEVELS: usize = 16;

    fn get_level(&self, level: usize) -> u8 {
        (self.sig[1 - level / 8] >> ((level % 8) * 8)) as u8
    }
}

impl<V: BinSafe> RadixKey for SigVal<[u64; 1], V> {
    const LEVELS: usize = 8;

    fn get_level(&self, level: usize) -> u8 {
        (self.sig[0] >> ((level % 8) * 8)) as u8
    }
}

impl<S: Sig + PartialEq, V: BinSafe> PartialEq for SigVal<S, V> {
    fn eq(&self, other: &Self) -> bool {
        self.sig == other.sig
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    MemDbg,
    MemSize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    FromBytes,
    IntoBytes,
)]
#[mem_size(flat)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    repr(C),
    epserde(zero_copy)
)]
/// Zero-sized placeholder value type for filter-mode construction.
///
/// Used internally by [`VFilter`] and `VBuilder::try_build_filter` when
/// building structures that map keys to hash-derived values rather than
/// caller-supplied values. All arithmetic operations ([`BitXor`],
/// [`BitXorAssign`]) are no-ops.
///
/// [`VFilter`]: crate::dict::VFilter
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

// Fake implementations to treat EmptyVal like a value.
impl From<EmptyVal> for u128 {
    fn from(_: EmptyVal) -> u128 {
        0
    }
}

impl<V: BinSafe + BitXor<Output: BinSafe>> BitXor<SigVal<[u64; 1], V>> for SigVal<[u64; 1], V> {
    type Output = SigVal<[u64; 1], V::Output>;

    fn bitxor(self, rhs: SigVal<[u64; 1], V>) -> Self::Output {
        SigVal {
            sig: [self.sig[0].bitxor(rhs.sig[0])],
            val: self.val.bitxor(rhs.val),
        }
    }
}

impl<V: BinSafe + BitXor<Output: BinSafe>> BitXor<SigVal<[u64; 2], V>> for SigVal<[u64; 2], V> {
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

impl<V: BinSafe + BitXorAssign> BitXorAssign<SigVal<[u64; 1], V>> for SigVal<[u64; 1], V> {
    fn bitxor_assign(&mut self, rhs: SigVal<[u64; 1], V>) {
        self.sig[0] ^= rhs.sig[0];
        self.val ^= rhs.val;
    }
}

impl<V: BinSafe + BitXorAssign> BitXorAssign<SigVal<[u64; 2], V>> for SigVal<[u64; 2], V> {
    fn bitxor_assign(&mut self, rhs: SigVal<[u64; 2], V>) {
        self.sig[0] ^= rhs.sig[0];
        self.sig[1] ^= rhs.sig[1];
        self.val ^= rhs.val;
    }
}

/// Trait for types that can be turned into a signature.
///
/// We provide implementations for all primitive types, `str`, `String`, `&str`,
/// `&String`, and slices of primitive types, by turning them into bytes.
///
/// Note that for efficiency reasons the implementations are not
/// endianness-independent.
pub trait ToSig<S> {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> S;
}

impl ToSig<[u64; 2]> for String {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&str>::to_sig(&**key.borrow(), seed)
    }
}

impl ToSig<[u64; 1]> for String {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        <&str>::to_sig(&**key.borrow(), seed)
    }
}

impl ToSig<[u64; 2]> for &String {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&str>::to_sig(&***key.borrow(), seed)
    }
}

impl ToSig<[u64; 1]> for &String {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        <&str>::to_sig(&***key.borrow(), seed)
    }
}

impl ToSig<[u64; 2]> for str {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&str>::to_sig(key.borrow(), seed)
    }
}

impl ToSig<[u64; 1]> for str {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        <&str>::to_sig(key.borrow(), seed)
    }
}

impl ToSig<[u64; 2]> for &str {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&[u8]>::to_sig(key.borrow().as_bytes(), seed)
    }
}

impl ToSig<[u64; 1]> for &str {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        <&[u8]>::to_sig(key.borrow().as_bytes(), seed)
    }
}

macro_rules! to_sig_prim {
    ($($ty:ty),*) => {$(
        impl ToSig<[u64; 2]> for $ty {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
                let bytes = key.borrow().to_ne_bytes();
                let mut hasher = xxh3::Xxh3::with_seed(seed);
                hasher.update(bytes.as_slice());
                <[u64; 2]>::from_hasher(&hasher)
            }
        }
        impl ToSig<[u64;1]> for $ty {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                let bytes = key.borrow().to_ne_bytes();
                let mut hasher = xxh3::Xxh3::with_seed(seed);
                hasher.update(bytes.as_slice());
                <[u64; 1]>::from_hasher(&hasher)
            }
        }
    )*};
}

to_sig_prim!(
    isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128
);

macro_rules! to_sig_slice {
    ($($ty:ty),*) => {$(
        impl ToSig<[u64; 2]> for &[$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
                // Alignment to u8 never fails or leave trailing/leading bytes
                let bytes = unsafe {key.borrow().align_to::<u8>().1 };
                let mut hasher = xxh3::Xxh3::with_seed(seed);
                hasher.update(bytes);
                <[u64; 2]>::from_hasher(&hasher)
            }
        }
        impl ToSig<[u64;1]> for &[$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                // Alignment to u8 never fails or leave trailing/leading bytes
                let bytes = unsafe {key.borrow().align_to::<u8>().1 };
                let mut hasher = xxh3::Xxh3::with_seed(seed);
                hasher.update(bytes);
                <[u64; 1]>::from_hasher(&hasher)
            }
        }

        impl ToSig<[u64; 2]> for [$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
                <&[$ty]>::to_sig(key.borrow(), seed)
            }
        }
        impl ToSig<[u64;1]> for [$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                <&[$ty]>::to_sig(key.borrow(), seed)
            }
        }
    )*};
}

to_sig_slice!(
    isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128
);

/// A signature store.
///
/// The purpose of this trait is that of avoiding clumsy `where` clauses when
/// passing around a signature store. There is only one implementation,
/// [`SigStoreImpl`], but it is implemented only for certain combinations of
/// type parameters. Having this trait greatly simplifies the type signatures.
pub trait SigStore<S: Sig + BinSafe, V: BinSafe> {
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
    /// [`max_shard_high_bits`] or this method will panic.
    ///
    /// [`max_shard_high_bits`]: SigStore::max_shard_high_bits
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
/// See the [module documentation] for more information.
///
/// [module documentation]: crate::utils::sig_store
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
/// [temporary directory], and the files will be deleted when the store is
/// dropped.
///
/// [temporary directory]: https://doc.rust-lang.org/std/env/fn.temp_dir.html
pub fn new_offline<S: BinSafe + Sig, V: BinSafe>(
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
pub fn new_online<S: BinSafe + Sig, V: BinSafe>(
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

impl<S: BinSafe + Sig + Send + Sync, V: BinSafe> SigStore<S, V>
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

        Ok(ShardStoreImpl {
            bucket_high_bits: self.buckets_high_bits,
            shard_high_bits,
            max_shard_high_bits: self.max_shard_high_bits,
            buckets: files,
            buf_sizes: self.bucket_sizes,
            fine_shard_sizes: self.shard_sizes,
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

impl<S: BinSafe + Sig + Send + Sync, V: BinSafe> SigStore<S, V>
    for SigStoreImpl<S, V, Vec<SigVal<S, V>>>
{
    type Error = core::convert::Infallible;

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
        Ok(ShardStoreImpl {
            bucket_high_bits: self.buckets_high_bits,
            shard_high_bits,
            max_shard_high_bits: self.max_shard_high_bits,
            buckets: files,
            buf_sizes: self.bucket_sizes,
            fine_shard_sizes: self.shard_sizes,
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

#[cfg(feature = "rayon")]
impl<S: BinSafe + Sig + Send + Sync, V: BinSafe + Send + Sync>
    SigStoreImpl<S, V, Vec<SigVal<S, V>>>
{
    /// Populates the store in parallel by calling `f(i)` for each index
    /// `0..n` across rayon's thread pool and depositing the resulting
    /// [`SigVal`] directly into its bucket.
    ///
    /// Each bucket is protected by a [`Mutex`]; at 256
    /// buckets the contention is negligible. Returns the maximum value
    /// seen (for bit-width computation).
    /// Populates the store in parallel by calling `f(i)` for each index
    /// `0..n` across rayon's thread pool and depositing the resulting
    /// [`SigVal`] directly into its bucket.
    ///
    /// Each bucket is protected by a [`Mutex`].
    /// Thread-local buffers (one per bucket) batch insertions to reduce
    /// lock acquisitions. Returns the maximum value seen (for bit-width
    /// computation).
    ///
    /// [`Mutex`]: std::sync::Mutex
    pub fn par_populate(
        &mut self,
        n: usize,
        max_num_threads: usize,
        f: impl Fn(usize) -> SigVal<S, V> + Send + Sync,
    ) -> V
    where
        V: Default + Ord + Send,
    {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let num_buckets = 1usize << self.buckets_high_bits;
        let num_shards = 1usize << self.max_shard_high_bits;
        let bhb = self.buckets_high_bits;
        let bmask = self.buckets_mask;
        let shb = self.max_shard_high_bits;
        let smask = self.max_shard_mask;

        // Wrap each bucket Vec in a Mutex.
        let mutexed_buckets: Vec<Mutex<Vec<SigVal<S, V>>>> =
            self.buckets.drain(..).map(Mutex::new).collect();

        // Each rayon task uses thread-local per-bucket mini-buffers
        // (ArrayVec, stack-allocated) to batch mutex acquisitions, and
        // thread-local counters for bucket/shard sizes (merged at the
        // end, avoiding all atomic operations in the hot loop).
        const CAP: usize = 48;

        use arrayvec::ArrayVec;

        let max_val = (0..n)
            .into_par_iter()
            .with_min_len((n / max_num_threads).max(1_000_000))
            .fold(
                || {
                    let bufs: Box<[ArrayVec<SigVal<S, V>, CAP>]> =
                        (0..num_buckets).map(|_| ArrayVec::new()).collect();
                    let bc: Box<[usize]> = vec![0usize; num_buckets].into();
                    let sc: Box<[usize]> = vec![0usize; num_shards].into();
                    (V::default(), bufs, bc, sc)
                },
                |(mut local_max, mut local_bufs, mut bc, mut sc): (
                    V,
                    Box<[ArrayVec<SigVal<S, V>, CAP>]>,
                    Box<[usize]>,
                    Box<[usize]>,
                ),
                 i| {
                    let sv = f(i);
                    local_max = Ord::max(local_max, sv.val);
                    let bucket = sv.sig.high_bits(bhb, bmask) as usize;
                    let shard = sv.sig.high_bits(shb, smask) as usize;
                    bc[bucket] += 1;
                    sc[shard] += 1;
                    local_bufs[bucket].push(sv);
                    if local_bufs[bucket].is_full() {
                        mutexed_buckets[bucket]
                            .lock()
                            .unwrap()
                            .extend(local_bufs[bucket].drain(..));
                    }
                    (local_max, local_bufs, bc, sc)
                },
            )
            .map(|(local_max, local_bufs, bc, sc)| {
                for (bucket, buf) in local_bufs.into_vec().into_iter().enumerate() {
                    if !buf.is_empty() {
                        mutexed_buckets[bucket].lock().unwrap().extend(buf);
                    }
                }
                (local_max, bc, sc)
            })
            .reduce(
                || {
                    let bc: Box<[usize]> = vec![0usize; num_buckets].into();
                    let sc: Box<[usize]> = vec![0usize; num_shards].into();
                    (V::default(), bc, sc)
                },
                |(max_a, mut bc_a, mut sc_a), (max_b, bc_b, sc_b)| {
                    let m = Ord::max(max_a, max_b);
                    for (a, b) in bc_a.iter_mut().zip(bc_b.iter()) {
                        *a += b;
                    }
                    for (a, b) in sc_a.iter_mut().zip(sc_b.iter()) {
                        *a += b;
                    }
                    (m, bc_a, sc_a)
                },
            );

        let (max_val, local_bc, local_sc) = max_val;

        // Move Vecs back out of Mutexes.
        self.buckets
            .extend(mutexed_buckets.into_iter().map(|m| m.into_inner().unwrap()));

        // Merge local counters into the store's counts.
        for (i, c) in local_bc.iter().enumerate() {
            self.bucket_sizes[i] += c;
        }
        for (i, c) in local_sc.iter().enumerate() {
            self.shard_sizes[i] += c;
        }

        self.len += n;
        max_val
    }
}

/// A container for the signatures and values accumulated by a [`SigStore`],
/// with the ability to [enumerate them grouped in shards].
///
/// [enumerate them grouped in shards]: ShardStore::iter
///
/// Also in this case, the purpose of this trait is that of avoiding clumsy
/// `where` clauses when passing around a signature store. There is only one
/// implementation, [`ShardStoreImpl`], but it is implemented only for certain
/// combinations of type parameters. Having this trait greatly simplifies the
/// type signatures.
pub trait ShardStore<S: Sig, V: BinSafe> {
    /// Returns an iterator over the shard sizes at the current granularity.
    ///
    /// Yields one value per shard (i.e., 2<sup>`shard_high_bits`</sup>
    /// values). Computed on the fly from fine-grained counters; callers
    /// needing random access should `.collect()`.
    fn shard_sizes(&self) -> Box<dyn Iterator<Item = usize> + '_>;

    /// Returns an iterator on shards.
    ///
    /// This method can be called multiple times; the store is not
    /// modified.
    fn iter(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_>;

    /// Returns an iterator on shards, draining the store.
    ///
    /// Like [`iter`], but frees each shard's memory as it is consumed.
    /// After draining, subsequent calls to `iter` or `drain` will yield
    /// no elements.
    ///
    /// [`iter`]: Self::iter
    fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_>;

    /// Changes the shard granularity.
    ///
    /// `new_bits` must be at most [`max_shard_high_bits`]. Both coarsening
    /// and refining are supported; the fine-grained counters from
    /// construction are never discarded.
    ///
    /// # Panics
    ///
    /// Panics if `new_bits` exceeds [`max_shard_high_bits`].
    ///
    /// [`max_shard_high_bits`]: Self::max_shard_high_bits
    fn set_shard_high_bits(&mut self, new_bits: u32);

    /// Returns the maximum value that can be passed to
    /// [`set_shard_high_bits`].
    ///
    /// [`set_shard_high_bits`]: Self::set_shard_high_bits
    fn max_shard_high_bits(&self) -> u32;

    /// Returns the number of signature/value pairs in the store.
    fn len(&self) -> usize {
        self.shard_sizes().sum()
    }
}

/// An implementation of [`ShardStore`].
///
/// See the [module documentation] for more information.
///
/// [module documentation]: crate::utils::sig_store
#[derive(Debug)]
pub struct ShardStoreImpl<S, V, B> {
    /// The number of high bits used for bucket sorting.
    bucket_high_bits: u32,
    /// The number of high bits defining a shard (the current view).
    shard_high_bits: u32,
    /// The maximum value of `shard_high_bits` (set at construction).
    max_shard_high_bits: u32,
    /// The buckets (files or vectors).
    buckets: Vec<B>,
    /// The number of keys in each bucket.
    buf_sizes: Vec<usize>,
    /// Per-shard key counts at `max_shard_high_bits` granularity (never changed).
    fine_shard_sizes: Vec<usize>,
    _marker: PhantomData<(S, V)>,
}

impl<S: BinSafe + Sig + Send + Sync, V: BinSafe + Send + Sync, B: Send + Sync> ShardStore<S, V>
    for ShardStoreImpl<S, V, B>
where
    for<'a> ShardIter<S, V, B, &'a mut Self>: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync,
{
    fn shard_sizes(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        let coarsen = 1usize << (self.max_shard_high_bits - self.shard_high_bits);
        Box::new(
            self.fine_shard_sizes
                .chunks(coarsen)
                .map(|c| c.iter().sum()),
        )
    }

    fn set_shard_high_bits(&mut self, new_bits: u32) {
        assert!(new_bits <= self.max_shard_high_bits);
        self.shard_high_bits = new_bits;
    }

    fn max_shard_high_bits(&self) -> u32 {
        self.max_shard_high_bits
    }

    fn iter(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        Box::new(ShardIter {
            store: self,
            borrowed: true,
            next_bucket: 0,
            next_shard: 0,
            shards: VecDeque::from(vec![]),
            _marker: PhantomData,
        })
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        Box::new(ShardIter {
            store: self,
            borrowed: false,
            next_bucket: 0,
            next_shard: 0,
            shards: VecDeque::from(vec![]),
            _marker: PhantomData,
        })
    }
}

/// An iterator on shards in a [`ShardStore`].
///
/// A [`ShardIter`] handles the mapping between buckets and shards. If a
/// shard is made by one or more buckets, it will aggregate them as necessary;
/// if a bucket contains several shards, it will split the bucket into shards.
#[derive(Debug)]
pub struct ShardIter<S: BinSafe + Sig, V: BinSafe, B, T: BorrowMut<ShardStoreImpl<S, V, B>>> {
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
    S: BinSafe + Sig + Send + Sync,
    V: BinSafe,
    T: BorrowMut<ShardStoreImpl<S, V, BufReader<File>>>,
> Iterator for ShardIter<S, V, BufReader<File>, T>
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

            let coarsen = 1usize << (store.max_shard_high_bits - store.shard_high_bits);
            let base = self.next_shard * coarsen;
            let len: usize = store.fine_shard_sizes[base..base + coarsen].iter().sum();
            let mut shard = Vec::<SigVal<S, V>>::with_capacity(len);

            // SAFETY: we just allocated this vector so it is safe to set the length,
            // and read_exact guarantees that the vector will be filled with data
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
                    if !self.borrowed {
                        let _ = store.buckets[i].get_mut().set_len(0);
                    }
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
                let coarsen = 1usize << (store.max_shard_high_bits - store.shard_high_bits);
                for shard in shard_offset..shard_offset + split_into {
                    let base = shard * coarsen;
                    let cap: usize = store.fine_shard_sizes[base..base + coarsen].iter().sum();
                    self.shards.push_back(Vec::with_capacity(cap));
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
    S: BinSafe + Sig + Send + Sync,
    V: BinSafe,
    T: BorrowMut<ShardStoreImpl<S, V, Arc<Vec<SigVal<S, V>>>>>,
> Iterator for ShardIter<S, V, Arc<Vec<SigVal<S, V>>>, T>
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

            let coarsen = 1usize << (store.max_shard_high_bits - store.shard_high_bits);
            let base = self.next_shard * coarsen;
            let len: usize = store.fine_shard_sizes[base..base + coarsen].iter().sum();
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
                let coarsen = 1usize << (store.max_shard_high_bits - store.shard_high_bits);
                for shard in shard_offset..shard_offset + split_into {
                    let base = shard * coarsen;
                    let cap: usize = store.fine_shard_sizes[base..base + coarsen].iter().sum();
                    self.shards.push_back(Vec::with_capacity(cap));
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
    S: BinSafe + Sig + Send + Sync,
    V: BinSafe,
    B: Send + Sync,
    T: BorrowMut<ShardStoreImpl<S, V, B>>,
> ExactSizeIterator for ShardIter<S, V, B, T>
where
    for<'a> ShardIter<S, V, B, T>: Iterator,
{
    #[inline(always)]
    fn len(&self) -> usize {
        (1usize << self.store.borrow().shard_high_bits) - self.next_shard
    }
}

impl<
    S: BinSafe + Sig + Send + Sync,
    V: BinSafe,
    B: Send + Sync,
    T: BorrowMut<ShardStoreImpl<S, V, B>>,
> FusedIterator for ShardIter<S, V, B, T>
where
    for<'a> ShardIter<S, V, B, T>: Iterator,
{
}

fn write_binary<S: BinSafe + Sig, V: BinSafe>(
    writer: &mut impl Write,
    tuples: &[SigVal<S, V>],
) -> std::io::Result<()> {
    let (pre, buf, post) = unsafe { tuples.align_to::<u8>() };
    debug_assert!(pre.is_empty());
    debug_assert!(post.is_empty());
    writer.write_all(buf)
}

/// A [`ShardStore`] wrapper that filters entries and re-shards to an
/// arbitrary granularity.
///
/// [`SigVal`]s for which the predicate returns `false` are excluded.
///
/// [`set_shard_high_bits`] is called on the inner store to set the desired
/// granularity (both coarsening and refining are supported up to
/// [`max_shard_high_bits`]).
///
/// [`set_shard_high_bits`]: ShardStore::set_shard_high_bits
/// [`max_shard_high_bits`]: ShardStore::max_shard_high_bits
pub struct FilteredShardStore<'a, SS: ?Sized, S, V, F> {
    /// The inner store.
    inner: &'a mut SS,
    /// The filter predicate.
    filter: F,
    /// Shard sizes after filtering.
    shard_sizes: Vec<usize>,
    _marker: std::marker::PhantomData<(S, V)>,
}

impl<'a, SS: ?Sized, S, V, F> FilteredShardStore<'a, SS, S, V, F>
where
    SS: ShardStore<S, V>,
    S: Sig + BinSafe + Send + Sync,
    V: BinSafe + Copy,
    F: Fn(&SigVal<S, V>) -> bool,
{
    /// Creates a new filtered view with pre-computed shard sizes,
    /// skipping the size-counting scan.
    ///
    /// `shard_sizes` must have one entry per shard (after
    /// re-aggregation to `shard_high_bits`), where each entry is the
    /// number of entries in that shard that pass `filter`.
    pub fn new(
        inner: &'a mut SS,
        shard_high_bits: u32,
        filter: F,
        shard_sizes: Vec<usize>,
    ) -> Self {
        inner.set_shard_high_bits(shard_high_bits);
        Self {
            inner,
            filter,
            shard_sizes,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, SS: ?Sized, S, V, F> ShardStore<S, V> for FilteredShardStore<'a, SS, S, V, F>
where
    SS: ShardStore<S, V>,
    S: Sig + BinSafe + Send + Sync,
    V: BinSafe + Copy + Send + Sync,
    F: Fn(&SigVal<S, V>) -> bool + Send + Sync,
{
    fn shard_sizes(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        Box::new(self.shard_sizes.iter().copied())
    }

    fn set_shard_high_bits(&mut self, new_bits: u32) {
        self.inner.set_shard_high_bits(new_bits);
        // Recompute filtered shard sizes after the inner store changed.
        self.shard_sizes = self
            .inner
            .iter()
            .map(|shard| shard.iter().filter(|sv| (self.filter)(sv)).count())
            .collect();
    }

    fn max_shard_high_bits(&self) -> u32 {
        self.inner.max_shard_high_bits()
    }

    fn iter(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        let filter = &self.filter;
        let inner_shards: Vec<_> = self.inner.iter().collect();
        Box::new(
            inner_shards
                .into_iter()
                .map(move |shard| {
                    let filtered: Vec<_> = shard.iter().filter(|sv| filter(sv)).copied().collect();
                    Arc::new(filtered)
                })
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        // FilteredShardStore always re-filters, so drain == iter.
        self.iter()
    }
}

impl<S: Sig, V: BinSafe> ShardStore<S, V> for Box<dyn ShardStore<S, V> + Send + Sync> {
    fn shard_sizes(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        (**self).shard_sizes()
    }

    fn set_shard_high_bits(&mut self, new_bits: u32) {
        (**self).set_shard_high_bits(new_bits)
    }

    fn max_shard_high_bits(&self) -> u32 {
        (**self).max_shard_high_bits()
    }

    fn iter(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        (**self).iter()
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        (**self).drain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    fn _test_sig_store<S: BinSafe + Sig + Send + Sync>(
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
        for shard in shard_store.drain() {
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

    fn _test_u8<S: BinSafe + Sig>(
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
