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
//! An [online] [`SigStore`] keeps the signature/value pairs in memory, while an
//! [offline] [`SigStore`] writes them to disk. After all key signatures and
//! values have been accumulated, [`into_shard_store`] will return a
//! [`ShardStore`]. [`into_shard_store`] takes the number of high bits to use
//! for grouping signatures into shards, and the necessary bucket splitting or
//! merging will be handled automatically, albeit the most efficient scenario is
//! the one in which the number of buckets is equal to the number of shards.
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
                let h = xxh3::xxh3_128_with_seed(&bytes, seed);
                [(h >> 64) as u64, h as u64]
            }
        }
        impl ToSig<[u64;1]> for $ty {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                let bytes = key.borrow().to_ne_bytes();
                [xxh3::xxh3_64_with_seed(&bytes, seed)]
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
                // Alignment to u8 never fails or leaves trailing/leading bytes
                let bytes = unsafe { key.borrow().align_to::<u8>().1 };
                let h = xxh3::xxh3_128_with_seed(bytes, seed);
                [(h >> 64) as u64, h as u64]
            }
        }
        impl ToSig<[u64;1]> for &[$ty] {
            fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
                // Alignment to u8 never fails or leaves trailing/leading bytes
                let bytes = unsafe { key.borrow().align_to::<u8>().1 };
                [xxh3::xxh3_64_with_seed(bytes, seed)]
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
    /// [`max_shard_high_bits`] or this method will panic. For an offline
    /// (file-backed) store, iterating the resulting shard store also panics
    /// if a temp-file read or seek fails (a disk error or a truncated
    /// temporary file); such a mid-build I/O failure is not recoverable.
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

    /// Reserves capacity in each bucket based on the given per-bucket counts.
    ///
    /// For in-memory stores, this pre-allocates the bucket vectors so that
    /// subsequent pushes avoid reallocations. For on-disk stores, this is a
    /// no-op.
    fn reserve_buckets(&mut self, _counts: impl AsRef<[usize]>) {}

    /// Merges per-bucket data and shard counts into this store,
    /// updating bucket sizes, shard sizes, and key count.
    ///
    /// # Safety
    ///
    /// `buckets` must contain exactly one vector per store bucket.
    /// `shard_sizes` must contain exactly one count per shard at
    /// [`max_shard_high_bits`](Self::max_shard_high_bits), and each count must
    /// equal the number of supplied signatures assigned to that shard.
    unsafe fn merge_from(
        &mut self,
        buckets: Box<[Vec<SigVal<S, V>>]>,
        shard_sizes: Box<[usize]>,
    ) -> Result<(), Self::Error>;
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
    // A mask for the lowest max_shard_high_bits bits.
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
/// [temporary directory], and the files will be deleted when the shard store
/// returned by [`into_shard_store`] is dropped.
///
/// [temporary directory]: https://doc.rust-lang.org/std/env/fn.temp_dir.html
/// [`into_shard_store`]: SigStore::into_shard_store
pub fn new_offline<S: BinSafe + Sig, V: BinSafe>(
    buckets_high_bits: u32,
    max_shard_high_bits: u32,
    _expected_num_keys: Option<usize>,
) -> Result<SigStoreImpl<S, V, BufWriter<File>>> {
    let temp_dir = tempfile::TempDir::new()?;
    anyhow::ensure!(
        buckets_high_bits < usize::BITS && max_shard_high_bits < usize::BITS,
        "high bit counts must be less than {} (got buckets {buckets_high_bits}, max shard {max_shard_high_bits})",
        usize::BITS
    );
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
    anyhow::ensure!(
        buckets_high_bits < usize::BITS && max_shard_high_bits < usize::BITS,
        "high bit counts must be less than {} (got buckets {buckets_high_bits}, max shard {max_shard_high_bits})",
        usize::BITS
    );
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
        // high_bits can be 0
        let buffer = sig_val
            .sig
            .high_bits(self.buckets_high_bits, self.buckets_mask) as usize;
        let shard = sig_val
            .sig
            .high_bits(self.max_shard_high_bits, self.max_shard_mask) as usize;

        // Account only after write_binary returns Ok, so a write error does not
        // leave the counters claiming records that were never accepted. This
        // fixes phantom counts; BufWriter still buffers, so it is not a
        // transactional guarantee and the store should be discarded after an
        // I/O error.
        write_binary(&mut self.buckets[buffer], std::slice::from_ref(&sig_val))?;
        self.len += 1;
        self.bucket_sizes[buffer] += 1;
        self.shard_sizes[shard] += 1;
        Ok(())
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
            _temp_dir: self.temp_dir,
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

    unsafe fn merge_from(
        &mut self,
        buckets: Box<[Vec<SigVal<S, V>>]>,
        shard_sizes: Box<[usize]>,
    ) -> Result<(), Self::Error> {
        assert_eq!(buckets.len(), self.buckets.len(), "bucket count mismatch");
        assert_eq!(
            shard_sizes.len(),
            self.shard_sizes.len(),
            "shard count mismatch"
        );
        for (b, mini_bucket) in buckets.into_vec().into_iter().enumerate() {
            // Account only after the write succeeds (see try_push): a failure on
            // bucket b must not inflate len/bucket_sizes for records not written.
            write_binary(&mut self.buckets[b], &mini_bucket)?;
            self.len += mini_bucket.len();
            self.bucket_sizes[b] += mini_bucket.len();
        }
        for (s, &c) in shard_sizes.iter().enumerate() {
            self.shard_sizes[s] += c;
        }
        Ok(())
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
            _temp_dir: self.temp_dir,
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

    fn reserve_buckets(&mut self, counts: impl AsRef<[usize]>) {
        for (bucket, &count) in self.buckets.iter_mut().zip(counts.as_ref()) {
            bucket.reserve(count);
        }
    }

    unsafe fn merge_from(
        &mut self,
        buckets: Box<[Vec<SigVal<S, V>>]>,
        shard_sizes: Box<[usize]>,
    ) -> Result<(), Self::Error> {
        assert_eq!(buckets.len(), self.buckets.len(), "bucket count mismatch");
        assert_eq!(
            shard_sizes.len(),
            self.shard_sizes.len(),
            "shard count mismatch"
        );
        for (b, mini_bucket) in buckets.into_vec().into_iter().enumerate() {
            self.len += mini_bucket.len();
            self.bucket_sizes[b] += mini_bucket.len();
            self.buckets[b].extend(mini_bucket);
        }
        for (s, &c) in shard_sizes.iter().enumerate() {
            self.shard_sizes[s] += c;
        }
        Ok(())
    }
}

#[cfg(feature = "rayon")]
impl<S: BinSafe + Sig + Send + Sync, V: BinSafe + Send + Sync>
    SigStoreImpl<S, V, Vec<SigVal<S, V>>>
{
    /// Populates the store in parallel by calling a provided closure for each
    /// index [0 . . `n`) across rayon's thread pool.
    ///
    /// Each thread populates a local store. The local stores are then merged
    /// into this store via [`merge_from`], which does a bulk copy per bucket.
    ///
    /// Returns the maximum value seen (for bit-width computation).
    ///
    /// [`merge_from`]: SigStore::merge_from
    pub fn par_populate(
        &mut self,
        n: usize,
        max_num_threads: usize,
        f: impl Fn(usize) -> SigVal<S, V> + Send + Sync,
    ) -> anyhow::Result<V>
    where
        V: Default + Ord + Send,
    {
        use rayon::prelude::*;

        // Never create more workers than there are records: an empty chunk still
        // allocates a full per-shard counter array (1 << max_shard_high_bits), so
        // cap the worker count by n to keep memory proportional to the input.
        let num_threads = max_num_threads.max(1).min(n.max(1));
        let chunk_size = n.div_ceil(num_threads);
        let num_buckets = 1usize << self.buckets_high_bits;
        let num_shards = 1usize << self.max_shard_high_bits;
        let bhb = self.buckets_high_bits;
        let bmask = self.buckets_mask;
        let mshb = self.max_shard_high_bits;
        let msmask = self.max_shard_mask;

        let mini_stores: Vec<_> = (0..num_threads)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk_size;
                let end = ((t + 1) * chunk_size).min(n);
                let per_bucket =
                    (end.saturating_sub(start).div_ceil(num_buckets) as f64 * 1.10) as usize;
                let mut buckets: Box<[Vec<SigVal<S, V>>]> = (0..num_buckets)
                    .map(|_| Vec::with_capacity(per_bucket))
                    .collect();
                let mut shard_sizes: Box<[usize]> = vec![0usize; num_shards].into_boxed_slice();
                let mut max_v: Option<V> = None;
                for i in start..end {
                    let sv = f(i);
                    max_v = Some(max_v.map_or(sv.val, |m| Ord::max(m, sv.val)));
                    let bucket = sv.sig.high_bits(bhb, bmask) as usize;
                    let shard = sv.sig.high_bits(mshb, msmask) as usize;
                    shard_sizes[shard] += 1;
                    buckets[bucket].push(sv);
                }
                (buckets, shard_sizes, max_v)
            })
            .collect();

        let bucket_totals: Vec<usize> = (0..num_buckets)
            .map(|b| mini_stores.iter().map(|(bs, _, _)| bs[b].len()).sum())
            .collect();
        self.reserve_buckets(&bucket_totals);

        let mut max_val: Option<V> = None;
        for (buckets, shard_sizes, local_max) in mini_stores {
            if let Some(lm) = local_max {
                max_val = Some(max_val.map_or(lm, |m| Ord::max(m, lm)));
            }
            // SAFETY: each mini-store uses the same bucket/shard geometry as
            // self, and increments shard_sizes for every value put in buckets.
            unsafe { self.merge_from(buckets, shard_sizes).unwrap() };
        }

        Ok(max_val.unwrap_or_default())
    }
}

/// A container for the signatures and values accumulated by a [`SigStore`],
/// with the ability to [enumerate them grouped in shards].
///
/// Also in this case, the purpose of this trait is that of avoiding clumsy
/// `where` clauses when passing around a signature store. There is only one
/// implementation, [`ShardStoreImpl`], but it is implemented only for certain
/// combinations of type parameters. Having this trait greatly simplifies the
/// type signatures.
///
/// [enumerate them grouped in shards]: ShardStore::iter
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
    /// Like [`iter`], but frees each shard's memory as it is consumed and
    /// drops the store's accounting for each drained shard, so `len` and
    /// `shard_sizes` track only the data left after a full or partial drain.
    /// After a full drain, subsequent `iter`/`drain` calls yield only empty
    /// shards (no signature/value pairs).
    ///
    /// [`iter`]: Self::iter
    fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_>;

    /// Changes the shard granularity.
    ///
    /// `new_bits` must be at most [`max_shard_high_bits`]. Both coarsening
    /// and refining are supported: changing the granularity re-aggregates the
    /// fine-grained per-shard counters rather than discarding them (draining,
    /// by contrast, does clear the counts for consumed shards).
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
    /// Per-shard key counts at `max_shard_high_bits` granularity. Changing the
    /// shard granularity re-aggregates these; draining clears the counts for
    /// consumed shards so `len`/`shard_sizes` track only what remains.
    fine_shard_sizes: Vec<usize>,
    /// Keeps the backing temporary directory alive while the bucket readers
    /// are open (offline stores only). It must be declared after `buckets` so
    /// that the files are closed before the directory is removed.
    _temp_dir: Option<tempfile::TempDir>,
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

            {
                let spare = &mut shard.spare_capacity_mut()[..len];
                // SAFETY: MaybeUninit<SigVal<S, V>> may be viewed as bytes;
                // u8 has alignment one and every byte pattern is valid.
                let (pre, mut buf, post) = unsafe { spare.align_to_mut::<u8>() };
                assert!(pre.is_empty());
                assert!(post.is_empty());
                for i in self.next_bucket..self.next_bucket + to_aggr {
                    let bytes = store.buf_sizes[i]
                        .checked_mul(core::mem::size_of::<SigVal<S, V>>())
                        .expect("offline signature store: bucket byte size overflow");
                    store.buckets[i]
                        .seek(SeekFrom::Start(0))
                        .expect("offline signature store: temp-file seek failed");
                    let target = buf
                        .get_mut(..bytes)
                        .expect("offline signature store: shard counts are inconsistent");
                    store.buckets[i].read_exact(target).expect(
                        "offline signature store: temp-file read failed (disk error or truncation)",
                    );
                    if !self.borrowed {
                        let _ = store.buckets[i].get_mut().set_len(0);
                        // Drop the store's accounting for this drained bucket so
                        // len() and any later iter() do not read a now-truncated
                        // temp file.
                        store.buf_sizes[i] = 0;
                    }
                    buf = &mut buf[bytes..];
                }
                assert!(
                    buf.is_empty(),
                    "offline signature store: shard counts are inconsistent"
                );
            }
            // SAFETY: every byte of all `len` elements was initialized by
            // read_exact above from previously written BinSafe SigVal records.
            unsafe { shard.set_len(len) };
            if !self.borrowed {
                // The whole current shard has been drained; zero its fine-shard
                // counters so the store's len()/shard_sizes() reflect the removal
                // even if iteration stops here.
                store.fine_shard_sizes[base..base + coarsen]
                    .iter_mut()
                    .for_each(|c| *c = 0);
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
                let shard_mask = (1 << store.shard_high_bits) - 1;
                store.buckets[self.next_bucket]
                    .seek(SeekFrom::Start(0))
                    .expect("offline signature store: temp-file seek failed");

                while len > 0 {
                    let to_read = buf_size.min(len);
                    let spare = &mut buffer.spare_capacity_mut()[..to_read];
                    // SAFETY: MaybeUninit<SigVal<S, V>> may be viewed as
                    // bytes; u8 has alignment one and every byte pattern is valid.
                    let (pre, buf, after) = unsafe { spare.align_to_mut::<u8>() };
                    debug_assert!(pre.is_empty());
                    debug_assert!(after.is_empty());

                    store.buckets[self.next_bucket].read_exact(buf).expect(
                        "offline signature store: temp-file read failed (disk error or truncation)",
                    );
                    // SAFETY: read_exact initialized every byte of the
                    // `to_read` BinSafe SigVal records in spare capacity.
                    unsafe { buffer.set_len(to_read) };

                    // We copy each signature/value pair into its shard.
                    for &value in &buffer {
                        let shard =
                            usize::try_from(value.sig.high_bits(store.shard_high_bits, shard_mask))
                                .expect("the configured shard index must fit in usize")
                                - shard_offset;
                        self.shards[shard].push(value);
                    }
                    buffer.clear();
                    len -= to_read;
                }

                if !self.borrowed {
                    let _ = store.buckets[self.next_bucket].get_mut().set_len(0);
                    // Reading the bucket materialized all of its shards into
                    // `self.shards` and truncated the file, so its data is gone
                    // from the store now; drop the accounting for the whole
                    // contiguous fine-shard range (an early iterator drop must
                    // not leave len() counting removed data).
                    let fine_start = shard_offset * coarsen;
                    let fine_end = (shard_offset + split_into) * coarsen;
                    store.buf_sizes[self.next_bucket] = 0;
                    store.fine_shard_sizes[fine_start..fine_end]
                        .iter_mut()
                        .for_each(|c| *c = 0);
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
                let taken = std::mem::take(&mut store.buckets[self.next_bucket]);
                // Draining removes the bucket; drop its accounting so
                // len()/shard_sizes() stay honest if iteration stops here.
                let coarsen = 1usize << (store.max_shard_high_bits - store.shard_high_bits);
                let base = self.next_shard * coarsen;
                store.buf_sizes[self.next_bucket] = 0;
                store.fine_shard_sizes[base..base + coarsen]
                    .iter_mut()
                    .for_each(|c| *c = 0);
                taken
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
                    store.buf_sizes[i] = 0;
                }
            }
            if !self.borrowed {
                // The current shard has been drained; zero its fine-shard
                // counters so len()/shard_sizes() reflect the removal.
                store.fine_shard_sizes[base..base + coarsen]
                    .iter_mut()
                    .for_each(|c| *c = 0);
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
                    // The whole bucket was materialized into `self.shards`; drop
                    // its accounting for the full contiguous fine-shard range so
                    // an early iterator drop cannot leave len() overcounting.
                    let fine_start = shard_offset * coarsen;
                    let fine_end = (shard_offset + split_into) * coarsen;
                    store.buf_sizes[self.next_bucket] = 0;
                    store.fine_shard_sizes[fine_start..fine_end]
                        .iter_mut()
                        .for_each(|c| *c = 0);
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
    // `SigVal` is `repr(Rust)`; when the value type is narrower than the
    // signature alignment (e.g. `u8`) the struct carries padding whose bytes
    // are never initialized. Byte-casting `&[SigVal]` with `align_to::<u8>()`
    // and writing it would read that uninitialized padding, which is undefined
    // behavior. Instead we copy only the initialized `sig`/`val` field bytes
    // into a zeroed buffer at their real struct offsets, so each on-disk record
    // is byte-identical to the in-memory layout (the read side reconstructs it
    // unchanged) but the padding is deterministic zero and never read.
    const CHUNK: usize = 1024;
    let elem_size = core::mem::size_of::<SigVal<S, V>>();
    let sig_off = core::mem::offset_of!(SigVal<S, V>, sig);
    let val_off = core::mem::offset_of!(SigVal<S, V>, val);
    let sig_size = core::mem::size_of::<S>();
    let val_size = core::mem::size_of::<V>();
    // Size the reusable buffer to the actual work, not a fixed CHUNK: a
    // single-record `try_push` (tuples.len() == 1) must not allocate and zero
    // space for CHUNK records.
    let mut buf = vec![0u8; elem_size * tuples.len().min(CHUNK)];
    for chunk in tuples.chunks(CHUNK) {
        for (i, sv) in chunk.iter().enumerate() {
            let base = i * elem_size;
            let src = (sv as *const SigVal<S, V>).cast::<u8>();
            // SAFETY: `sig` and `val` are initialized fields at offsets
            // `sig_off`/`val_off` within `*sv`, so `src.add(off)` reads exactly
            // that field's initialized bytes; the destination ranges lie inside
            // the zeroed `buf` because `base + off + size <= elem_size` and
            // `base + elem_size <= buf.len()`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src.add(sig_off),
                    buf.as_mut_ptr().add(base + sig_off),
                    sig_size,
                );
                core::ptr::copy_nonoverlapping(
                    src.add(val_off),
                    buf.as_mut_ptr().add(base + val_off),
                    val_size,
                );
            }
        }
        writer.write_all(&buf[..core::mem::size_of_val(chunk)])?;
    }
    Ok(())
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
        // Report the cached filtered count per shard, but zero for any shard the
        // inner store has already drained. Draining a bucket in the split case
        // removes several inner shards at once, so masking by
        // `inner.shard_sizes()` (which is zero exactly for drained or empty
        // shards) keeps len() honest after a partial drain without rescanning.
        Box::new(
            self.shard_sizes
                .iter()
                .copied()
                .zip(self.inner.shard_sizes())
                .map(|(filtered, inner)| if inner == 0 { 0 } else { filtered }),
        )
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
        // Filter one shard at a time, lazily, so the whole store is never
        // materialized in RAM at once.
        Box::new(
            self.inner.iter().map(move |shard| {
                Arc::new(shard.iter().filter(|sv| filter(sv)).copied().collect())
            }),
        )
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send + Sync + '_> {
        let filter = &self.filter;
        // Same filtering as iter, but drains the inner store so each shard's
        // memory is released as it is consumed. Post-drain counts stay correct
        // via shard_sizes(), which masks the cached filtered counts by the inner
        // store's now-drained shard sizes.
        Box::new(
            self.inner.drain().map(move |shard| {
                Arc::new(shard.iter().filter(|sv| filter(sv)).copied().collect())
            }),
        )
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

    #[test]
    fn test_high_bits_validation() {
        // High-bit counts at or above the word width would overflow the
        // 1 << bits allocations and masks; they must be rejected.
        assert!(new_offline::<[u64; 1], u64>(usize::BITS, 8, None).is_err());
        assert!(new_offline::<[u64; 1], u64>(8, usize::BITS, None).is_err());
        assert!(new_online::<[u64; 1], u64>(usize::BITS, 8, None).is_err());
        assert!(new_online::<[u64; 1], u64>(8, usize::BITS, None).is_err());
        // Valid parameters still succeed.
        assert!(new_online::<[u64; 1], u64>(4, 8, None).is_ok());
    }

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

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_populate_all_negative_max() -> anyhow::Result<()> {
        // par_populate must return the true maximum value even when every value
        // is negative; seeding the accumulator with V::default() (0) would
        // wrongly report 0 for an all-negative signed value type.
        let neg = |n: usize, threads: usize| -> anyhow::Result<i64> {
            let mut store = new_online::<[u64; 1], i64>(2, 2, None)?;
            store.par_populate(n, threads, |i| {
                let i = i64::try_from(i).expect("index fits i64");
                SigVal {
                    sig: [u64::try_from(i).expect("index fits u64") + 1],
                    val: -i - 1,
                }
            })
        };
        assert_eq!(neg(10, 4)?, -1, "max of -1..=-10 is -1, not the default 0");
        // Fewer values than threads leaves some worker chunks empty; the merge
        // must skip them rather than fold in a spurious default 0.
        assert_eq!(neg(2, 8)?, -1);
        // Empty input returns the default.
        let mut empty = new_online::<[u64; 1], i64>(2, 2, None)?;
        assert_eq!(empty.par_populate(0, 4, |_| unreachable!())?, 0);
        Ok(())
    }

    /// A failed bucket write in offline mode must not inflate the record
    /// counters (len/bucket_sizes/shard_sizes); otherwise into_shard_store would
    /// later read more records than were actually written. `/dev/full` fails
    /// every write with ENOSPC, and a zero-capacity BufWriter forwards writes to
    /// it without buffering.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_offline_try_push_write_error_keeps_counts_zero() -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        let mut store = new_offline::<[u64; 1], u8>(0, 0, None)?;
        store.buckets[0] =
            BufWriter::with_capacity(0, File::options().write(true).open("/dev/full")?);
        let err = store
            .try_push(SigVal {
                sig: [1u64],
                val: 0u8,
            })
            .unwrap_err();
        assert_eq!(
            err.raw_os_error(),
            Some(28),
            "expected ENOSPC from /dev/full"
        );
        assert_eq!(store.len, 0, "len must not count an unwritten record");
        assert_eq!(store.bucket_sizes[0], 0);
        assert_eq!(store.shard_sizes[0], 0);
        Ok(())
    }

    /// Same accounting invariant for the offline merge_from fast path.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_offline_merge_from_write_error_keeps_counts_zero() -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        let mut store = new_offline::<[u64; 1], u8>(0, 0, None)?;
        store.buckets[0] =
            BufWriter::with_capacity(0, File::options().write(true).open("/dev/full")?);
        let buckets = vec![vec![SigVal {
            sig: [1u64],
            val: 0u8,
        }]]
        .into_boxed_slice();
        let shard_sizes = vec![1usize].into_boxed_slice();
        // SAFETY: a single bucket and a single shard match the store's layout
        // (buckets_high_bits == max_shard_high_bits == 0); the write fails before
        // any counter is applied, which is precisely the invariant asserted below.
        let err = unsafe { store.merge_from(buckets, shard_sizes) }.unwrap_err();
        assert_eq!(
            err.raw_os_error(),
            Some(28),
            "expected ENOSPC from /dev/full"
        );
        assert_eq!(store.len, 0);
        assert_eq!(store.bucket_sizes[0], 0);
        assert_eq!(store.shard_sizes[0], 0);
        Ok(())
    }

    fn _check_drain<SS: ShardStore<[u64; 2], u64>>(
        label: &str,
        build: impl Fn() -> anyhow::Result<SS>,
        n: usize,
    ) -> anyhow::Result<()> {
        // A full drain must leave the store reporting itself empty and, for an
        // offline store, must not panic on a later iteration by reading a
        // now-truncated temp file.
        let mut ss = build()?;
        assert_eq!(ss.len(), n, "{label}: len before drain");
        let drained: usize = ss.drain().map(|s| s.len()).sum();
        assert_eq!(drained, n, "{label}: total drained");
        assert_eq!(
            ss.len(),
            0,
            "{label}: store must be empty after a full drain"
        );
        assert_eq!(
            ss.shard_sizes().sum::<usize>(),
            0,
            "{label}: shard_sizes after drain"
        );
        assert_eq!(
            ss.iter().map(|s| s.len()).sum::<usize>(),
            0,
            "{label}: re-iteration after drain"
        );

        // A partial drain (consume one shard, drop the iterator) must leave len()
        // matching the data still readable, even when the split path removed a
        // whole bucket at once.
        let mut ss = build()?;
        {
            let mut it = ss.drain();
            let _ = it.next();
        }
        let reported = ss.len();
        let actual: usize = ss.iter().map(|s| s.len()).sum();
        assert_eq!(
            reported, actual,
            "{label}: len must match readable data after partial drain"
        );
        Ok(())
    }

    #[test]
    fn test_drain_lifecycle_updates_counts() -> anyhow::Result<()> {
        // (buckets_high_bits, shard_high_bits, max_shard_high_bits)
        let cases = [
            (4u32, 4u32, 4u32), // shards == buckets, coarsen 1
            (4, 4, 8),          // shards == buckets, coarsen > 1
            (8, 4, 8),          // aggregate buckets into shards
            (8, 4, 4),          // aggregate, several buckets per fine shard
            (2, 4, 8),          // split buckets into shards
            (2, 4, 4),          // split, coarsen 1
        ];
        let n = 2000usize;
        for (b, s, m) in cases {
            let online = format!("online b={b} s={s} m={m}");
            _check_drain(
                &online,
                || {
                    let mut st = new_online::<[u64; 2], u64>(b, m, None)?;
                    let mut rand = SmallRng::seed_from_u64(0);
                    for _ in 0..n {
                        st.try_push(SigVal {
                            sig: [rand.random(), rand.random()],
                            val: rand.random(),
                        })?;
                    }
                    st.into_shard_store(s)
                },
                n,
            )?;
            let offline = format!("offline b={b} s={s} m={m}");
            _check_drain(
                &offline,
                || {
                    let mut st = new_offline::<[u64; 2], u64>(b, m, None)?;
                    let mut rand = SmallRng::seed_from_u64(0);
                    for _ in 0..n {
                        st.try_push(SigVal {
                            sig: [rand.random(), rand.random()],
                            val: rand.random(),
                        })?;
                    }
                    st.into_shard_store(s)
                },
                n,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod filtered_lazy_tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    use std::sync::atomic::{AtomicUsize, Ordering};

    type SV = SigVal<[u64; 2], u8>;

    /// An inner iterator that counts how many shards have been pulled.
    struct Counting {
        inner: std::vec::IntoIter<Arc<Vec<SV>>>,
        counter: Arc<AtomicUsize>,
    }
    impl Iterator for Counting {
        type Item = Arc<Vec<SV>>;
        fn next(&mut self) -> Option<Self::Item> {
            let next = self.inner.next();
            if next.is_some() {
                self.counter.fetch_add(1, Ordering::Relaxed);
            }
            next
        }
    }

    /// A minimal in-memory [`ShardStore`] with separate counters for the
    /// `iter` and `drain` paths.
    struct Fake {
        shards: Vec<Arc<Vec<SV>>>,
        iter_pulls: Arc<AtomicUsize>,
        drain_pulls: Arc<AtomicUsize>,
    }
    impl ShardStore<[u64; 2], u8> for Fake {
        fn shard_sizes(&self) -> Box<dyn Iterator<Item = usize> + '_> {
            let sizes: Vec<usize> = self.shards.iter().map(|s| s.len()).collect();
            Box::new(sizes.into_iter())
        }
        fn set_shard_high_bits(&mut self, _new_bits: u32) {}
        fn max_shard_high_bits(&self) -> u32 {
            0
        }
        fn iter(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SV>>> + Send + Sync + '_> {
            Box::new(Counting {
                inner: self.shards.clone().into_iter(),
                counter: self.iter_pulls.clone(),
            })
        }
        fn drain(&mut self) -> Box<dyn Iterator<Item = Arc<Vec<SV>>> + Send + Sync + '_> {
            Box::new(Counting {
                inner: std::mem::take(&mut self.shards).into_iter(),
                counter: self.drain_pulls.clone(),
            })
        }
    }

    #[test]
    fn filtered_iter_is_lazy_and_drain_uses_inner_drain() {
        let sv = |s: u64, v: u8| SigVal {
            sig: [s, 0],
            val: v,
        };
        let shards = vec![Arc::new(vec![sv(1, 0), sv(2, 1)]), Arc::new(vec![sv(3, 0)])];
        let iter_pulls = Arc::new(AtomicUsize::new(0));
        let drain_pulls = Arc::new(AtomicUsize::new(0));
        let mut fake = Fake {
            shards,
            iter_pulls: iter_pulls.clone(),
            drain_pulls: drain_pulls.clone(),
        };
        let mut fs = FilteredShardStore::new(&mut fake, 0, |sv: &SV| sv.val == 0, vec![1, 1]);

        {
            let mut it = fs.iter();
            // Building the iterator must not pull any inner shard.
            assert_eq!(iter_pulls.load(Ordering::Relaxed), 0);
            let first = it.next().unwrap();
            // Exactly one inner shard was pulled, filtered to val == 0.
            assert_eq!(iter_pulls.load(Ordering::Relaxed), 1);
            assert_eq!(first.len(), 1);
            assert_eq!(first[0].sig, [1, 0]);
        }

        // drain is lazy too and goes through the inner drain path.
        let mut drain_it = fs.drain();
        assert_eq!(drain_pulls.load(Ordering::Relaxed), 0);
        drain_it.next().unwrap();
        assert_eq!(drain_pulls.load(Ordering::Relaxed), 1);
        let rest: Vec<_> = drain_it.collect();
        assert_eq!(drain_pulls.load(Ordering::Relaxed), 2);
        // The iter path was not used by drain.
        assert_eq!(iter_pulls.load(Ordering::Relaxed), 1);
        assert_eq!(rest.len(), 1);
    }

    fn _check_filtered_drain<SS: ShardStore<[u64; 2], u64>>(
        label: &str,
        build: impl Fn() -> anyhow::Result<SS>,
        shard_bits: u32,
        pred: fn(&SigVal<[u64; 2], u64>) -> bool,
    ) -> anyhow::Result<()> {
        // A full drain empties the filtered view.
        let mut inner = build()?;
        inner.set_shard_high_bits(shard_bits);
        let counts: Vec<usize> = inner
            .iter()
            .map(|sh| sh.iter().filter(|sv| pred(sv)).count())
            .collect();
        let total: usize = counts.iter().sum();
        let mut fs = FilteredShardStore::new(&mut inner, shard_bits, pred, counts);
        assert_eq!(fs.len(), total, "{label}: filtered len before drain");
        let drained: usize = fs.drain().map(|s| s.len()).sum();
        assert_eq!(drained, total, "{label}: filtered drained count");
        assert_eq!(
            fs.len(),
            0,
            "{label}: filtered store empty after full drain"
        );
        drop(fs);

        // Partial drain then drop: the split path removes a whole inner bucket
        // (several shards) at once, so the wrapper's reported length must be
        // reconciled against the inner store's drained shard sizes rather than
        // its stale cached filtered counts.
        let mut inner = build()?;
        inner.set_shard_high_bits(shard_bits);
        let counts: Vec<usize> = inner
            .iter()
            .map(|sh| sh.iter().filter(|sv| pred(sv)).count())
            .collect();
        let mut fs = FilteredShardStore::new(&mut inner, shard_bits, pred, counts);
        {
            let mut it = fs.drain();
            let _ = it.next();
        }
        let reported = fs.len();
        let actual: usize = fs.iter().map(|s| s.len()).sum();
        assert_eq!(
            reported, actual,
            "{label}: filtered len must match readable data after partial drain"
        );
        Ok(())
    }

    #[test]
    fn test_filtered_drain_lifecycle() -> anyhow::Result<()> {
        // Inner bucket_high_bits (2) < filtered shard_high_bits (4): draining one
        // wrapped shard removes a whole inner bucket = several shards.
        let n = 2000usize;
        let pred: fn(&SigVal<[u64; 2], u64>) -> bool = |sv| sv.val % 2 == 0;
        let build_online = || -> anyhow::Result<_> {
            let mut st = new_online::<[u64; 2], u64>(2, 8, None)?;
            let mut rand = SmallRng::seed_from_u64(1);
            for _ in 0..n {
                st.try_push(SigVal {
                    sig: [rand.random(), rand.random()],
                    val: rand.random(),
                })?;
            }
            st.into_shard_store(4)
        };
        let build_offline = || -> anyhow::Result<_> {
            let mut st = new_offline::<[u64; 2], u64>(2, 8, None)?;
            let mut rand = SmallRng::seed_from_u64(1);
            for _ in 0..n {
                st.try_push(SigVal {
                    sig: [rand.random(), rand.random()],
                    val: rand.random(),
                })?;
            }
            st.into_shard_store(4)
        };
        _check_filtered_drain("online", build_online, 4, pred)?;
        _check_filtered_drain("offline", build_offline, 4, pred)?;
        Ok(())
    }
}
