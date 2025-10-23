/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Compressed string storage by rear-coded prefix omission.

use std::borrow::Borrow;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;

use crate::traits::{IndexedDict, IndexedSeq, IntoIteratorFrom, Types};
use crate::utils::RewindableIoLender;
use epserde::prelude::SerIter;
use epserde::ser::{Serialize, WriteNoStd};
use lender::for_;
use lender::{ExactSizeLender, IntoLender, Lender, Lending};
use log::info;
use mem_dbg::*;

#[derive(Debug, Clone, MemDbg, MemSize, Default)]
/// Statistics of the encoded data.
struct Stats {
    /// Maximum block size in bytes.
    pub max_block_bytes: usize,
    /// The total sum of the block size in bytes.
    pub sum_block_bytes: usize,

    /// Maximum shared prefix in bytes.
    pub max_lcp: usize,
    /// The total sum of the shared prefix in bytes.
    pub sum_lcp: usize,

    /// Maximum string length in bytes.
    pub max_str_len: usize,
    /// The total sum of the string lengths in bytes.
    pub sum_str_len: usize,

    /// The number of bytes used to store the rear lengths in data.
    pub code_bytes: usize,
    /// The number of bytes used to store the suffixes in data.
    pub suffixes_bytes: usize,

    /// The bytes wasted writing without compression the first string in block.
    pub redundancy: isize,
}

/// Immutable lists of strings compressed by rear-coded prefix omission.
///
/// Prefix omission compresses a list of strings omitting the common prefixes of
/// consecutive strings. To do so, it stores the length of what remains after
/// the common prefix (hence, rear coding). It is usually applied to lists
/// of strings sorted in ascending order. The strings can contain arbitrary data,
/// including `\0` bytes.
///
/// The encoding is done in blocks of `k` strings: in each block the first
/// string is encoded without compression, whereas the other strings are encoded
/// with the common prefix removed.
///
/// Besides the standard access by means of the [`IndexedSeq`] trait, this
/// structure also implements the [`get_in_place`](RearCodedList::get_in_place)
/// method, which allows to write the string directly into a user-provided
/// buffer, avoiding allocations and UTF-8 validity checks.
///
/// Rear-coded lists can be iterated upon using either an
/// [`Iterator`](RearCodedList::iter) or a [`Lender`](RearCodedList::lend).
/// In the first case there will be an allocation at each iteration, whereas in
/// second case a single buffer will be reused.
///
/// There are two versions of the structure, depending on a const boolean
/// parameter `SORTED`: if it is true, the strings must be sorted in ascending
/// order (this is the usual case for obtaining good compression), and an
/// implementation of [`IndexedDict`] will be available that finds the position
/// of strings in the list by binary search; if false, the strings can be in
/// arbitrary order, but no dictionary operations will be available (this
/// configuration is mainly useful for storing large lists of strings with quick
/// deserialization).
///
/// To build a [`RearCodedList`] you use a [`RearCodedListBuilder`], which
/// has a `SORTED` boolean parameter.
///
/// # Examples
///
/// ```rust
/// use sux::traits::{IndexedSeq, IndexedDict};
/// use sux::dict::RearCodedListBuilder;
/// let mut rclb = RearCodedListBuilder::<true>::new(4);
///
/// rclb.push("aa");
/// rclb.push("aab");
/// rclb.push("abc");
/// rclb.push("abdd");
/// rclb.push("abde\0f");
/// rclb.push("abdf");
///
/// let rcl = rclb.build();
/// assert_eq!(rcl.len(), 6);
/// assert_eq!(rcl.get(0), "aa");
/// assert_eq!(rcl.get(1), "aab");
/// assert_eq!(rcl.get(2), "abc");
/// assert_eq!(rcl.index_of("abde\0f"), Some(4));
/// assert_eq!(rcl.index_of("foo"), None);
/// ```
///
/// # Format
///
/// The rear-coded list keeps an array of pointers to the beginning of each block
/// in data. Due do the omission of prefixes, we can only start decoding from
/// the beginning of a block as we write the full string there.
///
/// The `data` portion contains strings in the following format:
/// - if the index of the string is a multiple of `k`, we write: `<len><bytes>`
/// - otherwise, we write: `<suffix_len><rear_len><suffix_bytes>`
///
/// where `<len>`, `<suffix_len>`, and `<rear_len>` are VByte-encoded integers
/// and `<bytes>` and `<suffix_bytes>` are the actual bytes of the string or suffix.
///
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RearCodedList<D = Box<[u8]>, P = Box<[usize]>, const SORTED: bool = true> {
    /// The number of strings in a block; this value trades off compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// The encoded strings, `\0`-terminated.
    data: D,
    /// The pointer to the starting string of each block.
    pointers: P,
}

pub fn serialize<'a, 'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool>(
    k: usize,
    mut lender: L,
    mut writer: impl WriteNoStd,
) -> anyhow::Result<usize> {
    let mut len = 0;
    let mut byte_len = 0;
    // First pass: count the number of strings and the byte length
    let mut builder = RearCodedListBuilder::<SORTED>::new(k);
    info!("Counting strings...");
    while let Some(s) = lender.next() {
        match s {
            Err(e) => return Err(e.into()),
            Ok(s) => {
                builder.push(s.borrow());
                len += 1;
                byte_len += builder.data.len();
                builder.data.truncate(0);
            }
        }
    }
    info!("Counted {} strings, compressed in {} bytes", len, byte_len);

    lender = lender.rewind().map_err(Into::into)?;

    let iter_state = RefCell::new(IterState::<_, _, SORTED> {
        builder: RearCodedListBuilder::<SORTED>::new(k),
        iter: lender,
        _marker: PhantomData,
    });

    let rear_coded_list = RearCodedList::<_, _, SORTED> {
        k,
        len,
        data: SerIter::new(BytesIter {
            iter_state: &iter_state,
            byte_len,
            pos: 0,
        }),
        pointers: SerIter::new(PointersIter {
            iter_state: &iter_state,
            pos: 0,
        }),
    };

    info!("Serializing...");
    let written = unsafe { rear_coded_list.serialize(&mut writer)? };
    info!("Completed.");
    Ok(written)
}

pub fn store<'a, 'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool>(
    k: usize,
    lender: L,
    filename: impl AsRef<std::path::Path>,
) -> anyhow::Result<usize> {
    let dst_file =
        std::fs::File::create(filename.as_ref()).expect("Cannot create destination file");
    let mut buf_writer = std::io::BufWriter::new(dst_file);
    serialize::<B, L, SORTED>(k, lender, &mut buf_writer)
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> RearCodedList<D, P, SORTED> {
    /// Returns the number of strings.
    ///
    /// This method is equivalent to [`IndexedSeq::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline]
    pub fn len(&self) -> usize {
        IndexedSeq::len(self)
    }

    /// Returns an [`Iterator`] over the strings starting from the given position.
    #[inline(always)]
    pub fn iter_from(&self, from: usize) -> Iter<'_, D, P, SORTED> {
        Iter {
            iter: Lend::new_from(self, from),
        }
    }

    /// Returns an [`Iterator`] over the strings.
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, D, P, SORTED> {
        self.iter_from(0)
    }

    /// Returns a [`Lender`] over the strings starting from the given position.
    #[inline(always)]
    pub fn lend_from(&self, from: usize) -> Lend<'_, D, P, SORTED> {
        Lend::new_from(self, from)
    }

    /// Returns a [`Lender`] over the strings.
    #[inline(always)]
    pub fn lend(&self) -> Lend<'_, D, P, SORTED> {
        self.lend_from(0)
    }

    /// Writes the index-th string to `result` as bytes. This is useful to avoid
    /// allocating a new string for every query and skipping the UTF-8 validity
    /// check.
    #[inline]
    pub fn get_in_place(&self, index: usize, result: &mut Vec<u8>) {
        result.clear();
        let block = index / self.k;
        let offset = index % self.k;

        let start = self.pointers.as_ref()[block];
        let mut data = &self.data.as_ref()[start..];

        // declare these vars so the decoding looks cleaner
        let (mut rear_length, mut len);
        // decode the first string in the block
        (len, data) = decode_int(data);
        result.extend_from_slice(&data[..len]);
        data = &data[len..];

        for _ in 0..offset {
            // get how much data to throw away
            (len, data) = decode_int(data);
            (rear_length, data) = decode_int(data);
            // throw away the data
            let lcp = result.len() - rear_length;
            result.truncate(lcp);
            // copy the new suffix
            result.extend_from_slice(&data[..len]);
            data = &data[len..];
        }
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> Types for RearCodedList<D, P, SORTED> {
    type Output<'a> = String;
    type Input = str;
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> IndexedSeq
    for RearCodedList<D, P, SORTED>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        let mut result = Vec::with_capacity(128);
        self.get_in_place(index, &mut result);
        String::from_utf8(result).unwrap()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>> IndexedDict for RearCodedList<D, P, true> {
    /// Uses a binary search on the blocks followed by a linear search within the block.
    #[inline(always)]
    fn contains(&self, value: impl Borrow<Self::Input>) -> bool {
        self.index_of(value).is_some()
    }

    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize> {
        let string = value.borrow().as_bytes();
        // first to a binary search on the blocks to find the block
        let block_idx = self.pointers.as_ref().binary_search_by(|block_ptr| {
            // do a quick in-place decode of the explicit string at the beginning of the block
            let data = &self.data.as_ref()[*block_ptr..];
            let (len, tmp) = decode_int(data);
            memcmp(string, &tmp[..len]).reverse()
        });

        if let Ok(block_idx) = block_idx {
            return Some(block_idx * self.k);
        }

        let mut block_idx = block_idx.unwrap_err();
        if block_idx == 0 || block_idx > self.pointers.as_ref().len() {
            // the string is before the first block
            return None;
        }
        block_idx -= 1;
        // finish by a linear search on the block
        let mut result = Vec::with_capacity(128);
        let start = self.pointers.as_ref()[block_idx];
        let mut data = &self.data.as_ref()[start..];

        let (mut to_copy, mut rear_length);
        // decode the first string in the block
        (to_copy, data) = decode_int(data);
        result.extend_from_slice(&data[..to_copy]);
        data = &data[to_copy..];

        let in_block = (self.k - 1).min(self.len - block_idx * self.k - 1);
        for idx in 0..in_block {
            // get how much data to throw away
            (to_copy, data) = decode_int(data);
            (rear_length, data) = decode_int(data);
            let lcp = result.len() - rear_length;
            // throw away the data
            result.truncate(lcp);
            // copy the new suffix
            result.extend_from_slice(&data[..to_copy]);
            data = &data[to_copy..];
            // TODO!: this can be optimized to avoid the copy
            match memcmp_rust(string, &result) {
                core::cmp::Ordering::Greater => {}
                core::cmp::Ordering::Equal => return Some(block_idx * self.k + idx + 1),
                core::cmp::Ordering::Less => return None,
            }
        }
        None
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> IntoLender
    for &'a RearCodedList<D, P, SORTED>
{
    type Lender = Lend<'a, D, P, SORTED>;
    #[inline(always)]
    fn into_lender(self) -> Lend<'a, D, P, SORTED> {
        Lend::new(self)
    }
}

/// Sequential [`Iterator`] over the strings.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct Iter<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> {
    iter: Lend<'a, D, P, SORTED>,
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> std::iter::ExactSizeIterator
    for Iter<'_, D, P, SORTED>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> std::iter::Iterator
    for Iter<'_, D, P, SORTED>
{
    type Item = String;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|v| unsafe { String::from_utf8_unchecked(Vec::from(v)) })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> IntoIterator
    for &'a RearCodedList<D, P, SORTED>
{
    type Item = String;
    type IntoIter = Iter<'a, D, P, SORTED>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> IntoIteratorFrom
    for &'a RearCodedList<D, P, SORTED>
{
    type IntoIterFrom = Iter<'a, D, P, SORTED>;
    #[inline(always)]
    fn into_iter_from(self, from: usize) -> Self::IntoIter {
        self.iter_from(from)
    }
}

/// Sequential [`Lender`] over the strings.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct Lend<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> {
    rca: &'a RearCodedList<D, P, SORTED>,
    buffer: Vec<u8>,
    data: &'a [u8],
    index: usize,
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> Lend<'a, D, P, SORTED> {
    /// Creates a new lender over the rear-coded list.
    pub fn new(rca: &'a RearCodedList<D, P, SORTED>) -> Self {
        Self {
            rca,
            buffer: Vec::with_capacity(128),
            data: rca.data.as_ref(),
            index: 0,
        }
    }

    /// Creates a new lender over the rear-coded list starting from the given
    /// position.
    pub fn new_from(rca: &'a RearCodedList<D, P, SORTED>, from: usize) -> Self {
        let block = from / rca.k;
        let offset = from % rca.k;

        let start = rca.pointers.as_ref()[block];
        let mut res = Lend {
            rca,
            index: block * rca.k,
            data: &rca.data.as_ref()[start..],
            buffer: Vec::with_capacity(128),
        };
        for _ in 0..offset {
            res.next();
        }
        res
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> Lending<'a>
    for Lend<'_, D, P, SORTED>
{
    type Lend = &'a str;
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> Lender for Lend<'_, D, P, SORTED> {
    #[inline]
    /// A next that returns a reference to the inner buffer containing the string.
    /// This is useful to avoid allocating a new string for every query if you
    /// don't need to keep the string around.
    fn next(&mut self) -> Option<&'_ str> {
        if self.index >= self.rca.len() {
            return None;
        }

        // figure out how much of the suffix we have to read
        let (to_copy, mut tmp) = decode_int(self.data);
        // figure out how much of the buffer we have to keep
        let lcp = if self.index % self.rca.k == 0 {
            0
        } else {
            let rear_length;
            (rear_length, tmp) = decode_int(tmp);
            self.buffer.len() - rear_length
        };
        // truncate the buffer to keep only the relevant part
        self.buffer.truncate(lcp);
        // copy the new suffix
        self.buffer.extend_from_slice(&tmp[..to_copy]);
        self.data = &tmp[to_copy..];
        self.index += 1;

        Some(unsafe { std::str::from_utf8_unchecked(&self.buffer) })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> ExactSizeLender
    for Lend<'_, D, P, SORTED>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.rca.len() - self.index
    }
}

/// Builder for a rear-coded list.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct RearCodedListBuilder<const SORTED: bool = true> {
    /// The number of strings in a block; this value trades compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// The encoded strings, `\0`-terminated.
    data: Vec<u8>,
    /// The total number of bytes written; this can be difference
    /// than the length of data in the low-memory construction,
    /// as we truncate data after each push
    written_bytes: usize,
    /// The pointer to the starting string of each block.
    pointers: Vec<usize>,
    /// Statistics of the encoded data.
    stats: Stats,
    /// Cache of the last encoded string for incremental encoding.
    last_str: Vec<u8>,
}

#[inline(always)]
/// Like memcmp
fn memcmp(string: &[u8], data: &[u8]) -> core::cmp::Ordering {
    for (a, b) in string.iter().zip(data.iter()) {
        let ord = a.cmp(b);
        if ord != core::cmp::Ordering::Equal {
            return ord;
        }
    }
    string.len().cmp(&data.len())
}

#[inline(always)]
/// Like memcmp, but both string are Rust strings.
fn memcmp_rust(string: &[u8], other: &[u8]) -> core::cmp::Ordering {
    for (a, b) in string.iter().zip(other.iter()) {
        let ord = a.cmp(b);
        if ord != core::cmp::Ordering::Equal {
            return ord;
        }
    }
    string.len().cmp(&other.len())
}

impl<const SORTED: bool> RearCodedListBuilder<SORTED> {
    /// Creates a builder for a rear-coded list with a block size of `k`.
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::with_capacity(1024),
            last_str: Vec::with_capacity(1024),
            pointers: Vec::new(),
            len: 0,
            written_bytes: 0,
            k,
            stats: Stats::default(),
        }
    }

    /// Builds the rear-coded list.
    pub fn build(self) -> RearCodedList<Box<[u8]>, Box<[usize]>, SORTED> {
        RearCodedList {
            data: self.data.into(),
            pointers: self.pointers.into(),
            len: self.len,
            k: self.k,
        }
    }

    /// Returns the number of strings in the list.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Prints in a human-readable format the statistics of the
    /// strings currently in the builder.
    pub fn print_stats(&self) {
        println!(
            "{:>20}: {:>10}",
            "max_block_bytes", self.stats.max_block_bytes
        );
        println!(
            "{:>20}: {:>10.3}",
            "avg_block_bytes",
            self.stats.sum_block_bytes as f64 / self.len as f64
        );

        println!("{:>20}: {:>10}", "max_lcp", self.stats.max_lcp);
        println!(
            "{:>20}: {:>10.3}",
            "avg_lcp",
            self.stats.sum_lcp as f64 / self.len as f64
        );

        println!("{:>20}: {:>10}", "max_str_len", self.stats.max_str_len);
        println!(
            "{:>20}: {:>10.3}",
            "avg_str_len",
            self.stats.sum_str_len as f64 / self.len as f64
        );

        let ptr_size: usize = self.pointers.len() * core::mem::size_of::<usize>();

        fn human(key: &str, x: usize) {
            const UOM: &[&str] = &["B", "KB", "MB", "GB", "TB"];
            let mut y = x as f64;
            let mut uom_idx = 0;
            while y > 1000.0 {
                uom_idx += 1;
                y /= 1000.0;
            }
            println!("{:>20}:{:>10.3}{}{:>20} ", key, y, UOM[uom_idx], x);
        }

        let total_size = ptr_size + self.data.len() + core::mem::size_of::<Self>();
        human("data_bytes", self.data.len());
        human("codes_bytes", self.stats.code_bytes);
        human("suffixes_bytes", self.stats.suffixes_bytes);
        human("ptrs_bytes", ptr_size);
        human("uncompressed_size", self.stats.sum_str_len);
        human("total_size", total_size);

        human(
            "optimal_size",
            (self.data.len() as isize - self.stats.redundancy) as usize,
        );
        human("redundancy", self.stats.redundancy as usize);
        let overhead = self.stats.redundancy + ptr_size as isize;
        println!(
            "overhead_ratio: {:>10}",
            overhead as f64 / (overhead + self.data.len() as isize) as f64
        );
        println!(
            "no_overhead_compression_ratio: {:.3}",
            (self.data.len() as isize - self.stats.redundancy) as f64
                / self.stats.sum_str_len as f64
        );

        println!(
            "compression_ratio: {:.3}",
            (ptr_size + self.data.len()) as f64 / self.stats.sum_str_len as f64
        );
    }

    fn push_impl(&mut self, string: &str, lcp: usize) {
        // compute how much to remove from the previous string
        let rear_length = self.last_str.len() - lcp;
        // and how long is this string without the lcp
        let suffix_len = string.len() - lcp;
        let length_before = self.data.len();

        // at every multiple of k we just encode the string as is
        let to_encode = if self.len % self.k == 0 {
            // compute the size in bytes of the previous block
            let last_ptr = self.pointers.last().copied().unwrap_or(0);
            let block_bytes = self.data.len() - last_ptr;
            // update stats
            self.stats.max_block_bytes = self.stats.max_block_bytes.max(block_bytes);
            self.stats.sum_block_bytes += block_bytes;
            // save a pointer to the start of the string
            assert_eq!(self.written_bytes, self.data.len());
            self.pointers.push(self.written_bytes);

            let prev_len = self.data.len();
            // encode the length of the string
            encode_int(string.len(), &mut self.data);
            // update stats
            self.stats.code_bytes += self.data.len() - prev_len;

            // compute the redundancy
            if self.len != 0 {
                self.stats.redundancy += lcp as isize;
                self.stats.redundancy += encode_int_len(string.len()) as isize;
                self.stats.redundancy -= encode_int_len(rear_length) as isize;
                self.stats.redundancy -= encode_int_len(suffix_len) as isize;
            }
            // just encode the whole string
            string.as_bytes()
        } else {
            // update the stats
            self.stats.max_lcp = self.stats.max_lcp.max(lcp);
            self.stats.sum_lcp += lcp;
            // encode the len of the bytes in data
            let prev_len = self.data.len();
            encode_int(suffix_len, &mut self.data);
            encode_int(rear_length, &mut self.data);
            // update stats
            self.stats.code_bytes += self.data.len() - prev_len;
            // return the delta suffix
            &string.as_bytes()[lcp..]
        };
        // Write the data to the buffer
        self.data.extend_from_slice(to_encode);
        self.written_bytes += self.data.len() - length_before;
        assert!(self.written_bytes == self.data.len());
        self.stats.suffixes_bytes += to_encode.len();

        // put the string as last_str for the next iteration
        self.last_str.truncate(lcp);
        self.last_str.extend_from_slice(&string.as_bytes()[lcp..]);
        self.len += 1;
    }
}

impl<const SORTED: bool> RearCodedListBuilder<SORTED> {
    /// Appends a string to the end of the list.
    ///
    /// # Panics
    ///
    /// Panics if `SORTED` is `true` and the string is not greater than or equal to the last string added.
    pub fn push(&mut self, string: impl Borrow<str>) {
        let string = string.borrow();
        // update stats
        self.stats.max_str_len = self.stats.max_str_len.max(string.len());
        self.stats.sum_str_len += string.len();

        let (lcp, order) = longest_common_prefix(&self.last_str, string.as_bytes());

        if SORTED && order == core::cmp::Ordering::Greater {
            panic!(
                "Strings must be sorted in ascending order when building a sorted RearCodedList. Got '{}' after '{}'",
                string,
                String::from_utf8_lossy(&self.last_str)
            );
        }

        self.push_impl(string, lcp);
    }

    /// Appends all the strings from a [`Lender`] to the end of the list.
    ///
    /// We prefer to implement extension via a [`Lender`] instead of an
    /// [`Iterator`] to avoid the need to allocate a new string for every string
    /// in the list. This is particularly useful when building large lists
    /// from files using, for example, a
    /// [`RewindableIoLender`](crate::utils::lenders::RewindableIoLender).
    ///
    /// # Panics
    ///
    /// Panics if `SORTED` is `true` and any string is not greater than or equal to the previous string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lender::*;
    /// use sux::dict::RearCodedListBuilder;
    /// let mut rclb = RearCodedListBuilder::<true>::new(4);
    /// let words = vec!["aa", "aab", "abc", "abdd", "abde", "abdf"];
    /// // We need the map because s has type &&str
    /// rclb.extend(words.iter().map(|s| *s).into_lender());
    /// let rcl = rclb.build();
    ///
    /// let mut rclb = RearCodedListBuilder::<false>::new(4);
    /// let words = vec!["aa".to_string(), "aab".to_string(), "abc".to_string(),
    ///     "abdd".to_string(), "abde".to_string(), "abdf".to_string()];
    /// // We need the map to turn String into &str
    /// rclb.extend(words.iter().map(|s| s.as_str()).into_lender());
    /// let rcl = rclb.build();
    /// ```
    pub fn extend<S: Borrow<str>, L: IntoLender>(&mut self, into_lender: L)
    where
        L::Lender: for<'lend> Lending<'lend, Lend = S>,
    {
        for_!(string in into_lender {
            self.push(string.borrow());
        });
    }
}

#[inline(always)]
/// Computes the longest common prefix between two strings as bytes.
fn longest_common_prefix(a: &[u8], b: &[u8]) -> (usize, core::cmp::Ordering) {
    let min_len = a.len().min(b.len());
    // normal lcp computation
    let mut i = 0;
    // we purposely don't do early stopping so it can be vectorized
    while i < min_len && a[i] == b[i] {
        i += 1;
    }
    // TODO!: try to optimize with vpcmpeqb pextrb and leading count ones
    if i < min_len {
        (i, a[i].cmp(&b[i]))
    } else {
        (i, a.len().cmp(&b.len()))
    }
}

/// Computes the length in bytes of value encoded as VByte
#[inline(always)]
fn encode_int_len(mut value: usize) -> usize {
    let mut len = 1;
    loop {
        value >>= 7;
        if value == 0 {
            return len;
        }
        value -= 1;
        len += 1;
    }
}

/// VByte encode an integer
#[inline(always)]
fn encode_int(mut value: usize, data: &mut Vec<u8>) {
    let mut buf = [0u8; 10];
    let mut pos = buf.len() - 1;
    buf[pos] = (value & 0x7F) as u8;
    value >>= 7;
    while value != 0 {
        value -= 1;
        pos -= 1;
        buf[pos] = 0x80 | (value & 0x7F) as u8;
        value >>= 7;
    }
    for &byte in &buf[pos..] {
        data.push(byte);
    }
}

#[inline(always)]
fn decode_int(mut data: &[u8]) -> (usize, &[u8]) {
    let mut byte = data[0];
    data = &data[1..];
    let mut value = (byte & 0x7F) as usize;
    while (byte >> 7) != 0 {
        value += 1;
        byte = data[0];
        data = &data[1..];
        value = (value << 7) | (byte & 0x7F) as usize;
    }
    (value, data)
}

struct IterState<B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool> {
    builder: RearCodedListBuilder<SORTED>,
    iter: L,
    _marker: PhantomData<B>,
}

struct BytesIter<'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool> {
    iter_state: &'b RefCell<IterState<B, L, SORTED>>,
    byte_len: usize,
    pos: usize,
}

impl<'a, 'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool> Iterator
    for BytesIter<'b, B, L, SORTED>
{
    type Item = u8;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let state = self.iter_state.borrow_mut();
        if self.pos < state.builder.data.len() {
            let byte = state.builder.data[self.pos];
            self.pos += 1;
            self.byte_len -= 1;
            Some(byte)
        } else {
            // We need separate mutable borrows for iter and builder,
            // but we just have a RefMut to the whole state. So we use
            // RefMut::map_split to split the borrow.
            let (mut iter, mut builder) =
                RefMut::map_split(state, |state| (&mut state.iter, &mut state.builder));
            iter.next().and_then(|item| match item {
                Ok(s) => {
                    // Empty the builder data and refill it
                    builder.data.truncate(0);
                    builder.push(s.borrow());
                    let byte = builder.data[0];
                    self.pos = 1;
                    self.byte_len -= 1;
                    Some(byte)
                }
                Err(_e) => {
                    panic!("Error while serializing RearCodedList")
                }
            })
        }
    }
}

impl<'a, 'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool>
    ExactSizeIterator for BytesIter<'b, B, L, SORTED>
{
    fn len(&self) -> usize {
        self.byte_len
    }
}

struct PointersIter<'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool> {
    iter_state: &'b RefCell<IterState<B, L, SORTED>>,
    pos: usize,
}

impl<'a, 'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool> Iterator
    for PointersIter<'b, B, L, SORTED>
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.iter_state.borrow();
        if self.pos == state.builder.pointers.len() {
            None
        } else {
            let ptr = state.builder.pointers[self.pos];
            self.pos += 1;
            Some(ptr)
        }
    }
}

impl<'a, 'b, B: ?Sized + Borrow<str>, L: RewindableIoLender<B>, const SORTED: bool>
    ExactSizeIterator for PointersIter<'b, B, L, SORTED>
{
    fn len(&self) -> usize {
        let state = self.iter_state.borrow();
        state.builder.pointers.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use epserde::utils::AlignedCursor;

    use super::*;

    #[test]
    fn test_memcmp() {
        assert_eq!(memcmp(b"abcd", b"abcd"), core::cmp::Ordering::Equal);
        assert_eq!(memcmp(b"abcd", b"abbd"), core::cmp::Ordering::Greater);
        assert_eq!(memcmp(b"abcd", b"abdd"), core::cmp::Ordering::Less);

        assert_eq!(memcmp(b"a", b"b\0"), core::cmp::Ordering::Less);
        assert_eq!(memcmp(b"b", b"a\0"), core::cmp::Ordering::Greater);
        assert_eq!(memcmp(b"abc", b"abc\0"), core::cmp::Ordering::Less);
        assert_eq!(memcmp(b"abc", b"ab\0"), core::cmp::Ordering::Greater);
        assert_eq!(memcmp(b"ab\0", b"abc"), core::cmp::Ordering::Less);
    }

    const UPPER_BOUND_1: usize = 128;
    const UPPER_BOUND_2: usize = 128_usize.pow(2) + UPPER_BOUND_1;
    const UPPER_BOUND_3: usize = 128_usize.pow(3) + UPPER_BOUND_2;
    const UPPER_BOUND_4: usize = 128_usize.pow(4) + UPPER_BOUND_3;
    const UPPER_BOUND_5: usize = 128_usize.pow(5) + UPPER_BOUND_4;
    const UPPER_BOUND_6: usize = 128_usize.pow(6) + UPPER_BOUND_5;
    const UPPER_BOUND_7: usize = 128_usize.pow(7) + UPPER_BOUND_6;
    const UPPER_BOUND_8: usize = 128_usize.pow(8) + UPPER_BOUND_7;

    #[test]
    fn test_encode_decode_int() {
        let values = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            UPPER_BOUND_1 - 1,
            UPPER_BOUND_1,
            UPPER_BOUND_1 + 1,
            UPPER_BOUND_2 - 1,
            UPPER_BOUND_2,
            UPPER_BOUND_2 + 1,
            UPPER_BOUND_3 - 1,
            UPPER_BOUND_3,
            UPPER_BOUND_3 + 1,
            UPPER_BOUND_4 - 1,
            UPPER_BOUND_4,
            UPPER_BOUND_4 + 1,
            UPPER_BOUND_5 - 1,
            UPPER_BOUND_5,
            UPPER_BOUND_5 + 1,
            UPPER_BOUND_6 - 1,
            UPPER_BOUND_6,
            UPPER_BOUND_6 + 1,
            UPPER_BOUND_7 - 1,
            UPPER_BOUND_7,
            UPPER_BOUND_7 + 1,
            UPPER_BOUND_8 - 1,
            UPPER_BOUND_8,
            UPPER_BOUND_8 + 1,
        ];
        let mut buffer = Vec::with_capacity(128);

        for i in &values {
            encode_int(*i, &mut buffer);
        }

        let mut data = &buffer[..];
        for i in &values {
            let (j, tmp) = decode_int(data);
            assert_eq!(data.len() - tmp.len(), encode_int_len(*i));
            data = tmp;
            assert_eq!(*i, j);
        }
    }

    #[test]
    fn test_longest_common_prefix() {
        let str1 = b"absolutely";
        let str2 = b"absorption";
        assert_eq!(
            longest_common_prefix(str1, str2),
            (4, core::cmp::Ordering::Less),
        );
        assert_eq!(
            longest_common_prefix(str1, str1),
            (str1.len(), core::cmp::Ordering::Equal)
        );
        assert_eq!(
            longest_common_prefix(str2, str2),
            (str2.len(), core::cmp::Ordering::Equal)
        );
    }

    #[cfg(test)]
    fn read_into_lender<L: IntoLender>(into_lender: L) -> usize
    where
        for<'a> <L::Lender as Lending<'a>>::Lend: AsRef<str>,
    {
        let mut iter = into_lender.into_lender();
        let mut c = 0;
        while let Some(s) = iter.next() {
            c += s.as_ref().len();
        }

        c
    }

    #[test]
    fn test_into_lend() {
        let mut builder = RearCodedListBuilder::<true>::new(4);
        builder.push("a");
        builder.push("b");
        builder.push("c");
        builder.push("d");
        let rcl = builder.build();
        read_into_lender::<&RearCodedList<Box<[u8]>, Box<[usize]>, true>>(&rcl);
    }

    #[test]
    fn test_zero_bytes() {
        let strings = vec![
            "\0\0\0\0a",
            "\0\0\0b",
            "\0\0c",
            "\0d",
            "e",
            "f\0",
            "g\0\0",
            "h\0\0\0",
        ];
        let mut builder = RearCodedListBuilder::<true>::new(4);
        for s in &strings {
            builder.push(s.borrow());
        }
        let rcl = builder.build();
        for i in 0..rcl.len() {
            let s = rcl.get(i);
            assert_eq!(s, strings[i]);
        }
    }

    #[cfg(feature = "epserde")]
    #[test]
    fn test_ser() -> anyhow::Result<()> {
        use epserde::deser::Deserialize;

        use crate::utils::FromIntoIterator;

        let v = ["a", "ab", "ab", "abc", "b", "bb"];
        let mut cursor = AlignedCursor::<maligned::A16>::new();
        serialize::<_, _, true>(4, FromIntoIterator::from(v), &mut cursor)?;

        cursor.set_position(0);
        let deser = unsafe {
            RearCodedList::<Box<[u8]>, Box<[usize]>, true>::deserialize_full(&mut cursor)?
        };
        assert_eq!(deser.len(), 6);
        for (i, s) in deser.iter().enumerate() {
            assert_eq!(s, v[i]);
        }
        assert_eq!(deser.get(0), "a");
        assert_eq!(deser.get(1), "ab");
        assert_eq!(deser.get(2), "ab");
        assert_eq!(deser.get(3), "abc");
        assert_eq!(deser.get(4), "b");
        assert_eq!(deser.get(5), "bb");

        Ok(())
    }
}
