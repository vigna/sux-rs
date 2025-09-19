/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Compressed string storage by rear-coded prefix omission.

use std::borrow::Borrow;

use crate::dict::elias_fano::{EfSeq, EliasFanoBuilder};
use crate::traits::{IndexedDict, IndexedSeq, IntoIteratorFrom, Types};
use lender::for_;
use lender::{ExactSizeLender, IntoLender, Lender, Lending};
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
/// strings sorted in ascending order.
///
/// The encoding is done in blocks of `k` strings: in each block the first
/// string is encoded without compression, wheres the other strings are encoded
/// with the common prefix removed.
///
/// Rear-coded lists can be iterated upon using either an
/// [`Iterator`](RearCodedList::iter) or a [`Lender`](RearCodedList::lend).
/// In the first case there will be an allocation at each iteration, whereas in
/// second case a single buffer will be reused.
///
/// To build a [`RearCodedList`] you use a [`RearCodedListBuilder`].
///
/// # Examples
///
/// ```rust
/// use sux::traits::IndexedSeq;
/// use sux::dict::RearCodedListBuilder;
/// let mut rclb = RearCodedListBuilder::new(4);
///
/// rclb.push("aa");
/// rclb.push("aab");
/// rclb.push("abc");
/// rclb.push("abdd");
/// rclb.push("abde");
/// rclb.push("abdf");
///
/// let rcl = rclb.build();
/// assert_eq!(rcl.len(), 6);
/// assert_eq!(rcl.get(0), "aa");
/// assert_eq!(rcl.get(1), "aab");
/// assert_eq!(rcl.get(2), "abc");
/// ```

#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RearCodedList<D: AsRef<[u8]> = Box<[u8]>, P: IndexedSeq = Box<[usize]>> {
    /// The number of strings in a block; this value trades off compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// Whether the strings are sorted.
    is_sorted: bool,
    /// The encoded strings, `\0`-terminated.
    data: D,
    /// The pointer to the starting string of each block.
    pointers: P,
}

impl<D: AsRef<[u8]>, P: IndexedSeq> RearCodedList<D, P>
where
    for<'a> P: IndexedSeq<Output<'a> = usize>,
{
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
    pub fn iter_from(&self, from: usize) -> Iter<'_, D, P> {
        Iter {
            iter: Lend::new_from(self, from),
        }
    }

    /// Returns an [`Iterator`] over the strings.
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, D, P> {
        self.iter_from(0)
    }

    /// Returns a [`Lender`] over the strings starting from the given position.
    #[inline(always)]
    pub fn lend_from(&self, from: usize) -> Lend<'_, D, P> {
        Lend::new_from(self, from)
    }

    /// Returns a [`Lender`] over the strings.
    #[inline(always)]
    pub fn lend(&self) -> Lend<'_, D, P> {
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

        let start = self.pointers.get(block);
        let data = &self.data.as_ref()[start..];

        // decode the first string in the block
        let mut data = strcpy(data, result);

        for _ in 0..offset {
            // get how much data to throw away
            let (len, tmp) = decode_int(data);
            // throw away the data
            result.resize(result.len() - len, 0);
            // copy the new suffix
            let tmp = strcpy(tmp, result);
            data = tmp;
        }
    }

    fn index_of_unsorted(&self, value: impl Borrow<<Self as Types>::Input>) -> Option<usize> {
        let key = value.borrow().as_bytes();
        let mut iter = self.into_lender().enumerate();
        while let Some((idx, string)) = iter.next() {
            if matches!(
                strcmp_rust(key, string.as_bytes()),
                core::cmp::Ordering::Equal
            ) {
                return Some(idx);
            }
        }
        None
    }

    fn index_of_sorted(&self, value: impl Borrow<<Self as Types>::Input>) -> Option<usize> {
        let string = value.borrow().as_bytes();
        // first to a binary search on the blocks to find the block
        let block_idx = self.pointers.binary_search_by(|block_ptr| {
            strcmp(string, &self.data.as_ref()[block_ptr..]).reverse()
        });

        if let Ok(block_idx) = block_idx {
            return Some(block_idx * self.k);
        }

        let mut block_idx = block_idx.unwrap_err();
        if block_idx == 0 || block_idx > self.pointers.len() {
            // the string is before the first block
            return None;
        }
        block_idx -= 1;
        // finish by a linear search on the block
        let mut result = Vec::with_capacity(128);
        let start = self.pointers.get(block_idx);
        let data = &self.data.as_ref()[start..];

        // decode the first string in the block
        let mut data = strcpy(data, &mut result);
        let in_block = (self.k - 1).min(self.len - block_idx * self.k - 1);
        for idx in 0..in_block {
            // get how much data to throw away
            let (len, tmp) = decode_int(data);
            let lcp = result.len() - len;
            // throw away the data
            result.resize(lcp, 0);
            // copy the new suffix
            let tmp = strcpy(tmp, &mut result);
            data = tmp;

            // TODO!: this can be optimized to avoid the copy
            match strcmp_rust(string, &result) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Equal => return Some(block_idx * self.k + idx + 1),
                core::cmp::Ordering::Greater => return None,
            }
        }
        None
    }
}

impl<D: AsRef<[u8]>, P: IndexedSeq> Types for RearCodedList<D, P>
where
    for<'a> P: IndexedSeq<Output<'a> = usize>,
{
    type Output<'a> = String;
    type Input = str;
}

impl<D: AsRef<[u8]>, P: IndexedSeq> IndexedSeq for RearCodedList<D, P>
where
    for<'a> P: IndexedSeq<Output<'a> = usize>,
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

impl<D: AsRef<[u8]>, P: IndexedSeq> IndexedDict for RearCodedList<D, P>
where
    for<'a> P: IndexedSeq<Output<'a> = usize>,
{
    /// If the strings in the list are sorted this is done with a binary search,
    /// otherwise it is done with a linear search.
    #[inline(always)]
    fn contains(&self, value: impl Borrow<Self::Input>) -> bool {
        self.index_of(value).is_some()
    }

    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize> {
        if self.is_sorted {
            self.index_of_sorted(value)
        } else {
            self.index_of_unsorted(value)
        }
    }
}

impl<'a, D: AsRef<[u8]>, P: IndexedSeq> IntoLender for &'a RearCodedList<D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    type Lender = Lend<'a, D, P>;
    #[inline(always)]
    fn into_lender(self) -> Lend<'a, D, P> {
        Lend::new(self)
    }
}

/// Sequential [`Iterator`] over the strings.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct Iter<'a, D: AsRef<[u8]>, P: IndexedSeq>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    iter: Lend<'a, D, P>,
}

impl<D: AsRef<[u8]>, P: IndexedSeq> std::iter::ExactSizeIterator for Iter<'_, D, P>
where
    for<'a> P: IndexedSeq<Output<'a> = usize>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<D: AsRef<[u8]>, P: IndexedSeq> std::iter::Iterator for Iter<'_, D, P>
where
    for<'a> P: IndexedSeq<Output<'a> = usize>,
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

impl<'a, D: AsRef<[u8]>, P: IndexedSeq> IntoIterator for &'a RearCodedList<D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    type Item = String;
    type IntoIter = Iter<'a, D, P>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, D: AsRef<[u8]>, P: IndexedSeq> IntoIteratorFrom for &'a RearCodedList<D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    type IntoIterFrom = Iter<'a, D, P>;
    #[inline(always)]
    fn into_iter_from(self, from: usize) -> Self::IntoIter {
        self.iter_from(from)
    }
}

/// Sequential [`Lender`] over the strings.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct Lend<'a, D: AsRef<[u8]>, P: IndexedSeq>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    rca: &'a RearCodedList<D, P>,
    buffer: Vec<u8>,
    data: &'a [u8],
    index: usize,
}

impl<'a, D: AsRef<[u8]>, P: IndexedSeq> Lend<'a, D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    pub fn new(rca: &'a RearCodedList<D, P>) -> Self {
        Self {
            rca,
            buffer: Vec::with_capacity(128),
            data: rca.data.as_ref(),
            index: 0,
        }
    }

    pub fn new_from(rca: &'a RearCodedList<D, P>, from: usize) -> Self {
        let block = from / rca.k;
        let offset = from % rca.k;

        let start = rca.pointers.get(block);
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

impl<'a, D: AsRef<[u8]>, P: IndexedSeq> Lending<'a> for Lend<'_, D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    type Lend = &'a str;
}

impl<D: AsRef<[u8]>, P: IndexedSeq> Lender for Lend<'_, D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    #[inline]
    /// A next that returns a reference to the inner buffer containing the string.
    /// This is useful to avoid allocating a new string for every query if you
    /// don't need to keep the string around.
    fn next(&mut self) -> Option<&'_ str> {
        if self.index >= self.rca.len() {
            return None;
        }

        if self.index % self.rca.k == 0 {
            // just copy the data
            self.buffer.clear();
            self.data = strcpy(self.data, &mut self.buffer);
        } else {
            let (len, tmp) = decode_int(self.data);
            self.buffer.resize(self.buffer.len() - len, 0);
            self.data = strcpy(tmp, &mut self.buffer);
        }
        self.index += 1;

        Some(unsafe { std::str::from_utf8_unchecked(&self.buffer) })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<D: AsRef<[u8]>, P: IndexedSeq> ExactSizeLender for Lend<'_, D, P>
where
    for<'b> P: IndexedSeq<Output<'b> = usize>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.rca.len() - self.index
    }
}

/// Workaround for [a compiler
/// bug](https://github.com/rust-lang/rust/issues/87755#issuecomment-2564811303)
pub trait BuiltPointers
where
    for<'a> Self: IndexedSeq<Output<'a> = usize, Input = usize>,
{
}

impl<P> BuiltPointers for P where for<'a> P: IndexedSeq<Output<'a> = usize, Input = usize> {}

/// Structure that can append `usize` and yield a [`IndexedSeq`]
///
/// # Safety
///
/// Values in the built `IndexedSeq` must match the input in the same order.
pub unsafe trait PointersBuilder {
    type Built: BuiltPointers;

    fn push(&mut self, pointer: usize);
    fn len(&self) -> usize;
    fn build(self) -> Self::Built;
}

unsafe impl PointersBuilder for Vec<usize> {
    type Built = Box<[usize]>;

    fn push(&mut self, pointer: usize) {
        Vec::push(self, pointer)
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn build(self) -> Self::Built {
        self.into()
    }
}

unsafe impl PointersBuilder for EliasFanoBuilder {
    type Built = EfSeq;

    fn push(&mut self, pointer: usize) {
        EliasFanoBuilder::push(self, pointer)
    }

    fn len(&self) -> usize {
        self.count
    }

    fn build(self) -> Self::Built {
        EliasFanoBuilder::build_with_seq(self)
    }
}

/// Builder for a rear-coded list.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct RearCodedListBuilder<P: PointersBuilder = Vec<usize>> {
    /// The number of strings in a block; this value trades compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// Whether the strings are sorted.
    is_sorted: bool,
    /// The encoded strings, `\0`-terminated.
    data: Vec<u8>,
    /// The pointer to the starting string of each block.
    pointers: P,
    last_pointer: Option<usize>,
    /// Statistics of the encoded data.
    stats: Stats,
    /// Cache of the last encoded string for incremental encoding.
    last_str: Vec<u8>,
}

/// Copies a string until the first `\0` from `data` to `result` and return the
/// remaining data.
#[inline(always)]
fn strcpy<'a>(mut data: &'a [u8], result: &mut Vec<u8>) -> &'a [u8] {
    loop {
        let c = data[0];
        data = &data[1..];
        if c == 0 {
            break;
        }
        result.push(c);
    }
    data
}

#[inline(always)]
/// Like strcmp, but `string` is a Rust string and data is a `\0`-terminated string.
fn strcmp(string: &[u8], data: &[u8]) -> core::cmp::Ordering {
    for (i, c) in string.iter().enumerate() {
        let ord = c.cmp(&data[i]);
        if ord != core::cmp::Ordering::Equal {
            return ord;
        }
    }

    if data[string.len()] == 0 {
        core::cmp::Ordering::Equal
    } else {
        core::cmp::Ordering::Less
    }
}

#[inline(always)]
/// Like strcmp, but both string are Rust strings.
fn strcmp_rust(string: &[u8], other: &[u8]) -> core::cmp::Ordering {
    for (i, c) in string.iter().enumerate() {
        match other.get(i).unwrap_or(&0).cmp(c) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    // string has an implicit final \0
    other.len().cmp(&string.len())
}

impl RearCodedListBuilder<Vec<usize>> {
    /// Creates a builder for a rear-coded list with a block size of `k`.
    pub fn new(k: usize) -> Self {
        Self::with_pointer_builder(k, Vec::new())
    }
}

impl<P: PointersBuilder> RearCodedListBuilder<P> {
    /// Creates a builder for a rear-coded list with a block size of `k`
    /// using a custom structure to store pointers, such as [`EfSeq`].
    pub fn with_pointer_builder(k: usize, pointers_builder: P) -> Self {
        Self {
            data: Vec::with_capacity(1024),
            last_str: Vec::with_capacity(1024),
            pointers: pointers_builder,
            last_pointer: None,
            len: 0,
            is_sorted: true,
            k,
            stats: Stats::default(),
        }
    }

    /// Builds the rear-coded list.
    pub fn build(self) -> RearCodedList<Box<[u8]>, P::Built> {
        RearCodedList {
            data: self.data.into(),
            pointers: self.pointers.build(),
            len: self.len,
            is_sorted: self.is_sorted,
            k: self.k,
        }
    }

    /// Returns the number of strings in the list.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Appends a string to the end of the list.
    pub fn push(&mut self, string: impl AsRef<str>) {
        let string = string.as_ref();
        // update stats
        self.stats.max_str_len = self.stats.max_str_len.max(string.len());
        self.stats.sum_str_len += string.len();

        let (lcp, order) = longest_common_prefix(&self.last_str, string.as_bytes());

        if order == core::cmp::Ordering::Greater {
            self.is_sorted = false;
        }

        // at every multiple of k we just encode the string as is
        let to_encode = if self.len % self.k == 0 {
            // compute the size in bytes of the previous block
            let last_pointer = self.last_pointer.unwrap_or(0);
            let block_bytes = self.data.len() - last_pointer;
            // update stats
            self.stats.max_block_bytes = self.stats.max_block_bytes.max(block_bytes);
            self.stats.sum_block_bytes += block_bytes;
            // save a pointer to the start of the string
            self.pointers.push(self.data.len());
            self.last_pointer = Some(self.data.len());

            // compute the redundancy
            let rear_length = self.last_str.len() - lcp;
            if self.len != 0 {
                self.stats.redundancy += lcp as isize;
                self.stats.redundancy -= encode_int_len(rear_length) as isize;
            }
            // just encode the whole string
            string.as_bytes()
        } else {
            // update the stats
            self.stats.max_lcp = self.stats.max_lcp.max(lcp);
            self.stats.sum_lcp += lcp;
            // encode the len of the bytes in data
            let rear_length = self.last_str.len() - lcp;
            let prev_len = self.data.len();
            encode_int(rear_length, &mut self.data);
            // update stats
            self.stats.code_bytes += self.data.len() - prev_len;
            // return the delta suffix
            &string.as_bytes()[lcp..]
        };
        // Write the data to the buffer
        self.data.extend_from_slice(to_encode);
        // push the \0 terminator
        self.data.push(0);
        self.stats.suffixes_bytes += to_encode.len() + 1;

        // put the string as last_str for the next iteration
        self.last_str.clear();
        self.last_str.extend_from_slice(string.as_bytes());
        self.len += 1;
    }

    /// Appends all the strings from a [`Lender`] to the end of the list.
    ///
    /// We prefer to implement extension via a [`Lender`] instead of an
    /// [`Iterator`] to avoid the need to allocate a new string for every string
    /// in the list. This is particularly useful when building large lists
    /// from files using, for example, a
    /// [`RewindableIoLender`](crate::utils::lenders::RewindableIoLender).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lender::*;
    /// use sux::dict::RearCodedListBuilder;
    /// let mut rclb = RearCodedListBuilder::new(4);
    /// let words = vec!["aa", "aab", "abc", "abdd", "abde", "abdf"];
    /// // We need the map because s has type &&str
    /// rclb.extend(words.iter().map(|s| *s).into_lender());
    /// let rcl = rclb.build();
    ///
    /// let mut rclb = RearCodedListBuilder::new(4);
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
}

#[inline(always)]
/// Computes the longest common prefix between two strings as bytes.
fn longest_common_prefix(a: &[u8], b: &[u8]) -> (usize, core::cmp::Ordering) {
    let min_len = a.len().min(b.len());
    // normal lcp computation
    let mut i = 0;
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
    let mut max = 1 << 7;
    while value >= max {
        len += 1;
        value -= max;
        max <<= 7;
    }
    len
}

const UPPER_BOUND_1: usize = 128;
const UPPER_BOUND_2: usize = 128_usize.pow(2) + UPPER_BOUND_1;
const UPPER_BOUND_3: usize = 128_usize.pow(3) + UPPER_BOUND_2;
const UPPER_BOUND_4: usize = 128_usize.pow(4) + UPPER_BOUND_3;
const UPPER_BOUND_5: usize = 128_usize.pow(5) + UPPER_BOUND_4;
const UPPER_BOUND_6: usize = 128_usize.pow(6) + UPPER_BOUND_5;
const UPPER_BOUND_7: usize = 128_usize.pow(7) + UPPER_BOUND_6;
const UPPER_BOUND_8: usize = 128_usize.pow(8) + UPPER_BOUND_7;

/// VByte encode an integer
#[inline(always)]
fn encode_int(mut value: usize, data: &mut Vec<u8>) {
    if value < UPPER_BOUND_1 {
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_2 {
        value -= UPPER_BOUND_1;
        debug_assert!((value >> 8) < (1 << 6));
        data.push(0x80 | (value >> 8) as u8);
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_3 {
        value -= UPPER_BOUND_2;
        debug_assert!((value >> 16) < (1 << 5));
        data.push(0xC0 | (value >> 16) as u8);
        data.push((value >> 8) as u8);
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_4 {
        value -= UPPER_BOUND_3;
        debug_assert!((value >> 24) < (1 << 4));
        data.push(0xE0 | (value >> 24) as u8);
        data.push((value >> 16) as u8);
        data.push((value >> 8) as u8);
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_5 {
        value -= UPPER_BOUND_4;
        debug_assert!((value >> 32) < (1 << 3));
        data.push(0xF0 | (value >> 32) as u8);
        data.push((value >> 24) as u8);
        data.push((value >> 16) as u8);
        data.push((value >> 8) as u8);
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_6 {
        value -= UPPER_BOUND_5;
        debug_assert!((value >> 40) < (1 << 2));
        data.push(0xF8 | (value >> 40) as u8);
        data.push((value >> 32) as u8);
        data.push((value >> 24) as u8);
        data.push((value >> 16) as u8);
        data.push((value >> 8) as u8);
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_7 {
        value -= UPPER_BOUND_6;
        debug_assert!((value >> 48) < (1 << 1));
        data.push(0xFC | (value >> 48) as u8);
        data.push((value >> 40) as u8);
        data.push((value >> 32) as u8);
        data.push((value >> 24) as u8);
        data.push((value >> 16) as u8);
        data.push((value >> 8) as u8);
        data.push(value as u8);
        return;
    }
    if value < UPPER_BOUND_8 {
        value -= UPPER_BOUND_7;
        data.push(0xFE);
        data.push((value >> 48) as u8);
        data.push((value >> 40) as u8);
        data.push((value >> 32) as u8);
        data.push((value >> 24) as u8);
        data.push((value >> 16) as u8);
        data.push((value >> 8) as u8);
        data.push(value as u8);
        return;
    }

    data.push(0xFF);
    data.push((value >> 56) as u8);
    data.push((value >> 48) as u8);
    data.push((value >> 40) as u8);
    data.push((value >> 32) as u8);
    data.push((value >> 24) as u8);
    data.push((value >> 16) as u8);
    data.push((value >> 8) as u8);
    data.push(value as u8);
}

#[inline(always)]
fn decode_int(data: &[u8]) -> (usize, &[u8]) {
    let x = data[0];
    if x < 0x80 {
        return (x as usize, &data[1..]);
    }
    if x < 0xC0 {
        let x = ((((x & !0xC0) as usize) << 8) | data[1] as usize) + UPPER_BOUND_1;
        return (x, &data[2..]);
    }
    if x < 0xE0 {
        let x = ((((x & !0xE0) as usize) << 16) | ((data[1] as usize) << 8) | data[2] as usize)
            + UPPER_BOUND_2;
        return (x, &data[3..]);
    }
    if x < 0xF0 {
        let x = ((((x & !0xF0) as usize) << 24)
            | ((data[1] as usize) << 16)
            | ((data[2] as usize) << 8)
            | data[3] as usize)
            + UPPER_BOUND_3;
        return (x, &data[4..]);
    }
    if x < 0xF8 {
        let x = ((((x & !0xF8) as usize) << 32)
            | ((data[1] as usize) << 24)
            | ((data[2] as usize) << 16)
            | ((data[3] as usize) << 8)
            | data[4] as usize)
            + UPPER_BOUND_4;
        return (x, &data[5..]);
    }
    if x < 0xFC {
        let x = ((((x & !0xFC) as usize) << 40)
            | ((data[1] as usize) << 32)
            | ((data[2] as usize) << 24)
            | ((data[3] as usize) << 16)
            | ((data[4] as usize) << 8)
            | data[5] as usize)
            + UPPER_BOUND_5;
        return (x, &data[6..]);
    }
    if x < 0xFE {
        let x = ((((x & !0xFE) as usize) << 48)
            | ((data[1] as usize) << 40)
            | ((data[2] as usize) << 32)
            | ((data[3] as usize) << 24)
            | ((data[4] as usize) << 16)
            | ((data[5] as usize) << 8)
            | data[6] as usize)
            + UPPER_BOUND_6;
        return (x, &data[7..]);
    }
    if x < 0xFF {
        let x = (((data[1] as usize) << 48)
            | ((data[2] as usize) << 40)
            | ((data[3] as usize) << 32)
            | ((data[4] as usize) << 24)
            | ((data[5] as usize) << 16)
            | ((data[6] as usize) << 8)
            | data[7] as usize)
            + UPPER_BOUND_7;
        return (x, &data[8..]);
    }

    let x = ((data[1] as usize) << 56)
        | ((data[2] as usize) << 48)
        | ((data[3] as usize) << 40)
        | ((data[4] as usize) << 32)
        | ((data[5] as usize) << 24)
        | ((data[6] as usize) << 16)
        | ((data[7] as usize) << 8)
        | data[8] as usize;
    (x, &data[9..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strcmp() {
        assert_eq!(strcmp(b"abcd", b"abcd\0"), core::cmp::Ordering::Equal);
        assert_eq!(strcmp(b"abcd", b"abbd\0"), core::cmp::Ordering::Greater);
        assert_eq!(strcmp(b"abcd", b"abdd\0"), core::cmp::Ordering::Less);

        assert_eq!(strcmp(b"a", b"b\0"), core::cmp::Ordering::Less);
        assert_eq!(strcmp(b"b", b"a\0"), core::cmp::Ordering::Greater);
        assert_eq!(strcmp(b"abc", b"abc\0"), core::cmp::Ordering::Equal);
        assert_eq!(strcmp(b"abc", b"abc\0abc"), core::cmp::Ordering::Equal);
        assert_eq!(strcmp(b"abc", b"abc\0ab"), core::cmp::Ordering::Equal);
        assert_eq!(strcmp(b"abc", b"ab\0"), core::cmp::Ordering::Greater);
        assert_eq!(strcmp(b"abc", b"ab\0abc"), core::cmp::Ordering::Greater);
        assert_eq!(strcmp(b"abc", b"ab\0ab"), core::cmp::Ordering::Greater);
        assert_eq!(strcmp(b"abc", b"ab\0"), core::cmp::Ordering::Greater);
        assert_eq!(strcmp(b"a", b"ab\0"), core::cmp::Ordering::Less);
        assert_eq!(strcmp(b"ab", b"ab\0"), core::cmp::Ordering::Equal);
    }

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
        let mut builder = RearCodedListBuilder::new(4);
        builder.push("a");
        builder.push("b");
        builder.push("c");
        builder.push("d");
        let rcl = builder.build();
        read_into_lender::<&RearCodedList>(&rcl);
    }
}
