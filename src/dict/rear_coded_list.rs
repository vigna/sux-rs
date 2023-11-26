/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Immutable lists of strings compressed by prefix omission via rear coding.

*/

use crate::traits::IndexedDict;
use epserde::*;
use lender::{ExactSizeLender, IntoLender, Lender, Lending};
use mem_dbg::*;

#[derive(Debug, Clone, Default)]
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
    /// The total sum of the strings length in bytes.
    pub sum_str_len: usize,

    /// The number of bytes used to store the rear lengths in data.
    pub code_bytes: usize,
    /// The number of bytes used to store the suffixes in data.
    pub suffixes_bytes: usize,

    /// The bytes wasted writing without compression the first string in block.
    pub redundancy: isize,
}

/**

Immutable lists of strings compressed by prefix omission via rear coding.

Prefix omission compresses a list of strings omitting the common prefixes
of consecutive strings. To do so, it stores the length of what remains
after the common prefix (hence, rear coding). It is usually applied
to lists strings sorted in ascending order.

The encoding is done in blocks of `k` strings: in each block the first string is encoded
without compression, wheres the other strings are encoded with the common prefix
removed.

*/
#[derive(Debug, Clone, Epserde, MemDbg, MemSize)]
pub struct RearCodedList<D: AsRef<[u8]> = Vec<u8>, P: AsRef<[usize]> = Vec<usize>> {
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

pub struct RearCodedListBuilder {
    /// The number of strings in a block; this value trades off compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// Whether the strings are sorted.
    is_sorted: bool,
    /// The encoded strings, `\0`-terminated.
    data: Vec<u8>,
    /// The pointer to the starting string of each block.
    pointers: Vec<usize>,
    /// Statistics of the encoded data.
    stats: Stats,
    /// Cache of the last encoded string for incremental encoding.
    last_str: Vec<u8>,
}

/// Copy a string until the first `\0` from `data` to `result` and return the
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
/// strcmp but string is a Rust string and data is a `\0`-terminated string.
fn strcmp(string: &[u8], data: &[u8]) -> core::cmp::Ordering {
    for (i, c) in string.iter().enumerate() {
        match data[i].cmp(c) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    // string has an implicit final \0
    data[string.len()].cmp(&0)
}

#[inline(always)]
/// strcmp but both string are Rust strings.
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

impl RearCodedListBuilder {
    #[inline]
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::with_capacity(1024),
            last_str: Vec::with_capacity(1024),
            pointers: Vec::new(),
            len: 0,
            is_sorted: true,
            k,
            stats: Default::default(),
        }
    }

    #[inline]
    pub fn build(self) -> RearCodedList<Vec<u8>, Vec<usize>> {
        RearCodedList {
            data: self.data,
            pointers: self.pointers,
            len: self.len,
            is_sorted: self.is_sorted,
            k: self.k,
        }
    }

    /// Re-allocate the data to remove wasted capacity in the structure
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
        self.pointers.shrink_to_fit();
        self.last_str.shrink_to_fit();
    }

    #[inline]
    /// Encode and append a string to the end of the list.
    pub fn push<S: AsRef<str>>(&mut self, string: S) {
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
            let last_ptr = self.pointers.last().copied().unwrap_or(0);
            let block_bytes = self.data.len() - last_ptr;
            // update stats
            self.stats.max_block_bytes = self.stats.max_block_bytes.max(block_bytes);
            self.stats.sum_block_bytes += block_bytes;
            // save a pointer to the start of the string
            self.pointers.push(self.data.len());

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

    #[inline]
    /// Append all the strings from an iterator to the end of the list
    pub fn extend<S: AsRef<str>, I: std::iter::Iterator<Item = S>>(&mut self, iter: I) {
        for string in iter {
            self.push(string);
        }
    }

    /// Print in an human readable format the statistics of the RCL
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

impl<D: AsRef<[u8]>, P: AsRef<[usize]>> RearCodedList<D, P> {
    /// Write the index-th string to `result` as bytes. This is useful to avoid
    /// allocating a new string for every query and skipping the UTF-8 validity
    /// check.
    #[inline(always)]
    pub fn get_inplace(&self, index: usize, result: &mut Vec<u8>) {
        result.clear();
        let block = index / self.k;
        let offset = index % self.k;

        let start = self.pointers.as_ref()[block];
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

    fn contains_unsorted(&self, key: &<Self as IndexedDict>::Input) -> bool {
        let key = key.as_bytes();
        let mut iter = self.into_lender();
        while let Some(string) = iter.next() {
            if matches!(strcmp(key, string.as_bytes()), core::cmp::Ordering::Equal) {
                return true;
            }
        }
        false
    }

    fn contains_sorted(&self, string: &<Self as IndexedDict>::Input) -> bool {
        let string = string.as_bytes();
        // first to a binary search on the blocks to find the block
        let block_idx = self
            .pointers
            .as_ref()
            .binary_search_by(|block_ptr| strcmp(string, &self.data.as_ref()[*block_ptr..]));

        if block_idx.is_ok() {
            return true;
        }

        let mut block_idx = block_idx.unwrap_err();
        if block_idx == 0 || block_idx > self.pointers.as_ref().len() {
            // the string is before the first block
            return false;
        }
        block_idx -= 1;
        // finish by a linear search on the block
        let mut result = Vec::with_capacity(128);
        let start = self.pointers.as_ref()[block_idx];
        let data = &self.data.as_ref()[start..];

        // decode the first string in the block
        let mut data = strcpy(data, &mut result);
        let in_block = (self.k - 1).min(self.len - block_idx * self.k - 1);
        for _ in 0..in_block {
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
                core::cmp::Ordering::Equal => return true,
                core::cmp::Ordering::Greater => return false,
            }
        }
        false
    }
}

impl<'a, 'all, D: AsRef<[u8]>, P: AsRef<[usize]>> Lending<'all> for &'a RearCodedList<D, P> {
    type Lend = &'all str;
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> IntoLender for &'a RearCodedList<D, P> {
    type Lender = Iterator<'a, D, P>;
    #[inline(always)]
    fn into_lender(self) -> Iterator<'a, D, P> {
        Iterator::new(self)
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>> IndexedDict for RearCodedList<D, P> {
    type Output = String;
    type Input = str;

    unsafe fn get_unchecked(&self, index: usize) -> Self::Output {
        let mut result = Vec::with_capacity(128);
        self.get_inplace(index, &mut result);
        String::from_utf8(result).unwrap()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    /// Return whether the string is contained in the array.
    /// If the strings in the list are sorted this is done with a binary search,
    /// otherwise it is done with a linear search.
    #[inline]
    fn contains(&self, string: &Self::Input) -> bool {
        if self.is_sorted {
            self.contains_sorted(string)
        } else {
            self.contains_unsorted(string)
        }
    }
}

/// Sequential iterator over the strings.
pub struct Iterator<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> {
    rca: &'a RearCodedList<D, P>,
    buffer: Vec<u8>,
    data: &'a [u8],
    index: usize,
}

pub struct ValueIterator<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> {
    iter: Iterator<'a, D, P>,
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> std::iter::Iterator for ValueIterator<'a, D, P> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|v| unsafe { String::from_utf8_unchecked(Vec::from(v)) })
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> Iterator<'a, D, P> {
    pub fn new(rca: &'a RearCodedList<D, P>) -> Self {
        Self {
            rca,
            buffer: Vec::with_capacity(128),
            data: rca.data.as_ref(),
            index: 0,
        }
    }

    pub fn new_from(rca: &'a RearCodedList<D, P>, start_index: usize) -> Self {
        let block = start_index / rca.k;
        let offset = start_index % rca.k;

        let start = rca.pointers.as_ref()[block];
        let mut res = Iterator {
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

impl<'a, 'b, D: AsRef<[u8]>, P: AsRef<[usize]>> Lending<'a> for Iterator<'b, D, P> {
    type Lend = &'a str;
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> Lender for Iterator<'a, D, P> {
    #[inline]
    /// A next that returns a reference to the inner buffer containg the string.
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
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> ExactSizeLender for Iterator<'a, D, P> {
    fn len(&self) -> usize {
        self.rca.len() - self.index
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>> IntoIterator for &'a RearCodedList<D, P> {
    type Item = String;
    type IntoIter = ValueIterator<'a, D, P>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        ValueIterator {
            iter: Iterator::new(self),
        }
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>> RearCodedList<D, P> {
    pub fn into_iter_from(&self, from: usize) -> ValueIterator<'_, D, P> {
        ValueIterator {
            iter: Iterator::new_from(self, from),
        }
    }
}

#[inline(always)]
/// Compute the longest common prefix between two strings as bytes.
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

/// Compute the length in bytes of value encoded as VByte
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
        let x = (((x & !0xC0) as usize) << 8 | data[1] as usize) + UPPER_BOUND_1;
        return (x, &data[2..]);
    }
    if x < 0xE0 {
        let x = (((x & !0xE0) as usize) << 16 | (data[1] as usize) << 8 | data[2] as usize)
            + UPPER_BOUND_2;
        return (x, &data[3..]);
    }
    if x < 0xF0 {
        let x = (((x & !0xF0) as usize) << 24
            | (data[1] as usize) << 16
            | (data[2] as usize) << 8
            | data[3] as usize)
            + UPPER_BOUND_3;
        return (x, &data[4..]);
    }
    if x < 0xF8 {
        let x = (((x & !0xF8) as usize) << 32
            | (data[1] as usize) << 24
            | (data[2] as usize) << 16
            | (data[3] as usize) << 8
            | data[4] as usize)
            + UPPER_BOUND_4;
        return (x, &data[5..]);
    }
    if x < 0xFC {
        let x = (((x & !0xFC) as usize) << 40
            | (data[1] as usize) << 32
            | (data[2] as usize) << 24
            | (data[3] as usize) << 16
            | (data[4] as usize) << 8
            | data[5] as usize)
            + UPPER_BOUND_5;
        return (x, &data[6..]);
    }
    if x < 0xFE {
        let x = (((x & !0xFE) as usize) << 48
            | (data[1] as usize) << 40
            | (data[2] as usize) << 32
            | (data[3] as usize) << 24
            | (data[4] as usize) << 16
            | (data[5] as usize) << 8
            | data[6] as usize)
            + UPPER_BOUND_6;
        return (x, &data[7..]);
    }
    if x < 0xFF {
        let x = ((data[1] as usize) << 48
            | (data[2] as usize) << 40
            | (data[3] as usize) << 32
            | (data[4] as usize) << 24
            | (data[5] as usize) << 16
            | (data[6] as usize) << 8
            | data[7] as usize)
            + UPPER_BOUND_7;
        return (x, &data[8..]);
    }

    let x = (data[1] as usize) << 56
        | (data[2] as usize) << 48
        | (data[3] as usize) << 40
        | (data[4] as usize) << 32
        | (data[5] as usize) << 24
        | (data[6] as usize) << 16
        | (data[7] as usize) << 8
        | data[8] as usize;
    (x, &data[9..])
}

#[cfg(test)]
#[cfg_attr(test, test)]
fn test_encode_decode_int() {
    const MAX: usize = 1 << 20;
    const MIN: usize = 0;
    let mut buffer = Vec::with_capacity(128);

    for i in MIN..MAX {
        encode_int(i, &mut buffer);
    }

    let mut data = &buffer[..];
    for i in MIN..MAX {
        let (j, tmp) = decode_int(data);
        assert_eq!(data.len() - tmp.len(), encode_int_len(i));
        data = tmp;
        assert_eq!(i, j);
    }
}

#[cfg(test)]
#[cfg_attr(test, test)]
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

#[cfg(test)]
#[cfg_attr(test, test)]
fn test_into_lend() {
    let mut builder = RearCodedListBuilder::new(4);
    builder.push("a");
    builder.push("b");
    builder.push("c");
    builder.push("d");
    let rcl = builder.build();
    read_into_lender::<&RearCodedList>(&rcl);
}
