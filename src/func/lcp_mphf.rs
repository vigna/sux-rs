/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! LCP-based monotone minimal perfect hash function for strings.

use crate::bits::BitFieldVec;
use crate::func::VFunc;
use mem_dbg::*;

/// A monotone minimal perfect hash function for lexicographically sorted
/// strings, based on longest common prefixes (LCPs).
///
/// Given *n* strings in lexicographic order, the structure maps each string
/// to its rank (0 to *n* − 1). Querying a string not in the original set
/// returns an arbitrary value (same contract as [`VFunc`]).
///
/// The strings are divided into buckets of size 2^*k*. For each bucket,
/// the longest common byte prefix (LCP) shared by all strings in the
/// bucket (and the last string of the previous bucket) is computed. The
/// rank of a key is then reconstructed as `bucket * bucket_size + offset`,
/// where the bucket is determined by the LCP and the offset is stored
/// directly.
///
/// Internally, the structure contains two [`VFunc`]s:
/// - `offset_lcp_length`: maps each key (`str`) to a packed value
///   encoding the LCP length and the offset within the bucket;
/// - `lcp2bucket`: maps each LCP byte prefix (`[u8]`) to its bucket
///   index.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMinPerfHashFunc {
    /// Number of keys.
    n: usize,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
    /// Maps each key to `(lcp_length << log2_bucket_size) | offset`.
    offset_lcp_length: VFunc<str, usize, BitFieldVec<Box<[usize]>>>,
    /// Maps each LCP byte prefix to its bucket index.
    lcp2bucket: VFunc<[u8], usize, BitFieldVec<Box<[usize]>>>,
}

impl LcpMinPerfHashFunc {
    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary
    /// (same contract as [`VFunc::get`]).
    #[inline]
    pub fn get(&self, key: &str) -> usize {
        let packed = self.offset_lcp_length.get(key);
        let lcp_length = packed >> self.log2_bucket_size;
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        let bucket = self.lcp2bucket.get(&key.as_bytes()[..lcp_length]);
        (bucket << self.log2_bucket_size) + offset
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}
