/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! LCP-based monotone minimal perfect hash function for strings.

use crate::bits::BitFieldVec;
use crate::func::VBuilder;
use crate::func::VFunc;
use crate::utils::*;
use anyhow::{bail, Result};
use dsi_progress_logger::ProgressLog;
use lender::*;
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

/// Returns the length of the longest common byte prefix of two byte slices.
fn lcp_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Computes the log2 of bucket size from *n* using the LCP-MPHF formula.
fn log2_bucket_size(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let ln_n = (n as f64).ln();
    let theoretical = 1.0 + 1.11 * f64::ln(2.0) + ln_n - (1.0 + ln_n).ln();
    let theoretical = theoretical.ceil().max(1.0) as usize;
    theoretical.next_power_of_two().ilog2() as usize
}

#[cfg(feature = "rayon")]
impl LcpMinPerfHashFunc {
    /// Creates a new LCP-based monotone minimal perfect hash function.
    ///
    /// The keys must be provided in strictly increasing lexicographic
    /// order.
    ///
    /// # Arguments
    ///
    /// * `keys`: a lender yielding `&str` references in sorted order.
    ///
    /// * `n`: the number of keys.
    ///
    /// * `pl`: a progress logger.
    pub fn new(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend str>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if n == 0 {
            // Build trivial empty VFuncs.
            let empty_keys_str: Vec<&str> = vec![];
            let empty_keys_u8: Vec<&[u8]> = vec![];
            let empty_vals: Vec<usize> = vec![];

            let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>>::default()
                .try_build_func::<str, &str>(
                    FromSlice::new(&empty_keys_str),
                    FromSlice::new(&empty_vals),
                    pl,
                )?;
            let lcp2bucket = VBuilder::<_, BitFieldVec<Box<[usize]>>>::default()
                .try_build_func::<[u8], &[u8]>(
                    FromSlice::new(&empty_keys_u8),
                    FromSlice::new(&empty_vals),
                    pl,
                )?;
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offset_lcp_length,
                lcp2bucket,
            });
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        // -- First pass: compute LCPs --

        let mut lcp_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut bucket_first_strings: Vec<String> = Vec::with_capacity(num_buckets);

        let mut prev_key = String::new();
        let mut curr_lcp_len: usize = 0;
        let mut i = 0usize;

        while let Some(key) = keys.next()? {
            let key: &str = key;

            // Verify strictly increasing order.
            if i > 0 && key <= prev_key.as_str() {
                bail!(
                    "Keys are not in strictly increasing lexicographic \
                     order at position {i}: {prev_key:?} >= {key:?}"
                );
            }

            let offset = i & bucket_mask;

            if offset == 0 {
                // Start of a new bucket.
                if i > 0 {
                    // Finalize previous bucket.
                    lcp_lengths.push(curr_lcp_len);
                }
                bucket_first_strings.push(key.to_owned());
                // LCP with the last key of the previous bucket.
                // For bucket 0, prev_key is empty, so lcp_bytes returns 0.
                curr_lcp_len = lcp_bytes(key.as_bytes(), prev_key.as_bytes());
            } else {
                // Inside a bucket: track minimum LCP.
                curr_lcp_len =
                    curr_lcp_len.min(lcp_bytes(key.as_bytes(), prev_key.as_bytes()));
            }

            prev_key.clear();
            prev_key.push_str(key);
            i += 1;
        }

        assert_eq!(i, n, "Expected {n} keys but got {i}");

        // Finalize the last bucket.
        lcp_lengths.push(curr_lcp_len);

        assert_eq!(lcp_lengths.len(), num_buckets);

        // -- Build packed values from LCP lengths --

        let packed_values: Vec<usize> = (0..n)
            .map(|idx| {
                let bucket = idx >> log2_bs;
                let offset = idx & bucket_mask;
                (lcp_lengths[bucket] << log2_bs) | offset
            })
            .collect();

        // -- Build offset_lcp_length VFunc --

        let keys = keys.rewind()?;

        let offset_lcp_length = VBuilder::<_, BitFieldVec<Box<[usize]>>>::default()
            .expected_num_keys(n)
            .try_build_func::<str, str>(keys, FromSlice::new(&packed_values), pl)?;

        // -- Build lcp2bucket VFunc --

        let lcps: Vec<Vec<u8>> = (0..num_buckets)
            .map(|b| bucket_first_strings[b].as_bytes()[..lcp_lengths[b]].to_vec())
            .collect();
        let lcp_refs: Vec<&[u8]> = lcps.iter().map(|v| v.as_slice()).collect();
        let bucket_indices: Vec<usize> = (0..num_buckets).collect();

        let lcp2bucket = VBuilder::<_, BitFieldVec<Box<[usize]>>>::default()
            .expected_num_keys(num_buckets)
            .try_build_func::<[u8], &[u8]>(
                FromSlice::new(&lcp_refs),
                FromSlice::new(&bucket_indices),
                pl,
            )?;

        Ok(Self {
            n,
            log2_bucket_size: log2_bs,
            offset_lcp_length,
            lcp2bucket,
        })
    }
}
