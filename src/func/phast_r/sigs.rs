/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The signatures of the keys of a level, distributed in memory into parts
//! of consecutive buckets.

use super::builder::*;
use super::sweep::*;
use super::*;

/// The signatures of the keys of a level, distributed into parts (see
/// [`group`]).
///
/// Parts are made of a power of two of consecutive buckets. For each part
/// and for each *chunk* of the keys (one per thread), a *region* contains
/// the signatures of the keys of the chunk whose bucket is in the part, in
/// the order of the keys. Regions are sized before computing the
/// signatures, assuming that they are uniform, and they have some slack:
/// the few signatures that do not fit their region, if any, are kept
/// separately.
///
/// The signatures of a part are distributed into buckets only when a sweep
/// needs the part (see [`load`]): thus, besides the signatures, which take
/// eight bytes per key, the construction needs just a byte per key recording
/// the part of each key, which makes it possible to scan the signatures in
/// the order of the keys (see [`walk`]).
///
/// [`load`]: Self::load
/// [`walk`]: Self::walk
pub(super) struct Signatures {
    /// The regions, in part-major order (see `region_begin`).
    pub(super) sigs: Vec<u64>,
    /// For each part, and for each chunk, the start of the region of the
    /// part and the chunk in `sigs`, followed by the length of `sigs`.
    pub(super) region_begin: Vec<usize>,
    /// For each part, and for each chunk, the number of signatures in the
    /// region.
    pub(super) region_len: Vec<usize>,
    /// The part of each key.
    pub(super) parts: Vec<u8>,
    /// For each chunk, the signatures that did not fit their region, with
    /// their part, sorted by part (and thus, by key within a part).
    pub(super) extra: Vec<Vec<(u8, u64)>>,
    /// The number of keys of a chunk (except the last one).
    pub(super) chunk_len: usize,
    /// The number of chunks.
    pub(super) chunks: usize,
    /// The number of parts.
    pub(super) num_parts: usize,
    /// The part of a bucket is its index shifted right by this amount.
    pub(super) shift: u32,
}

/// The base-2 logarithm of the number of keys of the parts into which
/// [`group`] distributes signatures, if they are not more than
/// [`MAX_PARTS`] (otherwise, parts are larger): small parts fit the cache,
/// and they reduce the size of the buffers of the sweeps (tests use even
/// smaller parts, so that sweeps go through many parts).
#[cfg(not(test))]
const LOG2_PART_KEYS: u32 = 14;
#[cfg(test)]
const LOG2_PART_KEYS: u32 = 12;
/// The maximum number of parts of [`group`], and thus of regions written at
/// the same time by a thread: the lines of the cache and the entries of the
/// TLB used by so many sequential writes are few for any architecture
/// (also, the number of parts must fit a byte).
const MAX_PARTS: usize = 256;
/// The base-2 logarithm of the number of keys of the smaller parts into
/// which [`Signatures::load`] distributes a part before distributing them
/// into buckets: with their copy and the positions of their buckets, they
/// take about 150 KB, which fits the second-level cache of any
/// architecture.
#[cfg(not(test))]
const LOG2_CACHE_KEYS: u32 = 13;
#[cfg(test)]
const LOG2_CACHE_KEYS: u32 = 8;
/// The base-2 logarithm of the maximum number of smaller parts: beyond this
/// number (that is, with parts of more than 2<sup>22</sup> keys), smaller
/// parts do not fit the cache.
const MAX_LOG2_SUBPARTS: u32 = 9;
/// The number of standard deviations of slack in the size of a region: the
/// probability that a region overflows is about 10<sup>-9</sup>.
const SLACK: f64 = 6.0;

/// Computes the signatures of the keys of a level and distributes them into
/// parts (see [`Signatures`]); the signature of the key of index *i* in the
/// level is `sig(i)`.
///
/// Signatures need not be sorted: the thread processing a chunk of the keys
/// writes each signature at the end of the region of its part, and parts
/// (which are at most [`MAX_PARTS`], so a thread writes to a bounded
/// number of places at the same time) are distributed into buckets later,
/// a part at a time, when they are needed by a sweep.
pub(super) fn group(n: usize, sig: impl Fn(usize) -> u64 + Sync, g: &Geometry) -> Signatures {
    group_with(n, sig, g, LOG2_PART_KEYS, MAX_PARTS, SLACK)
}

/// Implements [`group`] with the given parameters in place of
/// [`LOG2_PART_KEYS`], [`MAX_PARTS`] and [`SLACK`].
fn group_with(
    n: usize,
    sig: impl Fn(usize) -> u64 + Sync,
    g: &Geometry,
    log2_part_keys: u32,
    max_parts: usize,
    slack: f64,
) -> Signatures {
    let nb = g.buckets;
    let log2_parts = (n >> log2_part_keys)
        .next_power_of_two()
        .ilog2()
        .min(max_parts.ilog2());
    // The part of a bucket is given by its index shifted right by this
    // amount
    let shift = (usize::BITS - (nb - 1).leading_zeros()).saturating_sub(log2_parts);
    let num_parts = ((nb - 1) >> shift) + 1;
    let part = |h: u64| g.bucket(h) >> shift;
    #[cfg(feature = "rayon")]
    let par = parallel(n);
    #[cfg(feature = "rayon")]
    let chunk_len = if par {
        n.div_ceil(rayon::current_num_threads())
    } else {
        n
    };
    #[cfg(not(feature = "rayon"))]
    let chunk_len = n;
    let chunk_len = chunk_len.max(1);
    let chunks = n.div_ceil(chunk_len);

    // The size of a region is the expected number of signatures of its
    // chunk falling in its part, plus some standard deviations
    let capacity = |chunk: usize| {
        let len = ((chunk + 1) * chunk_len).min(n) - chunk * chunk_len;
        let mean = len as f64 / num_parts as f64;
        (mean + slack * mean.sqrt()).ceil() as usize + 32
    };
    let mut region_begin = Vec::with_capacity(num_parts * chunks + 1);
    let mut end = 0;
    for _ in 0..num_parts {
        for chunk in 0..chunks {
            region_begin.push(end);
            end += capacity(chunk);
        }
    }
    region_begin.push(end);
    // Pages of regions that are never written cost nothing
    let mut sigs = vec![0u64; end];
    let mut parts = vec![0u8; n];
    // The regions of each chunk
    let mut regions: Vec<Vec<&mut [u64]>> =
        (0..chunks).map(|_| Vec::with_capacity(num_parts)).collect();
    let mut rest = &mut sigs[..];
    for p in 0..num_parts {
        for (chunk, regions) in regions.iter_mut().enumerate() {
            let (region, r) = rest.split_at_mut(
                region_begin[p * chunks + chunk + 1] - region_begin[p * chunks + chunk],
            );
            regions.push(region);
            rest = r;
        }
    }
    type Chunk<'a, 'b, 'c> = (usize, (&'a mut Vec<&'b mut [u64]>, &'c mut [u8]));
    let distribute = |(chunk, (regions, parts)): Chunk| -> (Vec<usize>, Vec<(u8, u64)>) {
        let mut len = vec![0usize; num_parts];
        let mut extra = vec![];
        for (k, parts) in parts.iter_mut().enumerate() {
            let h = sig(chunk * chunk_len + k);
            let p = part(h);
            *parts = p as u8;
            if let Some(x) = regions[p].get_mut(len[p]) {
                *x = h;
                len[p] += 1;
            } else {
                extra.push((p as u8, h));
            }
        }
        extra.sort_by_key(|x| x.0);
        (len, extra)
    };
    #[cfg(feature = "rayon")]
    let results: Vec<(Vec<usize>, Vec<(u8, u64)>)> = if par {
        use rayon::prelude::*;
        regions
            .par_iter_mut()
            .zip(parts.par_chunks_mut(chunk_len))
            .enumerate()
            .map(distribute)
            .collect()
    } else {
        regions
            .iter_mut()
            .zip(parts.chunks_mut(chunk_len))
            .enumerate()
            .map(distribute)
            .collect()
    };
    #[cfg(not(feature = "rayon"))]
    let results: Vec<(Vec<usize>, Vec<(u8, u64)>)> = regions
        .iter_mut()
        .zip(parts.chunks_mut(chunk_len))
        .enumerate()
        .map(distribute)
        .collect();
    drop(regions);
    let mut region_len = vec![0usize; num_parts * chunks];
    let mut extra = Vec::with_capacity(chunks);
    for (chunk, (len, chunk_extra)) in results.into_iter().enumerate() {
        for (p, &len) in len.iter().enumerate() {
            region_len[p * chunks + chunk] = len;
        }
        extra.push(chunk_extra);
    }
    Signatures {
        sigs,
        region_begin,
        region_len,
        parts,
        extra,
        chunk_len,
        chunks,
        num_parts,
        shift,
    }
}

impl Signatures {
    /// Returns the part of bucket `b`.
    #[inline(always)]
    pub(super) fn part(&self, b: usize) -> usize {
        b >> self.shift
    }

    /// Returns the first bucket of part `p`.
    #[inline(always)]
    pub(super) fn first_bucket(&self, p: usize) -> usize {
        p << self.shift
    }

    /// Returns the number of buckets of part `p`.
    pub(super) fn buckets(&self, p: usize, g: &Geometry) -> usize {
        (g.buckets - self.first_bucket(p)).min(1 << self.shift)
    }

    /// Returns the region of part `p` and chunk `chunk`.
    #[inline(always)]
    pub(super) fn region(&self, p: usize, chunk: usize) -> &[u64] {
        let i = p * self.chunks + chunk;
        &self.sigs[self.region_begin[i]..][..self.region_len[i]]
    }

    /// Returns the signatures of part `p` that did not fit their region, for
    /// each chunk.
    pub(super) fn extra(&self, p: usize) -> impl Iterator<Item = u64> + '_ {
        self.extra.iter().flat_map(move |extra| {
            let begin = extra.partition_point(|x| (x.0 as usize) < p);
            extra[begin..]
                .iter()
                .take_while(move |x| x.0 as usize == p)
                .map(|x| x.1)
        })
    }

    /// Distributes into buckets the signatures of part `p` whose bucket is
    /// in `range` (which must contain the buckets of the part only, but
    /// can be a proper subset of them), writing them to `sigs` and the
    /// position in `sigs` of the first signature of each bucket of the
    /// range, followed by the number of signatures, to `begin`; `temp` is a
    /// temporary buffer (see [`distribute`]).
    pub(super) fn load(
        &self,
        p: usize,
        g: &Geometry,
        range: std::ops::Range<usize>,
        sigs: &mut Vec<u64>,
        begin: &mut Vec<usize>,
        temp: &mut Vec<u64>,
    ) {
        let whole = range == (self.first_bucket(p)..self.first_bucket(p) + self.buckets(p, g));
        // The signatures to distribute, as a sequence of runs
        let extra: Vec<u64> = self.extra(p).collect();
        if whole {
            let mut source: Vec<&[u64]> = (0..self.chunks)
                .map(|chunk| self.region(p, chunk))
                .collect();
            source.push(&extra);
            distribute(&source, g, range, sigs, begin, temp);
        } else {
            // We gather the signatures of the range using the storage of
            // the temporary buffer, which we take back afterwards if it is
            // larger than the current one
            let mut filtered = std::mem::take(temp);
            filtered.clear();
            for chunk in 0..self.chunks {
                filtered.extend(
                    self.region(p, chunk)
                        .iter()
                        .filter(|&&h| range.contains(&g.bucket(h))),
                );
            }
            filtered.extend(extra.iter().filter(|&&h| range.contains(&g.bucket(h))));
            distribute(&[&filtered], g, range, sigs, begin, temp);
            if temp.capacity() < filtered.capacity() {
                *temp = filtered;
            }
        }
    }

    /// Calls `f` on the index and the signature of each key of the given
    /// chunk, in the order of the keys.
    pub(super) fn walk(&self, chunk: usize, mut f: impl FnMut(usize, u64)) {
        // For each part, the next signature of its region and the end of
        // the region
        let mut region: Vec<(usize, usize)> = (0..self.num_parts)
            .map(|p| {
                let i = p * self.chunks + chunk;
                (
                    self.region_begin[i],
                    self.region_begin[i] + self.region_len[i],
                )
            })
            .collect();
        // For each part, the next signature that did not fit the region
        let extra = &self.extra[chunk];
        let mut next_extra = vec![0usize; self.num_parts + 1];
        for x in extra {
            next_extra[x.0 as usize + 1] += 1;
        }
        for p in 0..self.num_parts {
            next_extra[p + 1] += next_extra[p];
        }
        let first = chunk * self.chunk_len;
        for (k, &p) in self.parts[first..].iter().take(self.chunk_len).enumerate() {
            let p = p as usize;
            let (next, end) = &mut region[p];
            let h = if *next < *end {
                *next += 1;
                self.sigs[*next - 1]
            } else {
                next_extra[p] += 1;
                extra[next_extra[p] - 1].1
            };
            f(first + k, h);
        }
    }
}

impl PartSource for Signatures {
    type Loader = ();

    #[inline(always)]
    fn log2_min_buckets(&self) -> u32 {
        self.shift
    }

    #[inline(always)]
    fn part(&self, b: usize) -> usize {
        Signatures::part(self, b)
    }

    #[inline(always)]
    fn first_bucket(&self, p: usize) -> usize {
        Signatures::first_bucket(self, p)
    }

    #[inline(always)]
    fn buckets(&self, p: usize, g: &Geometry) -> usize {
        Signatures::buckets(self, p, g)
    }

    #[inline(always)]
    fn load(
        &self,
        _loader: &mut (),
        p: usize,
        g: &Geometry,
        range: std::ops::Range<usize>,
        sigs: &mut Vec<u64>,
        begin: &mut Vec<usize>,
        temp: &mut Vec<u64>,
    ) {
        Signatures::load(self, p, g, range, sigs, begin, temp)
    }
}

/// Distributes into buckets a sequence of runs of signatures whose buckets
/// are in `range`, writing them to `sigs` and the position in `sigs` of the
/// first signature of each bucket of the range, followed by the number of
/// signatures, to `begin`; `temp` is a temporary buffer.
///
/// Signatures are distributed first into smaller parts that fit the cache,
/// and then each smaller part into its buckets.
pub(super) fn distribute(
    source: &[&[u64]],
    g: &Geometry,
    range: std::ops::Range<usize>,
    sigs: &mut Vec<u64>,
    begin: &mut Vec<usize>,
    temp: &mut Vec<u64>,
) {
    let (first_bucket, buckets) = (range.start, range.len());
    let len = source.iter().map(|run| run.len()).sum::<usize>();
    // Buffers are reused, so we avoid clearing them
    if sigs.len() < len {
        sigs.resize(len, 0);
    }
    begin.clear();
    begin.resize(buckets + 1, 0);
    let log2_subparts = (len >> LOG2_CACHE_KEYS)
        .next_power_of_two()
        .ilog2()
        .min(MAX_LOG2_SUBPARTS)
        .min(buckets.ilog2());
    if log2_subparts == 0 {
        into_buckets(
            source,
            &mut sigs[..len],
            &mut begin[..buckets],
            first_bucket,
            0,
            g,
        );
        begin[buckets] = len;
        return;
    }
    // The smaller parts are made of 2^sub_shift consecutive buckets,
    // starting from the first bucket of the range; as for regions, we
    // reserve for each of them its expected size plus some slack, and
    // we keep separately the signatures that do not fit
    let sub_shift = (buckets.ilog2() + 1).saturating_sub(log2_subparts + 1);
    let subparts = ((buckets - 1) >> sub_shift) + 1;
    let subpart = |h: u64| (g.bucket(h) - first_bucket) >> sub_shift;
    let mean = len as f64 / subparts as f64;
    let capacity = (mean + SLACK * mean.sqrt()).ceil() as usize + 32;
    if sigs.len() < subparts * capacity {
        sigs.resize(subparts * capacity, 0);
    }
    let mut next: Vec<usize> = (0..subparts).map(|s| s * capacity).collect();
    let mut extra: Vec<(usize, u64)> = vec![];
    for run in source {
        for &h in *run {
            let s = subpart(h);
            if next[s] < (s + 1) * capacity {
                sigs[next[s]] = h;
                next[s] += 1;
            } else {
                extra.push((s, h));
            }
        }
    }
    extra.sort_by_key(|x| x.0);
    let extra_sigs: Vec<u64> = extra.iter().map(|x| x.1).collect();
    if temp.len() < len {
        temp.resize(len, 0);
    }
    let mut offset = 0;
    for (s, begin) in begin[..buckets].chunks_mut(1 << sub_shift).enumerate() {
        let extra_begin = extra.partition_point(|x| x.0 < s);
        let extra_end = extra.partition_point(|x| x.0 <= s);
        let source = [
            &sigs[s * capacity..next[s]],
            &extra_sigs[extra_begin..extra_end],
        ];
        let sub_len = source[0].len() + source[1].len();
        into_buckets(
            &source,
            &mut temp[offset..offset + sub_len],
            begin,
            first_bucket + (s << sub_shift),
            offset,
            g,
        );
        offset += sub_len;
    }
    begin[buckets] = len;
    std::mem::swap(sigs, temp);
}

/// Distributes into buckets a sequence of runs of signatures whose buckets
/// are consecutive and start from `first_bucket`.
///
/// The signatures are written to `sigs`, and the position of the first
/// signature of each bucket to `begin`, assuming that `sigs` starts at
/// position `offset`.
#[inline(always)]
fn into_buckets(
    source: &[&[u64]],
    sigs: &mut [u64],
    begin: &mut [usize],
    first_bucket: usize,
    offset: usize,
    g: &Geometry,
) {
    for run in source {
        for &h in *run {
            begin[g.bucket(h) - first_bucket] += 1;
        }
    }
    let mut sum = 0;
    for begin in begin.iter_mut() {
        let size = *begin;
        *begin = sum;
        sum += size;
    }
    for run in source {
        for &h in *run {
            let next = &mut begin[g.bucket(h) - first_bucket];
            sigs[*next] = h;
            *next += 1;
        }
    }
    // Each element is now the end of its bucket, that is, the beginning of
    // the following one
    let mut prev = 0;
    for begin in begin.iter_mut() {
        let end = *begin;
        *begin = offset + prev;
        prev = end;
    }
}

/// Returns the indices of the keys whose bucket has no seed, in increasing
/// order.
///
/// We scan the signatures in the order of the keys (see
/// [`Signatures::walk`]), and we check the bucket of each signature in a
/// bit vector.
pub(super) fn bumped(sigs: &Signatures, seeds: &[u16], g: &Geometry) -> Vec<usize> {
    let word = |seeds: &[u16]| {
        seeds
            .iter()
            .enumerate()
            .fold(0u64, |word, (i, &seed)| word | (((seed == 0) as u64) << i))
    };
    let scan = |bits: &[u64], chunk: usize| {
        let mut out = vec![];
        sigs.walk(chunk, |i, h| {
            let b = g.bucket(h);
            if bits[b / 64] >> (b % 64) & 1 != 0 {
                out.push(i);
            }
        });
        out
    };
    #[cfg(feature = "rayon")]
    if parallel(sigs.parts.len()) {
        use rayon::prelude::*;
        let bits: Vec<u64> = seeds.par_chunks(64).map(word).collect();
        let parts: Vec<Vec<usize>> = (0..sigs.chunks)
            .into_par_iter()
            .map(|chunk| scan(&bits, chunk))
            .collect();
        return parts.concat();
    }
    let bits: Vec<u64> = seeds.chunks(64).map(word).collect();
    (0..sigs.chunks)
        .flat_map(|chunk| scan(&bits, chunk))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group() {
        // One or several parts, smaller parts, and regions with little or
        // no slack (so that many signatures do not fit their region), with
        // skewed signatures, too
        for (n, log2_part_keys, max_parts, slack) in [
            (1, 20, 256, 6.0),
            (1000, 20, 256, 6.0),
            (100_000, 8, 256, 6.0),
            (100_000, 4, 16, 0.0),
            (300_000, 10, 4, 1.0),
            (300_000, 6, 256, 0.0),
        ] {
            for skew in [false, true] {
                let g = PHastRBuilder::default().geometry(n, n, 4.5);
                let sig = |i: usize| {
                    let h = mix(i as u64 + 1, 0x9E37_79B9_7F4A_7C15);
                    if skew { h >> (i % 8) } else { h }
                };
                let sigs = group_with(n, sig, &g, log2_part_keys, max_parts, slack);
                // Walking the signatures in the order of the keys
                let mut seen = vec![false; n];
                for chunk in 0..sigs.chunks {
                    sigs.walk(chunk, |i, h| {
                        assert_eq!(h, sig(i));
                        assert!(!std::mem::replace(&mut seen[i], true));
                    });
                }
                assert!(seen.iter().all(|&x| x));
                // Loading parts
                let (mut part, mut begin, mut temp) = (vec![], vec![], vec![]);
                let mut all = vec![];
                for p in 0..sigs.num_parts {
                    let first = sigs.first_bucket(p);
                    let range = first..first + sigs.buckets(p, &g);
                    sigs.load(p, &g, range, &mut part, &mut begin, &mut temp);
                    assert_eq!(begin.len(), sigs.buckets(p, &g) + 1);
                    assert_eq!(begin[0], 0);
                    let len = begin[begin.len() - 1];
                    for (b, w) in begin.windows(2).enumerate() {
                        for &h in &part[w[0]..w[1]] {
                            assert_eq!(g.bucket(h), sigs.first_bucket(p) + b);
                        }
                    }
                    all.extend_from_slice(&part[..len]);
                }
                let mut expected: Vec<u64> = (0..n).map(sig).collect();
                expected.sort_unstable();
                all.sort_unstable();
                assert_eq!(all, expected);
                // The keys of the buckets without a seed
                let seeds: Vec<u16> = (0..g.buckets).map(|b| (b % 3) as u16).collect();
                let expected: Vec<usize> = (0..n).filter(|&i| g.bucket(sig(i)) % 3 == 0).collect();
                assert_eq!(bumped(&sigs, &seeds, &g), expected);
            }
        }
    }
}
