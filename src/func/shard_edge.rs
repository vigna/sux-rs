/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Strategies to shard keys and to generate edges in hypergraphs.
//!
//! The code for [`VFunc`](crate::func::VFunc)/[`VFilter`](crate::dict::VFilter)
//! and [`VBuilder`](crate::func::VBuilder) is generic with respect to the size
//! of signatures and to the logic that possibly shards keys and generates edges
//! from keys; such a logic is defined by a [`ShardEdge`] implementation. While
//! the default should be a reasonable choice in most cases, there are possible
//! alternatives.
//!
//! The amount of sharding is partially controlled by the parameter ε of
//! [`ShardEdge::set_up_shards`], which specifies the target space loss due to
//! ε-cost sharding. Usually, the larger the loss, the finer the sharding, but
//! depending on the technique there are other bounds involved that might limit
//! the amount of sharding.
//!
//! Here we discuss pros and cons of the main alternatives; further options are
//! available, but they are mostly interesting for benchmarking or for
//! historical reasons. More details can be found in the documentation of each
//! implementation, and in “ε-Cost Sharding: Scaling Hypergraph-Based Static
//! Functions and Filters to Trillions of Keys”.
//!
//! - [`FuseLge3Shards`] with 128-bit signatures and 64-bit local signatures:
//!   this is the default choice. It shards keys using ε-cost sharding and
//!   generates edges from 64-bit local signatures using a fuse 3-hypergraph
//!   with a 10.5% (ε = 0.001) space overhead for key sets above a few million
//!   keys. Below 800000 keys, it switches to lazy Gaussian elimination, which
//!   increases construction time but still contains space overhead to 12.5%.
//!   Sharding makes parallelism possible above a few dozen million keys.
//!   Depending on the amount of sharding, functions with more than a few dozen
//!   billion keys might incur in duplicate local signatures, which requires
//!   switching to [`FuseLge3FullSigs`].
//!
//! - [`FuseLge3NoShards`] with 64-bit signatures: this choice provides fast
//!   queries on key sets of less than two billion keys. Construction, however,
//!   cannot be parallelized. It generates edges using a fuse 3-hypergraph with
//!   a 10.5% space overhead for key sets above a few million keys: the overhead
//!   increases slightly for smaller key sets, but below 100000 keys we use lazy
//!   Gaussian elimination to keep overhead within 12.5%. Above that threshold,
//!   this logic is that of a standard fuse 3-hypergraph and thus mostly
//!   equivalent to that described in [“Binary Fuse Filters: Fast and Smaller
//!   Than Xor Filters”](https://doi.org/10.1145/3510449).
//!
//! - [`FuseLge3FullSigs`]: When building functions with more than a few dozen
//!   billion keys, depending on the amount of sharding [`FuseLge3Shards`] might
//!   incur in duplicate local signatures. This logic is slightly slower and
//!   uses almost double memory during construction, but uses full local
//!   signatures, which cannot lead to duplicates with overwhelming probability.
//!   Filter do not have this problem as local signatures can be deduplicated
//!   without affecting the semantics of the filter.
//!
//! - `Mwhc3Shards` with 128-bit signatures (requires the `mwhc`feature): this
//!   choice gives much worse overhead (23%) but can be sharded very finely.
//!   With ε = 0.01 (and thus 24% space overhead) sharding can already happen at
//!   very small sizes, providing the fastest parallel construction. Query speed
//!   is similar to [`FuseLge3Shards`].

use crate::utils::Sig;
use common_traits::{CastableFrom, UnsignedInt, UpcastableInto};
use epserde::prelude::*;
use mem_dbg::*;
use rdst::RadixKey;
use std::fmt::Display;

use epserde::{
    deser::DeserializeInner,
    ser::SerializeInner,
    traits::{AlignHash, TypeHash},
};

/// Shard and edge logic.
///
/// This trait is used to derive shards and edges from key signatures. Instances
/// are stored, for example, in a [`VFunc`](crate::func::VFunc) or in a
/// [`VBuilder`](crate::func::VBuilder). They contain the data and logic that
/// turns a signature into an edge, and possibly a shard.
///
/// This trait makes it possible to test different types of generation
/// techniques for hypergraphs. Moreover, it decouples entirely the sharding and
/// edge-generation logic from the rest of the code.
///
/// If you compile with the `mwhc` feature, you will get additional
/// implementations for the classic [MWHC
/// construction](https://doi.org/10.1093/comjnl/39.6.547).
///
/// There are a few different implementations depending on the type of graphs,
/// on the size of signatures, and on whether sharding is used. See, for
/// example, [`FuseLge3Shards`]. The [module
/// documentation](crate::func::shard_edge) contains some discussion.
///
/// The implementation of the [`Display`] trait should return the relevant
/// information about the sharding and edge logic.
///
/// Sometimes sorting signatures by a certain key improves performance when
/// generating a hypergraph. In this case, the implementation of
/// [`num_sort_keys`](ShardEdge::num_sort_keys) should return the number of keys
/// used for sorting, and [`sort_key`](ShardEdge::sort_key) should return the
/// key to be used for sorting.
pub trait ShardEdge<S, const K: usize>:
    Default
    + Display
    + Clone
    + Copy
    + Send
    + Sync
    + SerializeInner
    + DeserializeInner
    + TypeHash
    + AlignHash
{
    /// The type to use for sorting signature when looking for duplicate edges.
    ///
    /// This type must be [transmutable](std::mem::transmute) with `SigVal<S,
    /// V>`, but it must implement [`PartialEq`] and [`RadixKey`] so that equal
    /// `SortSigVal` generate identical edges and after radix sort equal
    /// `SortSigVal` are adjacent. Using `SigVal<S, V>` always works, but it
    /// might be possible to use less information. See, for example,
    /// [`LowSortSigVal`].
    ///
    /// Note that [`VBuilder`](crate::func::VBuilder) assumes internally that
    /// sorting the signatures by this type gives an order very similar to that
    /// obtained sorting by the key returned by
    /// [`sort_key`](ShardEdge::sort_key).
    type SortSigVal<V: ZeroCopy + Send + Sync>: RadixKey + Send + Sync + Copy + PartialEq;

    /// The type of local signatures used to generate local edges.
    ///
    /// In general, local edges will depend on a local signature, which might
    /// depend only on a fraction of the global signature bits. The method
    /// [`local_sig`](ShardEdge::local_sig), which returns this type, returns
    /// the local signature.
    type LocalSig: Sig;

    /// The type representing vertices local to a shard.
    ///
    /// [`set_up_graphs`](ShardEdge::set_up_graphs) should panic
    /// if the number of vertices in a shard is larger than the maximum value
    /// representable by this type.
    ///
    /// Note that since all our graphs have more vertices than edges, this
    /// type is also used to represent an edge by its index (i.e., the
    /// index of the associated key).
    type Vertex: UnsignedInt + CastableFrom<usize> + UpcastableInto<usize>;

    /// Sets up the sharding logic for the given number of keys.
    ///
    /// `eps` is the target relative space overhead. See “ε-Cost Sharding:
    /// Scaling Hypergraph-Based Static Functions and Filters to Trillions of
    /// Keys” for more information.
    ///
    /// This method can be called multiple times. For example, it can be used to
    /// precompute the number of shards so to optimize a
    ///  [`SigStore`](crate::utils::SigStore) by using the same number of
    /// buckets.
    ///
    /// After this call, [`shard_high_bits`](ShardEdge::shard_high_bits) will
    /// and [`num_shards`](ShardEdge::num_shards) contain sharding information.
    fn set_up_shards(&mut self, n: usize, eps: f64);

    /// Sets up the edge logic for the given number of keys and maximum shard
    /// size.
    ///
    /// This methods must be called after
    /// [`set_up_shards`](ShardEdge::set_up_shards), albeit some no-sharding
    /// implementation might not require it. It returns the expansion factor and
    /// whether the graph will need lazy Gaussian elimination.
    ///
    /// This method can be called multiple times. For example, it can be used to
    /// precompute data and then refine it.
    fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool);

    /// Returns the number of high bits used for sharding.
    fn shard_high_bits(&self) -> u32;

    /// Returns the number of sorting keys to be used for count sorting
    /// signatures before processing.
    /// 
    /// If no count sorting is needed, this method should return 1.
    fn num_sort_keys(&self) -> usize;

    /// Returns the sort key for the given signature.
    /// 
    /// If no count sorting is needed, this method should return 0.
    fn sort_key(&self, sig: S) -> usize;

    /// Extracts a 64-bit hash from a local signature.
    fn edge_hash(&self, sig: Self::LocalSig) -> u64;

    /// Returns the number of shards.
    fn num_shards(&self) -> usize {
        1 << self.shard_high_bits()
    }

    /// Returns the number of vertices in a shard.
    ///
    /// If there is no sharding, this method returns the overall
    /// number of vertices.
    ///
    /// This method returns a `usize`, but vertices must be
    /// representable by the [`Vertex`](ShardEdge::Vertex) type.
    fn num_vertices(&self) -> usize;

    /// Returns the shard assigned to a signature.
    ///
    /// This method is mainly used for testing and debugging, as
    /// [`edge`](ShardEdge::edge) already takes sharding
    /// into consideration.
    fn shard(&self, sig: S) -> usize;

    /// Extracts the signature used to generate a local edge.
    fn local_sig(&self, sig: S) -> Self::LocalSig;

    /// Returns the local edge generated by a [local
    /// signature](ShardEdge::LocalSig).
    ///
    /// The edge returned is local to the shard the signature belongs to. If
    /// there is no sharding, this method has the same value as
    /// [`edge`](ShardEdge::edge).
    fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; K];

    /// Returns the global edge assigned to a signature.
    ///
    /// The edge returned is global, that is, its vertices are absolute indices
    /// into the backend. If there is no sharding, this method has the same
    /// value as [`edge`](ShardEdge::edge).
    fn edge(&self, sig: S) -> [usize; K];
}

/// Fixed-point arithmetic range reduction.
///
/// This macro computes ⌊⍺ * *n*⌋, where ⍺ ∈ [0..1), using 64-bit fixed-point
/// arithmetic. ⍺ is represented by a 32-bit unsigned integer *x*. In
/// fixed-point arithmetic, this amounts to computing ⌊*x* * *n* / 2³²⌋.
macro_rules! fixed_point_reduce_64 {
    ($x:expr, $n:expr) => {
        (($x as u64 * $n as u64) >> 32) as usize
    };
}

/// Fixed-point arithmetic range reduction.
///
/// This macro computes ⌊⍺ * *n*⌋, where ⍺ ∈ [0..1), using 128-bit fixed-point
/// arithmetic. ⍺ is represented by a 64-bit unsigned integer *x*. In
/// fixed-point arithmetic, this amounts to computing ⌊*x* * *n* / 2⁶⁴⌋.
macro_rules! fixed_point_reduce_128 {
    ($x:expr, $n:expr) => {
        (($x as u128 * $n as u128) >> 64) as usize
    };
}

/// Returns the maximum number of high bits for sharding the given number of keys
/// so that the overhead of the maximum shard size with respect to the average
/// shard size is with high probability `eps`.
///
/// From “ε-Cost Sharding: Scaling Hypergraph-Based Static Functions and
/// Filters to Trillions of Keys”
fn sharding_high_bits(n: usize, eps: f64) -> u32 {
    // Bound from balls and bins problem
    let t = (n as f64 * eps * eps / 2.0).max(1.);
    (t.log2() - t.ln().max(1.).log2()).floor() as u32
}

#[cfg(feature = "mwhc")]
mod mwhc {
    use crate::utils::SigVal;
    use epserde::Epserde;

    use super::*;

    /// ε-cost sharded 3-hypergraph [MWHC
    /// construction](https://doi.org/10.1093/comjnl/39.6.547).
    ///
    /// This construction uses uses ε-cost sharding (“ε-Cost Sharding:
    /// Scaling Hypergraph-Based Static Functions and Filters to Trillions of
    /// Keys”) to shard keys and then random peelable 3-hypergraphs on sharded
    /// keys, giving a 23% space overhead. Duplicate edges are not possible,
    /// which makes it possible to shard keys with a finer grain than with [fuse
    /// graphs](crate::func::shard_edge::FuseLge3Shards).
    ///
    /// The MWHC construction has mostly been obsoleted by [fuse
    /// graphs](crate::func::shard_edge::FuseLge3Shards), but it is still useful
    /// for benchmarking and comparison. It also provides slightly faster
    /// queries due to the simpler edge-generation logic, albeit construction is
    /// slower due to cache-unfriendly accesses.
    #[derive(Epserde, Debug, MemDbg, MemSize, Clone, Copy)]
    #[deep_copy]
    pub struct Mwhc3Shards {
        // One third of the number of vertices in a shard
        seg_size: usize,
        shard_bits_shift: u32,
    }

    impl Default for Mwhc3Shards {
        fn default() -> Self {
            Self {
                seg_size: 0,
                shard_bits_shift: 63,
            }
        }
    }

    impl Display for Mwhc3Shards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "MWHC (shards) Number of shards: 2^{} Number of vertices per shard: {}",
                self.shard_high_bits(),
                self.seg_size * 3
            )
        }
    }

    /// We use the lower 32 bits of sig[0] for the first vertex, the higher 32
    /// bits of sig[1], and the lower 32 bits of sig[1] for the third vertex.
    fn edge(shard: usize, seg_size: usize, sig: [u64; 2]) -> [usize; 3] {
        let mut start = shard * seg_size * 3;
        let v0 = fixed_point_reduce_64!(sig[0] as u32, seg_size) + start;
        start += seg_size;
        let v1 = fixed_point_reduce_64!(sig[1] >> 32, seg_size) + start;
        start += seg_size;
        let v2 = fixed_point_reduce_64!(sig[1] as u32, seg_size) + start;
        [v0, v1, v2]
    }

    /// Returns the maximum number of high bits for sharding the given number of
    /// keys so that the probability of a duplicate edge in a hypergraph is at
    /// most `eta`.
    ///
    /// From “ε-Cost Sharding: Scaling Hypergraph-Based Static Functions and
    /// Filters to Trillions of Keys”.
    fn dup_edge_high_bits(arity: usize, n: usize, c: f64, eta: f64) -> u32 {
        let n = n as f64;
        match arity {
            3 => (0.5
                * (n.log2() + 1. + 3. * c.log2() - 3. * 3_f64.log2() + (-(1. - eta).ln()).log2()))
            .floor() as u32,
            _ => unimplemented!(),
        }
    }

    impl ShardEdge<[u64; 2], 3> for Mwhc3Shards {
        type SortSigVal<V: ZeroCopy + Send + Sync> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64) {
            self.shard_bits_shift =
                63 - sharding_high_bits(n, eps).min(dup_edge_high_bits(3, n, 1.23, 0.001));
        }

        fn set_up_graphs(&mut self, _n: usize, max_shard: usize) -> (f64, bool) {
            self.seg_size = ((max_shard as f64 * 1.23) / 3.).ceil() as usize;
            if self.shard_high_bits() != 0 {
                self.seg_size = self.seg_size.next_multiple_of(128);
            }

            assert!(self.seg_size * 3 <= Self::Vertex::MAX as usize + 1);
            (1.23, false)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            63 - self.shard_bits_shift
        }

        fn num_sort_keys(&self) -> usize {
            1
        }

        #[inline(always)]
        fn sort_key(&self, _sig: [u64; 2]) -> usize {
            0
        }

        #[inline(always)]
        fn edge_hash(&self, sig: [u64; 2]) -> u64 {
            sig[1]
        }

        #[inline(always)]
        fn shard(&self, sig: [u64; 2]) -> usize {
            (sig[0] >> self.shard_bits_shift >> 1) as usize
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            self.seg_size * 3
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            sig
        }

        #[inline(always)]
        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            edge(0, self.seg_size, local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            edge(self.shard(sig), self.seg_size, sig)
        }
    }

    /// Unsharded 3-hypergraph [MWHC
    /// construction](https://doi.org/10.1093/comjnl/39.6.547).
    ///
    /// This construction uses random peelable 3-hypergraphs, giving a 23% space
    /// overhead. See [`Mwhc3Shards`] for more information.
    #[derive(Epserde, Default, Debug, MemDbg, MemSize, Clone, Copy)]
    #[deep_copy]
    pub struct Mwhc3NoShards {
        // One third of the number of vertices in a shard
        seg_size: usize,
    }

    impl Display for Mwhc3NoShards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "MWHC (no shards) Number of vertices per shard: {}",
                self.seg_size * 3
            )
        }
    }

    impl ShardEdge<[u64; 2], 3> for Mwhc3NoShards {
        type SortSigVal<V: ZeroCopy + Send + Sync> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = usize;

        fn set_up_shards(&mut self, _n: usize, _eps: f64) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            self.seg_size = ((n as f64 * 1.23) / 3.).ceil() as usize;
            (1.23, false)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            0
        }

        fn num_sort_keys(&self) -> usize {
            1
        }

        #[inline(always)]
        fn sort_key(&self, _sig: [u64; 2]) -> usize {
            0
        }

        #[inline(always)]
        fn edge_hash(&self, sig: [u64; 2]) -> u64 {
            sig[1]
        }

        #[inline(always)]
        fn shard(&self, _sig: [u64; 2]) -> usize {
            0
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            self.seg_size * 3
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            sig
        }

        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            // We use fixed-point arithmetic on sig[0] and sig[1] for the first
            // two vertices. Then, we reuse the two lower halves for the third
            // vertex.
            let seg_size = self.seg_size;
            let v0 = fixed_point_reduce_128!(local_sig[0], seg_size);
            let v1 = fixed_point_reduce_128!(local_sig[1], seg_size) + seg_size;
            let v2 = fixed_point_reduce_128!(local_sig[0] ^ local_sig[1], seg_size) + 2 * seg_size;

            [v0, v1, v2]
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            self.local_edge(sig)
        }
    }
}

#[cfg(feature = "mwhc")]
pub use mwhc::*;

mod fuse {
    use crate::utils::SigVal;

    use super::*;
    use lambert_w::lambert_w0;
    use rdst::RadixKey;
    /// ε-cost sharded fuse 3-hypergraphs with lazy Gaussian elimination
    /// using 64-bit local signatures.
    ///
    /// This construction uses ε-cost sharding (“ε-Cost Sharding: Scaling
    /// Hypergraph-Based Static Functions and Filters to Trillions of Keys”) to
    /// shard keys and then fuse 3-hypergraphs (see “[Dense Peelable Random
    /// Uniform Hypergraphs](https://doi.org/10.4230/LIPIcs.ESA.2019.38)”) on
    /// sharded keys, giving a 10.5% space overhead for large key sets; smaller
    /// key sets have a slightly larger overhead. Duplicate edges are possible,
    /// which limits the amount of possible sharding.
    ///
    /// In a fuse graph there are 𝓁 + 2 *segments* of size *s*. A random edge
    /// is chosen by selecting a first segment *f* uniformly at random among the
    /// first 𝓁, and then choosing uniformly and at random a vertex in the
    /// segments *f*, *f* + 1 and *f* + 2. The probability of duplicates thus
    /// increases as segments gets smaller. This construction uses new empirical
    /// estimate of segment sizes to obtain much better sharding than previously
    /// possible.
    ///
    /// Below a few million keys, fuse graphs have a much higher space overhead.
    /// This construction in that case switches to sharding and lazy Gaussian
    /// elimination to provide a 12.5% overhead. The construction time per keys
    /// increases by an order of magnitude, but since the number of keys is
    /// small, the impact is limited.
    ///
    /// When building functions over key sets above a few dozen billion keys
    /// (but this depends on the sharding parameter ε) we suggest to use
    /// [`FuseLge3FullSigs`]. It uses full signatures as local signatures,
    /// making the probability of a duplicate local signature negligible. As a
    /// result, it is slightly slower and uses more space at construction time.
    #[derive(Epserde, Debug, MemDbg, MemSize, Clone, Copy)]
    #[deep_copy]
    pub struct FuseLge3Shards {
        shard_bits_shift: u32,
        log2_seg_size: u32,
        l: u32,
    }

    impl Default for FuseLge3Shards {
        fn default() -> Self {
            Self {
                shard_bits_shift: 63,
                log2_seg_size: 0,
                l: 0,
            }
        }
    }

    impl Display for FuseLge3Shards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Fuse (shards) Number of shards: 2^{} Segment size: 2^{} Number of segments: {}",
                self.shard_high_bits(),
                self.log2_seg_size,
                self.l + 2
            )
        }
    }

    fn edge_1(shard: usize, log2_seg_size: u32, l: u32, sig: [u64; 1]) -> [usize; 3] {
        let start = (shard * (l as usize + 2)) << log2_seg_size;
        let v0 = start + fixed_point_reduce_128!(sig[0], l << log2_seg_size);
        let seg_size = 1 << log2_seg_size;
        let seg_size_mask = seg_size - 1;

        let mut v1 = v0 + seg_size;
        v1 ^= (sig[0] as usize) & seg_size_mask;
        let mut v2 = v1 + seg_size;
        v2 ^= ((sig[0] >> log2_seg_size) as usize) & seg_size_mask;
        [v0, v1, v2]
    }

    fn edge_2(log2_seg_size: u32, l: u32, sig: [u64; 2]) -> [usize; 3] {
        // This strategy will work up to 10^16 keys
        let v0 = fixed_point_reduce_128!(sig[0], l << log2_seg_size);
        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        let mut v1 = v0 + segment_size;
        v1 ^= (sig[1] >> 32) as usize & segment_mask;
        let mut v2 = v1 + segment_size;
        v2 ^= sig[1] as u32 as usize & segment_mask;
        [v0, v1, v2]
    }

    impl FuseLge3Shards {
        /// The maximum size intended for linear solving. Under this size we change
        /// our sharding strategy and we try to make shards as big as possible,
        /// within a maximum size of 2 * [`Self::HALF_MAX_LIN_SHARD_SIZE`]. Above
        /// this threshold we do not shard unless we can create shards of at least
        /// [`Self::MIN_FUSE_SHARD`].
        const MAX_LIN_SIZE: usize = 800_000;
        /// We try to keep shards large enough so that they are solvable and that
        /// the size of the largest shard is close to the target average size, but
        /// also small enough so that we can exploit parallelism. See
        /// [`Self::MAX_LIN_SIZE`].
        const HALF_MAX_LIN_SHARD_SIZE: usize = 50_000;
        /// When we shard, we never create a shard smaller then this.
        const MIN_FUSE_SHARD: usize = 10_000_000;
        /// The log₂ of the maximum number of shards.
        const LOG2_MAX_SHARDS: u32 = 16;

        /// Returns the expansion factor for fuse graphs.
        ///
        /// Handcrafted, and meaningful for more than 2 * [`Self::MAX_LIN_SIZE`]
        /// keys only.
        fn c(arity: usize, n: usize) -> f64 {
            match arity {
                3 => {
                    debug_assert!(n > 2 * Self::HALF_MAX_LIN_SHARD_SIZE);
                    if n <= Self::MIN_FUSE_SHARD / 2 {
                        1.125
                    } else if n <= Self::MIN_FUSE_SHARD {
                        1.12
                    } else if n <= 2 * Self::MIN_FUSE_SHARD {
                        1.11
                    } else {
                        1.105
                    }
                }

                _ => unimplemented!(),
            }
        }

        /// Returns the maximum log₂ of segment size for fuse graphs that makes the
        /// graphs solvable with high probability.
        ///
        /// This function should not be called for graphs larger than 2 *
        /// [`Self::HALF_MAX_LIN_SHARD_SIZE`].
        fn lin_log2_seg_size(arity: usize, n: usize) -> u32 {
            match arity {
                3 => {
                    debug_assert!(n <= 2 * Self::HALF_MAX_LIN_SHARD_SIZE);
                    (0.85 * (n.max(1) as f64).ln()).floor().max(1.) as u32
                }
                _ => unimplemented!(),
            }
        }

        const A: f64 = 0.41;
        const B: f64 = -3.0;

        /// Returns the maximum log₂ of segment size for fuse graphs that makes the
        /// graphs peelable with high probability.
        pub fn log2_seg_size(arity: usize, n: usize) -> u32 {
            match arity {
                3 => if n <= 2 * Self::MIN_FUSE_SHARD {
                    let n = n.max(1) as f64;
                    // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
                    // https://doi.org/10.1145/3510449
                    //
                    // This estimate is correct for c(arity, n).
                    n.ln() / (3.33_f64).ln() + 2.25
                } else {
                    let n = n.max(1) as f64;
                    // From “ε-Cost Sharding: Scaling Hypergraph-Based
                    // Static Functions and Filters to Trillions of Keys”
                    //
                    // This estimate is correct for c = 1.105.
                    Self::A * n.ln() * n.ln().max(1.).ln() + Self::B
                }
                .floor() as u32,
                _ => unimplemented!(),
            }
        }

        /// Returns the maximum number of high bits for sharding the given number of
        /// keys so that the probability of a duplicate edge in a fuse graph with
        /// segments defined by [`FuseLge3Shards::log2_seg_size`] is at most `eta`.
        ///
        /// From “ε-Cost Sharding: Scaling Hypergraph-Based Static Functions and
        /// Filters to Trillions of Keys”.
        fn dup_edge_high_bits(arity: usize, n: usize, c: f64, eta: f64) -> u32 {
            let n = n as f64;
            match arity {
                3 => {
                    let subexpr = (1. / (2. * Self::A))
                        * (-n / (2. * c * (1. - eta).ln()) - 2. * Self::B).log2();
                    (n.log2() - subexpr / (2.0_f64.ln() * lambert_w0(subexpr))).floor() as u32
                }
                _ => unimplemented!(),
            }
        }
    }

    /// A newtype for sorting by the second value of a `[u64; 2]` signature.
    #[derive(Epserde, Debug, MemDbg, MemSize, Clone, Copy)]
    #[repr(transparent)]
    pub struct LowSortSigVal<V: ZeroCopy + Send + Sync>(SigVal<[u64; 2], V>);

    impl<V: ZeroCopy + Send + Sync> RadixKey for LowSortSigVal<V> {
        const LEVELS: usize = 8;

        fn get_level(&self, level: usize) -> u8 {
            (self.0.sig[1] >> ((level % 8) * 8)) as u8
        }
    }

    impl<V: ZeroCopy + Send + Sync> PartialEq for LowSortSigVal<V> {
        fn eq(&self, other: &Self) -> bool {
            self.0.sig[1] == other.0.sig[1]
        }
    }

    impl ShardEdge<[u64; 2], 3> for FuseLge3Shards {
        type SortSigVal<V: ZeroCopy + Send + Sync> = LowSortSigVal<V>;
        type LocalSig = [u64; 1];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64) {
            self.shard_bits_shift = 63
                - if n <= Self::MAX_LIN_SIZE {
                    // We just try to make shards as big as possible,
                    // within a maximum size of 2 * MAX_LIN_SHARD_SIZE
                    (n / Self::HALF_MAX_LIN_SHARD_SIZE).max(1).ilog2()
                } else {
                    sharding_high_bits(n, eps)
                        .min(Self::dup_edge_high_bits(3, n, 1.105, 0.001))
                        .min(Self::LOG2_MAX_SHARDS) // We don't really need too many shards
                        .min((n / Self::MIN_FUSE_SHARD).max(1).ilog2()) // Shards can't be smaller than MIN_FUSE_SHARD
                };
        }

        fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
            let (c, lge);
            (c, self.log2_seg_size, lge) = if n <= Self::MAX_LIN_SIZE {
                (1.12, Self::lin_log2_seg_size(3, max_shard), true)
            } else {
                (
                    Self::c(3, max_shard),
                    Self::log2_seg_size(3, max_shard),
                    false,
                )
            };

            self.l = ((c * max_shard as f64).ceil() as usize)
                .div_ceil(1 << self.log2_seg_size)
                .saturating_sub(2)
                .max(1)
                .try_into()
                .unwrap();

            assert!((self.l as usize + 2) << self.log2_seg_size <= u32::MAX as usize + 1);
            (c, lge)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            63 - self.shard_bits_shift
        }

        fn num_sort_keys(&self) -> usize {
            self.l as usize
        }

        #[inline(always)]
        fn sort_key(&self, sig: [u64; 2]) -> usize {
            fixed_point_reduce_128!(sig[1], self.l)
        }

        #[inline(always)]
        fn edge_hash(&self, sig: Self::LocalSig) -> u64 {
            sig[0]
        }

        #[inline(always)]
        fn shard(&self, sig: [u64; 2]) -> usize {
            (sig[0] >> self.shard_bits_shift >> 1) as usize
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            (self.l as usize + 2) << self.log2_seg_size
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            [sig[1]]
        }

        #[inline(always)]
        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            edge_1(0, self.log2_seg_size, self.l, local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            edge_1(self.shard(sig), self.log2_seg_size, self.l, [sig[1]])
        }
    }

    /// Unsharded fuse 3-hypergraphs with lazy Gaussian elimination.
    ///
    /// See [`FuseLge3Shards`] for a general description of fuse graphs.
    ///
    /// This construction does not use sharding, so it has a higher space
    /// overhead for a small number of keys, albeit it uses lazy Gaussian
    /// elimination in the smaller cases to improve the overhead.
    ///
    /// This construction, coupled `[u64; 1]` signatures, is the fastest for
    /// small sets of keys, but it does not scale beyond two billion keys or so.
    /// It is mostly equivalent to that described in [“Binary Fuse Filters: Fast
    /// and Smaller Than Xor Filters”](https://doi.org/10.1145/3510449).
    #[derive(Epserde, Default, Debug, MemDbg, MemSize, Clone, Copy)]
    #[deep_copy]
    pub struct FuseLge3NoShards {
        log2_seg_size: u32,
        l: u32,
    }

    impl Display for FuseLge3NoShards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Fuse (no shards) Segment size: 2^{} Number of segments: {}",
                self.log2_seg_size,
                self.l + 2
            )
        }
    }

    impl FuseLge3NoShards {
        /// Returns the expansion factor for fuse graphs.
        ///
        /// Handcrafted, and meaningful for more than 2 *
        /// [`FuseLge3Shards::HALF_MAX_LIN_SHARD_SIZE`] keys only.
        fn c(arity: usize, n: usize) -> f64 {
            match arity {
                3 => {
                    if n <= FuseLge3Shards::MAX_LIN_SIZE {
                        // Exhaustively verified for all inputs from 100000 to 800000
                        // Retries: 1:1218 2:234 3:419 4:121
                        0.168 + (300000_f64).ln().ln() / (n as f64 + 200000.).ln().max(1.).ln()
                    } else {
                        FuseLge3Shards::c(3, n)
                    }
                }

                _ => unimplemented!(),
            }
        }

        /// Returns the maximum log₂ of segment size for fuse graphs that makes the
        /// graphs peelable with high probability.
        pub fn log2_seg_size(arity: usize, n: usize) -> u32 {
            let n = n.max(1) as f64;
            match arity {
                3 =>
                // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
                // https://doi.org/10.1145/3510449
                //
                // This estimate is correct for c(arity, n).
                {
                    (n.ln() / (3.33_f64).ln() + 2.25).floor() as u32
                }
                _ => unimplemented!(),
            }
        }

        fn set_up_graphs(&mut self, n: usize) -> (f64, bool) {
            let (c, lge);

            (c, self.log2_seg_size, lge) = if n <= 2 * FuseLge3Shards::HALF_MAX_LIN_SHARD_SIZE {
                (1.13, FuseLge3Shards::lin_log2_seg_size(3, n), true)
            } else {
                // The minimization has only effect on very large inputs. By
                // reducing the segment size, we increase locality. The speedup in
                // the peeling phase for 2B keys is 1.5x. We cannot to the same in
                // the sharded case because we would significantly increase the
                // probability of duplicate edges.
                (Self::c(3, n), Self::log2_seg_size(3, n).min(18), false)
            };

            self.l = ((c * n as f64).ceil() as u64)
                .div_ceil(1 << self.log2_seg_size)
                .saturating_sub(2)
                .max(1)
                .try_into()
                .unwrap();

            (c, lge)
        }
    }

    impl ShardEdge<[u64; 2], 3> for FuseLge3NoShards {
        type SortSigVal<V: ZeroCopy + Send + Sync> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = usize;

        fn set_up_shards(&mut self, _n: usize, _eps: f64) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            FuseLge3NoShards::set_up_graphs(self, n)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            0
        }

        fn num_sort_keys(&self) -> usize {
            self.l as usize
        }

        #[inline(always)]
        fn sort_key(&self, sig: [u64; 2]) -> usize {
            fixed_point_reduce_128!(sig[0], self.l)
        }

        #[inline(always)]
        fn edge_hash(&self, sig: [u64; 2]) -> u64 {
            sig[1]
        }

        #[inline(always)]
        fn shard(&self, _sig: [u64; 2]) -> usize {
            0
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            (self.l as usize + 2) << self.log2_seg_size
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            sig
        }

        #[inline(always)]
        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            edge_2(self.log2_seg_size, self.l, local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            edge_2(self.log2_seg_size, self.l, sig)
        }
    }

    impl ShardEdge<[u64; 1], 3> for FuseLge3NoShards {
        type SortSigVal<V: ZeroCopy + Send + Sync> = SigVal<[u64; 1], V>;
        type LocalSig = [u64; 1];
        type Vertex = usize;

        fn set_up_shards(&mut self, _n: usize, _eps: f64) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            FuseLge3NoShards::set_up_graphs(self, n)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            0
        }

        fn num_sort_keys(&self) -> usize {
            self.l as usize
        }

        #[inline(always)]
        fn sort_key(&self, sig: [u64; 1]) -> usize {
            fixed_point_reduce_128!(sig[0], self.l)
        }

        #[inline(always)]
        fn edge_hash(&self, sig: [u64; 1]) -> u64 {
            sig[0]
        }

        #[inline(always)]
        fn shard(&self, _sig: [u64; 1]) -> usize {
            0
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 1]) -> Self::LocalSig {
            sig
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            (self.l as usize + 2) << self.log2_seg_size
        }

        #[inline(always)]
        fn local_edge(&self, sig: Self::LocalSig) -> [usize; 3] {
            edge_1(0, self.log2_seg_size, self.l, sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 1]) -> [usize; 3] {
            edge_1(0, self.log2_seg_size, self.l, sig)
        }
    }

    /// ε-cost sharded fuse 3-hypergraphs with lazy Gaussian elimination using
    /// full local signatures.
    ///
    ///
    /// This construction should be preferred to [`FuseLge3Shards`] when
    /// building functions over key sets above a few dozen billion keys (but
    /// this depends on the sharding parameter ε). It uses full signatures as
    /// local signatures, making the probability of a duplicate local signature
    /// negligible. As a result, it is slightly slower and uses more space at
    /// construction time.
    ///
    /// The rest of the logic is identical.
    #[derive(Epserde, Debug, MemDbg, MemSize, Clone, Copy)]
    #[deep_copy]
    #[derive(Default)]
    pub struct FuseLge3FullSigs(FuseLge3Shards);

    impl Display for FuseLge3FullSigs {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    fn edge_2_big(
        shard: usize,
        shard_high_bits: u32,
        log2_seg_size: u32,
        l: u32,
        sig: [u64; 2],
    ) -> [usize; 3] {
        // This strategy will work up to 10^16 keys
        let start = (shard * (l as usize + 2)) << log2_seg_size;
        let v0 = start
            + fixed_point_reduce_128!(
                sig[0].rotate_right(shard_high_bits).rotate_right(1),
                l << log2_seg_size
            );

        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        let mut v1 = v0 + segment_size;
        v1 ^= (sig[1] >> 32) as usize & segment_mask;
        let mut v2 = v1 + segment_size;
        v2 ^= sig[1] as u32 as usize & segment_mask;
        [v0, v1, v2]
    }

    impl ShardEdge<[u64; 2], 3> for FuseLge3FullSigs {
        type SortSigVal<V: ZeroCopy + Send + Sync> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64) {
            self.0.set_up_shards(n, eps);
        }

        fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
            self.0.set_up_graphs(n, max_shard)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            self.0.shard_high_bits()
        }

        fn num_sort_keys(&self) -> usize {
            self.0.num_sort_keys()
        }

        #[inline(always)]
        fn sort_key(&self, sig: [u64; 2]) -> usize {
            fixed_point_reduce_64!(
                sig[0].rotate_right(self.0.shard_bits_shift).rotate_right(1) >> 32,
                self.0.l
            )
        }

        #[inline(always)]
        fn edge_hash(&self, sig: [u64; 2]) -> u64 {
            sig[1]
        }

        #[inline(always)]
        fn shard(&self, sig: [u64; 2]) -> usize {
            (sig[0] >> self.0.shard_bits_shift >> 1) as usize
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            self.0.num_vertices()
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            sig
        }

        #[inline(always)]
        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            edge_2_big(
                0,
                self.0.shard_bits_shift,
                self.0.log2_seg_size,
                self.0.l,
                local_sig,
            )
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            edge_2_big(
                self.shard(sig),
                self.0.shard_bits_shift,
                self.0.log2_seg_size,
                self.0.l,
                sig,
            )
        }
    }
}

pub use fuse::*;
