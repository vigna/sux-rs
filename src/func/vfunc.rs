/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use crate::traits::bit_field_slice::*;
use crate::utils::*;
use epserde::prelude::*;
use lambert_w::lambert_w0;
use mem_dbg::*;
use std::fmt::Display;

/// Static functions with low space overhead, fast parallel construction, and
/// fast queries.
///
/// Space overhead with respect to the optimum depends on the [`ShardEdge`]
/// type.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`). Please see the documentation of
/// [`VBuilder`](crate::func::VBuilder) for examples.
///
/// # Generics
///
/// * `T`: The type of the keys.
/// * `W`: The word used to store the data, which is also the output type. It
///        can be any unsigned type.
/// * `D`: The backend storing the function data. It can be a
///        [`BitFieldVec<W>`](crate::bits::BitFieldVec) or a `Box<[W]>`. In the
///        first case, the data is stored using exactly the number of bits
///        needed, but access is slightly slower, while in the second case the
///        data is stored `W`, thus limiting the number of bits to the number of
///        bits of `W`, but access will be faster.
/// * `S`: The signature type. The default is `[u64; 2]`. You can switch
///        to `[u64; 1]` for slightly faster construction and queries, but
///        the construction will not scale beyond a billion keys or so.
/// * `E`: The sharding and edge logic type. The default is [`FuseLge3Shards`].
///        For small sets of keys you might try [`FuseLge3NoShards`], possibly
///        coupled with `[u64; 1]` signatures.
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFunc<
    T: ?Sized + ToSig<S>,
    W: ZeroCopy + Word = usize,
    D: BitFieldSlice<W> = Box<[W]>,
    S: Sig = [u64; 2],
    E: ShardEdge<S, 3> = FuseLge3Shards,
> {
    pub(in crate) shard_edge: E,
    pub(in crate) seed: u64,
    pub(in crate) num_keys: usize,
    pub(in crate) data: D,
    pub(in crate) _marker_t: std::marker::PhantomData<T>,
    pub(in crate) _marker_w: std::marker::PhantomData<W>,
    pub(in crate) _marker_s: std::marker::PhantomData<S>,
}

/// Return the maximum number of high bits for sharding the given number of keys
/// so that the overhead of the maximum shard size with respect to the average
/// shard size is with high probability `eps`.
///
/// From “Zero–Cost Sharding: Scaling Hypergraph-Based Static Functions and
/// Filters to Trillions of Keys”
fn sharding_high_bits(n: usize, eps: f64) -> u32 {
    // Bound from balls and bins problem
    let t = (n as f64 * eps * eps / 2.0).max(1.);
    (t.log2() - t.ln().max(1.).log2()).floor() as u32
}

/// Shard and edge logic.
///
/// This trait is used to derive shards and edges from key signatures. Instances
/// are stored in a [`VFunc`]/[`VFilter`], and they contain the data and logic
/// that turns a signature into an edge, and possibly a shard.
///
/// Having this trait makes it possible to test different types of generation
/// techniques for `K`-uniform hypergraphs.
///
/// There are a few different implementations depending on the type of graphs,
/// on the size of signatures, and on whether sharding is used. See, for
/// example, [`FuseLge3Shards`].
///
/// The implementation of the [`Display`] trait should return the relevant
/// information about the sharding and edge logic.
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
    /// Sets up the sharding logic for the given number of keys.
    ///
    /// This method can be called multiple times. For example, it can be used to
    /// precompute the number of shards so to optimize a [`SigStore`] by using
    /// the same number of buckets.
    ///
    /// After this call, [`shard_high_bits`](ShardEdge::shard_high_bits) will
    /// and [`num_shards`](ShardEdge::num_shards) contain sharding information.
    fn set_up_shards(&mut self, n: usize);

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

    /// Return the number of shards.
    fn num_shards(&self) -> usize {
        1 << self.shard_high_bits()
    }

    /// Return the number of vertices in a shard.
    ///
    /// If there is no sharding, this method returns the overall
    /// number of vertices.
    fn num_vertices(&self) -> usize;

    /// Return the shard assigned to a signature.
    ///
    /// This method is mainly used for testing and debugging, as
    /// [`edge`](ShardEdge::edge) already takes sharding
    /// into consideration.
    fn shard(&self, sig: S) -> usize;

    /// Return the local edge assigned to a signature.
    ///
    /// The edge returned is local to the shard the signature belongs to. If
    /// there is no sharding, this method has the same value as
    /// [`edge`](ShardEdge::edge).
    fn local_edge(&self, sig: S) -> [usize; K];

    /// Return the edge assigned to a signature.
    ///
    /// The edge returned is global, that is, its vertices are absolute indices
    /// into the backend. If there is no sharding, this method has the same
    /// value as [`edge`](ShardEdge::edge).
    fn edge(&self, sig: S) -> [usize; K];
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

#[cfg(feature = "mwhc")]
mod mwhc {
    use super::*;

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
    /// Zero-cost sharded 3-hypergraph MWHC construction.
    ///
    /// This construction uses random peelable 3-hypergraphs on sharded keys,
    /// giving a 23% space overhead. Duplicate edges are not possible, which
    /// makes it possible to shard keys with a finer grain than with [fuse
    /// graphs](crate::func::FuseLge3Shards).
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

    impl Mwhc3Shards {
        /// We use the lower 32 bits of sig[0] for the first vertex, the higher 32
        /// bits of sig[1], and the lower 32 bits of sig[1] for the third vertex.
        #[inline(always)]
        fn _edge_2(&self, shard: usize, sig: [u64; 2]) -> [usize; 3] {
            let seg_size = self.seg_size;
            let mut start = shard * seg_size * 3;
            let v0 = fixed_point_reduce_64!(sig[0] as u32, seg_size) + start;
            start += seg_size;
            let v1 = fixed_point_reduce_64!(sig[1] >> 32, seg_size) + start;
            start += seg_size;
            let v2 = fixed_point_reduce_64!(sig[1] as u32, seg_size) + start;
            [v0, v1, v2]
        }
    }

    impl ShardEdge<[u64; 2], 3> for Mwhc3Shards {
        fn set_up_shards(&mut self, n: usize) {
            self.shard_bits_shift = 63 - sharding_high_bits(n, 0.001);
        }

        fn set_up_graphs(&mut self, _n: usize, max_shard: usize) -> (f64, bool) {
            self.seg_size = ((max_shard as f64 * 1.23) / 3.).ceil() as usize;
            if self.shard_high_bits() != 0 {
                self.seg_size = self.seg_size.next_multiple_of(128);
            }
            (1.23, false)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            63 - self.shard_bits_shift
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
        fn local_edge(&self, sig: [u64; 2]) -> [usize; 3] {
            self._edge_2(0, sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            self._edge_2(self.shard(sig), sig)
        }
    }

    /// Unsharded 3-hypergraph MWHC construction.
    ///
    /// This construction uses random peelable 3-hypergraphs, giving a 23% space
    /// overhead. Due to very low locality, this construction is mainly
    /// useful for comparison and testing.
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
        fn set_up_shards(&mut self, _n: usize) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            self.seg_size = ((n as f64 * 1.23) / 3.).ceil() as usize;
            (1.23, false)
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            0
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
        fn local_edge(&self, sig: [u64; 2]) -> [usize; 3] {
            // We use the upper 32 bits of sig[0] for the first vertex, the
            // lower 32 bits of sig[0] for the second vertex, and the upper 32
            // bits of sig[1] for the third vertex.
            let seg_size = self.seg_size;
            let v0 = fixed_point_reduce_64!(sig[0] >> 32, seg_size);
            let v1 = fixed_point_reduce_64!(sig[0] as u32, seg_size) + seg_size;
            let v2 = fixed_point_reduce_64!(sig[1] >> 32, seg_size) + 2 * seg_size;

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

/// Zero-cost sharded fuse 3-hypergraphs with lazy Gaussian elimination.
///
/// This construction uses fuse 3-hypergraphs (see ”[Dense Peelable Random
/// Uniform Hypergraphs](https://doi.org/10.4230/LIPIcs.ESA.2019.38)”) on
/// sharded keys, giving a 10.5% space overhead. Duplicate edges are possible,
/// which limits the amount of possible sharding.
///
/// In a fuse graph there are 𝓁 + 2 *segments* of size *s*. A random edge is
/// chosen by selecting a first segment *f* uniformly at random among the first
/// 𝓁, and then choosing uniformly and at random a vertex in the segments *f*,
/// *f* + 1 and *f* + 2. The probability of duplicates thus increases as segments
/// gets smaller. This construction uses new empirical estimate of segment
/// sizes to obtain much better sharding than previously possible. See
/// “Zero–Cost Sharding: Scaling Hypergraph-Based Static Functions and
/// Filters to Trillions of Keys”.
///
/// Below a few million keys, fuse graphs have a much higher space overhead.
/// This construction in that case switches to sharding and lazy Gaussian
/// elimination to provide a close, albeit slightly larger, space overhead. The
/// construction time per keys increases by an order of magnitude, but since the
/// number of keys is small, the impact is limited.

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

impl FuseLge3Shards {
    /// The maximum size intended for linear solving. While linear solving is
    /// always active, under this size we change our sharding strategy and we
    /// try to make shards as big as possible, within a maximum size of 2 *
    /// [`Self::HALF_MAX_LIN_SHARD_SIZE`]. Above this threshold we do not shard
    /// unless we can create shards of at least [`Self::MIN_FUSE_SHARD`].
    const MAX_LIN_SIZE: usize = 800_000;
    /// We try to keep shards large enough so that they are solvable and that
    /// the size of the largest shard is close to the target average size, but
    /// also small enough so that we can exploit parallelism. See
    /// [`Self::MAX_LIN_SIZE`].
    const HALF_MAX_LIN_SHARD_SIZE: usize = 50_000;
    /// When we shard, we never create a shard smaller then this.
    const MIN_FUSE_SHARD: usize = 20_000_000;
    /// The log₂ of the maximum number of shards.
    const LOG2_MAX_SHARDS: u32 = 11;

    const A: f64 = 0.41;
    const B: f64 = -3.0;

    /// The expansion factor for fuse graphs.
    ///
    /// Handcrafted, and meaningful for more than 2 *
    /// [`Self::MAX_LIN_SIZE`] keys only.
    fn c(arity: usize, n: usize) -> f64 {
        match arity {
            3 => {
                debug_assert!(n > 2 * Self::HALF_MAX_LIN_SHARD_SIZE);
                if n <= Self::MIN_FUSE_SHARD / 4 {
                    1.125
                } else if n <= Self::MIN_FUSE_SHARD / 2 {
                    1.12
                } else if n <= Self::MIN_FUSE_SHARD {
                    1.11
                } else {
                    1.105
                }
            }

            _ => unimplemented!(),
        }
    }

    /// Return the maximum log₂ of segment size for fuse graphs that makes the
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

    /// Return the maximum log₂ of segment size for fuse graphs that makes the
    /// graphs peelable with high probability.
    fn log2_seg_size(arity: usize, n: usize) -> u32 {
        match arity {
            3 => if n <= Self::MIN_FUSE_SHARD {
                let n = n.max(1) as f64;
                // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
                // https://doi.org/10.1145/3510449
                //
                // TODO: maybe more
                // This estimate is correct for c(arity, n).
                n.ln() / (3.33_f64).ln() + 2.25
            } else {
                let n = n.max(1) as f64;
                // From “Zero–Cost Sharding: Scaling Hypergraph-Based
                // Static Functions and Filters to Trillions of Keys”
                //
                // This estimate is correct for c = 1.105.
                Self::A * n.ln() * n.ln().max(1.).ln() + Self::B
            }
            .floor() as u32,
            _ => unimplemented!(),
        }
    }

    /// Return the maximum number of high bits for sharding the given number of
    /// keys so that the probability of a duplicate edge in a fuse graph with
    /// segments defined by [`FuseLge3Shards::log2_seg_size`] is at most `eps`.
    ///
    /// From “Zero–Cost Sharding: Scaling Hypergraph-Based Static Functions and
    /// Filters to Trillions of Keys”
    fn dup_edge_high_bits(arity: usize, n: usize, c: f64, eps: f64) -> u32 {
        let n = n as f64;
        match arity {
            3 => {
                let subexpr =
                    (1. / (2. * Self::A)) * (-n / (2. * c * (1. - eps).ln()) - 2. * Self::B).log2();
                (n.log2() - subexpr / (2.0_f64.ln() * lambert_w0(subexpr))).floor() as u32
            }
            _ => unimplemented!(),
        }
    }

    /// In this implementation, which is common to the sharded and non-sharded
    /// implementations:
    /// - the `shard_high_bits()` most significant bits of the first component
    ///   are used to select a shard;
    /// - the next bit is not used
    /// - the following 32 bits of the first component are used to select the
    ///   first segment using fixed-point arithmetic;
    /// - the lower 32 bits of first component are used to select the first
    ///   vertex;
    /// - the upper 32 bits of the second component are used to select the
    ///   second vertex;
    /// - the lower 32 bits of the second component are used to select the third
    ///   vertex.
    ///
    /// Note that the lower `shard_high_bits()` + 1 of the bits used to select
    /// the first segment are the same as the upper `shard_high_bits()` of the
    /// bits used to select the first element of the edge, but being the result
    /// of fixed-point arithmetic mostly sensitive to the high bits, this is not
    /// a problem.
    #[inline(always)]
    fn _edge_2(
        shard: usize,
        shard_bits_shift: u32,
        log2_seg_size: u32,
        l: u32,
        sig: [u64; 2],
    ) -> [usize; 3] {
        // Note that we're losing here a random bit at the bottom because we
        // would need a right rotation of one to move exactly the shard high
        // bits to the bottom, but in this way we save an operation, and there
        // are enough random bits anyway.
        let first_segment = fixed_point_reduce_128!(sig[0].rotate_right(shard_bits_shift), l);
        let mut start = (shard * (l as usize + 2) + first_segment) << log2_seg_size;
        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        let v0 = (sig[0] as usize & segment_mask) + start;
        start += segment_size;
        let v1 = ((sig[1] >> 32) as usize & segment_mask) + start;
        start += segment_size;
        let v2 = (sig[1] as usize & segment_mask) + start;
        [v0, v1, v2]
    }

    fn _set_up_shards(&mut self, n: usize) {
        self.shard_bits_shift = 63
            - if n <= Self::MAX_LIN_SIZE {
                // We just try to make shards as big as possible,
                // within a maximum size of 2 * MAX_LIN_SHARD_SIZE
                (n / Self::HALF_MAX_LIN_SHARD_SIZE).max(1).ilog2()
            } else {
                sharding_high_bits(n, 0.001)
                    .min(Self::dup_edge_high_bits(3, n, 1.105, 0.001)) // No duplicate edges
                    .min(Self::LOG2_MAX_SHARDS) // We don't really need too many shards
                    .min((n / Self::MIN_FUSE_SHARD).max(1).ilog2()) // Shards can't be smaller than MIN_FUSE_SHARD
            };
    }

    fn _set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
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

        (c, lge)
    }
}

impl ShardEdge<[u64; 2], 3> for FuseLge3Shards {
    fn set_up_shards(&mut self, n: usize) {
        self._set_up_shards(n);
    }

    fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
        self._set_up_graphs(n, max_shard)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        63 - self.shard_bits_shift
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
    fn local_edge(&self, sig: [u64; 2]) -> [usize; 3] {
        FuseLge3Shards::_edge_2(0, self.shard_bits_shift, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
        FuseLge3Shards::_edge_2(
            self.shard(sig),
            self.shard_bits_shift,
            self.log2_seg_size,
            self.l,
            sig,
        )
    }
}

/// Unsharded fuse 3-hypergraphs with lazy Gaussian elimination.
///
/// See [`FuseLge3Shards`] for a general description of fuse graphs.
///
/// This construction does not use sharding, so it has a higher space overhead
/// for a small number of keys, albeit it uses lazy Gaussian elimination
/// in the smaller cases to improve the overhead.
///
/// This construction, coupled `[u64; 1]` signatures, is the fastest for small
/// sets of keys, but it does not scale beyond a billion keys or so.
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
    /// The expansion factor for fuse graphs.
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

    /// Return the maximum log₂ of segment size for fuse graphs that makes the
    /// graphs peelable with high probability.
    fn log2_seg_size(arity: usize, n: usize) -> u32 {
        let n = n.max(1) as f64;
        match arity {
            3 =>
            // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
            // https://doi.org/10.1145/3510449
            //
            // TODO: maybe more
            // TODO: check floor
            // This estimate is correct for c(arity, n).
            {
                (n.ln() / (3.33_f64).ln() + 2.25).floor() as u32
            }
            _ => unimplemented!(),
        }
    }

    fn _set_up_graphs(&mut self, n: usize) -> (f64, bool) {
        let (c, lge);

        (c, self.log2_seg_size, lge) = if n <= 2 * FuseLge3Shards::HALF_MAX_LIN_SHARD_SIZE {
            (1.13, FuseLge3Shards::lin_log2_seg_size(3, n), true)
        } else {
            // TODO: better bounds (with some repeats)
            (Self::c(3, n), Self::log2_seg_size(3, n), false)
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
    fn set_up_shards(&mut self, _n: usize) {}

    fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
        self._set_up_graphs(n)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        0
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
    fn local_edge(&self, sig: [u64; 2]) -> [usize; 3] {
        let first_segment = fixed_point_reduce_128!(sig[0], self.l);
        let mut start = first_segment << self.log2_seg_size;
        let segment_size = 1 << self.log2_seg_size;
        let segment_mask = segment_size - 1;

        let v0 = (sig[0] as u32 as usize & segment_mask) + start;
        start += segment_size;
        let v1 = ((sig[1] >> 32) as usize & segment_mask) + start;
        start += segment_size;
        let v2 = (sig[1] as u32 as usize & segment_mask) + start;
        [v0, v1, v2]
    }

    #[inline(always)]
    fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
        self.local_edge(sig)
    }
}

impl ShardEdge<[u64; 1], 3> for FuseLge3NoShards {
    fn set_up_shards(&mut self, _n: usize) {}

    fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
        self._set_up_graphs(n)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn shard(&self, _sig: [u64; 1]) -> usize {
        0
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        (self.l as usize + 2) << self.log2_seg_size
    }

    #[inline(always)]
    fn local_edge(&self, sig: [u64; 1]) -> [usize; 3] {
        // From https://github.com/ayazhafiz/xorf
        let hash = sig[0];
        let v0 = fixed_point_reduce_128!(hash, self.l << self.log2_seg_size);
        let seg_size = 1 << self.log2_seg_size;
        let mut v1 = v0 + seg_size;
        let mut v2 = v1 + seg_size;
        let seg_size_mask = seg_size - 1;
        v1 ^= (hash as usize >> 18) & seg_size_mask;
        v2 ^= (hash as usize) & seg_size_mask;
        [v0, v1, v2]
    }

    #[inline(always)]
    fn edge(&self, sig: [u64; 1]) -> [usize; 3] {
        self.local_edge(sig)
    }
}

impl<T: ?Sized + ToSig<S>, W: ZeroCopy + Word, D: BitFieldSlice<W>, S: Sig, E: ShardEdge<S, 3>>
    VFunc<T, W, D, S, E>
{
    /// Return the value associated with the given signature, or a random value
    /// if the signature is not the signature of a key .
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: S) -> W {
        let edge = self.shard_edge.edge(sig);

        unsafe {
            self.data.get_unchecked(edge[0])
                ^ self.data.get_unchecked(edge[1])
                ^ self.data.get_unchecked(edge[2])
        }
    }

    /// Return the value associated with the given key, or a random value if the
    /// key is not present.
    #[inline(always)]
    pub fn get(&self, key: &T) -> W {
        self.get_by_sig(T::to_sig(key, self.seed))
    }

    /// Return the number of keys in the function.
    pub fn len(&self) -> usize {
        self.num_keys
    }

    /// Return whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

