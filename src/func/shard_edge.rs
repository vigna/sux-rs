/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Strategies to shard keys and to generate edges in hypergraphs.
//!
//! The code for [`VFunc`], [`VFilter`], [`LcpMmphf`], etc., is generic with respect
//! to the size of signatures, to the logic that possibly shards keys and
//! generates edges from keys, and to the solving strategy; these aspects are
//! defined by a [`ShardEdge`] implementation. While the default should be a
//! reasonable choice in most cases, there are possible alternatives.
//!
//! The amount of sharding is partially controlled by the parameter ε of
//! [`ShardEdge::set_up_shards`], which specifies the target space loss due to
//! ε-cost sharding. Usually, the larger the loss, the finer the sharding, but
//! depending on the technique there are other bounds involved that might limit
//! the amount of sharding.
//!
//! # Choosing a shard/edge logic
//!
//! - Use the default, [`FuseAug3Shards`], unless you have specific needs: it
//!   has a low space overhead at all sizes, and sharding makes construction
//!   parallel and scalable on large key sets.
//!
//! - Use [`Fuse3Shards`] if construction time on small key sets matters more
//!   than space: up to 2 · 10⁷ keys [`FuseAug3Shards`] is slower and sometimes
//!   needs more than one construction attempt. [`Fuse3Shards`] is also the
//!   default of [`CompVFunc`], which does not support augmented peeling.
//!
//! - Use [`FuseAug3FullSigs`] if the construction of a function fails with
//!   [`BuildError::DuplicateLocalSignatures`], which might happen with more
//!   than a few dozen billion keys.
//!
//! - Use [`Fuse3NoShards`] (possibly with `[u64; 1]` signatures) for small or
//!   medium key sets, as it avoids the small cost of sharding at query time.
//!
//! - The MWHC implementations (`mwhc` feature) are mainly useful for
//!   benchmarking.
//!
//! - [`FuseLge3Shards`] and [`FuseLge3FullSigs`] are deprecated equivalents of
//!   [`FuseAug3Shards`] and [`FuseAug3FullSigs`], which are needed only to
//!   deserialize structures serialized by previous versions.
//!
//! # The main alternatives
//!
//! Here we discuss pros and cons of the main alternatives; further options are
//! available, but they are mostly interesting for benchmarking or for
//! historical reasons. More details can be found in the documentation of each
//! implementation, and in “[ε–Cost Sharding: Scaling Hypergraph–Based Data
//! Structures to Trillions of Keys]”.
//!
//! - [`FuseAug3Shards`] with 128-bit signatures and 64-bit local signatures:
//!   this is the default choice. It shards keys using ε-cost sharding and
//!   generates edges from 64-bit local signatures using a fuse 3-hypergraph
//!   with a space overhead of about 11% (ε = 0.001) for key sets above a
//!   hundred million keys, and of 11–12% for key sets above 2 · 10⁷ keys. Up to
//!   2 · 10⁷ keys, it uses up to sixteen smaller shards solved by augmented
//!   peeling, which keeps the space overhead at 9–12% above 30000 keys (14% at
//!   10⁴ keys, 22% at 1000 keys), and makes construction parallel above 400000
//!   keys. Depending on the amount of sharding, functions with more than a few
//!   dozen billion keys might incur in duplicate local signatures, which
//!   requires switching to [`FuseAug3FullSigs`].
//!
//! - [`Fuse3Shards`] is like [`FuseAug3Shards`] but does not use augmented
//!   peeling, so on small key sets the construction will be faster but the
//!   resulting structure will be bigger (≈+10% on small key sets, ≈+2% on a
//!   few million keys).
//!
//! - [`Fuse3NoShards`] is like [`Fuse3Shards`] but does not use sharding. It is
//!   thus is mostly equivalent to [“Binary Fuse Filters: Fast and Smaller Than
//!   Xor Filters”], albeit for large key sets the space overhead will be lower,
//!   as we use a better expansion factor.
//!
//! - [`FuseAug3FullSigs`]: When building functions with more than a few dozen
//!   billion keys, depending on the amount of sharding [`FuseAug3Shards`] might
//!   incur in duplicate local signatures. This logic is slightly slower and
//!   uses almost double memory during construction, but uses full local
//!   signatures, which cannot lead to duplicates with overwhelming probability.
//!   Filter do not have this problem as local signatures can be deduplicated
//!   without affecting the semantics of the filter.
//!
//! - `Mwhc3Shards` with 128-bit signatures (requires the `mwhc` feature): this
//!   choice gives much worse overhead (23%) but can be sharded very finely.
//!   With ε = 0.01 (and thus 24% space overhead) sharding can already happen at
//!   very small sizes, providing the fastest parallel construction. Query speed
//!   is similar to [`Fuse3Shards`].
//!
//! Note that the solving strategy (e.g., using or not augmented peeling) has
//! no effect on query time. Sharding adds a few nanoseconds to
//! query time, but without sharding one cannot go beyond a few billion keys.
//!
//! [“Binary Fuse Filters: Fast and Smaller Than Xor Filters”]: https://doi.org/10.1145/3510449
//! [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
//! [`VBuilder`]: crate::func::VBuilder
//! [`VFilter`]: crate::dict::VFilter
//! [`VFunc`]: crate::func::VFunc
//! [`LcpMmphf`]: crate::func::LcpMmphf
//! [`CompVFunc`]: crate::func::CompVFunc
//! [`BuildError::DuplicateLocalSignatures`]: crate::func::BuildError::DuplicateLocalSignatures

use crate::utils::{BinSafe, Sig};
use mem_dbg::*;
use num_primitive::{PrimitiveNumberAs, PrimitiveUnsigned};
use rdst::RadixKey;
use std::fmt::Display;

/// Shard and edge logic.
///
/// This trait is used to derive shards and edges from key signatures. Instances
/// are stored, for example, in a [`VFunc`] or in a [`VBuilder`]. They contain
/// the data and logic that turns a signature into an edge, and possibly a
/// shard.
///
/// This trait makes it possible to test different types of generation
/// techniques for hypergraphs. Moreover, it decouples entirely the sharding and
/// edge-generation logic from the rest of the code.
///
/// If you compile with the `mwhc` feature, you will get additional
/// implementations for the classic [MWHC construction].
///
/// There are a few different implementations depending on the type of graphs,
/// on the size of signatures, and on whether sharding is used. See, for
/// example, [`FuseAug3Shards`]. The [module documentation] contains some
/// discussion.
///
/// The implementation of the [`Display`] trait should return the relevant
/// information about the sharding and edge logic.
///
/// Sometimes sorting signatures by a certain key improves performance when
/// generating a hypergraph. In this case, the implementation of
/// [`num_sort_keys`] should return the number of keys used for sorting, and
/// [`sort_key`] should return the key to be used for sorting.
///
/// [`sort_key`]: ShardEdge::sort_key
/// [`num_sort_keys`]: ShardEdge::num_sort_keys
/// [module documentation]: crate::func::shard_edge
/// [MWHC construction]: https://doi.org/10.1093/comjnl/39.6.547
/// [`VBuilder`]: crate::func::VBuilder
/// [`VFunc`]: crate::func::VFunc
/// # Safety
///
/// After the setup methods return, `num_shards() * num_vertices()` must fit in
/// `usize`; `shard(sig)` must be smaller than `num_shards()`; every index from
/// `edge(sig)` must be smaller than that product; and every index from
/// `local_edge(local_sig)` must be smaller than `num_vertices()`.
///
/// For every `V`, `SortSigVal<V>` must have the same size, alignment, and valid
/// representation as `SigVal<S, V>` so mutable-reference transmutation is
/// sound. Its equality and radix order must satisfy the requirements below.
pub unsafe trait ShardEdge<S, const K: usize>:
    Default + Display + Clone + Copy + Send + Sync
{
    /// The type to use for sorting signature when looking for duplicate edges.
    ///
    /// This type must be [transmutable] with `SigVal<S, V>`, but it must
    /// implement [`PartialEq`] and [`RadixKey`] so that equal `SortSigVal`
    /// generate identical edges and after radix sort equal `SortSigVal` are
    /// adjacent. Using `SigVal<S, V>` always works, but it might be possible to
    /// use less information. See, for example, [`LowSortSigVal`].
    ///
    /// Note that [`VBuilder`] assumes internally that sorting the signatures by
    /// this type gives an order very similar to that obtained sorting by the
    /// key returned by [`sort_key`].
    ///
    /// [`sort_key`]: ShardEdge::sort_key
    /// [`VBuilder`]: crate::func::VBuilder
    /// [transmutable]: std::mem::transmute
    type SortSigVal<V: BinSafe>: RadixKey + Send + Sync + Copy + PartialEq;

    /// The type of local signatures used to generate local edges.
    ///
    /// In general, local edges will depend on a local signature, which might
    /// depend only on a fraction of the global signature bits. The method
    /// [`local_sig`], which returns this type, returns the local signature.
    ///
    /// [`local_sig`]: ShardEdge::local_sig
    type LocalSig: Sig;

    /// The type representing vertices local to a shard.
    ///
    /// [`set_up_graphs`] should panic if the number of vertices in a shard is
    /// larger than the maximum value representable by this type.
    ///
    /// Note that since all our graphs have more vertices than edges, this type
    /// is also used to represent an edge by its index (i.e., the index of the
    /// associated key).
    ///
    /// [`set_up_graphs`]: ShardEdge::set_up_graphs
    type Vertex: PrimitiveUnsigned + PrimitiveNumberAs<usize>;

    /// Sets up the sharding logic for the given number of keys.
    ///
    /// `eps` is the target relative space overhead. See “[ε–Cost Sharding:
    /// Scaling Hypergraph–Based Data Structures to Trillions of Keys]” for more
    /// information.
    ///
    /// `retry_prob` is the maximum acceptable probability that a construction
    /// attempt has to be retried because a shard contains duplicate edges
    /// (duplicate local signatures are not covered by this bound). Sharded
    /// implementations should limit sharding so that this bound is
    /// satisfied; implementations that do not shard can ignore it.
    ///
    /// This method can be called multiple times. For example, it can be used to
    /// precompute the number of shards so to optimize a [`SigStore`] by using
    /// the same number of buckets.
    ///
    /// After this call, [`shard_high_bits`] and [`num_shards`] will contain
    /// sharding information. Implementations must never select more than
    /// [`LOG2_MAX_SHARDS`] shard high bits, as this is the maximum supported
    /// by the [`SigStore`] implementations used by
    /// [`VBuilder`](crate::func::VBuilder).
    ///
    /// [`num_shards`]: ShardEdge::num_shards
    /// [`shard_high_bits`]: ShardEdge::shard_high_bits
    /// [`SigStore`]: crate::utils::SigStore
    /// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
    fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64);

    /// Sets up the edge logic for the given number of keys and maximum shard
    /// size.
    ///
    /// This method must be called after [`set_up_shards`], albeit some
    /// no-sharding implementation might not require it. It returns the
    /// expansion factor and whether the graph must be solved by augmented
    /// peeling, that is, peeling that removes an edge whenever it gets stuck,
    /// followed by the solution of a small dense linear system that
    /// satisfies the removed edges. Graphs solved by augmented peeling can be
    /// much denser than peelable graphs.
    ///
    /// This method can be called multiple times. For example, it can be used to
    /// precompute data and then refine it.
    ///
    /// [`set_up_shards`]: ShardEdge::set_up_shards
    fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool);

    /// Sets up the edge logic for a *correlated multi-edge* build, where each
    /// key contributes several correlated edges, as in
    /// [`CompVFunc`](crate::func::CompVFunc).
    ///
    /// # Arguments
    ///
    /// * `num_keys` - total number of keys (regime selector,
    ///   analogous to `n` in [`set_up_graphs`]).
    ///
    /// * `max_shard_keys` - largest per-shard key count.
    ///
    /// * `max_shard_edges` - largest per-shard edge count.
    ///
    /// * `clump` - the largest per-shard value of Σ *k*ᵢ² / Σ *k*ᵢ, where
    ///   *k*ᵢ is the number of edges contributed by the *i*-th key of the
    ///   shard (the average number of edges contributed by the key of a
    ///   random edge, which is one when each key contributes a single edge);
    ///   correlated edges raise the peelability threshold proportionally to
    ///   the square root of this quantity.
    ///
    /// * `num_failures` - number of previous construction attempts that
    ///   failed because a shard was unsolvable; implementations should make
    ///   the graph sparser at each failure, so that the construction
    ///   eventually succeeds even on distributions of edges that make the
    ///   graph harder to peel than expected.
    ///
    /// Returns `(c, aug)` analogously to [`set_up_graphs`] and stores per-shard
    /// layout in `self`.
    ///
    /// The default implementation fails at compile time: every [`ShardEdge`]
    /// used with [`CompVFunc`] must override this method. Implementing types
    /// that do not support correlated multi-edge builds are in this way automatically
    /// prevented to be used as type parameters.
    ///
    /// [`set_up_graphs`]: ShardEdge::set_up_graphs
    /// [`CompVFunc`]: crate::func::CompVFunc
    fn set_up_corr_graphs(
        &mut self,
        _num_keys: usize,
        _max_shard_keys: usize,
        _max_shard_edges: usize,
        _clump: f64,
        _num_failures: usize,
    ) -> (f64, bool) {
        const {
            assert!(
                // This useless call forces compilation of this assert at
                // monomorphization time
                align_of::<Self>() == 0,
                "set_up_corr_graphs must be implemented for use with CompVFunc"
            );
        }
        unreachable!()
    }

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
    /// If there is no sharding, this method returns the overall number of
    /// vertices.
    ///
    /// This method returns a `usize`, but vertices must be representable by the
    /// [`Vertex`] type.
    ///
    /// [`Vertex`]: ShardEdge::Vertex
    fn num_vertices(&self) -> usize;

    /// Returns the shard assigned to a signature.
    ///
    /// This method is mainly used for testing and debugging, as [`edge`]
    /// already takes sharding into consideration.
    ///
    /// [`edge`]: ShardEdge::edge
    fn shard(&self, sig: S) -> usize;

    /// Extracts the signature used to generate a local edge.
    fn local_sig(&self, sig: S) -> Self::LocalSig;

    /// Returns the local edge generated by a [local signature].
    ///
    /// The edge returned is local to the shard the signature belongs to. If
    /// there is no sharding, this method has the same value as [`edge`].
    ///
    /// [`edge`]: ShardEdge::edge
    /// [local signature]: ShardEdge::LocalSig
    fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; K];

    /// Returns the global edge assigned to a signature.
    ///
    /// The edge returned is global, that is, its vertices are absolute indices
    /// into the backend. If there is no sharding, this method has the same
    /// value as [`edge`].
    ///
    /// [`edge`]: ShardEdge::edge
    fn edge(&self, sig: S) -> [usize; K];

    /// Derives a 64-bit remixed hash from a signature.
    ///
    /// This composes [`local_sig`], [`edge_hash`], and the finalization step of
    /// Austin Appleby's [MurmurHash3].
    ///
    /// The result is the canonical hash used to verify membership by
    /// [`VFilter`] and by the [`SignedFunc`] wrappers of structures based on
    /// a [`ShardEdge`].
    ///
    /// [`VFilter`]: crate::dict::VFilter
    /// [`SignedFunc`]: crate::func::SignedFunc
    /// [MurmurHash3]: http://code.google.com/p/smhasher/
    /// [`edge_hash`]: Self::edge_hash
    /// [`local_sig`]: Self::local_sig
    #[inline(always)]
    fn remixed_hash(&self, sig: S) -> u64 {
        crate::utils::mix64(self.edge_hash(self.local_sig(sig)))
    }
}

/// Inversion by 64-bit fixed-point arithmetic.
///
/// This macro computes the inversion ⌊⍺*n*⌋, where ⍺ ∈ [0 . . 1), using 64-bit
/// fixed-point arithmetic, that is, computing ⌊*xn* / 2³²⌋, where *x* is 32-bit
/// unsigned integer representing ⍺.
macro_rules! fixed_point_inv_64 {
    ($x:expr, $n:expr) => {
        (($x as u64 * $n as u64) >> 32) as usize
    };
}

/// Inversion by 128-bit fixed-point arithmetic.
///
/// This macro computes the inversion ⌊⍺*n*⌋, where ⍺ ∈ [0 . . 1), using 128-bit
/// fixed-point arithmetic, that is, computing ⌊*xn* / 2⁶⁴⌋, where *x* is 64-bit
/// unsigned integer representing ⍺.
macro_rules! fixed_point_inv_128 {
    ($x:expr, $n:expr) => {
        (($x as u128 * $n as u128) >> 64) as usize
    };
}

/// The maximum number of shard high bits supported by the [`SigStore`]
/// implementations used by [`VBuilder`](crate::func::VBuilder).
///
/// Implementations of [`ShardEdge::set_up_shards`] must never select more
/// shard high bits than this value.
///
/// [`SigStore`]: crate::utils::SigStore
pub const LOG2_MAX_SHARDS: u32 = 16;

/// Returns the maximum number of high bits for sharding the given number of
/// keys so that the overhead of the maximum shard size with respect to the
/// average shard size is with high probability `eps`.
///
/// From “[ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to
/// Trillions of Keys]”.
///
/// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
pub fn sharding_high_bits(n: usize, eps: f64) -> u32 {
    // Bound from balls and bins problem
    let t = (n as f64 * eps * eps / 2.0).max(1.);
    (t.log2() - t.ln().max(1.).log2()).floor() as u32
}

/// Returns the minimum number of shard high bits such that the shards of a
/// fuse graph on `n` keys can be indexed by `u32` vertices.
///
/// We assume conservatively that the largest shard is at most 10% larger
/// than the average, and that a shard needs at most 1.18 vertices per key.
///
/// # Panics
///
/// If the number of high bits needed is larger than [`LOG2_MAX_SHARDS`];
/// `name` is the name of the [`ShardEdge`] used in the panic message.
fn min_fuse_high_bits(name: &str, n: usize) -> u32 {
    let min_high_bits = (n as f64 * 1.1 * 1.18 / (1u64 << 32) as f64)
        .log2()
        .ceil()
        .max(0.0) as u32;
    assert!(
        min_high_bits <= LOG2_MAX_SHARDS,
        "{name} does not support {n} keys: shards would need more than 2^32 vertices"
    );
    min_high_bits
}

/// Returns the number of vertices ⌈`c` · `num_edges`⌉ of a graph with
/// `num_edges` edges and expansion factor `c`.
///
/// # Panics
///
/// If the number of vertices is larger than `max_vertices` + 1; `name` is
/// the name of the [`ShardEdge`] used in the panic message.
fn num_vertices(name: &str, c: f64, num_edges: usize, max_vertices: u128) -> u128 {
    let num_vertices = (c * num_edges as f64).ceil() as u128;
    assert!(
        num_vertices <= max_vertices + 1,
        "{name} does not support more than {} vertices, but you are requesting {num_vertices}",
        max_vertices + 1
    );
    num_vertices
}

/// Returns the number 𝓁 of segments in which the first vertex of an edge can
/// lie in a fuse graph with segments of size 2<sup>`log2_seg_size`</sup> and
/// at least `num_vertices` vertices (the graph has 𝓁 + 2 segments).
fn fuse_l(num_vertices: u128, log2_seg_size: u32) -> u128 {
    num_vertices
        .div_ceil(1 << log2_seg_size)
        .saturating_sub(2)
        .max(1)
}

/// The maximum number of edges of the fuse graphs whose number of segments
/// is adjusted by [`fuse_l_peelable`].
const PEELABLE_MAX_EDGES: usize = 100_000;

/// Returns the smallest number of segments 𝓁′ ≥ `l` in which the first
/// vertex of an edge can lie such that a fuse graph with `num_edges` edges
/// and segments of size *s* = 2<sup>`log2_seg_size`</sup>, sharded into
/// `num_shards` shards, is peelable with probability at least ≈70%.
///
/// The peelability of a fuse graph depends on the density 𝓁 · *s* /
/// `num_edges` of its first 𝓁 segments. Since 𝓁 is obtained by rounding
/// the number of vertices to a multiple of *s* and then subtracting the two
/// final segments, this density can be up to 2*s* / `num_edges` smaller than
/// the expansion factor. On graphs with a few dozen segments the difference
/// is large enough to make almost all construction attempts fail at some
/// sizes, so we compare the density with an empirical threshold law: an
/// asymptotic threshold, minus a bonus for the lightly loaded end segments
/// of short graphs, plus a term for the fluctuations of the load of the 𝓁 ·
/// `num_shards` segments, which decreases with *s*.
///
/// On graphs with more than [`PEELABLE_MAX_EDGES`] edges the rounding has
/// negligible effects and `l` is returned unchanged.
fn fuse_l_peelable(l: u128, num_edges: usize, log2_seg_size: u32, num_shards: usize) -> u128 {
    if num_edges > PEELABLE_MAX_EDGES {
        return l;
    }
    // Constants of the threshold law: asymptotic threshold, bonus of the
    // final segments and its decay, and fluctuation term
    const C_INF: f64 = 1.0908012824839475;
    const END_BONUS: f64 = 0.426709947971467;
    const END_DECAY: f64 = 3.0276193196246166;
    const FLUCT: f64 = 0.2048487576749646;
    const FLUCT_SEG_EXP: f64 = 0.44627684139096313;
    const FLUCT_LOG_EXP: f64 = 0.8862899551924789;
    let s = (1_u64 << log2_seg_size) as f64;
    let e = num_edges.max(1) as f64;
    let threshold = |l: f64| {
        let segments = (l * num_shards as f64).max(2.0);
        C_INF - END_BONUS * (-(l - 2.0) / END_DECAY).exp()
            + FLUCT * s.powf(-FLUCT_SEG_EXP) * segments.ln().powf(FLUCT_LOG_EXP)
    };
    let mut peelable_l = 1.0;
    while peelable_l * s / e < threshold(peelable_l) {
        peelable_l += 1.0;
    }
    l.max(peelable_l as u128)
}

/// Returns the number of segments in which the first vertex of an edge can
/// lie for a fuse graph with *correlated* edges, as in
/// [`CompVFunc`](crate::func::CompVFunc), given the number of segments `l`
/// obtained from the expansion factor.
///
/// The graph has `num_edges` edges and segments of size *s* =
/// 2<sup>`log2_seg_size`</sup>, and it is sharded into `num_shards`
/// shards; see [`ShardEdge::set_up_corr_graphs`] for the meaning of `clump`
/// and `num_failures`.
///
/// Correlated edges make the load of segments fluctuate more, raising the
/// peelability threshold by a quantity proportional to √(`clump` / *s*). The
/// number of segments is increased, if necessary, so that the density 𝓁 ·
/// *s* / `num_edges` of the first 𝓁 segments reaches an empirical threshold
/// law at which ≈70% of the construction attempts succeed. Moreover, each
/// previous failure increases the density by 0.01, so that the construction
/// eventually succeeds even if the law underestimates the threshold.
fn fuse_l_corr(
    l: u128,
    num_edges: usize,
    log2_seg_size: u32,
    num_shards: usize,
    clump: f64,
    num_failures: usize,
) -> u128 {
    let s = (1_u64 << log2_seg_size) as f64;
    let e = num_edges.max(1) as f64;
    // An estimate of the overall number of segments
    let est_l = (1.1 * e / s * num_shards as f64).max(3.0);
    let threshold = 1.0949 + 0.1844 * (clump / s).sqrt() * est_l.ln().powf(1.05735);
    l.max((threshold * e / s).ceil() as u128) + num_failures as u128 * (0.01 * e / s).ceil() as u128
}

/// The complementary error function, with fractional error less than
/// 1.2 · 10⁻⁷ (from “Numerical Recipes”).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// The boundary term δ(𝓁) of the rank threshold of fuse 3-hypergraphs, for
/// 𝓁 = 1, 2, …, 23 (see [`aug_rank_threshold`]).
const AUG_DELTA: [f64; 23] = [
    2.0, 1.844074, 1.678319, 1.552089, 1.454049, 1.3769, 1.315092, 1.264774, 1.223203, 1.188394,
    1.158889, 1.133599, 1.111705, 1.092578, 1.075731, 1.060784, 1.047433, 1.035436, 1.024598,
    1.014758, 1.005784, 0.997566, 0.990013,
];

/// Returns the rank threshold, as the density 𝓁*s* / *m*, of a fuse
/// 3-hypergraph with *m* edges and 𝓁 segments in which the first vertex of
/// an edge can lie, in the limit *s* → ∞.
///
/// The threshold is computed by density evolution of the 2-core: the
/// incidence matrix has full rank as long as the core has more vertices than
/// edges. It has the form *c*<sub>3</sub>𝓁 / (𝓁 + δ(𝓁)), where *c*<sub>3</sub>
/// ≈ 1.0894 is the inverse of the threshold of random 3-XORSAT: the end
/// segments, which are lightly loaded, are worth δ(𝓁) segments. We have δ(1)
/// = 2, as all segments are equally loaded, and δ(𝓁) → 0.805 as 𝓁 → ∞; we use
/// exact values up to 𝓁 = 23 and an asymptotic expansion afterwards (the
/// relative error on the threshold is less than 4 · 10⁻⁵).
fn aug_rank_threshold(l: f64) -> f64 {
    const C3: f64 = 1.0 / 0.9179352766580832;
    let delta = if l <= 23.0 {
        AUG_DELTA[l as usize - 1]
    } else {
        0.8048619809906162 + 4.841716620896959 / l - 14.04584820008975 / (l * l)
    };
    C3 * l / (l + delta)
}

/// Returns an estimate of the probability that a construction attempt by
/// augmented peeling succeeds for a fuse graph with `num_edges` edges, 𝓁
/// = `l` segments in which the first vertex of an edge can lie, and
/// segments of size *s* = 2<sup>`log2_seg_size`</sup>.
///
/// The attempt fails if there are duplicate edges (the graph has 𝓁*s*³
/// possible edges) or if the linear system is unsolvable. We model the
/// probability that the system is solvable as Φ((𝓁*s* / `num_edges` − μ) /
/// σ), where:
///
/// - μ is the [rank threshold] for *s* → ∞ plus a correction proportional to
///   1 / *s*, which grows slowly with 𝓁 as the load of some segments is
///   larger than average;
///
/// - σ combines the finite-size window of a system with about 𝓁*s*
///   variables, which grows with 𝓁 as the end segments make the threshold
///   fluctuate, and a fluctuation term that decreases quickly with *s*.
///
/// The parameters of the corrections have been fitted on about 1.6 million
/// simulated systems with 2³ ≤ *s* ≤ 2¹⁴ and 1 ≤ 𝓁 ≤ 1536.
///
/// [rank threshold]: aug_rank_threshold
fn aug_success_prob(num_edges: usize, l: u128, log2_seg_size: u32) -> f64 {
    const A0: f64 = 0.7549025339737417;
    const A1: f64 = 0.08505870512858636;
    const A2: f64 = 1.5574013144125352;
    const A3: f64 = 3.2518393977370414;
    const G: f64 = 0.5258773539624467;
    const LAMBDA: f64 = 4.186702858369681;
    const G_SHIFT: f64 = -0.48274975870686254;
    const H: f64 = 0.6795275068300325;
    const H_S_EXP: f64 = 1.8526507762564222;
    let s = (1_u64 << log2_seg_size) as f64;
    let l = l as f64;
    let m = num_edges.max(1) as f64;
    let no_dup = (-m * (m - 1.0) / (2.0 * l * s * s * s)).exp();
    let mu = aug_rank_threshold(l) + (A0 + A1 * l.ln().powf(A2)) * (1.0 + A3 / s) / s;
    let g = G * (1.0 - (-(l + G_SHIFT) / LAMBDA).exp());
    let sigma = (g * g / (l * s) + H * H / s.powf(H_S_EXP)).sqrt();
    let solvable = 0.5 * erfc(-(l * s / m - mu) / (sigma * std::f64::consts::SQRT_2));
    no_dup * solvable
}

/// Returns an estimate of the fraction of edges removed by augmented peeling
/// near the threshold for a fuse graph with 𝓁 = `l` segments in which the
/// first vertex of an edge can lie and segments of size *s* =
/// 2<sup>`log2_seg_size`</sup>.
///
/// The first term accounts for the load of the end segments, the second one
/// for the fluctuations of the load of the segments; the estimate is within
/// 13% for 90% of the simulated configurations.
fn aug_removed_frac(l: u128, log2_seg_size: u32) -> f64 {
    let s = (1_u64 << log2_seg_size) as f64;
    let l = l as f64;
    0.05879304076717531 * (1.0 + l / 11.007559385010206).powf(-1.1533203174756244)
        + 0.09756436184694474
            * s.powf(-0.43041243276810787)
            * (1.0 - (-l / 113.29602572810826).exp())
}

/// The maximum estimated number of edges removed by augmented peeling for
/// the graphs sized by [`aug_fuse_params`] (see [`aug_removed_frac`]).
///
/// The cost of the final dense system grows with the cube of the number of
/// removed edges. The bound has no effect on the shards of
/// [`FuseAug3Shards`], which contain fewer than 1.3 million keys, but it
/// avoids pathologically slow constructions if the sizing is used for larger
/// graphs.
const AUG_MAX_REMOVED: f64 = 8192.0;

/// The probability that a construction attempt succeeds (that is, that all
/// shards are solvable) targeted by [`FuseAug3Shards`] and
/// [`FuseAug3FullSigs`] when using augmented peeling.
const AUG_SUCCESS_PROB: f64 = 0.7;

/// Returns the log₂ of the segment size and the number 𝓁 of segments in
/// which the first vertex of an edge can lie that minimize the number of
/// vertices (𝓁 + 2)*s* of a fuse graph with `num_edges` edges, subject to
/// the condition that a construction attempt by augmented peeling succeeds
/// with probability at least `success_prob` (see [`aug_success_prob`]).
///
/// Graphs whose estimated number of removed edges is larger than
/// [`AUG_MAX_REMOVED`] are discarded, unless no graph satisfies the bound.
fn aug_fuse_params(num_edges: usize, success_prob: f64) -> (u32, u128) {
    let m = num_edges.max(1);
    // Smaller segments have too many duplicate edges
    let min_log2_seg_size = ((m as f64).sqrt().log2() as u32).saturating_sub(1).max(1);
    let max_log2_seg_size = m.ilog2().max(min_log2_seg_size) + 1;
    // The best graph overall, and the best one whose estimated number of
    // removed edges is within AUG_MAX_REMOVED
    let mut best_any: Option<(u128, u32, u128)> = None;
    let mut best: Option<(u128, u32, u128)> = None;
    for log2_seg_size in min_log2_seg_size..=max_log2_seg_size {
        let max_l = 4 * (m as u128 >> log2_seg_size) + 16;
        let Some(l) = (1..=max_l).find(|&l| aug_success_prob(m, l, log2_seg_size) >= success_prob)
        else {
            continue;
        };
        let num_vertices = (l + 2) << log2_seg_size;
        if best_any.is_none_or(|(v, _, _)| num_vertices < v) {
            best_any = Some((num_vertices, log2_seg_size, l));
        }
        if m as f64 * aug_removed_frac(l, log2_seg_size) <= AUG_MAX_REMOVED
            && best.is_none_or(|(v, _, _)| num_vertices < v)
        {
            best = Some((num_vertices, log2_seg_size, l));
        }
    }
    let (_, log2_seg_size, l) = best.or(best_any).expect("no feasible fuse graph");
    (log2_seg_size, l)
}

/// Checks that the vertices of a shard can be represented by a `u32` and
/// that the overall number of vertices fits a `usize`.
///
/// The global edge index is `shard * num_vertices + local`, so the total
/// vertex count across all shards must fit `usize`; the check is performed
/// in `u128` (lossless widening) so that it stays correct on 32-bit
/// targets, where the product can otherwise wrap.
///
/// # Panics
///
/// If one of the conditions above is not satisfied; `name` is the name of
/// the [`ShardEdge`] used in the panic message.
fn check_total_vertices(name: &str, num_vertices: usize, num_shards: usize) {
    assert!(num_vertices - 1 <= u32::MAX as usize);
    let total_vertices = num_vertices as u128 * num_shards as u128;
    assert!(
        total_vertices <= usize::MAX as u128,
        "{name} total vertices ({total_vertices}) overflow usize on this target"
    );
}

// The deprecated FuseLge3Shards and FuseLge3FullSigs are defined (and used by
// derived implementations) in this module.
#[allow(deprecated)]
mod fuse {
    use crate::utils::{BinSafe, SigVal};

    use super::*;
    use rdst::RadixKey;

    /// [ε-cost sharded] [fuse 3-hypergraphs] with augmented peeling using
    /// 64-bit local signatures.
    ///
    /// This construction uses ε-cost sharding (“[ε–Cost Sharding: Scaling
    /// Hypergraph–Based Data Structures to Trillions of Keys]”) to shard keys
    /// and then fuse 3-hypergraphs (see “[Dense Peelable Random Uniform
    /// Hypergraphs]”) on sharded keys, giving a space overhead of about 11% for
    /// large key sets (above a hundred million keys); smaller key sets have a
    /// slightly larger overhead (11–12%). Duplicate edges are possible, which
    /// limits the amount of possible sharding.
    ///
    /// To keep the expansion factor close to the minimums (1.105), in the
    /// peeling regime shards are never smaller than 10⁷ keys. In the
    /// augmented-peeling regime (up to 2 · 10⁷ keys, see below) shards are
    /// much smaller, and the probability of a construction retry caused by a
    /// duplicate edge is not limited by the configured bound: it is instead
    /// accounted for, together with the probability of an unsolvable system,
    /// when sizing the graphs.
    ///
    /// In a fuse graph there are 𝓁 + 2 *segments* of size *s*. A random edge is
    /// chosen by selecting a first segment *f* uniformly at random among the
    /// first 𝓁, and then choosing uniformly and at random a vertex in the
    /// segments *f*, *f* + 1 and *f* + 2. The probability of duplicates thus
    /// increases as segments gets smaller. This implementation uses a new
    /// empirical estimate of segment sizes to obtain much better sharding than
    /// previously possible.
    ///
    /// Edges are generated from 64-bit local signatures, so a fuse graph
    /// has at most 2⁶⁴ distinct edges. When 𝓁s³ ≥ 2⁶⁴ the map from local
    /// signatures to edges is injective: a duplicate edge can only come
    /// from a duplicate local signature. With the segment-size estimate of
    /// this implementation, the injective regime starts at shards of ≈10⁸
    /// keys. For smaller shards the amount of sharding is limited so that
    /// the probability of a construction retry caused by a duplicate edge
    /// is at most [`Fuse3Shards::DUP_EDGE_RETRY_PROB`] (or the bound set with
    /// [`VBuilder::retry_prob`](crate::func::VBuilder::retry_prob)).
    ///
    /// Below a few dozen million keys, peelable fuse graphs have a higher space
    /// overhead, and since shards cannot be smaller than 10⁷ keys, construction
    /// is sequential. Up to 2 · 10⁷ keys this construction thus uses up to
    /// eight shards of at least 200000 keys, or sixteen shards of at least
    /// 400000 keys, and, following a suggestion by Stefan Walzer, Stefan
    /// Hermann, and Markus Wünstel, solves them by *augmented peeling*:
    /// whenever peeling gets stuck, it removes the first remaining edge in the
    /// leftmost segment containing edges and continues; at the end, the removed
    /// edges are satisfied by solving a small dense linear system whose
    /// unknowns are the vertices left without edges after the first removal.
    /// Augmented peeling fails only if the system is unsolvable, so graphs can
    /// be much denser than peelable graphs. The segment size and the number of
    /// segments are chosen to minimize the number of vertices using a law for
    /// the threshold of solvability, derived by density evolution and corrected
    /// for finite segments, so that a construction attempt succeeds with
    /// probability ≈70%, duplicate edges included. The resulting space overhead
    /// is 9–12% above 30000 keys (≈14% at 10⁴ keys, ≈22% at 1000 keys), and
    /// construction is parallel above 400000 keys.
    ///
    /// Augmented peeling is a transposed form of the *inactivation decoding*
    /// of Raptor codes, described by Amin Shokrollahi and Michael Luby in
    /// “[Raptor Codes]”, 2011: if we transpose the linear system, so that
    /// vertices become equations and edges become unknowns, peeling a vertex
    /// of degree one becomes solving an equation with a single unknown, and
    /// removing an edge becomes inactivating an unknown, that is, setting it
    /// aside as a symbolic value to be computed at the end.
    ///
    /// When a construction attempt fails the construction is retried with a
    /// new seed: on average, between 1.1 and 1.35 attempts are necessary,
    /// depending on the number of keys. [`VBuilder`](crate::func::VBuilder)
    /// allows up to 1000 attempts in this regime (instead of 100). If you
    /// need the fastest construction, you can use [`Fuse3Shards`] instead,
    /// trading some space for a faster construction.
    ///
    /// When building functions over key sets above a few dozen billion keys
    /// (but this depends on the sharding parameter ε) we suggest to use
    /// [`FuseAug3FullSigs`]. It uses full signatures as local signatures,
    /// making the probability of a duplicate local signature negligible. As a
    /// result, it is slightly slower and uses more space at construction time.
    /// Filters do not have this problem, as local signatures can be
    /// deduplicated without affecting the semantics of the filter.
    ///
    /// [Dense Peelable Random Uniform Hypergraphs]: https://doi.org/10.4230/LIPIcs.ESA.2019.38
    /// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
    /// [fuse 3-hypergraphs]: https://doi.org/10.4230/LIPIcs.ESA.2019.38
    /// [ε-cost sharded]: https://arxiv.org/abs/2503.18397
    /// [Raptor Codes]: https://doi.org/10.1561/0100000060
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct FuseAug3Shards {
        shard_bits_shift: u32,
        log2_seg_size: u32,
        l: u32,
    }

    impl Default for FuseAug3Shards {
        fn default() -> Self {
            Self {
                shard_bits_shift: 63,
                log2_seg_size: 0,
                l: 0,
            }
        }
    }

    impl Display for FuseAug3Shards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Fuse3 (augmented peeling, shards); shards: 2^{}; vertices per shard: {}; segment size: 2^{}; segments: {}",
                self.shard_high_bits(),
                (self.l as usize + 2) << self.log2_seg_size,
                self.log2_seg_size,
                self.l + 2
            )
        }
    }

    fn edge_64(shard: usize, log2_seg_size: u32, l: u32, sig: [u64; 1]) -> [usize; 3] {
        let start = (shard * (l as usize + 2)) << log2_seg_size;
        let v0 = start + fixed_point_inv_128!(sig[0], (l as u64) << log2_seg_size);
        let seg_size = 1 << log2_seg_size;
        let seg_size_mask = seg_size - 1;

        let mut v1 = v0 + seg_size;
        v1 ^= (sig[0] as usize) & seg_size_mask;
        let mut v2 = v1 + seg_size;
        v2 ^= ((sig[0] >> log2_seg_size) as usize) & seg_size_mask;
        [v0, v1, v2]
    }

    impl FuseAug3Shards {
        /// The maximum number of keys for which we use augmented peeling.
        ///
        /// Up to this size we solve shards by augmented peeling, using either
        /// at most 2<sup>[`Self::LOG2_AUG_SHARDS`] − 1</sup> shards of at least
        /// [`Self::MIN_AUG_SHARD_SIZE`] keys, or
        /// 2<sup>[`Self::LOG2_AUG_SHARDS`]</sup> shards of at least
        /// [`Self::MIN_MAX_AUG_SHARD_SIZE`] keys. Above this size we use
        /// peeling, and we do not shard unless we can create shards of at
        /// least [`Self::MIN_FUSE_SHARD`] keys.
        const MAX_AUG_SIZE: usize = 20_000_000;
        /// The minimum number of keys of a shard solved by augmented peeling.
        ///
        /// Smaller shards are solved faster, but since segments must be large
        /// enough to make duplicate edges rare, they would contain too few
        /// segments, and the lightly loaded end segments would increase the
        /// space overhead.
        const MIN_AUG_SHARD_SIZE: usize = 200_000;
        /// The minimum number of keys of a shard solved by augmented peeling
        /// when using 2<sup>[`Self::LOG2_AUG_SHARDS`]</sup> shards.
        ///
        /// To make duplicate edges rare, the size of segments must grow with
        /// the number of keys, whatever the number of shards: above about 3.3
        /// · 10⁶ keys segments of size 2¹² are necessary, and with the maximum
        /// number of shards and fewer keys per shard shards would contain
        /// fewer than eighty segments, increasing the space overhead to 13%.
        const MIN_MAX_AUG_SHARD_SIZE: usize = 400_000;
        /// The log₂ of the maximum number of shards solved by augmented
        /// peeling.
        ///
        /// More shards would increase parallelism, but they would have too
        /// few segments (see [`Self::MIN_AUG_SHARD_SIZE`]).
        const LOG2_AUG_SHARDS: u32 = 4;
        /// When we shard and use peeling, we never create a shard smaller than
        /// this.
        const MIN_FUSE_SHARD: usize = 10_000_000;

        /// Returns the number of shard high bits for `n` keys when using
        /// augmented peeling.
        fn aug_high_bits(n: usize) -> u32 {
            if n >= Self::MIN_MAX_AUG_SHARD_SIZE << Self::LOG2_AUG_SHARDS {
                Self::LOG2_AUG_SHARDS
            } else {
                (n / Self::MIN_AUG_SHARD_SIZE)
                    .max(1)
                    .ilog2()
                    .min(Self::LOG2_AUG_SHARDS - 1)
            }
        }

        /// Implements [`ShardEdge::set_up_graphs`] for this structure and for
        /// [`FuseAug3FullSigs`]; `name` is used in panic messages.
        fn set_up_graphs_named(&mut self, name: &str, n: usize, max_shard: usize) -> (f64, bool) {
            let num_shards = <Self as ShardEdge<[u64; 2], 3>>::num_shards(self);
            if n <= Self::MAX_AUG_SIZE {
                // An attempt succeeds only if all shards are solvable
                let l;
                (self.log2_seg_size, l) =
                    aug_fuse_params(max_shard, AUG_SUCCESS_PROB.powf(1.0 / num_shards as f64));
                self.l = l.try_into().unwrap();
                let num_vertices = <Self as ShardEdge<[u64; 2], 3>>::num_vertices(self);
                check_total_vertices(name, num_vertices, num_shards);
                return (num_vertices as f64 / max_shard.max(1) as f64, true);
            }

            let c = Fuse3NoShards::c(max_shard);
            self.log2_seg_size = Fuse3Shards::log2_seg_size(max_shard);
            let num_vertices = num_vertices(
                name,
                c,
                max_shard,
                <Self as ShardEdge<[u64; 2], 3>>::Vertex::MAX as u128,
            );
            self.l = fuse_l(num_vertices, self.log2_seg_size).try_into().unwrap();
            check_total_vertices(
                name,
                <Self as ShardEdge<[u64; 2], 3>>::num_vertices(self),
                num_shards,
            );
            (c, false)
        }
    }

    /// A newtype for sorting by the second value of a `[u64; 2]` signature.
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[repr(transparent)]
    pub struct LowSortSigVal<V: BinSafe>(SigVal<[u64; 2], V>);

    impl<V: BinSafe> RadixKey for LowSortSigVal<V> {
        const LEVELS: usize = 8;

        fn get_level(&self, level: usize) -> u8 {
            (self.0.sig[1] >> ((level % 8) * 8)) as u8
        }
    }

    impl<V: BinSafe> PartialEq for LowSortSigVal<V> {
        fn eq(&self, other: &Self) -> bool {
            self.0.sig[1] == other.0.sig[1]
        }
    }

    // SAFETY: setup bounds all generated shard/global/local indices, and
    // LowSortSigVal is a transparent SigVal<[u64; 2], V> wrapper.
    unsafe impl ShardEdge<[u64; 2], 3> for FuseAug3Shards {
        type SortSigVal<V: BinSafe> = LowSortSigVal<V>;
        type LocalSig = [u64; 1];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64) {
            self.shard_bits_shift = 63
                - if n <= Self::MAX_AUG_SIZE {
                    Self::aug_high_bits(n)
                } else {
                    // Shards can't be smaller than MIN_FUSE_SHARD, and we
                    // cannot use more than LOG2_MAX_SHARDS shard bits
                    let max_high_bits = sharding_high_bits(n, eps)
                        .min((n / Self::MIN_FUSE_SHARD).max(1).ilog2())
                        .min(LOG2_MAX_SHARDS);
                    Fuse3Shards::dup_edge_high_bits(n, retry_prob, Some(64), max_high_bits)
                        .max(min_fuse_high_bits("FuseAug3Shards", n))
                };
        }

        fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
            self.set_up_graphs_named("FuseAug3Shards", n, max_shard)
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
            fixed_point_inv_128!(sig[1], self.l)
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
            edge_64(0, self.log2_seg_size, self.l, local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            edge_64(self.shard(sig), self.log2_seg_size, self.l, [sig[1]])
        }
    }

    /// Unsharded [fuse 3-hypergraphs] without augmented peeling.
    ///
    /// In the case of 64-bit signatures this implementation is equivalent to
    /// that described in "[Binary Fuse Filters: Fast and Smaller Than Xor
    /// Filters]"; however, with 64-bit signatures it is not possible to build
    /// functions beyond a few billion keys, so the usage is limited to small to
    /// medium key sets (up to 2³² keys due to `u32` vertices).
    ///
    /// Moreover, for small key sets (up to 800k) we use the expansion factor from
    /// Table 1 of "[Binary Fuse Filters: Fast and Smaller Than Xor
    /// Filters]". After that, we use an [experimentally derived smaller
    /// factor] that is close to the real peelability threshold (*c* = 1.105).
    /// Up to 100000 keys, the number of segments is increased, if necessary,
    /// so that the density of the segments containing the first vertex of
    /// edges reaches an empirical peelability threshold: otherwise, the
    /// rounding of the number of segments would make most construction
    /// attempts fail at some sizes. The same holds for [`Fuse3Shards`].
    ///
    /// We provide also an implementation for 128-bit signatures without these
    /// limitations, but we strongly suggest to use sharding as very large
    /// graphs are very slow to peel due sparse memory access and lack of
    /// parallel processing.
    ///
    /// [Binary Fuse Filters: Fast and Smaller Than Xor Filters]: https://doi.org/10.1145/3510449
    /// [fuse 3-hypergraphs]: https://doi.org/10.4230/LIPIcs.ESA.2019.38
    /// [experimentally derived smaller factor]: Self::c
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[derive(Default)]
    pub struct Fuse3NoShards {
        log2_seg_size: u32,
        l: u32,
    }

    impl Display for Fuse3NoShards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Fuse3 (no augmented peeling, no shards); vertices: {}; segment size: 2^{}; segments: {}",
                (self.l as usize + 2) << self.log2_seg_size,
                self.log2_seg_size,
                self.l + 2
            )
        }
    }

    impl Fuse3NoShards {
        /// Returns the expansion factor for fuse 3-hypergraphs.
        ///
        /// For small key sets (up to 800k) this function uses Table 1 of
        /// "[Binary Fuse Filters: Fast and Smaller Than Xor Filters]". After
        /// that, it uses an experimentally derived smaller factor that is close
        /// to the real peelability threshold; in particular, after 20 million
        /// keys it returns 1.105.
        ///
        /// [Binary Fuse Filters: Fast and Smaller Than Xor Filters]: https://doi.org/10.1145/3510449
        pub fn c(n: usize) -> f64 {
            if n <= 800_000 {
                // From Table 1 (3-wise) of "Binary Fuse Filters: Fast and
                // Smaller Than Xor Filters"
                0.875
                    + 0.25
                        * (1.0_f64)
                            .max((1e6_f64).ln() / ((n as f64).max(core::f64::consts::E)).ln())
            } else if n <= 5_000_000 {
                1.125
            } else if n <= 10_000_000 {
                // Beyond this point the formula above gives too large a value, losing space
                1.12
            } else if n <= 20_000_000 {
                1.11
            } else {
                1.105
            }
        }

        /// Returns the log₂ of segment size for fuse 3-hypergraphs.
        ///
        /// From "[Binary Fuse Filters: Fast and Smaller Than Xor Filters]".
        ///
        /// [Binary Fuse Filters: Fast and Smaller Than Xor Filters]: https://doi.org/10.1145/3510449
        pub fn log2_seg_size(n: usize) -> u32 {
            let n = n.max(1) as f64;
            (n.ln() / (3.33_f64).ln() + 2.25).floor() as u32
        }

        fn edge_128(log2_seg_size: u32, l: u32, sig: [u64; 2]) -> [usize; 3] {
            // This strategy will work up to about 10^15 keys (l is a u32
            // and the segment-size clamp caps segments at 2^18 vertices)
            let v0 = fixed_point_inv_128!(sig[0], (l as u64) << log2_seg_size);
            let segment_size = 1 << log2_seg_size;
            let segment_mask = segment_size - 1;

            let mut v1 = v0 + segment_size;
            v1 ^= (sig[1] >> 32) as usize & segment_mask;
            let mut v2 = v1 + segment_size;
            v2 ^= sig[1] as u32 as usize & segment_mask;
            [v0, v1, v2]
        }

        fn set_up_graphs(&mut self, n: usize, max_vertices: u128) -> (f64, bool) {
            let c = Self::c(n);
            // The minimization has only effect on very large inputs. By
            // reducing the segment size, we increase locality. The speedup in
            // the peeling phase for 2B keys is 1.5x. We cannot to the same in
            // the sharded case because we would significantly increase the
            // probability of duplicate edges.
            self.log2_seg_size = Self::log2_seg_size(n).min(18);

            let num_vertices = num_vertices("Fuse3NoShards", c, n, max_vertices);
            // The adjustment applies only to small graphs, so the vertex
            // count stays within bounds
            self.l = fuse_l_peelable(
                fuse_l(num_vertices, self.log2_seg_size),
                n,
                self.log2_seg_size,
                1,
            )
            .try_into()
            .unwrap();

            (c, false) // false = no Gaussian elimination
        }

        fn set_up_corr_graphs(
            &mut self,
            _num_keys: usize,
            _max_shard_keys: usize,
            max_shard_edges: usize,
            clump: f64,
            num_failures: usize,
            max_vertices: u128,
        ) -> (f64, bool) {
            // The expansion factor of independent edges is a floor; the effect
            // of correlation is accounted for by fuse_l_corr
            let c = Self::c(max_shard_edges);
            self.log2_seg_size = Self::log2_seg_size(max_shard_edges);

            let num_vertices = num_vertices("Fuse3NoShards", c, max_shard_edges, max_vertices);
            let l = fuse_l_corr(
                fuse_l(num_vertices, self.log2_seg_size),
                max_shard_edges,
                self.log2_seg_size,
                1,
                clump,
                num_failures,
            );
            let num_vertices = (l + 2) << self.log2_seg_size;
            assert!(
                num_vertices <= max_vertices + 1,
                "Fuse3NoShards does not support more than {} vertices, but you are requesting {num_vertices}",
                max_vertices + 1
            );
            self.l = l.try_into().unwrap();

            (c, false)
        }
    }

    // SAFETY: setup bounds all generated indices, and SortSigVal is exactly
    // SigVal<[u64; 1], V>.
    unsafe impl ShardEdge<[u64; 1], 3> for Fuse3NoShards {
        type SortSigVal<V: BinSafe> = SigVal<[u64; 1], V>;
        type LocalSig = [u64; 1];
        type Vertex = u32;

        fn set_up_shards(&mut self, _n: usize, _eps: f64, _retry_prob: f64) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            Fuse3NoShards::set_up_graphs(self, n, Self::Vertex::MAX as u128)
        }

        fn set_up_corr_graphs(
            &mut self,
            num_keys: usize,
            max_shard_keys: usize,
            max_shard_edges: usize,
            clump: f64,
            num_failures: usize,
        ) -> (f64, bool) {
            Fuse3NoShards::set_up_corr_graphs(
                self,
                num_keys,
                max_shard_keys,
                max_shard_edges,
                clump,
                num_failures,
                Self::Vertex::MAX as u128,
            )
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
            fixed_point_inv_128!(sig[0], self.l)
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
            edge_64(0, self.log2_seg_size, self.l, sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 1]) -> [usize; 3] {
            edge_64(0, self.log2_seg_size, self.l, sig)
        }
    }

    // SAFETY: setup bounds all generated indices, and SortSigVal is exactly
    // SigVal<[u64; 2], V>.
    unsafe impl ShardEdge<[u64; 2], 3> for Fuse3NoShards {
        type SortSigVal<V: BinSafe> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = usize;

        fn set_up_shards(&mut self, _n: usize, _eps: f64, _retry_prob: f64) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            Fuse3NoShards::set_up_graphs(self, n, Self::Vertex::MAX as u128)
        }

        fn set_up_corr_graphs(
            &mut self,
            num_keys: usize,
            max_shard_keys: usize,
            max_shard_edges: usize,
            clump: f64,
            num_failures: usize,
        ) -> (f64, bool) {
            Fuse3NoShards::set_up_corr_graphs(
                self,
                num_keys,
                max_shard_keys,
                max_shard_edges,
                clump,
                num_failures,
                Self::Vertex::MAX as u128,
            )
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
            fixed_point_inv_128!(sig[1], self.l)
        }

        #[inline(always)]
        fn edge_hash(&self, sig: [u64; 2]) -> u64 {
            sig[0]
        }

        #[inline(always)]
        fn shard(&self, _sig: [u64; 2]) -> usize {
            0
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            sig
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            (self.l as usize + 2) << self.log2_seg_size
        }

        #[inline(always)]
        fn local_edge(&self, sig: Self::LocalSig) -> [usize; 3] {
            Self::edge_128(self.log2_seg_size, self.l, sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            Self::edge_128(self.log2_seg_size, self.l, sig)
        }
    }

    /// [ε-cost sharded] [fuse 3-hypergraphs] without augmented peeling.
    ///
    /// This is a sharded variant of [`Fuse3NoShards`]. See the comments
    /// on sharding and on the injective regime in the documentation of
    /// [`FuseAug3Shards`], which apply to this variant as well.
    ///
    /// Compared to [`FuseAug3Shards`], this variant avoids augmented peeling
    /// (which slows down construction) at the cost of higher space overhead
    /// for small shard sizes (e.g., ≈28% instead of ≈14% at ten thousand
    /// keys, ≈19% instead of ≈11% at a hundred thousand keys, and ≈13%
    /// instead of ≈11.5% at a few million keys; the two variants coincide
    /// above 2 · 10⁷ keys).
    ///
    /// [Binary Fuse Filters: Fast and Smaller Than Xor Filters]: https://doi.org/10.1145/3510449
    /// [fuse 3-hypergraphs]: https://doi.org/10.4230/LIPIcs.ESA.2019.38
    /// [ε-cost sharded]: https://arxiv.org/abs/2503.18397
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Fuse3Shards {
        shard_bits_shift: u32,
        log2_seg_size: u32,
        l: u32,
    }

    impl Default for Fuse3Shards {
        fn default() -> Self {
            Self {
                shard_bits_shift: 63,
                log2_seg_size: 0,
                l: 0,
            }
        }
    }

    impl Display for Fuse3Shards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Fuse3 (no augmented peeling, shards); shards: 2^{}; vertices per shard: {}; segment size: 2^{}; segments: {}",
                self.shard_high_bits(),
                (self.l as usize + 2) << self.log2_seg_size,
                self.log2_seg_size,
                self.l + 2
            )
        }
    }

    impl Fuse3Shards {
        /// Minimum shard size. Shards are never smaller than this to keep
        /// the expansion factor near its asymptotic optimum.
        const MIN_SHARD: usize = 10_000_000;
        /// Threshold above which the ε-cost sharding segment size formula
        /// is used instead of the standard fuse formula.
        ///
        /// This switch is necessary because the standard fuse formula provides
        /// smaller segments which measurably reduce construction time and query
        /// time. However, at large sizes the segments are too small to keep the
        /// probability of duplicate edges low.
        const LARGE_SEGMENT_THRESHOLD: usize = 20_000_000;

        /// The multiplicative constant α of the estimate α ln *n* ln ln *n* + β
        /// of the log₂ of segment size used by
        /// [`log2_large_seg_size`](Self::log2_large_seg_size).
        pub const A: f64 = 0.41;
        /// The additive constant β of the estimate α ln *n* ln ln *n* + β of
        /// the log₂ of segment size used by
        /// [`log2_large_seg_size`](Self::log2_large_seg_size).
        pub const B: f64 = -3.0;

        /// Returns the log₂ of segment size for fuse 3-hypergraphs for large
        /// shards.
        ///
        /// This formula grows asymptotically faster than
        /// [`Fuse3NoShards::log2_seg_size`], reducing the possibility of
        /// duplicate edges for large key sets.
        ///
        /// From “[ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to
        /// Trillions of Keys]”.
        ///
        /// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
        pub fn log2_large_seg_size(n: usize) -> u32 {
            let n = n.max(1) as f64;
            (Self::A * n.ln() * n.ln().max(1.).ln() + Self::B).floor() as u32
        }

        /// Returns the log₂ of segment size that makes the graphs peelable with
        /// high probability, keeping segments as large as possible to reduce
        /// the probability of duplicate edges.
        ///
        /// This function returns [`Fuse3NoShards::log2_seg_size`] for less than
        /// 20 millions, and [`Fuse3Shards::log2_large_seg_size`] for
        /// larger value.
        pub fn log2_seg_size(n: usize) -> u32 {
            if n <= Self::LARGE_SEGMENT_THRESHOLD {
                Fuse3NoShards::log2_seg_size(n)
            } else {
                Self::log2_large_seg_size(n)
            }
        }

        /// The maximum probability of a construction retry caused by
        /// duplicate edges.
        ///
        /// Two distinct local signatures can generate the same edge, which
        /// makes a shard unsolvable; in that case the construction retries with
        /// a new seed. Sharding is limited so that the probability of this
        /// event stays below this bound.
        ///
        /// This is the default value of the bound; it can be changed using
        /// [`VBuilder::retry_prob`](crate::func::VBuilder::retry_prob).
        pub const DUP_EDGE_RETRY_PROB: f64 = 0.05;

        /// Returns the probability that a construction of `n` keys divided
        /// into `2^log2_num_shards` shards must be retried because two
        /// distinct local signatures generate the same edge.
        ///
        /// A fuse graph with 𝓁 first segments of size *s* has 𝓁s³ distinct
        /// edges. If `local_sig_bits` is `Some(t)`, edges are a
        /// deterministic image of *t*-bit local signatures, so the
        /// collision probability of a pair of distinct signatures is at
        /// most 1/𝓁s³ − 2⁻ᵗ; in particular, when 𝓁s³ ≥ 2ᵗ the map from
        /// local signatures to edges is injective and duplicate edges are
        /// impossible. If `local_sig_bits` is `None` (full signatures),
        /// the uniform estimate 1/𝓁s³ is used.
        ///
        /// From “[ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to
        /// Trillions of Keys]”.
        ///
        /// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
        pub fn dup_edge_retry_prob(
            n: usize,
            log2_num_shards: u32,
            local_sig_bits: Option<u32>,
        ) -> f64 {
            // Average shard size; the maximum shard is larger by a factor
            // 1 + ε at most, which is irrelevant at this level of
            // precision.
            let m = (n >> log2_num_shards).max(1);
            let c = Fuse3NoShards::c(m);
            let log2_seg_size = Self::log2_seg_size(m);
            let l = fuse_l((c * m as f64).ceil() as u128, log2_seg_size);
            let log2_num_edges = (l as f64).log2() + 3.0 * log2_seg_size as f64;
            let mut pair_prob = (-log2_num_edges).exp2();
            if let Some(t) = local_sig_bits {
                pair_prob = (pair_prob - (-(t as f64)).exp2()).max(0.0);
            }
            let m = m as f64;
            let expected_dups = (log2_num_shards as f64).exp2() * (m * m / 2.0) * pair_prob;
            -f64::exp_m1(-expected_dups)
        }

        /// Returns the maximum number of high bits for sharding `n` keys,
        /// starting the search at `max_high_bits`, so that the probability
        /// of a retry caused by a duplicate edge
        /// ([`Self::dup_edge_retry_prob`]) is at most `eta`.
        fn dup_edge_high_bits(
            n: usize,
            eta: f64,
            local_sig_bits: Option<u32>,
            max_high_bits: u32,
        ) -> u32 {
            (0..=max_high_bits)
                .rev()
                .find(|&h| Self::dup_edge_retry_prob(n, h, local_sig_bits) <= eta)
                .unwrap_or(0)
        }
    }

    // SAFETY: setup bounds all generated shard/global/local indices, and
    // LowSortSigVal is a transparent SigVal<[u64; 2], V> wrapper.
    unsafe impl ShardEdge<[u64; 2], 3> for Fuse3Shards {
        type SortSigVal<V: BinSafe> = LowSortSigVal<V>;
        type LocalSig = [u64; 1];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64) {
            self.shard_bits_shift = 63
                - if n <= Self::MIN_SHARD {
                    // No sharding below the minimum shard size.
                    0
                } else {
                    let max_high_bits = sharding_high_bits(n, eps)
                        .min((n / Self::MIN_SHARD).max(1).ilog2())
                        .min(LOG2_MAX_SHARDS);
                    Self::dup_edge_high_bits(n, retry_prob, Some(64), max_high_bits)
                        .max(min_fuse_high_bits("Fuse3Shards", n))
                };
        }

        fn set_up_graphs(&mut self, _n: usize, max_shard: usize) -> (f64, bool) {
            let c = Fuse3NoShards::c(max_shard);
            self.log2_seg_size = Self::log2_seg_size(max_shard);

            let num_vertices = num_vertices("Fuse3Shards", c, max_shard, Self::Vertex::MAX as u128);
            self.l = fuse_l_peelable(
                fuse_l(num_vertices, self.log2_seg_size),
                max_shard,
                self.log2_seg_size,
                self.num_shards(),
            )
            .try_into()
            .unwrap();
            check_total_vertices("Fuse3Shards", self.num_vertices(), self.num_shards());
            (c, false) // false = no Gaussian elimination
        }

        fn set_up_corr_graphs(
            &mut self,
            _num_keys: usize,
            _max_shard_keys: usize,
            max_shard_edges: usize,
            clump: f64,
            num_failures: usize,
        ) -> (f64, bool) {
            // The expansion factor of independent edges is a floor; the effect
            // of correlation is accounted for by fuse_l_corr
            let c = Fuse3NoShards::c(max_shard_edges);
            self.log2_seg_size = Fuse3Shards::log2_seg_size(max_shard_edges);

            let num_vertices =
                num_vertices("Fuse3Shards", c, max_shard_edges, Self::Vertex::MAX as u128);
            self.l = fuse_l_corr(
                fuse_l(num_vertices, self.log2_seg_size),
                max_shard_edges,
                self.log2_seg_size,
                self.num_shards(),
                clump,
                num_failures,
            )
            .try_into()
            .unwrap();
            check_total_vertices("Fuse3Shards", self.num_vertices(), self.num_shards());
            (c, false)
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
            fixed_point_inv_128!(sig[1], self.l)
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
            edge_64(0, self.log2_seg_size, self.l, local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            edge_64(self.shard(sig), self.log2_seg_size, self.l, [sig[1]])
        }
    }

    /// [ε-cost sharded] fuse 3-hypergraphs with augmented peeling using full
    /// local signatures.
    ///
    /// This construction should be preferred to [`FuseAug3Shards`] when
    /// building functions over key sets above a few dozen billion keys (but
    /// this depends on the sharding parameter ε). It uses full signatures as
    /// local signatures, making the probability of a duplicate local signature
    /// negligible. Filters do not have this problem as local signatures can be
    /// deduplicated without affecting the semantics of the filter.
    ///
    /// As a result, it is slightly slower and uses more space at
    /// construction time. The rest of the logic is identical, with one
    /// exception: since local signatures have 128 bits, the map from local
    /// signatures to edges never becomes injective, so, contrarily to
    /// [`FuseAug3Shards`], duplicate edges are possible at every scale.
    /// Sharding is limited so that the probability of a construction retry
    /// caused by a duplicate edge is at most
    /// [`Fuse3Shards::DUP_EDGE_RETRY_PROB`] (or the bound set with
    /// [`VBuilder::retry_prob`](crate::func::VBuilder::retry_prob)). As a
    /// consequence, this structure sometimes shards less than
    /// [`FuseAug3Shards`].
    ///
    /// [ε-cost sharded]: https://arxiv.org/abs/2503.18397
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[derive(Default)]
    pub struct FuseAug3FullSigs(FuseAug3Shards);

    impl Display for FuseAug3FullSigs {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    fn edge_2_big(
        shard: usize,
        shard_bits_shift: u32,
        log2_seg_size: u32,
        l: u32,
        sig: [u64; 2],
    ) -> [usize; 3] {
        // This strategy will work up to 10^16 keys
        let start = (shard * (l as usize + 2)) << log2_seg_size;
        let v0 = start
            + fixed_point_inv_128!(
                sig[0].rotate_right(shard_bits_shift).rotate_right(1),
                (l as u64) << log2_seg_size
            );

        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        let mut v1 = v0 + segment_size;
        v1 ^= (sig[1] >> 32) as usize & segment_mask;
        let mut v2 = v1 + segment_size;
        v2 ^= sig[1] as u32 as usize & segment_mask;
        [v0, v1, v2]
    }

    // SAFETY: the delegated setup bounds all generated indices, and
    // SortSigVal is exactly SigVal<[u64; 2], V>.
    unsafe impl ShardEdge<[u64; 2], 3> for FuseAug3FullSigs {
        type SortSigVal<V: BinSafe> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64) {
            // Same logic as FuseAug3Shards, except that full local
            // signatures never enter the injective regime, so the uniform
            // duplicate-edge estimate applies at every scale.
            self.0.shard_bits_shift = 63
                - if n <= FuseAug3Shards::MAX_AUG_SIZE {
                    FuseAug3Shards::aug_high_bits(n)
                } else {
                    let max_high_bits = sharding_high_bits(n, eps)
                        .min((n / FuseAug3Shards::MIN_FUSE_SHARD).max(1).ilog2())
                        .min(LOG2_MAX_SHARDS);
                    // With full signatures the duplicate-edge bound limits
                    // sharding at every scale, so on huge key sets it is the
                    // vertex bound that determines the number of shards
                    Fuse3Shards::dup_edge_high_bits(n, retry_prob, None, max_high_bits)
                        .max(min_fuse_high_bits("FuseAug3FullSigs", n))
                };
        }

        fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
            self.0.set_up_graphs_named("FuseAug3FullSigs", n, max_shard)
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
            fixed_point_inv_64!(
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

    /// A deprecated equivalent of [`FuseAug3Shards`].
    ///
    /// This type was the default shard/edge logic before augmented peeling
    /// replaced lazy Gaussian elimination. It builds exactly the same
    /// structures as [`FuseAug3Shards`], but it is a different type, and it is
    /// kept only because [ε-serde] type hashes include type names: structures
    /// serialized by previous versions with the default shard/edge logic can
    /// be deserialized by specifying this type explicitly (e.g.,
    /// `VFunc<K, D, [u64; 2], FuseLge3Shards>`).
    ///
    /// [ε-serde]: https://crates.io/crates/epserde
    #[deprecated(
        since = "0.15.3",
        note = "use FuseAug3Shards; this type is needed only to deserialize structures serialized by previous versions"
    )]
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct FuseLge3Shards {
        shard_bits_shift: u32,
        log2_seg_size: u32,
        l: u32,
    }

    impl FuseLge3Shards {
        /// Returns the equivalent [`FuseAug3Shards`].
        #[inline(always)]
        fn aug(&self) -> FuseAug3Shards {
            FuseAug3Shards {
                shard_bits_shift: self.shard_bits_shift,
                log2_seg_size: self.log2_seg_size,
                l: self.l,
            }
        }

        /// Returns the equivalent of the given [`FuseAug3Shards`].
        fn from_aug(aug: FuseAug3Shards) -> Self {
            Self {
                shard_bits_shift: aug.shard_bits_shift,
                log2_seg_size: aug.log2_seg_size,
                l: aug.l,
            }
        }
    }

    impl Default for FuseLge3Shards {
        fn default() -> Self {
            Self::from_aug(FuseAug3Shards::default())
        }
    }

    impl Display for FuseLge3Shards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.aug().fmt(f)
        }
    }

    // SAFETY: every method delegates to FuseAug3Shards, which has the same
    // fields.
    unsafe impl ShardEdge<[u64; 2], 3> for FuseLge3Shards {
        type SortSigVal<V: BinSafe> = LowSortSigVal<V>;
        type LocalSig = [u64; 1];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64) {
            let mut aug = self.aug();
            aug.set_up_shards(n, eps, retry_prob);
            *self = Self::from_aug(aug);
        }

        fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
            let mut aug = self.aug();
            let result = aug.set_up_graphs(n, max_shard);
            *self = Self::from_aug(aug);
            result
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            self.aug().shard_high_bits()
        }

        fn num_sort_keys(&self) -> usize {
            self.aug().num_sort_keys()
        }

        #[inline(always)]
        fn sort_key(&self, sig: [u64; 2]) -> usize {
            self.aug().sort_key(sig)
        }

        #[inline(always)]
        fn edge_hash(&self, sig: Self::LocalSig) -> u64 {
            self.aug().edge_hash(sig)
        }

        #[inline(always)]
        fn shard(&self, sig: [u64; 2]) -> usize {
            self.aug().shard(sig)
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            self.aug().num_vertices()
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            self.aug().local_sig(sig)
        }

        #[inline(always)]
        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            self.aug().local_edge(local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            self.aug().edge(sig)
        }
    }

    /// A deprecated equivalent of [`FuseAug3FullSigs`].
    ///
    /// This type is kept only to deserialize structures serialized by
    /// previous versions (see [`FuseLge3Shards`]).
    #[deprecated(
        since = "0.15.3",
        note = "use FuseAug3FullSigs; this type is needed only to deserialize structures serialized by previous versions"
    )]
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[derive(Default)]
    pub struct FuseLge3FullSigs(FuseLge3Shards);

    impl FuseLge3FullSigs {
        /// Returns the equivalent [`FuseAug3FullSigs`].
        #[inline(always)]
        fn aug(&self) -> FuseAug3FullSigs {
            FuseAug3FullSigs(self.0.aug())
        }

        /// Returns the equivalent of the given [`FuseAug3FullSigs`].
        fn from_aug(aug: FuseAug3FullSigs) -> Self {
            Self(FuseLge3Shards::from_aug(aug.0))
        }
    }

    impl Display for FuseLge3FullSigs {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.aug().fmt(f)
        }
    }

    // SAFETY: every method delegates to FuseAug3FullSigs, which has the same
    // fields.
    unsafe impl ShardEdge<[u64; 2], 3> for FuseLge3FullSigs {
        type SortSigVal<V: BinSafe> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64) {
            let mut aug = self.aug();
            aug.set_up_shards(n, eps, retry_prob);
            *self = Self::from_aug(aug);
        }

        fn set_up_graphs(&mut self, n: usize, max_shard: usize) -> (f64, bool) {
            let mut aug = self.aug();
            let result = aug.set_up_graphs(n, max_shard);
            *self = Self::from_aug(aug);
            result
        }

        #[inline(always)]
        fn shard_high_bits(&self) -> u32 {
            self.aug().shard_high_bits()
        }

        fn num_sort_keys(&self) -> usize {
            self.aug().num_sort_keys()
        }

        #[inline(always)]
        fn sort_key(&self, sig: [u64; 2]) -> usize {
            self.aug().sort_key(sig)
        }

        #[inline(always)]
        fn edge_hash(&self, sig: Self::LocalSig) -> u64 {
            self.aug().edge_hash(sig)
        }

        #[inline(always)]
        fn shard(&self, sig: [u64; 2]) -> usize {
            self.aug().shard(sig)
        }

        #[inline(always)]
        fn num_vertices(&self) -> usize {
            self.aug().num_vertices()
        }

        #[inline(always)]
        fn local_sig(&self, sig: [u64; 2]) -> Self::LocalSig {
            self.aug().local_sig(sig)
        }

        #[inline(always)]
        fn local_edge(&self, local_sig: Self::LocalSig) -> [usize; 3] {
            self.aug().local_edge(local_sig)
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            self.aug().edge(sig)
        }
    }

    #[cfg(all(test, target_pointer_width = "64"))]
    mod tests {
        use super::super::{AUG_SUCCESS_PROB, ShardEdge, aug_rank_threshold, aug_success_prob};
        use super::{Fuse3Shards, FuseAug3FullSigs, FuseAug3Shards};

        #[test]
        fn test_deprecated_types_are_equivalent() {
            use super::{FuseLge3FullSigs, FuseLge3Shards};
            for n in [10, 1000, 300_000, 3_000_000, 30_000_000, 3_000_000_000] {
                let mut a = FuseAug3Shards::default();
                let mut b = FuseLge3Shards::default();
                a.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                b.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                let max_shard = n.div_ceil(a.num_shards());
                assert_eq!(a.set_up_graphs(n, max_shard), b.set_up_graphs(n, max_shard));
                assert_eq!(a.num_vertices(), b.num_vertices());
                assert_eq!(a.num_sort_keys(), b.num_sort_keys());
                assert_eq!(a.to_string(), b.to_string());
                let mut c = FuseAug3FullSigs::default();
                let mut d = FuseLge3FullSigs::default();
                c.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                d.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                let max_shard = n.div_ceil(c.num_shards());
                assert_eq!(c.set_up_graphs(n, max_shard), d.set_up_graphs(n, max_shard));
                let mut x = n as u64;
                for _ in 0..1000 {
                    x = x.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                    let sig = [x, x.rotate_left(29) ^ 0xA5A5_A5A5_A5A5_A5A5];
                    assert_eq!(a.shard(sig), b.shard(sig));
                    assert_eq!(a.edge(sig), b.edge(sig));
                    assert_eq!(a.sort_key(sig), b.sort_key(sig));
                    assert_eq!(c.shard(sig), d.shard(sig));
                    assert_eq!(c.edge(sig), d.edge(sig));
                    assert_eq!(c.sort_key(sig), d.sort_key(sig));
                }
            }
        }

        #[cfg(feature = "epserde")]
        #[test]
        fn test_deprecated_types_type_hash() {
            // The ε-serde type hashes of the deprecated types must not
            // change, or structures serialized by previous versions could
            // not be deserialized
            use super::{FuseLge3FullSigs, FuseLge3Shards};
            use epserde::traits::{CryptoHasher, TypeHash};
            fn digest<T: TypeHash>() -> String {
                let mut hasher = CryptoHasher::new();
                T::type_hash(&mut hasher);
                hasher
                    .finalize()
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect()
            }
            assert_eq!(
                digest::<FuseLge3Shards>(),
                "42c4a5cf189c060654db23bce75aa6efedc815ecc7973b90be45ab1c9a4cfbbc"
            );
            assert_eq!(
                digest::<FuseLge3FullSigs>(),
                "fe7984f5193abc8bea72daefd9c7f786c25a57f98125db807de4062863c2b77e"
            );
        }

        #[test]
        fn test_aug_rank_threshold() {
            // With one segment the graph is a random 3-partite hypergraph,
            // whose rank threshold is that of random 3-XORSAT
            let c3 = 1.0 / 0.9179352766580832;
            assert!((3.0 * aug_rank_threshold(1.0) - c3).abs() < 1e-9);
            // The threshold increases to c3 with the number of segments,
            // and the asymptotic expansion joins the table smoothly
            let mut prev = 0.0;
            for l in 1..10_000 {
                let x = aug_rank_threshold(l as f64);
                assert!(x > prev && x < c3, "l = {l}");
                prev = x;
            }
            assert!(c3 - aug_rank_threshold(4096.0) < 6e-4);
        }

        #[test]
        fn test_aug_sizing() {
            // The augmented-peeling regime uses up to 8 shards of at least
            // 200000 keys, or 16 shards of at least 400000 keys, sized so
            // that an attempt succeeds with probability at least
            // AUG_SUCCESS_PROB, with a reasonable expansion factor
            let mut n = 1_usize;
            while n <= FuseAug3Shards::MAX_AUG_SIZE {
                let mut se = FuseAug3Shards::default();
                se.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                let num_shards = se.num_shards();
                assert!(num_shards <= 16, "n = {n}");
                assert!(num_shards == 1 || n / num_shards >= 200_000, "n = {n}");
                assert!(num_shards < 16 || n / num_shards >= 400_000, "n = {n}");
                let max_shard = n.div_ceil(num_shards);
                let (c, aug) = se.set_up_graphs(n, max_shard);
                assert!(aug, "n = {n}");
                let p = aug_success_prob(max_shard, se.l as u128, se.log2_seg_size);
                assert!(
                    p.powi(num_shards as i32) >= AUG_SUCCESS_PROB - 1e-9,
                    "n = {n}"
                );
                if n >= 30_000 {
                    assert!(c < 1.12, "n = {n}, c = {c}");
                } else if n >= 1000 {
                    assert!(c < 1.25, "n = {n}, c = {c}");
                }
                // The 128-bit variant shards in the same way
                let mut se = FuseAug3FullSigs::default();
                se.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                assert_eq!(se.num_shards(), num_shards, "n = {n}");
                n = n * 3 / 2 + 1;
            }
            // Above the regime we switch to peeling
            let mut se = FuseAug3Shards::default();
            let n = FuseAug3Shards::MAX_AUG_SIZE + 1;
            se.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
            let (_, aug) = se.set_up_graphs(n, n.div_ceil(se.num_shards()));
            assert!(!aug);
        }

        #[test]
        fn test_dup_edge_retry_prob() {
            // Configurations from the benchmarks of the ε-cost sharding
            // paper. Shards of ≈10⁸ keys or more are in the injective
            // regime: duplicate edges are impossible.
            assert_eq!(
                Fuse3Shards::dup_edge_retry_prob(10_usize.pow(11), 9, Some(64)),
                0.0
            );
            assert_eq!(
                Fuse3Shards::dup_edge_retry_prob(10_usize.pow(12), 13, Some(64)),
                0.0
            );
            // 10¹⁰ keys in 2⁷ shards of ≈7.8 × 10⁷ keys are below the
            // injective regime; the retry probability is ≈4.4%.
            let p = Fuse3Shards::dup_edge_retry_prob(10_usize.pow(10), 7, Some(64));
            assert!(p > 0.03 && p < 0.05, "{p}");
            // For reference, the model matches measurements: 800000 keys
            // in 16 shards with segments of size 2⁹ (the configuration used
            // before augmented peeling) fail with probability
            // ≈74% (73% measured over 37 attempts), although with such small
            // shards the segment-size estimate of the model differs, so the
            // model is only accurate in order of magnitude there.
            //
            // Full signatures (None) never enter the injective regime, so
            // even at 10¹² keys heavy sharding keeps a high retry
            // probability.
            assert!(Fuse3Shards::dup_edge_retry_prob(10_usize.pow(12), 13, None) > 0.5);
        }

        #[test]
        fn test_shard_bounds_match_retry_budget() {
            // The chosen sharding always respects the retry budget (in the
            // augmented-peeling regime, up to 2 · 10⁷ keys, duplicate edges
            // are instead accounted for by the sizing of the graphs).
            let mut se = FuseAug3Shards::default();
            for lg_n in [25, 27, 30, 33, 36, 40] {
                se.set_up_shards(1 << lg_n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                let h = se.shard_high_bits();
                assert!(
                    Fuse3Shards::dup_edge_retry_prob(1 << lg_n, h, Some(64))
                        <= Fuse3Shards::DUP_EDGE_RETRY_PROB,
                    "lg n = {lg_n}, h = {h}"
                );
            }
        }

        #[test]
        fn test_shards_match_paper_configs() {
            // The corrected bound reproduces the sharding of the paper's
            // benchmarks at 10¹⁰ and 10¹¹ keys, and improves it at 10¹²
            // (8192 shards, all in the injective regime).
            let mut se = FuseAug3Shards::default();
            for (n, expected_h) in [
                (10_usize.pow(10), 7),
                (10_usize.pow(11), 9),
                (10_usize.pow(12), 13),
            ] {
                se.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
                assert_eq!(se.shard_high_bits(), expected_h, "n = {n}");
            }
            // Full signatures shard less: no injective regime.
            let mut se = FuseAug3FullSigs::default();
            se.set_up_shards(10_usize.pow(12), 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
            assert!(se.shard_high_bits() < 13);
            assert!(
                Fuse3Shards::dup_edge_retry_prob(10_usize.pow(12), se.shard_high_bits(), None)
                    <= Fuse3Shards::DUP_EDGE_RETRY_PROB
            );
        }

        #[test]
        fn test_shards_fit_u32_vertices() {
            // With full signatures the duplicate-edge bound alone would
            // allow shards needing more than 2³² vertices.
            let n = 1_usize << 45;
            let mut se = FuseAug3FullSigs::default();
            se.set_up_shards(n, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
            let avg_shard = (n >> se.shard_high_bits()) as f64;
            assert!(avg_shard * 1.1 * 1.18 <= (1_u64 << 32) as f64);
        }

        #[test]
        #[should_panic(expected = "does not support")]
        fn test_too_many_keys_fail_early() {
            let mut se = FuseAug3FullSigs::default();
            se.set_up_shards(1_usize << 50, 0.001, Fuse3Shards::DUP_EDGE_RETRY_PROB);
        }
    }
}

pub use fuse::*;

#[cfg(feature = "mwhc")]
mod mwhc {
    use crate::utils::SigVal;

    use super::*;

    /// Returns the number of edges for which a graph must be sized so that,
    /// with an expansion factor of 1.23, the density of a graph with
    /// `num_edges` correlated edges increases by 0.01 for each of the
    /// `num_failures` previous failed construction attempts.
    ///
    /// The effect of correlated edges on the peelability of random
    /// 3-hypergraphs has not been studied, so we do not adjust the initial
    /// expansion factor, but the increase guarantees that the construction
    /// eventually succeeds.
    fn mwhc_corr_edges(num_edges: usize, num_failures: usize) -> usize {
        (num_edges as f64 * (1.23 + 0.01 * num_failures as f64) / 1.23).ceil() as usize
    }

    /// Unsharded 3-hypergraph [MWHC construction].
    ///
    /// This construction uses random peelable 3-hypergraphs, giving a 23% space
    /// overhead. See [`Mwhc3Shards`] for more information.
    ///
    /// [MWHC construction]: https://doi.org/10.1093/comjnl/39.6.547
    #[derive(Debug, Clone, Copy, MemSize, MemDbg, Default)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Mwhc3NoShards {
        // One third of the number of vertices in a shard
        seg_size: usize,
    }

    impl Display for Mwhc3NoShards {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MWHC3 (no shards); vertices: {}", self.seg_size * 3)
        }
    }

    // SAFETY: setup bounds all generated indices, and SortSigVal is exactly
    // SigVal<[u64; 2], V>.
    unsafe impl ShardEdge<[u64; 2], 3> for Mwhc3NoShards {
        type SortSigVal<V: BinSafe> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = usize;

        fn set_up_shards(&mut self, _n: usize, _eps: f64, _retry_prob: f64) {}

        fn set_up_graphs(&mut self, n: usize, _max_shard: usize) -> (f64, bool) {
            // The lower bound guarantees a nonzero number of vertices even for
            // an empty key set, mirroring the fuse implementations: a zero
            // vertex count would make the builder chunk the backing array with
            // chunk size zero, which panics. Moreover, with segments of size
            // one all keys would have the same edge.
            self.seg_size = (((n as f64 * 1.23) / 3.).ceil() as usize).max(2);
            // num_vertices = seg_size * 3 is used directly as the vertex-index bound
            // and must fit usize; the check promotes to u128 (lossless widening) so
            // the multiply cannot overflow, e.g. on 32-bit targets.
            assert!(
                self.seg_size as u128 * 3 <= usize::MAX as u128,
                "Mwhc3NoShards total vertices overflow usize on this target"
            );
            (1.23, false)
        }

        fn set_up_corr_graphs(
            &mut self,
            _num_keys: usize,
            _max_shard_keys: usize,
            max_shard_edges: usize,
            _clump: f64,
            num_failures: usize,
        ) -> (f64, bool) {
            self.set_up_graphs(mwhc_corr_edges(max_shard_edges, num_failures), 0)
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
            // two vertices. For the third vertex we mix the XOR of the two
            // words: using the XOR directly would make the high bits that
            // determine the third vertex the XOR of those determining the
            // first two, so the edge would depend essentially on two values.
            let seg_size = self.seg_size;
            let v0 = fixed_point_inv_128!(local_sig[0], seg_size);
            let v1 = fixed_point_inv_128!(local_sig[1], seg_size) + seg_size;
            let v2 =
                fixed_point_inv_128!(crate::utils::mix64(local_sig[0] ^ local_sig[1]), seg_size)
                    + 2 * seg_size;

            [v0, v1, v2]
        }

        #[inline(always)]
        fn edge(&self, sig: [u64; 2]) -> [usize; 3] {
            self.local_edge(sig)
        }
    }

    /// [ε-cost sharded] 3-hypergraph [MWHC construction].
    ///
    /// This construction uses ε-cost sharding (“[ε–Cost Sharding: Scaling
    /// Hypergraph–Based Data Structures to Trillions of Keys]”) to shard keys
    /// and then random peelable 3-hypergraphs on sharded keys, giving a 23%
    /// space overhead. Duplicate edges are much less likely than with [fuse
    /// graphs], which makes it possible to shard keys with a finer grain.
    ///
    /// The MWHC construction has mostly been obsoleted by [fuse graphs], but it
    /// is still useful for benchmarking and comparison. It also provides
    /// slightly faster queries due to the simpler edge-generation logic, albeit
    /// construction is slower due to cache-unfriendly accesses.
    ///
    /// [fuse graphs]: crate::func::shard_edge::FuseAug3Shards
    /// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
    /// [MWHC construction]: https://doi.org/10.1093/comjnl/39.6.547
    /// [ε-cost sharded]: https://arxiv.org/abs/2503.18397
    #[derive(Debug, Clone, Copy, MemSize, MemDbg)]
    #[mem_size(flat)]
    #[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(deep_copy))]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
                "MWHC3 (shards); shards: 2^{}; vertices per shard: {}",
                self.shard_high_bits(),
                self.seg_size * 3
            )
        }
    }

    /// We use the lower 32 bits of `sig[0]` for the first vertex, the higher 32
    /// bits of `sig[1]`, and the lower 32 bits of `sig[1]` for the third vertex.
    fn edge(shard: usize, seg_size: usize, sig: [u64; 2]) -> [usize; 3] {
        let mut start = shard * seg_size * 3;
        let v0 = fixed_point_inv_64!(sig[0] as u32, seg_size) + start;
        start += seg_size;
        let v1 = fixed_point_inv_64!(sig[1] >> 32, seg_size) + start;
        start += seg_size;
        let v2 = fixed_point_inv_64!(sig[1] as u32, seg_size) + start;
        [v0, v1, v2]
    }

    /// Returns the maximum number of high bits for sharding the given number of
    /// keys so that the probability of a duplicate edge in a hypergraph is at
    /// most `eta`.
    ///
    /// From “[ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to
    /// Trillions of Keys]”.
    ///
    /// [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397
    fn dup_edge_high_bits(arity: usize, n: usize, c: f64, eta: f64) -> u32 {
        let n = n as f64;
        match arity {
            3 => (0.5
                * (n.log2() + 1. + 3. * c.log2() - 3. * 3_f64.log2() + (-(1. - eta).ln()).log2()))
            .floor() as u32,
            _ => unimplemented!("only arity 3 is supported, got {arity}"),
        }
    }

    // SAFETY: setup bounds all generated shard/global/local indices, and
    // SortSigVal is exactly SigVal<[u64; 2], V>.
    unsafe impl ShardEdge<[u64; 2], 3> for Mwhc3Shards {
        type SortSigVal<V: BinSafe> = SigVal<[u64; 2], V>;
        type LocalSig = [u64; 2];
        type Vertex = u32;

        fn set_up_shards(&mut self, n: usize, eps: f64, retry_prob: f64) {
            self.shard_bits_shift = 63
                - sharding_high_bits(n, eps)
                    .min(dup_edge_high_bits(3, n, 1.23, retry_prob))
                    .min(LOG2_MAX_SHARDS);
        }

        fn set_up_graphs(&mut self, _n: usize, max_shard: usize) -> (f64, bool) {
            // See Mwhc3NoShards::set_up_graphs for the lower bound.
            self.seg_size = (((max_shard as f64 * 1.23) / 3.).ceil() as usize).max(2);
            if self.shard_high_bits() != 0 {
                // Round up to a multiple of 128 without checked_next_multiple_of
                // (Rust 1.87; crate MSRV is 1.85): (n + 127) / 128 * 128, with the
                // add checked so an oversized seg_size panics instead of wrapping.
                self.seg_size = self
                    .seg_size
                    .checked_add(127)
                    .map(|n| n / 128 * 128)
                    .expect("Mwhc3Shards segment size overflows usize");
            }

            // Per-shard num_vertices = seg_size * 3; its largest index
            // (num_vertices - 1) must fit Vertex (u32), and the total across all
            // shards must fit usize, since edge() computes shard * num_vertices +
            // local. The checks promote to u128 (all casts are lossless widening) so
            // they stay overflow-free for empty input (seg_size == 0) and on 32-bit
            // targets where seg_size * 3 can exceed usize.
            let num_vertices = self.seg_size as u128 * 3;
            assert!(
                num_vertices <= Self::Vertex::MAX as u128 + 1,
                "Mwhc3Shards supports at most {} vertices per shard, but {num_vertices} were requested",
                Self::Vertex::MAX as u128 + 1
            );
            let total = num_vertices * self.num_shards() as u128;
            assert!(
                total <= usize::MAX as u128,
                "Mwhc3Shards total vertices ({total}) overflow usize on this target"
            );
            (1.23, false)
        }

        fn set_up_corr_graphs(
            &mut self,
            _num_keys: usize,
            _max_shard_keys: usize,
            max_shard_edges: usize,
            _clump: f64,
            num_failures: usize,
        ) -> (f64, bool) {
            self.set_up_graphs(0, mwhc_corr_edges(max_shard_edges, num_failures))
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

    #[cfg(test)]
    mod tests {
        use super::super::ShardEdge;
        use super::{Mwhc3NoShards, Mwhc3Shards};

        #[test]
        fn empty_input_does_not_underflow() {
            // max_shard == 0 (empty input) makes the raw segment size zero;
            // the old seg_size * 3 - 1 bound underflowed usize here, and a
            // zero vertex count would later panic in the builder, which
            // chunks the backing array by the number of vertices. As in the
            // fuse implementations, empty input now gets a minimum segment
            // (of two vertices, so that distinct keys can have distinct edges).
            let mut shards = Mwhc3Shards::default();
            shards.set_up_graphs(0, 0);
            assert_eq!(shards.num_vertices(), 6);

            let mut no_shards = Mwhc3NoShards { seg_size: 0 };
            no_shards.set_up_graphs(0, 0);
            assert_eq!(no_shards.num_vertices(), 6);
        }

        #[test]
        #[should_panic(expected = "Mwhc3NoShards total vertices")]
        fn no_shards_total_vertices_is_bounded() {
            // seg_size ~ usize::MAX * 1.23 / 3, so seg_size * 3 exceeds usize;
            // the check is portable (no shift literals) and holds on 32 and 64
            // bits. The old code had no guard here.
            let mut no_shards = Mwhc3NoShards { seg_size: 0 };
            no_shards.set_up_graphs(usize::MAX, 0);
        }

        #[cfg(target_pointer_width = "64")]
        #[test]
        #[should_panic(expected = "vertices per shard")]
        fn per_shard_vertex_bound_is_enforced() {
            // A shard needing more than Vertex::MAX + 1 (2^32) vertices must be
            // rejected: seg_size = ceil(2^32 * 1.23 / 3) gives seg_size * 3 > 2^32.
            let mut shards = Mwhc3Shards::default();
            shards.set_up_graphs(0, 1usize << 32);
        }

        #[cfg(target_pointer_width = "64")]
        #[test]
        #[should_panic(expected = "overflow usize")]
        fn total_vertex_product_is_enforced() {
            // num_vertices (~2.5e7, within Vertex::MAX) times num_shards (2^40)
            // exceeds usize::MAX; the old code had no total check and silently
            // produced an invalid setup.
            let mut shards = Mwhc3Shards {
                seg_size: 0,
                shard_bits_shift: 23, // shard_high_bits = 63 - 23 = 40
            };
            shards.set_up_graphs(0, 20_500_000);
        }
    }
}

#[cfg(feature = "mwhc")]
pub use mwhc::*;
