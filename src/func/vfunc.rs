/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use crate::bits::*;
use crate::traits::bit_field_slice::*;
use crate::utils::*;
use common_traits::CastableInto;
use epserde::prelude::*;
use lender::Fuse;
use mem_dbg::*;
use std::borrow::Borrow;
use std::fmt::Display;
use std::ops::Index;

/// The log₂ of the segment size in the linear regime.
fn lin_log2_seg_size(arity: usize, n: usize) -> u32 {
    match arity {
        3 => {
            if n >= 100_000 {
                10
            } else {
                (0.85 * (n.max(1) as f64).ln()).floor() as u32
            }
        }
        _ => unimplemented!(),
    }
}

/// The log₂ of the segment size in the fuse-graphs regime.
fn fuse_log2_seg_size(arity: usize, n: usize) -> u32 {
    // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
    // https://doi.org/10.1145/3510449
    match arity {
        3 => ((n.max(1) as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32,
        4 => ((n.max(1) as f64).ln() / (2.91_f64).ln() - 0.5).floor() as u32,
        _ => unimplemented!(),
    }
}

/// The expansion factor in the unsharded fuse-graphs regime.
fn fuse_c(arity: usize, n: usize) -> f64 {
    // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
    // https://doi.org/10.1145/3510449
    match arity {
        3 => 1.125_f64.max(0.875 + 0.25 * (1000000_f64).ln() / (n as f64).ln()),
        4 => 1.075_f64.max(0.77 + 0.305 * (600000_f64).ln() / (n as f64).ln()),
        _ => unimplemented!(),
    }
}

/// Static functions with 10%-11% space overhead for large key sets, fast
/// parallel construction, and fast queries.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`).
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFunc<
    T: ?Sized + ToSig<S>,
    W: ZeroCopy + Word = usize,
    D: BitFieldSlice<W> = BitFieldVec<W>,
    S: Sig = [u64; 2],
    E: ShardEdge<S, 3> = FuseShards,
> {
    pub(in crate::func) shard_edge: E,
    pub(in crate::func) seed: u64,
    pub(in crate::func) num_keys: usize,
    pub(in crate::func) data: D,
    pub(in crate::func) _marker_t: std::marker::PhantomData<T>,
    pub(in crate::func) _marker_w: std::marker::PhantomData<W>,
    pub(in crate::func) _marker_s: std::marker::PhantomData<S>,
}

/// Filters with 10%-11% space overhead for large key sets, fast parallel
/// construction, and fast queries.
///
/// Instances of this structure are immutable; they are built using a
/// [`VBuilder`](crate::func::VBuilder) and can be serialized using
/// [ε-serde](`epserde`).
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFilter<W: ZeroCopy + Word, F> {
    pub(in crate::func) func: F,
    pub(in crate::func) filter_mask: W,
}

/// Shard and edge information.
///
/// This trait is used to derive shards and edges from key signatures. The
/// `shard` method returns the shard; the `edge` method is used to derive the
/// three vertices of the edge associated with the key; the `shard_edge` method
/// is used to derive the three vertices of the edge associated with a key
/// inside the shard the key belongs to (its default implementation just calls
/// `edge` with shard zero).
///
/// Edges must follow the generation method of fuse graphs (see ”[Dense Peelable
/// Random Uniform Hypergraphs](https://doi.org/10.4230/LIPIcs.ESA.2019.38)”.
/// Using the signature bits as a seed, one extracts a pseudorandom first
/// segment between 0 and `l`, and then chooses a pseudorandom position in the
/// first segment and in the following two (or more, in the case of higher
/// degree).
pub trait ShardEdge<S, const K: usize>:
    Default + Display + Clone + Copy + Send + Sync + SerializeInner + DeserializeInner + TypeHash + AlignHash
{
    fn set_up_shards(&mut self, n: usize) -> (f64, bool);

    fn shard(&self, sig: &S) -> usize;

    fn local_edge(&self, sig: &S) -> [usize; K];

    fn edge(&self, sig: &S) -> [usize; K];

    fn shard_high_bits(&self) -> u32;

    fn num_shards(&self) -> usize {
        1 << self.shard_high_bits()
    }

    fn num_vertices(&self) -> usize;
}

#[derive(Epserde, Default, Debug, MemDbg, MemSize, Clone, Copy)]
#[deep_copy]
pub struct MwhcShards {
    shard_high_bits: u32,
    shard_mask: u32,
    seg_size: usize,
}

impl Display for MwhcShards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MWHC (shards) Number of shards: 2^{} Segment size: {}", self.shard_high_bits, self.seg_size)
    }
}

impl MwhcShards {
    #[inline(always)]
    fn _edge_1(&self, shard: usize, sig: &[u64; 1]) -> [usize; 3] {
        let seg_size: u64 = self.seg_size as _;
        let start = shard as u64 * self.seg_size as u64 * 3;
        let xor = sig[0] as u32 ^ (sig[0] >> 32) as u32;

        [
            (((sig[0] as u32 as u64 * seg_size) >> 32) + start) as _,
            ((((sig[0] >> 32) * seg_size) >> 32) + start + seg_size) as _,
            ((((xor as u64) * seg_size) >> 32) + start + 2 * seg_size) as _,
        ]
    }

    #[inline(always)]
    fn _edge_2(&self, shard: usize, sig: &[u64; 2]) -> [usize; 3] {
        let seg_size = self.seg_size as u64;
        let start = shard as u64 * seg_size * 3;

        [
            (((sig[0] as u32 as u64 * seg_size) >> 32) + start) as _,
            ((((sig[1] >> 32) * seg_size) >> 32) + start + seg_size) as _,
            ((((sig[1] as u32 as u64) * seg_size) >> 32) + start + 2 * seg_size) as _,
        ]
    }
}
#[derive(Epserde, Default, Debug, MemDbg, MemSize, Clone, Copy)]
#[deep_copy]
pub struct MwhcNoShards {
    seg_size: usize,
}

impl Display for MwhcNoShards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MWHC (no shards) Segment size: {}", self.seg_size)
    }
}

impl ShardEdge<[u64; 2], 3> for MwhcShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        let eps = 0.001; // Tolerance for deviation from the average shard size
        self.shard_high_bits = {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - t.ln()) / 2_f64.ln()).ceil().max(0.) as u32
            } else {
                0
            }
        };

        self.shard_mask = (1u32 << self.shard_high_bits) - 1;
        let num_shards = 1 << self.shard_high_bits;
        self.seg_size = ((n as f64 * 1.23).ceil() as usize)
            .div_ceil(num_shards)
            .div_ceil(3);

        (1.23, false)
    }

    #[inline(always)]
    fn shard(&self, sig: &[u64; 2]) -> usize {
        (sig[0].rotate_left(self.shard_high_bits) & self.shard_mask as u64) as usize
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        self._edge_2(0, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        self._edge_2(self.shard(sig), sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        self.shard_high_bits
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        self.seg_size * 3
    }
}

impl ShardEdge<[u64; 1], 3> for MwhcShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        let eps = 0.001; // Tolerance for deviation from the average shard size
        self.shard_high_bits = {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - t.ln()) / 2_f64.ln()).ceil() as u32
            } else {
                0
            }
        };

        self.shard_mask = (1u32 << self.shard_high_bits) - 1;
        let num_shards = 1 << self.shard_high_bits;
        self.seg_size = ((n as f64 * 1.23).ceil() as usize)
            .div_ceil(num_shards)
            .div_ceil(3);

        (1.23, false)
    }

    #[inline(always)]
    fn shard(&self, sig: &[u64; 1]) -> usize {
        // This must work even when shard_high_bits is zero
        let xor = sig[0] as u32 ^ (sig[0] >> 32) as u32;
        (xor.rotate_left(self.shard_high_bits) & self.shard_mask) as usize
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        self._edge_1(0, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        self._edge_1(self.shard(sig), sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        self.shard_high_bits
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        self.seg_size * 3
    }
}

impl ShardEdge<[u64; 2], 3> for MwhcNoShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        self.seg_size = ((n as f64 * 1.23).ceil() as usize).div_ceil(3);
        (1.23, false)
    }

    #[inline(always)]
    fn shard(&self, _sig: &[u64; 2]) -> usize {
        0
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        let seg_size: u64 = self.seg_size as _;

        [
            ((sig[0] as u32 as u64 * seg_size) >> 32) as _,
            (((sig[1] >> 32) * seg_size >> 32) + seg_size) as _,
            (((sig[1] as u32 as u64) * seg_size >> 32) + 2 * seg_size) as _,
        ]
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        self.local_edge(sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        self.seg_size * 3
    }
}

impl ShardEdge<[u64; 1], 3> for MwhcNoShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        self.seg_size = ((n as f64 * 1.23).ceil() as usize).div_ceil(3);
        (1.23, false)
    }

    #[inline(always)]
    fn shard(&self, _sig: &[u64; 1]) -> usize {
        0
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        let seg_size = self.seg_size as u64;
        let xor = sig[0] as u32 ^ (sig[0] >> 32) as u32;

        [
            ((sig[0] as u32 as u64 * seg_size) >> 32) as _,
            ((((sig[0] >> 32) * seg_size) >> 32) + seg_size) as _,
            ((((xor as u64) * seg_size) >> 32) + 2 * seg_size) as _,
        ]
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        self.local_edge(sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        self.seg_size * 3
    }
}

#[derive(Epserde, Default, Debug, MemDbg, MemSize, Clone, Copy)]
#[deep_copy]
pub struct FuseShards {
    shard_high_bits: u32,
    shard_mask: u32,
    log2_seg_size: u32,
    l: u32,
}

impl Display for FuseShards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fuse (shards) Number of shards: 2^{} Segment size: 2^{} Number of segments: {}", self.shard_high_bits, self.log2_seg_size, self.l + 2)
    }
}

#[inline(always)]
fn fuse_edge_2(
    shard: usize,
    shard_high_bits: u32,
    log2_seg_size: u32,
    l: u32,
    sig: &[u64; 2],
) -> [usize; 3] {
    let first_segment = (((sig[0] << shard_high_bits >> 32) * l as u64) >> 32) as usize;
    let shard_offset = shard * ((l as usize + 2) << log2_seg_size);
    let start = shard_offset + (first_segment << log2_seg_size);
    let segment_size = 1 << log2_seg_size;
    let segment_mask = segment_size - 1;

    [
        (sig[0] as usize & segment_mask) + start,
        ((sig[1] >> 32) as usize & segment_mask) + start + segment_size,
        (sig[1] as usize & segment_mask) + start + 2 * segment_size,
    ]
}

#[inline(always)]
fn fuse_edge_1(
    shard: usize,
    shard_high_bits: u32,
    log2_seg_size: u32,
    l: u32,
    sig: &[u64; 1],
) -> [usize; 3] {
    let xor = sig[0] as u32 ^ (sig[0] >> 32) as u32;
    let first_segment = ((xor.rotate_left(shard_high_bits) as u64 * l as u64) >> 32) as usize;
    let shard_offset = shard * ((l as usize + 2) << log2_seg_size);
    let start = shard_offset + (first_segment << log2_seg_size);
    let segment_size = 1 << log2_seg_size;
    let segment_mask = segment_size - 1;

    [
        (sig[0] as usize & segment_mask) + start,
        ((sig[0] >> 21) as usize & segment_mask) + start + segment_size,
        ((sig[0] >> 42) as usize & segment_mask) + start + 2 * segment_size,
    ]
}

impl FuseShards {
    const MAX_LIN_SIZE: usize = 1_000_000;
    const MAX_LIN_SHARD_SIZE: usize = 100_000;
    const MIN_FUSE_SHARD: usize = 10_000_000;
    const LOG2_MAX_SHARDS: u32 = 10;
}

/// In this implementation:
/// - the `shard_high_bits` most significant bits of the first component are
///   used to select a shard;
/// - the following 32 bits of the first component are used to select the first
///   segment using fixed-point arithmetic;
/// - the lower 32 bits of first component are used to select the first vertex;
/// - the upper 32 bits of the second component are used to select the second
///   vertex;
/// - the lower 32 bits of the second component are used to select the third
///   vertex.
///
/// Note that the lower `shard_high_bits` of the bits used to select the first
/// segment are the same as the upper `shard_high_bits` of the bits used to
/// select the first segment, but being the result mostly sensitive to the high
/// bits, this is not a problem.
impl ShardEdge<[u64; 2], 3> for FuseShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        let eps = 0.001; // Tolerance for deviation from the average shard size
        self.shard_high_bits = if n <= Self::MAX_LIN_SIZE {
            // We just try to make shards as big as possible,
            // within a maximum size of 2 * MAX_LIN_SHARD_SIZE
            (n / Self::MAX_LIN_SHARD_SIZE).max(1).ilog2()
        } else {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                // We correct the estimate to increase slightly the shard size
                ((t - 1.92 * t.ln() - 1.22 * t.ln().ln()) / 2_f64.ln())
                    .ceil()
                    .max(3.) as u32
            } else {
                0
            }
            .min(Self::LOG2_MAX_SHARDS) // We don't really need too many shards
            .min((n / Self::MIN_FUSE_SHARD).max(1).ilog2()) // Shards can't smaller than MIN_FUSE_SHARD
        };

        self.shard_mask = (1u32 << self.shard_high_bits) - 1;
        let num_shards = 1 << self.shard_high_bits;

        let shard_size = n.div_ceil(num_shards);
        let lazy_gaussian = n <= Self::MAX_LIN_SIZE;

        let c;
        (c, self.log2_seg_size) = if lazy_gaussian {
            (1.10, lin_log2_seg_size(3, shard_size))
        } else {
            (1.105, fuse_log2_seg_size(3, shard_size))
        };

        self.l = ((c * shard_size as f64).ceil() as usize)
            .div_ceil(1 << self.log2_seg_size)
            .try_into()
            .unwrap();

        (c, lazy_gaussian)
    }

    #[inline(always)]
    fn shard(&self, sig: &[u64; 2]) -> usize {
        // This must work even when shard_high_bits is zero
        (sig[0].rotate_left(self.shard_high_bits) & self.shard_mask as u64) as usize
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        fuse_edge_2(0, self.shard_high_bits, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        fuse_edge_2(self.shard(sig), self.shard_high_bits, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        self.shard_high_bits
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        (self.l as usize + 2) << self.log2_seg_size
    }
}

/// In this implementation:
/// - the `shard_high_bits` most significant bits of the two 32-bit halves XOR'd
///   together are used to select a shard;
/// - the two 32-bit halves XOR'd together and rotated to the left by
///   `shard_high_bits` are used to select the first segment using fixed-point
///   arithmetic;
/// - the lower 21 bits are used to select the first vertex;
/// - the next 21 bits are used to select the second vertex;
/// - the next 21 bits are used to select the third vertex.
///
/// Note that the lower `shard_high_bits` of the bits used to select the first
/// segment are the same as the bits used to select the shard, but being the
/// result mostly sensitive to the high bits, this is not a problem.
impl ShardEdge<[u64; 1], 3> for FuseShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        let eps = 0.001; // Tolerance for deviation from the average shard size
        self.shard_high_bits = if n <= Self::MAX_LIN_SIZE {
            // We just try to make shards as big as possible,
            // within a maximum size of 2 * MAX_LIN_SHARD_SIZE
            (n / Self::MAX_LIN_SHARD_SIZE).max(1).ilog2()
        } else {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                // We correct the estimate to increase slightly the shard size
                ((t - 1.92 * t.ln() - 1.22 * t.ln().ln()) / 2_f64.ln())
                    .ceil()
                    .max(3.) as u32
            } else {
                0
            }
            .min(Self::LOG2_MAX_SHARDS) // We don't really need too many shards
            .min((n / Self::MIN_FUSE_SHARD).max(1).ilog2()) // Shards can't smaller than MIN_FUSE_SHARD
        };

        self.shard_mask = (1u32 << self.shard_high_bits) - 1;

        let num_shards = 1 << self.shard_high_bits;

        let shard_size = n.div_ceil(num_shards);
        let lazy_gaussian = n <= Self::MAX_LIN_SIZE;

        let c;
        (c, self.log2_seg_size) = if lazy_gaussian {
            (1.10, lin_log2_seg_size(3, shard_size))
        } else {
            (1.105, fuse_log2_seg_size(3, shard_size))
        };

        self.l = ((c * shard_size as f64).ceil() as usize)
            .div_ceil(1 << self.log2_seg_size)
            .try_into()
            .unwrap();

        (c, lazy_gaussian)
    }

    #[inline(always)]
    fn shard(&self, sig: &[u64; 1]) -> usize {
        // This must work even when shard_high_bits is zero
        let xor = sig[0] as u32 ^ (sig[0] >> 32) as u32;
        (xor.rotate_left(self.shard_high_bits) & self.shard_mask) as usize
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        fuse_edge_1(0, self.shard_high_bits, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        fuse_edge_1(self.shard(sig), self.shard_high_bits, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        self.shard_high_bits
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        (self.l as usize + 2) << self.log2_seg_size
    }
}

#[derive(Epserde, Default, Debug, MemDbg, MemSize, Clone, Copy)]
#[deep_copy]
pub struct FuseNoShards {
    log2_seg_size: u32,
    l: u32,
}

impl Display for FuseNoShards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fuse (no shards) Segment size: 2^{} Number of segments: {}", self.log2_seg_size, self.l + 2)
    }
}


impl ShardEdge<[u64; 2], 3> for FuseNoShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        let c = if n <= FuseShards::MAX_LIN_SIZE {
            1.105
        } else {
            fuse_c(3, n)
        };

        self.log2_seg_size = fuse_log2_seg_size(3, n);
        self.l = ((c * n as f64).ceil() as u64)
            .div_ceil(1 << self.log2_seg_size)
            .try_into()
            .unwrap();

        (c, false)
    }

    #[inline(always)]
    fn shard(&self, _sig: &[u64; 2]) -> usize {
        0
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        fuse_edge_2(0, 0, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 2]) -> [usize; 3] {
        self.local_edge(sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        (self.l as usize + 2) << self.log2_seg_size
    }
}

impl ShardEdge<[u64; 1], 3> for FuseNoShards {
    fn set_up_shards(&mut self, n: usize) -> (f64, bool) {
        let c = if n < FuseShards::MIN_FUSE_SHARD {
            1.105
        } else {
            fuse_c(3, n)
        };

        self.log2_seg_size = fuse_log2_seg_size(3, n);
        self.l = ((c * n as f64).ceil() as u64)
            .div_ceil(1 << self.log2_seg_size)
            .try_into()
            .unwrap();

        (c, false)
    }

    #[inline(always)]
    fn shard(&self, _sig: &[u64; 1]) -> usize {
        0
    }

    #[inline(always)]
    fn local_edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        fuse_edge_1(0, 0, self.log2_seg_size, self.l, sig)
    }

    #[inline(always)]
    fn edge(&self, sig: &[u64; 1]) -> [usize; 3] {
        self.local_edge(sig)
    }

    #[inline(always)]
    fn shard_high_bits(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn num_vertices(&self) -> usize {
        (self.l as usize + 2) << self.log2_seg_size
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
    pub fn get_by_sig(&self, sig: &S) -> W {
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
        self.get_by_sig(&T::to_sig(key, self.seed))
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

impl<T: ?Sized + ToSig<S>, W: ZeroCopy + Word, D: BitFieldSlice<W>, S: Sig, E: ShardEdge<S, 3>>
    VFilter<W, VFunc<T, W, D, S, E>>
where
    u64: CastableInto<W>,
{
    /// Return the value associated with the given signature by the underlying
    /// function, or a random value if the signature is not the signature of a
    /// key .
    ///
    /// The user should not normally call this method, but rather
    /// [`contains_by_sig`](VFilter::contains_by_sig).
    #[inline(always)]
    pub fn get_by_sig(&self, sig: &S) -> W {
        self.func.get_by_sig(sig)
    }

    /// Return the value associated with the given key by the underlying
    /// function, or a random value if the key is not present.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline]
    pub fn get(&self, key: &T) -> W {
        self.func.get(key)
    }

    /// Return whether a signature is contained in the filter.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline(always)]
    pub fn contains_by_sig(&self, sig: &S) -> bool {
        self.func.get_by_sig(sig) == sig.sig_u64().cast() & self.filter_mask
    }

    /// Return whether a key is contained in the filter.
    ///
    /// The user should not normally call this method, but rather
    /// [`contains`](VFilter::contains).
    #[inline]
    pub fn contains(&self, key: &T) -> bool {
        self.contains_by_sig(&T::to_sig(key, self.func.seed))
    }

    /// Return the number of keys in the filter.
    pub fn len(&self) -> usize {
        self.func.len()
    }

    /// Return whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty()
    }
}

impl<
        T: ?Sized + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W>,
        S: Sig,
        E: ShardEdge<S, 3>,
        B: Borrow<T>,
    > Index<B> for VFilter<W, VFunc<T, W, D, S, E>>
where
    u64: CastableInto<W>,
{
    type Output = bool;

    #[inline(always)]
    fn index(&self, key: B) -> &Self::Output {
        if self.contains(key.borrow()) {
            &true
        } else {
            &false
        }
    }
}

#[cfg(test)]
mod tests {
    use dsi_progress_logger::no_logging;
    use epserde::{
        deser::Deserialize,
        ser::Serialize,
        traits::TypeHash,
        utils::AlignedCursor,
    };
    use rdst::RadixKey;

    use crate::{
        func::VBuilder,
        utils::{FromIntoIterator, Sig, SigVal, ToSig},
    };

    use super::{fuse_log2_seg_size, FuseShards, ShardEdge, VFilter, VFunc};

    #[test]
    fn test_filter_func() -> anyhow::Result<()> {
        _test_filter_func::<[u64; 2]>()?;
        _test_filter_func::<[u64; 2]>()?;
        Ok(())
    }

    fn _test_filter_func<S: Sig + Send + Sync>() -> anyhow::Result<()>
    where
        usize: ToSig<S>,
        SigVal<S, ()>: RadixKey,
        FuseShards: ShardEdge<S, 3>,
        VFunc<usize, u8, Vec<u8>, S, FuseShards>: Serialize + TypeHash, // Weird
        VFilter<u8, VFunc<usize, u8, Vec<u8>, S, FuseShards>>: Serialize,
    {
        for n in [0_usize, 10, 1000, 100_000, 3_000_000] {
            let filter = VBuilder::<_, _, Vec<u8>, S, FuseShards, ()>::default()
                .log2_buckets(4)
                .offline(false)
                .try_build_filter(FromIntoIterator::from(0..n), no_logging![])?;
            let mut cursor = <AlignedCursor<maligned::A16>>::new();
            filter.serialize(&mut cursor).unwrap();
            cursor.set_position(0);
            // TODO: This does not work with deserialize_eps
            let filter =
                VFilter::<u8, VFunc<_, _, Vec<u8>, S, FuseShards>>::deserialize_full(&mut cursor)?;
            for i in 0..n {
                let sig = ToSig::<S>::to_sig(&i, filter.func.seed);
                assert_eq!(sig.sig_u64() & 0xFF, filter.get(&i) as u64);
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "slow_tests")]
    fn test_lin_log2_seg_size() -> anyhow::Result<()> {
        use super::*;
        // Manually tested values:
        // 200000 -> 9
        // 150000 -> 9
        // 112500 -> 9
        // 100000 -> 8
        // 75000 -> 8
        // 50000 -> 8
        // 40000 -> 8
        // 35000 -> 7
        // 30000 -> 7
        // 25000 -> 7
        // 12500 -> 7
        // 10000 -> 6
        // 1000 -> 5
        // 100 -> 3
        // 10 -> 2

        for n in 0..MAX_LIN_SHARD_SIZE * 2 {
            if lin_log2_seg_size(3, n) != lin_log2_seg_size(3, n + 1) {
                eprintln!(
                    "Bulding function with {} keys (log₂ segment size = {})...",
                    n,
                    lin_log2_seg_size(3, n)
                );
                let _func = VBuilder::<_, _, BitFieldVec<_>, [u64; 2], true>::default().build(
                    FromIntoIterator::from(0..n),
                    FromIntoIterator::from(0_usize..),
                    no_logging![],
                )?;

                eprintln!(
                    "Bulding function with {} keys (log₂ segment size = {})...",
                    n + 1,
                    lin_log2_seg_size(3, n + 1)
                );
                let _func = VBuilder::<_, _, BitFieldVec<_>, [u64; 2], true>::default().build(
                    FromIntoIterator::from(0..n + 1),
                    FromIntoIterator::from(0_usize..),
                    no_logging![],
                )?;
            }
        }

        Ok(())
    }

    #[test]
    fn compare_funcs() {
        let eps = 0.001;
        let bound = |n: usize, corr: f64| {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - t.ln()) / 2_f64.ln() / corr).ceil() as u32
            } else {
                0
            }
        };

        let bound2 = |n: usize, corr0: f64, corr1: f64| {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - corr0 * t.ln() - corr1 * t.ln().ln()) / 2_f64.ln())
                    .ceil()
                    .max(2.) as u32
            } else {
                0
            }
        };

        let mut t = 1024;
        for _ in 0..50 {
            if t >= FuseShards::MIN_FUSE_SHARD {
                eprintln!(
                    "n: {t}, 1.1: {} 1.5: {} bound2(2.05, 0): {} bound2(2, 1): {}",
                    bound(t, 1.1),
                    bound(t, 1.5),
                    bound2(t, 2.05, 0.),
                    bound2(t, 1.7, 1.)
                );
            }
            t = t * 3 / 2;
        }
    }

    #[test]
    fn search_funcs() {
        let data_points = vec![
            (11491938_usize, 2),
            (17237907, 2),
            (25856860, 3),
            (38785290, 3),
            (58177935, 3),
            (87266902, 4),
            (130900353, 4),
            (196350529, 4),
            (294525793, 4),
            (441788689, 4),
            (662683033, 4),
            (994024549, 4),
            (1491036823, 4),
            (2236555234, 4),
            (3354832851, 5),
            (5032249276, 5),
            (7548373914, 5),
            (11322560871, 6),
            (16983841306, 6),
            (25475761959, 6),
            (38213642938, 7),
            (57320464407, 7),
            (85980696610, 8),
            (128971044915, 8),
            (193456567372, 9),
            (290184851058, 9),
            (435277276587, 10),
        ];

        let eps = 0.001;

        let bound = |n: usize, corr0: f64, corr1: f64| {
            // Bound from urns and balls problem
            let t = (n as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - corr0 * t.ln() - corr1 * t.ln().ln()) / 2_f64.ln())
                    .ceil()
                    .max(2.) as u32
            } else {
                0
            }
        };

        let mut max_err = usize::MAX;
        let mut _best_corr0 = 0.0;
        let mut _best_corr1 = 0.0;

        for corr0 in 0..300 {
            for corr1 in 0..300 {
                let corr0 = corr0 as f64 / 100.0;
                let corr1 = corr1 as f64 / 100.0;
                let mut err = 0;
                for &(n, log2) in data_points.iter() {
                    let log2 = log2;

                    let bound = bound(n, corr0, corr1) as usize;
                    if log2 != bound {
                        if bound > log2 {
                            err += (bound - log2) * 2;
                        } else {
                            err += log2 - bound;
                        }
                    }
                }

                if err < max_err {
                    eprintln!("corr0: {} corr1: {} err: {}", corr0, corr1, err);
                    max_err = err;
                    _best_corr0 = corr0;
                    _best_corr1 = corr1;

                    for &(n, log2) in data_points.iter() {
                        let log2 = log2;

                        let bound = bound(n, _best_corr0, _best_corr1) as usize;
                        eprintln!("n: {} log2: {} bound: {}", n, log2, bound);
                    }
                }
            }
        }
    }

    fn log2_seg_size(n: usize) -> u32 {
        fuse_log2_seg_size(3, n)
    }

    #[test]
    fn test_log2_seg_size() {
        let mut shard_size = 1024;
        for _ in 0..50 {
            if shard_size >= 1_000_000 {
                let l2ss = log2_seg_size(shard_size);
                let c = 1.105;
                let l = ((c * shard_size as f64).ceil() as usize).div_ceil(1 << l2ss);
                let ideal_num_vertices = c * shard_size as f64;
                let num_vertices = (1 << l2ss) * (l + 2);
                eprintln!(
                    "n: {shard_size} log₂ seg size: {} ideal: {} actual m: {} ratio: {}",
                    l2ss,
                    ideal_num_vertices,
                    num_vertices,
                    100.0 * num_vertices as f64 / ideal_num_vertices
                );
            }
            shard_size = shard_size * 3 / 2;
        }
    }
}
