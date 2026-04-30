/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::bits::{BitVec, BitVecU, test_unaligned_any_pos};
use crate::func::VBuilder;
use crate::func::codec::{Codec, Coder, Decoder, HuffmanCoder, HuffmanConf, HuffmanDecoder};
use crate::func::peeling::{DoubleStack, FastStack, XorGraph, remove_edge};
use crate::func::shard_edge::{Fuse3Shards, ShardEdge};
use crate::traits::bit_vec_ops::{BitVecOps, BitVecOpsMut};
use crate::traits::{
    Backend, BitLength, BitVecValueOps, TryIntoUnaligned, UnalignedConversionError, Word,
};
use crate::utils::sig_store::ShardStore;
use crate::utils::{BinSafe, FallibleRewindableLender, Sig, SigVal, ToSig};
use anyhow::{Result, anyhow, bail};
use core::error::Error;
use dsi_progress_logger::ProgressLog;
use lender::FallibleLending;
use mem_dbg::{FlatType, MemDbg, MemSize, SizeFlags};
use num_primitive::PrimitiveNumber;
use rdst::RadixKey;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

// ── CompVFunc struct ────────────────────────────────────────────────

/// A static function whose values are stored in compressed form.
///
/// [`CompVFunc`] maps `n` keys to values of type `D::Word`, representing each
/// value with a [prefix-free codeword] (by default a length-limited [Huffman
/// code]) and storing the codewords by peeling a fuse graph, as in the case of
/// [`VFunc`].
///
/// When the value distribution is skewed, this uses much less space than
/// [`VFunc`]: roughly the empirical entropy of the value list plus ≈11%
/// overhead. This estimate, however, does not take into consideration the
/// storage of the decoder, which can be significant when the number of
/// keys is very small.
///
/// Instances of this structure are immutable, except for the [`branchless`]
/// setting of the internal decoder they are built using [`try_new`] or one of
/// its variants, and can be serialized using [ε-serde] or [`serde`].
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Implementation notes
///
/// This structure follows roughly the design described by Marco Genuzio,
/// Giuseppe Ottaviano, and Sebastiano Vigna in “[Fast scalable construction of
/// (\[compressed\] static | minimal perfect hash)
/// functions](https://doi.org/10.1016/j.ic.2020.104517)”, Information and
/// Computation, 273:104517, 2020, but is based on [`VFunc`] rather than
/// on the static functions described in the paper.
///
/// # Generics
///
/// * `K` - the key type.
///
/// * `D` - the data backend: the output value type is [`D::Word`]; defaults to
///   [`BitVec<Box<[usize]>>`](crate::bits::BitVec).
///
/// * `S` - the signature type; defaults to `[u64; 2]` (see [`VFunc` for
///   details).
///
/// * `E` - the [`ShardEdge`] used for sharding and local hashing;
///   defaults to [`Fuse3Shards`] (see [`VFunc`] for details; note that
///   variants using Lazy Gaussian Elimination (LGE) are not supported).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use sux::func::CompVFunc;
/// # use dsi_progress_logger::no_logging;
/// # use sux::utils::FromCloneableIntoIterator;
/// # use mem_dbg::{MemSize, SizeFlags};
/// // Maps keys to their trailing zeros, which are distributed geometrically
/// let func = <CompVFunc<usize>>::try_new(
///     FromCloneableIntoIterator::new(0..100_usize),
///     FromCloneableIntoIterator::new((1..101_usize).map(|x| x.trailing_zeros() as usize)),
///     no_logging![],
/// )?;
///
/// for i in 0..100 {
///     assert_eq!(func.get(i), (i + 1).trailing_zeros() as usize);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// [`VFunc`]: crate::func::VFunc
/// [`VBuilder`]: crate::func::VBuilder
/// [`ShardEdge`]: crate::func::shard_edge::ShardEdge
/// [`ToSig`]: crate::utils::ToSig
/// [`try_new`]: Self::try_new
/// [ε-serde]: https://docs.rs/epserde/latest/epserde/
/// [`serde`]: https://crates.io/crates/serde
/// [`D::Word`]: Backend::Word
/// [`branchless`]: Self::branchless
/// [prefix-free codeword]: crate::func::codec::Codec
/// [Huffman code]: crate::func::codec::HuffmanConf
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(
    feature = "epserde",
    epserde(bound(
        deser = "D::Word: for<'a> epserde::deser::DeserInner<DeserType<'a> = D::Word>, \
                 for<'a> <D as epserde::deser::DeserInner>::DeserType<'a>: Backend<Word = D::Word>"
    ))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D: serde::Serialize, D::Word: serde::Serialize, E: serde::Serialize",
        deserialize = "D: serde::Deserialize<'de>, D::Word: serde::Deserialize<'de>, E: serde::Deserialize<'de>"
    ))
)]
pub struct CompVFunc<K: ?Sized, D: Backend = BitVec<Box<[usize]>>, S = [u64; 2], E = Fuse3Shards> {
    pub(crate) shard_edge: E,
    pub(crate) seed: u64,
    pub(crate) num_keys: usize,
    pub(crate) shard_size: usize,
    pub(crate) codeword_mask: D::Word,
    pub(crate) data: D,
    pub(crate) decoder: HuffmanDecoder<D::Word>,
    pub(crate) _marker: PhantomData<(*const K, S)>,
}

// ── Query path ──────────────────────────────────────────────────────

impl<
    K: ?Sized + ToSig<S>,
    D: Backend<Word: Word + BinSafe> + AsRef<[D::Word]> + BitLength + BitVecValueOps<D::Word>,
    S: Sig,
    E: ShardEdge<S, 3>,
> CompVFunc<K, D, S, E>
{
    /// Returns the value associated with `key`, or an arbitrary value
    /// if `key` is not in the original key set.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> D::Word {
        self.get_by_sig(K::to_sig(key.borrow(), self.seed))
    }

    /// Returns the value associated with the given signature, or an
    /// arbitrary value if no key has that signature.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: S) -> D::Word {
        // Here we compute manually the edge vertices. We cannot use
        // ShardEdge::edge because its edge computation derives the number of
        // vertices from the segment size, whereas here we have to includes a
        // bit padding and a rounding to a usize::BITS multiple, which is why we
        // cache the value in shard_size.
        let shard = self.shard_edge.shard(sig);
        let shard_offset = shard * self.shard_size;
        let esym_len = self.decoder.escaped_symbols_len() as usize;
        let local_sig = self.shard_edge.local_sig(sig);
        let local_edge = self.shard_edge.local_edge(local_sig);

        let v0 = shard_offset + local_edge[0];
        let v1 = shard_offset + local_edge[1];
        let v2 = shard_offset + local_edge[2];
        // The codeword is stored at the high end of the per-key layout
        // (offsets [esym_len..esym_len + w)), so we read at v + esym_len.
        // SAFETY: the bit vector is padded.
        let value = unsafe {
            (self.data.get_unaligned_unchecked(v0 + esym_len)
                ^ self.data.get_unaligned_unchecked(v1 + esym_len)
                ^ self.data.get_unaligned_unchecked(v2 + esym_len))
                & self.codeword_mask
        };
        if let Some(decoded) = self.decoder.decode(value.as_to::<usize>()) {
            return decoded;
        }
        // The literal sits at the low end [0..esym_len).
        // SAFETY: the bit vector is padded.
        unsafe {
            self.data.get_value_unchecked(v0, esym_len)
                ^ self.data.get_value_unchecked(v1, esym_len)
                ^ self.data.get_value_unchecked(v2, esym_len)
        }
    }
}

impl<K: ?Sized, D: Backend<Word: Word>, S, E> CompVFunc<K, D, S, E> {
    /// Number of keys in the function.
    pub const fn len(&self) -> usize {
        self.num_keys
    }

    /// Whether the function has no keys.
    pub const fn is_empty(&self) -> bool {
        self.num_keys == 0
    }

    /// Provides access to the underlying [`HuffmanDecoder`].
    ///
    /// This is useful for inspecting the code properties (e.g., max codeword
    /// length, length of escaped symbols, etc.).
    ///
    /// If you need to set at runtime a branchy or branchless decoding strategy,
    /// please use [`branchless`]
    ///
    /// [`branchless`]: Self::branchless
    pub fn decoder(&self) -> &HuffmanDecoder<D::Word> {
        &self.decoder
    }

    /// Sets at runtime a branchy or branchless decoding strategy.
    ///
    /// Delegates to the underlying decoder [`branchless`] method.
    ///
    /// [`branchless`]: HuffmanDecoder::branchless
    pub fn branchless(&mut self, branchless: bool) {
        self.decoder.branchless(branchless);
    }
}

// ── Entry points ────────────────────────────────────────────────────
//
// CompVFunc shares its parallel infrastructure with
// [`VBuilder`](crate::func::VBuilder): callers pass a
// VBuilder<BitVec<Box<[W]>>, S, E> configured with the usual VBuilder knobs
// (offline, check-dups, low-mem, threads, eps, seed). The only
// CompVFunc-specific configuration is the Huffman codec used for values; the
// default is unlimited-length Huffman.
//
// Both build- and query-side edge generation go through
// ShardEdge::local_edge(shard_edge.local_sig(sig)), which provides fuse-graph
// structured base positions in [0..num_vertices). The multi-edge layout then
// adds (w − 1 − l) per codeword bit to each base position.

impl<K, W, S, E> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe + Hash + MemSize + FlatType,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
    SigVal<S, W>: RadixKey,
    BitVec<Box<[W]>>: MemSize + FlatType,
    HuffmanDecoder<W>: MemSize + FlatType,
{
    /// Builds a [`CompVFunc`] from keys and values using a default
    /// [`HuffmanConf`] and [`VBuilder`]
    ///
    /// Keys and values are consumed one at a time through their respective
    /// lenders; this path is the right choice for input coming from disk or
    /// synthetic ranges. Neither the key set nor the value set needs to live in
    /// memory at once: the values lender is rewound at least once during
    /// construction.
    ///
    /// This is a convenience wrapper around [`try_new_with_builder`] with
    /// `VBuilder::default()`.
    ///
    /// If keys and values are available as slices and you want to parallelize
    /// the hashing phase, use [`try_par_new`](Self::try_par_new) instead.
    ///
    /// [`try_new_with_builder`]: Self::try_new_with_builder
    pub fn try_new<B: ?Sized + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, values, HuffmanConf::new(), VBuilder::default(), pl)
    }

    /// Builds a [`CompVFunc`] from lenders of keys and values using
    /// the given [`HuffmanConf`] and [`VBuilder`].
    ///
    /// See [`try_new`](Self::try_new) for the streaming semantics.
    /// The `builder` argument controls every VBuilder-side
    /// construction knob (offline mode, thread count, sharding ε,
    /// PRNG seed, etc.); the [`HuffmanConf`] argument controls the codec
    /// used for the values.
    pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
        huffman: HuffmanConf,
        builder: VBuilder<BitVec<Box<[W]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        build_inner_seq::<K, W, B, _, _, _, S, E>(huffman, builder, keys, values, pl)
    }
}

impl<K, W, S, E> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    W: Word + BinSafe + Hash + MemSize + FlatType,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
    SigVal<S, W>: RadixKey,
    BitVec<Box<[W]>>: MemSize + FlatType,
    HuffmanDecoder<W>: MemSize + FlatType,
{
    /// Builds a [`CompVFunc`] from parallel slices of keys and values using
    /// default [`VBuilder`] and [`HuffmanConf`].
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new`] instead.
    ///
    /// This is a convenience wrapper around [`try_par_new_with_builder`] with
    /// `VBuilder::default()`.
    ///
    /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
    /// [`try_new`]: Self::try_new
    pub fn try_par_new<B: Borrow<K> + Sync>(
        keys: &[B],
        values: &[W],
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_par_new_with_builder(keys, values, HuffmanConf::new(), VBuilder::default(), pl)
    }

    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using the given [`HuffmanConf`] and [`VBuilder`].
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new_with_builder`] instead.
    ///
    /// [`try_new_with_builder`]: Self::try_new_with_builder
    pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
        keys: &[B],
        values: &[W],
        huffman: HuffmanConf,
        builder: VBuilder<BitVec<Box<[W]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if keys.len() != values.len() {
            bail!(
                "keys and values must have the same length ({} vs {})",
                keys.len(),
                values.len()
            );
        }
        if keys.is_empty() {
            return Ok(empty_comp_vfunc::<K, W, S, E>());
        }
        build_inner_par::<K, W, B, _, S, E>(huffman, builder, keys, values, pl)
    }
}

// ── Builder core ───────────────────────────────────────────────────
//
// Both `build_inner_seq` and `build_inner_par` are thin adapters
// around VBuilder's retry loop: they delegate the sig-store population,
// duplicate check, and shard solving to VBuilder, and contribute a
// CompVFunc-specific `build_fn` closure that (a) calls
// `set_up_corr_graphs` on the shard edge with *equation* counts
// rather than key counts and (b) calls [`VBuilder::par_solve`] with
// a per-shard closure doing the multi-edge expansion, peeling, and
// reverse-peel assignment.
//
// The only difference between the two is the VBuilder entry point:
// `try_populate_and_build` for the lender-based sequential path,
// `try_par_populate_and_build` for the slice-based parallel path.
// Everything else — the Huffman setup, the build_fn closure, the
// output re-wrapping — is shared via `build_coder_from_frequencies`
// and `make_build_fn` below.
//
// Storage during construction is `BitVec<Box<[W]>>`, which
// implements `SliceByValueMut<Value = bool>` with a word-aligned
// `try_chunks_mut` — exactly what VBuilder's `par_solve` requires.
// The same `BitVec` is handed to `finish_build` unchanged.

/// Builds a [`HuffmanCoder`] from a frequency map. The parallel
/// path populates the map by iterating a value slice; the sequential
/// path populates it by iterating a value lender (see
/// [`build_inner_seq`]).
///
/// Returns an error if the resulting code has a maximum codeword length exceeding
/// the unaligned-read limit of `W::BITS - 7`.
fn build_coder_from_frequencies<W: Word + Hash>(
    huffman: HuffmanConf,
    frequencies: &HashMap<W, usize>,
) -> Result<HuffmanCoder<W>> {
    let coder = huffman.build_coder(frequencies);
    // The codeword is read via get_unaligned_unchecked, which reads a
    // full W and shifts right by up to 7 bits.
    let w = coder.max_codeword_len();
    let max = W::BITS - 7;
    if w > max {
        return Err(anyhow!(
            "max codeword length {w} exceeds the unaligned-read limit of {max}: try again after limiting the Huffman codec"
        ));
    }
    Ok(coder)
}

/// Slice fast-path for the frequency map used by [`build_inner_par`].
fn frequencies_from_slice<W: Word + Hash>(values: &[W]) -> HashMap<W, usize> {
    let mut frequencies: HashMap<W, usize> = HashMap::new();
    for &v in values {
        *frequencies.entry(v).or_insert(0) += 1;
    }
    frequencies
}

/// Returns the `build_fn` closure shared by both entry points.
///
/// The closure is called once per retry by VBuilder's populate-and-
/// build loop: it (a) recomputes the per-shard total codeword length
/// and re-sizes the fuse graph via `set_up_corr_graphs`,
/// (b) allocates the data array with the correct per-shard stride,
/// and (c) dispatches `par_solve` with the multi-edge per-shard
/// solver.
#[allow(clippy::type_complexity)]
fn make_build_fn<'c, W, S, E, P>(
    coder: &'c HuffmanCoder<W>,
) -> impl FnMut(
    &mut VBuilder<BitVec<Box<[W]>>, S, E>,
    u64,
    Box<dyn ShardStore<S, W> + Send + Sync>,
    W,
    usize,
    &mut P,
    &mut (),
) -> Result<(BitVec<Box<[W]>>, u64, usize)>
+ 'c
where
    W: Word + BinSafe + Hash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, W>: RadixKey,
    P: dsi_progress_logger::ProgressLog + Clone + Send + Sync,
{
    let max_codeword_len = coder.max_codeword_len();
    let escaped_symbol_length = coder.escaped_symbols_len();
    let w = max_codeword_len + escaped_symbol_length;
    let escape_codeword = coder.escape_codeword();

    let encode_val = move |v: W| -> (W, usize, u32) {
        match coder.encode(v) {
            Some(cw) => (W::ZERO, cw, coder.codeword_len(v)),
            None => {
                let lit = if escaped_symbol_length == 0 {
                    W::ZERO
                } else {
                    v.reverse_bits() >> (W::BITS - escaped_symbol_length)
                };
                (lit, escape_codeword, w)
            }
        }
    };

    move |vb: &mut VBuilder<BitVec<Box<[W]>>, S, E>,
          attempt_seed: u64,
          mut store: Box<dyn ShardStore<S, W> + Send + Sync>,
          _max_val: W,
          num_keys: usize,
          pl: &mut P,
          _state: &mut ()| {
        // Compute per-shard sums of codeword lengths and key counts. The
        // ShardEdge has to be resized against the actual edge count and key
        // count.
        let mut total_edges: usize = 0;
        let mut max_shard_edges: usize = 0;
        let mut max_shard_keys: usize = 0;
        for shard in store.iter() {
            let mut edges_in_shard: usize = 0;
            let keys_in_shard = shard.len();
            for sv in shard.iter() {
                edges_in_shard += coder.encoded_len(sv.val) as usize;
            }
            total_edges += edges_in_shard;
            max_shard_edges = max_shard_edges.max(edges_in_shard);
            max_shard_keys = max_shard_keys.max(keys_in_shard);
        }

        // Call the correlated-graph setup on the shard edge.`c is keyed at
        // max_shard_keys, log2_seg_size at max_shard_edges.
        let (c, lge) =
            vb.shard_edge
                .set_up_corr_graphs(total_edges, max_shard_keys, max_shard_edges);
        // This should never really happen--we have static checks
        assert!(!lge, "CompVFunc does not support LGE");
        vb.c = c;
        vb.lge = false;

        // Compute the per-shard stride and allocate
        let num_vertices_per_shard = vb.shard_edge.num_vertices();
        let num_shards = vb.shard_edge.num_shards();
        vb.num_threads = num_shards.min(vb.max_num_threads).max(1);

        pl.info(format_args!(
            "Huffman: entropy: {:.3}; max codeword length {}; length of escaped symbols: {}",
            coder.entropy(),
            coder.max_codeword_len(),
            coder.escaped_symbols_len()
        ));
        pl.info(format_args!(
            "Keys: {num_keys}; max shard keys: {max_shard_keys}; edges: {total_edges}; max shard edges: {max_shard_edges}; average codeword length: {:.3}",
            total_edges as f64 / num_keys as f64
        ));
        pl.info(format_args!(
            "{}; signatures: {}",
            vb.shard_edge,
            core::any::type_name::<S>()
        ));
        // Padding for multi-edge expansion
        let raw_stride = num_vertices_per_shard + w as usize;
        // `par_solve` chunks the data via `BitVec::try_chunks_mut`,
        // which requires `chunk_size` to be a multiple of `W::BITS`.
        let stride = raw_stride.next_multiple_of(W::BITS as usize);

        pl.info(format_args!(
            "c: {}; overhead (vertices/edges): {:+.3}%; number of threads: {}",
            c,
            100.0 * ((stride * num_shards) as f64 / (total_edges as f64) - 1.),
            vb.num_threads
        ));
        let padding = stride - num_vertices_per_shard;
        let total_bits = num_shards
            .checked_mul(stride)
            .ok_or_else(|| anyhow!("data size overflow"))?;

        let mut data = BitVec::<Box<[W]>>::new_padded(total_bits);

        // Two peeler families:
        //
        // peel_by_edge_high_mem/peel_by_data_low_mem: edges are written
        // into a XorGraph<PackedEdge> which keeps in a [u32; 3] both
        // the edge and the associated bit, limiting the vertex count
        // to `PackedEdge::MAX_VERTEX`; however, the representation is
        // compact and peeling is fast.
        //
        // peel_by_index materialises edges as a Vec<[u32; 3]> and BitVec
        // upfront, then peels by edge index. Used when per-shard vertex count
        // exceeds `PackedEdge::MAX_VERTEX`.
        //
        // Note that the tradeoffs here are very different than in VBuilder,
        // because we have a very large number of edges whose associated
        // value is a single bit.
        //
        // None of the peelers use a Gaussian elimination fallback as in
        // VBuilder: on any unpeeled remainder the shard is retried with a new
        // seed.
        let force_index_peeler = raw_stride >= PackedEdge::MAX_VERTICES as usize;
        let use_low_mem = vb.low_mem == Some(true)
            || (vb.low_mem.is_none() && vb.num_threads > 3 && num_shards > 2);
        let solve_shard = |this: &VBuilder<BitVec<Box<[W]>>, S, E>,
                           shard_index: usize,
                           shard: std::sync::Arc<Vec<SigVal<S, W>>>,
                           mut shard_data: BitVec<&mut [W]>,
                           pl: &mut P|
         -> Result<(), ()> {
            if this.failed.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(());
            }
            let shard_edge = &this.shard_edge;

            if force_index_peeler {
                pl.item_name("edge");
                pl.start(format!(
                    "Generating graph for shard {}/{} (index peeler)...",
                    shard_index + 1,
                    num_shards
                ));
                let mut edges: Vec<[u32; 3]> = Vec::with_capacity(max_shard_edges);
                let mut rhs: BitVec = BitVec::with_capacity(max_shard_edges);
                for sv in shard.iter() {
                    let (lit, cw, len) = encode_val(sv.val);
                    let local_sig = shard_edge.local_sig(sv.sig);
                    let base = shard_edge.local_edge(local_sig);
                    let cw_bits = len.min(max_codeword_len) as usize;
                    for l in 0..cw_bits {
                        let off = w as usize - 1 - l;
                        edges.push([
                            (base[0] + off) as u32,
                            (base[1] + off) as u32,
                            (base[2] + off) as u32,
                        ]);
                        rhs.push(((cw >> l) & 1) != 0);
                    }
                    let lit_bits = (len as usize).saturating_sub(max_codeword_len as usize);
                    for l in 0..lit_bits {
                        let off = escaped_symbol_length as usize - 1 - l;
                        edges.push([
                            (base[0] + off) as u32,
                            (base[1] + off) as u32,
                            (base[2] + off) as u32,
                        ]);
                        rhs.push(((lit >> l as u32) & W::ONE) != W::ZERO);
                    }
                }
                pl.done_with_count(edges.len());
                if this.failed.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(());
                }
                peel_by_index(
                    raw_stride,
                    &edges,
                    rhs,
                    &mut shard_data,
                    pl,
                    shard_index,
                    num_shards,
                )
                .map_err(|_| ())?;
            } else if use_low_mem {
                peel_by_edge_low_mem(
                    shard,
                    shard_edge,
                    &encode_val,
                    raw_stride,
                    w as usize,
                    max_codeword_len as usize,
                    escaped_symbol_length as usize,
                    &mut shard_data,
                    pl,
                    shard_index,
                    num_shards,
                )?;
            } else {
                peel_by_edge_high_mem(
                    shard,
                    shard_edge,
                    &encode_val,
                    raw_stride,
                    w as usize,
                    max_codeword_len as usize,
                    escaped_symbol_length as usize,
                    &mut shard_data,
                    pl,
                    shard_index,
                    num_shards,
                )?;
            }
            Ok(())
        };

        pl.log_level(log::Level::Trace);
        vb.par_solve(
            store.drain(),
            &mut data,
            padding,
            solve_shard,
            &mut pl.concurrent(),
            pl,
        )
        .map_err(anyhow::Error::from)?;
        pl.log_level(log::Level::Info);

        Ok((data, attempt_seed, stride))
    }
}

/// Finalizes a successful build by packing the construction-side [`BitVec`]
/// into a [`CompVFunc`].
fn finish_build<K: ?Sized, W: Word + MemSize + FlatType, S, E: MemSize + FlatType>(
    shard_edge: E,
    coder: HuffmanCoder<W>,
    data: BitVec<Box<[W]>>,
    seed_used: u64,
    num_keys: usize,
    shard_size: usize,
    pl: &mut impl ProgressLog,
) -> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    HuffmanDecoder<W>: MemSize + FlatType,
    BitVec<Box<[W]>>: MemSize + FlatType,
{
    let entropy = coder.entropy();
    let decoder = coder.into_decoder();
    let w = decoder.max_codeword_len();
    let codeword_mask = if w == 0 {
        W::ZERO
    } else {
        W::MAX >> (W::BITS - w)
    };

    pl.info(format_args!(
        "Decoder: {} bits",
        decoder.mem_size(SizeFlags::empty())
    ));

    let comp_vfunc = CompVFunc {
        shard_edge,
        seed: seed_used,
        num_keys,
        shard_size,
        codeword_mask,
        data,
        decoder,
        _marker: PhantomData,
    };

    let size = comp_vfunc.mem_size(SizeFlags::default()) as f64 * 8.0;

    pl.info(format_args!(
        "Bits/key: {:.3} ({:+.3}% with respect to entropy)",
        size / num_keys as f64,
        100.0 * (size / (num_keys as f64 * entropy) - 1.),
    ));

    comp_vfunc
}

/// Empty-function short-circuit shared by both entry points.
fn empty_comp_vfunc<K, W, S, E>() -> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized,
    W: Word + Hash,
    E: ShardEdge<S, 3>,
{
    // Build a single-symbol Huffman coder so the decoder has a valid
    // block and never falls through.  The data is all zeros, so every
    // XOR read yields 0 and decode(0) returns Some(W::ZERO).
    let mut freqs = HashMap::new();
    freqs.insert(W::ZERO, 1);
    let coder = HuffmanConf::new().build_coder(&freqs);
    let decoder = coder.into_decoder();
    let w = decoder.max_codeword_len();
    let codeword_mask = if w == 0 {
        W::ZERO
    } else {
        W::MAX >> (W::BITS - w)
    };
    let mut shard_edge = E::default();
    shard_edge.set_up_shards(0, 1.0);
    shard_edge.set_up_graphs(0, 1);
    let shard_size = shard_edge.num_vertices();
    CompVFunc {
        shard_edge,
        seed: 0,
        num_keys: 0,
        shard_size,
        codeword_mask,
        data: BitVec::<Box<[W]>>::new_padded(shard_size),
        decoder,
        _marker: PhantomData,
    }
}

/// Lender-based sequential path.
fn build_inner_seq<
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe + Hash + MemSize + FlatType,
    B: ?Sized + Borrow<K>,
    V: FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
    L: FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    P: ProgressLog + Clone + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
>(
    huffman: HuffmanConf,
    builder: VBuilder<BitVec<Box<[W]>>, S, E>,
    keys: L,
    mut values: V,
    pl: &mut P,
) -> Result<CompVFunc<K, BitVec<Box<[W]>>, S, E>>
where
    SigVal<S, W>: RadixKey,
    BitVec<Box<[W]>>: MemSize + FlatType,
    HuffmanDecoder<W>: MemSize + FlatType,
{
    let mut frequencies = <HashMap<W, usize>>::new();
    while let Some(&v) = values.next()? {
        *frequencies.entry(v).or_insert(0) += 1;
    }
    let n: usize = frequencies.values().sum();
    if n == 0 {
        return Ok(empty_comp_vfunc());
    }

    let coder = build_coder_from_frequencies(huffman, &frequencies)?;
    let total_edges: usize = frequencies
        .iter()
        .map(|(&v, &count)| coder.encoded_len(v) as usize * count)
        .sum();
    values = values.rewind()?;

    let mut builder = builder.shard_size_hint(total_edges);
    let mut build_fn = make_build_fn(&coder);
    let ((data, seed_used, shard_size), _keys) =
        builder.try_populate_and_build(keys, values, &mut build_fn, pl, ())?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build(
        shard_edge, coder, data, seed_used, n, shard_size, pl,
    ))
}

/// Slice-based parallel path.
fn build_inner_par<
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    W: Word + BinSafe + Hash + MemSize + FlatType,
    B: Borrow<K> + Sync,
    P: ProgressLog + Clone + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
>(
    huffman: HuffmanConf,
    builder: VBuilder<BitVec<Box<[W]>>, S, E>,
    keys: &[B],
    values: &[W],
    pl: &mut P,
) -> Result<CompVFunc<K, BitVec<Box<[W]>>, S, E>>
where
    SigVal<S, W>: RadixKey,
    BitVec<Box<[W]>>: MemSize + FlatType,
    HuffmanDecoder<W>: MemSize + FlatType,
{
    let n = keys.len();
    let coder = build_coder_from_frequencies(huffman, &frequencies_from_slice(values))?;
    let total_edges: usize = values.iter().map(|v| coder.encoded_len(*v) as usize).sum();
    let mut builder = builder.expected_num_keys(n).shard_size_hint(total_edges);
    let mut build_fn = make_build_fn(&coder);
    let (data, seed_used, shard_size) =
        builder.try_par_populate_and_build(keys, &|i: usize| values[i], &mut build_fn, pl, ())?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build(
        shard_edge, coder, data, seed_used, n, shard_size, pl,
    ))
}

// ── Index-based peeler + solver ────────────────────────────────────

/// Peels a 3-uniform hypergraph by edge index and assign values, mirroring
/// [`VBuilder::peel_by_index`](crate::func::vbuilder).
///
/// This builder is used when the per-shard vertex count exceeds
/// `PackedEdge::MAX_VERTEX` and we cannot use the other two peelers. It
/// materializes the edge list in a `Vec<[u32; 3]>` and the right-hand side in a
/// `BitVec`, then peels by edge index.
fn peel_by_index<W: Word>(
    num_variables: usize,
    edges: &[[u32; 3]],
    rhs: BitVec,
    solution: &mut BitVec<&mut [W]>,
    pl: &mut impl ProgressLog,
    shard_index: usize,
    num_shards: usize,
) -> Result<()> {
    pl.start(format!(
        "Peeling graph for shard {}/{}...",
        shard_index + 1,
        num_shards
    ));
    let (double_stack, sides_stack) = _peel_by_index(num_variables, edges);
    let n_peeled = double_stack.upper_len();
    let n_total = edges.len();

    if n_peeled < n_total {
        pl.debug(format_args!(
            "Peeling failed for shard {}/{} (peeled {} out of {} edges, {:.2}% unpeeled)",
            shard_index + 1,
            num_shards,
            n_peeled,
            n_total,
            100.0 * (n_total - n_peeled) as f64 / n_total as f64,
        ));
        pl.done();
        bail!(
            "peeling failed ({} unpeeled out of {})",
            n_total - n_peeled,
            n_total
        );
    }
    pl.done();

    pl.start(format!(
        "Assigning values for shard {}/{}...",
        shard_index + 1,
        num_shards
    ));
    for (&edge_idx, &side) in double_stack.iter_upper().zip(sides_stack.iter().rev()) {
        let [a, b, c] = edges[edge_idx as usize];
        let r = rhs[edge_idx as usize];
        let (pivot, val) = unsafe {
            match side {
                0 => (
                    a,
                    r ^ solution.get_unchecked(b as usize) ^ solution.get_unchecked(c as usize),
                ),
                1 => (
                    b,
                    r ^ solution.get_unchecked(a as usize) ^ solution.get_unchecked(c as usize),
                ),
                2 => (
                    c,
                    r ^ solution.get_unchecked(a as usize) ^ solution.get_unchecked(b as usize),
                ),
                // SAFETY: `side` is a 2-bit field in `XorGraph::degrees_sides`.
                _ => std::hint::unreachable_unchecked(),
            }
        };
        // SAFETY: pivot < num_variables = solution.len() by edge construction.
        unsafe {
            solution.set_unchecked(pivot as usize, val);
        }
    }

    pl.done_with_count(n_total);

    #[cfg(debug_assertions)]
    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let val = unsafe {
            solution.get_unchecked(a as usize)
                ^ solution.get_unchecked(b as usize)
                ^ solution.get_unchecked(c as usize)
        };
        debug_assert_eq!(
            val, rhs[i],
            "edge {i} not satisfied: vars=[{a},{b},{c}] rhs={} got={}",
            rhs[i] as u8, val as u8,
        );
    }

    Ok(())
}

/// Peels a 3-uniform hypergraph by edge index, mirroring
/// [`VBuilder::peel_by_index`](crate::func::vbuilder), but does not assign
/// values.
///
/// The caller checks `double_stack.upper_len()` against `edges.len()` to decide
/// whether peeling succeeded.
fn _peel_by_index(num_variables: usize, edges: &[[u32; 3]]) -> (DoubleStack<u32>, Vec<u8>) {
    let n_edges = edges.len();

    // Payload per vertex is the edge index (0-based). We never read
    // `xor_graph.edges[v]` unless `degree(v) == 1`, at which point it
    // holds the sole remaining incident edge's index.
    let mut xor_graph: XorGraph<u32> = XorGraph::new(num_variables);
    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let idx = i as u32;
        xor_graph.add(a as usize, idx, 0);
        xor_graph.add(b as usize, idx, 1);
        xor_graph.add(c as usize, idx, 2);
    }

    let mut double_stack: DoubleStack<u32> = DoubleStack::new(num_variables);
    let mut sides_stack: Vec<u8> = Vec::with_capacity(n_edges);

    assert!(!xor_graph.overflow, "Degree overflow");

    // Seed the visit frontier with every degree-1 vertex.
    for (v, degree) in xor_graph.degrees().enumerate() {
        if degree == 1 {
            double_stack.push_lower(v as u32);
        }
    }

    while let Some(v) = double_stack.pop_lower() {
        let v = v as usize;
        if xor_graph.degree(v) == 0 {
            continue;
        }
        debug_assert_eq!(xor_graph.degree(v), 1);
        let (edge_idx, side) = xor_graph.edge_and_side(v);
        xor_graph.zero(v);
        double_stack.push_upper(edge_idx);
        sides_stack.push(side as u8);

        let e = edges[edge_idx as usize];
        remove_edge!(
            xor_graph,
            e,
            side,
            edge_idx,
            double_stack,
            push_lower,
            |x: u32| x
        );
    }

    drop(xor_graph);
    (double_stack, sides_stack)
}

/// A triple of `u32` values packing an edge and the associated bit.
///
/// The associated bit is stored in the high bit of `v2`. This gives us a very
/// compact memory layout, at the cost of being limited to a maximum of
/// [`PackedEdge::MAX_VERTICES`] = 2³¹ vertices.
#[derive(Clone, Copy, Default, Debug)]
struct PackedEdge {
    /// First vertex.
    v0: u32,
    /// Second vertex.
    v1: u32,
    /// Top bit: associated bit; low 31 bits: third vertex.
    v2_bit: u32,
}

impl PackedEdge {
    /// Maximum number of representable vertices, due to packing of the
    /// associated bit into the high bit of the third vertex.
    const MAX_VERTICES: u32 = (1 << 31);

    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, rhs: bool) -> Self {
        debug_assert!(
            v0 < Self::MAX_VERTICES && v1 < Self::MAX_VERTICES && v2 < Self::MAX_VERTICES,
            "PackedEdge vertex index exceeds 2³¹ – 1"
        );
        Self {
            v0,
            v1,
            v2_bit: v2 | ((rhs as u32) << 31),
        }
    }

    /// Returns the three vertex indices and the rhs bit.
    #[inline(always)]
    fn unpack(self) -> ([u32; 3], bool) {
        let v2 = self.v2_bit & (u32::MAX >> 1);
        let rhs = (self.v2_bit >> 31) != 0;
        ([self.v0, self.v1, v2], rhs)
    }
}

impl std::ops::BitXorAssign for PackedEdge {
    #[inline(always)]
    fn bitxor_assign(&mut self, other: Self) {
        self.v0 ^= other.v0;
        self.v1 ^= other.v1;
        self.v2_bit ^= other.v2_bit;
    }
}

/// Populates an [`XorGraph<PackedEdge>`] by iterating over the shard, calling
/// `encode_val` to produce the codeword and optional escaped literal for each
/// [`SigVal`], and deriving the hyperedges.
///
/// Returns the populated graph and the total number of edges inserted
/// (which is the sum of codeword lengths across the shard and is also
/// the upper bound for the reverse-peel stack)
///
/// [`XorGraph<PackedEdge>`]: super::peeling::XorGraph
/// [`SigVal`]: crate::func::SigVal
fn populate_data_graph<W: BinSafe + Word, S, E>(
    shard: &[SigVal<S, W>],
    shard_edge: &E,
    encode_val: &impl Fn(W) -> (W, usize, u32),
    num_variables: usize,
    w: usize,
    escape_length: usize,
    escaped_symbol_length: usize,
) -> (XorGraph<PackedEdge>, usize)
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let mut xor_graph: XorGraph<PackedEdge> = XorGraph::new(num_variables);
    let mut n_edges: usize = 0;

    macro_rules! add_edge {
        ($base:expr, $off:expr, $rhs_bit:expr) => {
            let v0 = ($base[0] + $off) as u32;
            let v1 = ($base[1] + $off) as u32;
            let v2 = ($base[2] + $off) as u32;
            let pe = PackedEdge::new(v0, v1, v2, $rhs_bit);
            xor_graph.add(v0 as usize, pe, 0);
            xor_graph.add(v1 as usize, pe, 1);
            xor_graph.add(v2 as usize, pe, 2);
            n_edges += 1;
        };
    }

    for sv in shard.iter() {
        let (lit, cw, len) = encode_val(sv.val);
        let local_sig = shard_edge.local_sig(sv.sig);
        let base = shard_edge.local_edge(local_sig);
        let cw_bits = (len as usize).min(escape_length);
        for l in 0..cw_bits {
            let off = w - 1 - l;
            add_edge!(base, off, ((cw >> l) & 1) != 0);
        }
        let lit_bits = (len as usize).saturating_sub(escape_length);
        for l in 0..lit_bits {
            let off = escaped_symbol_length - 1 - l;
            add_edge!(base, off, ((lit >> l as u32) & W::ONE) != W::ZERO);
        }
    }

    assert!(!xor_graph.overflow, "Degree overflow",);
    (xor_graph, n_edges)
}

/// Reverse-peel assignment for a single [`PackedEdge`].
#[inline(always)]
unsafe fn reverse_peel_assign<W: Word>(
    solution: &mut BitVec<&mut [W]>,
    pe: PackedEdge,
    side: usize,
) {
    let ([a, b, c], r) = pe.unpack();
    let (pivot, val) = unsafe {
        match side {
            0 => (
                a,
                r ^ solution.get_unchecked(b as usize) ^ solution.get_unchecked(c as usize),
            ),
            1 => (
                b,
                r ^ solution.get_unchecked(a as usize) ^ solution.get_unchecked(c as usize),
            ),
            2 => (
                c,
                r ^ solution.get_unchecked(a as usize) ^ solution.get_unchecked(b as usize),
            ),
            // SAFETY: `side` is a 2-bit field in `XorGraph::degrees_sides`.
            _ => std::hint::unreachable_unchecked(),
        }
    };
    // SAFETY: `pivot < num_variables = solution.len()` by edge construction.
    unsafe {
        solution.set_unchecked(pivot as usize, val);
    }
}

/// Peels a 3-uniform hypergraph by edge (high-memory variant).
///
/// Mirrors [`VBuilder::peel_by_sig_vals_high_mem`]: after the graph is
/// populated, peeled-edge metadata lands in a [`FastStack<PackedEdge>`] for
/// cheap reverse-peel access. Faster than [`peel_by_edge_low_mem`] at the cost
/// of an extra `n_edges × 16` bytes of peel-time memory.
fn peel_by_edge_high_mem<W: Word + BinSafe, S, E>(
    shard: std::sync::Arc<Vec<SigVal<S, W>>>,
    shard_edge: &E,
    encode_val: &impl Fn(W) -> (W, usize, u32),
    num_variables: usize,
    w: usize,
    escape_length: usize,
    escaped_symbol_length: usize,
    solution: &mut BitVec<&mut [W]>,
    pl: &mut impl ProgressLog,
    shard_index: usize,
    num_shards: usize,
) -> Result<(), ()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    pl.start(format!(
        "Generating graph for shard {}/{} (high-mem)...",
        shard_index + 1,
        num_shards
    ));
    let (mut xor_graph, n_edges) = populate_data_graph(
        &shard,
        shard_edge,
        encode_val,
        num_variables,
        w,
        escape_length,
        escaped_symbol_length,
    );
    pl.done_with_count(n_edges);

    drop(shard);

    pl.start(format!(
        "Peeling graph for shard {}/{} (high-mem)...",
        shard_index + 1,
        num_shards
    ));
    let mut peeled_edges_stack: FastStack<PackedEdge> = FastStack::new(n_edges);
    let mut sides_stack: FastStack<u8> = FastStack::new(n_edges);
    let mut visit_stack: Vec<u32> = Vec::with_capacity(num_variables / 3);

    for (v, degree) in xor_graph.degrees().enumerate() {
        if degree == 1 {
            visit_stack.push(v as u32);
        }
    }

    while let Some(v) = visit_stack.pop() {
        let vu = v as usize;
        if xor_graph.degree(vu) == 0 {
            continue;
        }
        debug_assert_eq!(xor_graph.degree(vu), 1);
        let (pe, side) = xor_graph.edge_and_side(vu);
        xor_graph.zero(vu);
        peeled_edges_stack.push(pe);
        sides_stack.push(side as u8);

        let (e, _rhs) = pe.unpack();
        remove_edge!(xor_graph, e, side, pe, visit_stack, push, |x: u32| x);
    }

    drop(xor_graph);

    if peeled_edges_stack.len() != n_edges {
        pl.debug(format_args!(
            "Peeling failed for shard {}/{} (peeled {} out of {} edges, {:.2}% unpeeled)",
            shard_index + 1,
            num_shards,
            peeled_edges_stack.len(),
            n_edges,
            100.0 * (n_edges - peeled_edges_stack.len()) as f64 / n_edges as f64,
        ));
        pl.done();
        return Err(());
    }
    pl.done();

    pl.start(format!(
        "Assigning values for shard {}/{} (high-mem)...",
        shard_index + 1,
        num_shards
    ));
    for (&pe, &side) in peeled_edges_stack
        .iter()
        .rev()
        .zip(sides_stack.iter().rev())
    {
        unsafe {
            reverse_peel_assign(solution, pe, side as usize);
        }
    }
    pl.done_with_count(n_edges);

    Ok(())
}

/// Peels a 3-uniform hypergraph by edge (low-memory variant).
///
/// Mirrors [`VBuilder::peel_by_sig_vals_low_mem`]: after the graph is
/// populated, a single [`DoubleStack<u32>`] holds the visit queue in its lower
/// half and peeled pivot vertices in its upper half.
///
/// Uses roughly half the memory of [`peel_by_edge_high_mem`], but has
/// worse locality on reverse-peel assignment.
///
/// [`peel_by_edge_high_mem`]: peel_by_edge_high_mem
fn peel_by_edge_low_mem<W: Word + BinSafe, S, E>(
    shard: std::sync::Arc<Vec<SigVal<S, W>>>,
    shard_edge: &E,
    encode_val: &impl Fn(W) -> (W, usize, u32),
    num_variables: usize,
    w: usize,
    escape_length: usize,
    escaped_symbol_length: usize,
    solution: &mut BitVec<&mut [W]>,
    pl: &mut impl ProgressLog,
    shard_index: usize,
    num_shards: usize,
) -> Result<(), ()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    pl.start(format!(
        "Generating graph for shard {}/{} (low-mem)...",
        shard_index + 1,
        num_shards
    ));
    let (mut xor_graph, n_edges) = populate_data_graph(
        &shard,
        shard_edge,
        encode_val,
        num_variables,
        w,
        escape_length,
        escaped_symbol_length,
    );
    pl.done_with_count(n_edges);

    drop(shard);

    if xor_graph.overflow {
        return Err(());
    }

    pl.start(format!(
        "Peeling graph for shard {}/{} (low-mem)...",
        shard_index + 1,
        num_shards
    ));
    let mut double_stack: DoubleStack<u32> = DoubleStack::new(num_variables.max(1));

    for (v, degree) in xor_graph.degrees().enumerate() {
        if degree == 1 {
            double_stack.push_lower(v as u32);
        }
    }

    while let Some(v) = double_stack.pop_lower() {
        let vu = v as usize;
        if xor_graph.degree(vu) == 0 {
            continue;
        }
        debug_assert_eq!(xor_graph.degree(vu), 1);
        let (pe, side) = xor_graph.edge_and_side(vu);
        xor_graph.zero(vu);
        double_stack.push_upper(v);

        let (e, _rhs) = pe.unpack();
        remove_edge!(xor_graph, e, side, pe, double_stack, push_lower, |x: u32| x);
    }

    if double_stack.upper_len() != n_edges {
        pl.debug(format_args!(
            "Peeling failed for shard {}/{} (peeled {} out of {} edges, {:.2}% unpeeled)",
            shard_index + 1,
            num_shards,
            double_stack.upper_len(),
            n_edges,
            100.0 * (n_edges - double_stack.upper_len()) as f64 / n_edges as f64,
        ));
        pl.done();
        return Err(());
    }
    pl.done();

    pl.start(format!(
        "Assigning values for shard {}/{} (low-mem)...",
        shard_index + 1,
        num_shards
    ));
    for &pivot_v in double_stack.iter_upper() {
        let (pe, side) = xor_graph.edge_and_side(pivot_v as usize);
        unsafe {
            reverse_peel_assign(solution, pe, side);
        }
    }
    pl.done_with_count(n_edges);

    Ok(())
}

// ── Aligned ↔ Unaligned conversions ────────────────────────────────

impl<K: ?Sized, W: Word, S, E> TryIntoUnaligned for CompVFunc<K, BitVec<Box<[W]>>, S, E> {
    type Unaligned = CompVFunc<K, BitVecU<Box<[W]>>, S, E>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        let esym = Decoder::escaped_symbols_len(&self.decoder) as usize;
        if esym > 0 && !test_unaligned_any_pos!(W, esym) {
            return Err(UnalignedConversionError(format!(
                "escaped-symbol bit width {esym} does not satisfy the constraints \
                 for arbitrary-position unaligned reads on {} (must be <= {})",
                core::any::type_name::<W>(),
                W::BITS as usize - 7
            )));
        }
        Ok(CompVFunc {
            shard_edge: self.shard_edge,
            seed: self.seed,
            num_keys: self.num_keys,
            shard_size: self.shard_size,
            codeword_mask: self.codeword_mask,
            data: self.data.try_into_unaligned()?,
            decoder: self.decoder,
            _marker: PhantomData,
        })
    }
}

impl<K: ?Sized, W: Word, S, E> From<CompVFunc<K, BitVecU<Box<[W]>>, S, E>>
    for CompVFunc<K, BitVec<Box<[W]>>, S, E>
{
    fn from(u: CompVFunc<K, BitVecU<Box<[W]>>, S, E>) -> Self {
        CompVFunc {
            shard_edge: u.shard_edge,
            seed: u.seed,
            num_keys: u.num_keys,
            shard_size: u.shard_size,
            codeword_mask: u.codeword_mask,
            data: u.data.into(),
            decoder: u.decoder,
            _marker: PhantomData,
        }
    }
}
