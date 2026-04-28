/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A compressed static function.
//!
//! [`CompVFunc`] maps `n` keys to values of type `D::Word`, representing each
//! value with a [prefix-free codeword][crate::func::codec::Codec] (by
//! default a length-limited [Huffman code][crate::func::codec::Huffman])
//! and storing the codewords by peeling a fuse graph, as in the case of
//! [`VFunc`].
//!
//! When the value distribution is skewed, this uses much less space
//! than [`VFunc`]: roughly the empirical entropy of the value list plus
//! ≈10% overhead.
//!
//! Within each shard, every key contributes `L` (= codeword length) linear
//! equations, all sharing the same three base vertex positions and shifted by
//! `l = 0 . . L − 1`. At query time we read three *w*-bit windows (with *w* the
//! length of the longest keyword) at the per-shard base positions, XOR them,
//! and [decode].
//!
//! [`VFunc`]: crate::func::VFunc
//! [`VBuilder`]: crate::func::VBuilder
//! [`ShardEdge`]: crate::func::shard_edge::ShardEdge
//! [`ToSig`]: crate::utils::ToSig
//! [decode]: crate::func::codec::Decoder::decode

use crate::bits::{BitVec, BitVecU, test_unaligned_any_pos};
use crate::func::VBuilder;
use crate::func::codec::{Codec, Coder, Decoder, Huffman, HuffmanCoder, HuffmanDecoder};
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
use mem_dbg::{MemDbg, MemSize};
use num_primitive::PrimitiveNumber;
use rdst::RadixKey;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

// ── CompVFunc struct ────────────────────────────────────────────────

/// A static function whose values are stored in compressed form.
///
/// See the [module documentation](crate::func::comp_vfunc) for an overview.
///
/// Instances of this structure are immutable; they are built using [`try_new`]
/// or one of its variants, and can be serialized using [ε-serde] or [`serde`].
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Generics
///
/// * `K` - the key type.
/// * `D` - the data backend; defaults to [`BitVec<Box<[usize]>>`](crate::bits::BitVec).
///   The output value type is [`D::Word`](crate::traits::Backend::Word).
/// * `S` - the signature type; defaults to `[u64; 2]`.
/// * `E` - the [`ShardEdge`] used for sharding and local hashing;
///   defaults to [`Fuse3Shards`].
///
/// [`try_new`]: Self::try_new
/// [ε-serde]: https://docs.rs/epserde/latest/epserde/
/// [`serde`]: https://crates.io/crates/serde
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
        if self.num_keys == 0 {
            return D::Word::ZERO;
        }
        // Here we compute manually the edge vertices. We cannot use
        // ShardEdge::edge because its edge computation derives the number of
        // vertices from the segment size, whereas here we have to includes a
        // bit padding and a rounding to a usize::BITS multiple, which is why we
        // cache the value in shard_size.
        let shard = self.shard_edge.shard(sig);
        let shard_offset = shard * self.shard_size;
        let esym_len = self.decoder.escaped_symbols_length() as usize;
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
        // The literal sits at the low end [0 . . esym_len).
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

    /// The maximum codeword length used by the underlying code (`w`).
    pub fn global_max_codeword_length(&self) -> u32 {
        self.decoder.max_codeword_length()
    }

    /// Length of the escape codeword (0 when there are no escaped
    /// symbols).
    pub fn escape_length(&self) -> u32 {
        Decoder::max_codeword_length(&self.decoder)
    }

    /// Width in bits of the literal field used to encode escaped
    /// symbols (0 when there are no escaped symbols).
    pub fn escaped_symbol_length(&self) -> u32 {
        Decoder::escaped_symbols_length(&self.decoder)
    }

    /// Returns `true` if the underlying [`HuffmanDecoder`] is using
    /// the branchless decode strategy.
    pub const fn is_decoder_branchless(&self) -> bool {
        self.decoder.is_branchless()
    }

    /// Forces the underlying [`HuffmanDecoder`] to use the branchy
    /// (early-exit) or branchless (always-touch-all-blocks) decode
    /// strategy. The default is chosen at construction time based on
    /// the number of length blocks; this method overrides it.
    pub fn decoder_branchless(&mut self, branchless: bool) -> &mut Self {
        self.decoder.branchless(branchless);
        self
    }
}

// ── Aligned ↔ Unaligned conversions ────────────────────────────────

impl<K: ?Sized, W: Word, S, E> TryIntoUnaligned for CompVFunc<K, BitVec<Box<[W]>>, S, E> {
    type Unaligned = CompVFunc<K, BitVecU<Box<[W]>>, S, E>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        let esym = Decoder::escaped_symbols_length(&self.decoder) as usize;
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

// ── Entry points ────────────────────────────────────────────────────
//
// CompVFunc shares its parallel infrastructure with
// [`VBuilder`](crate::func::VBuilder): callers pass a
// `VBuilder<BitVec<Box<[usize]>>, S, E>` configured with the
// usual VBuilder knobs (offline, check-dups, low-mem, threads, eps,
// seed). The only CompVFunc-specific configuration is the
// [`Huffman`] codec used for values; the default is unlimited-length
// Huffman.
//
// Both build- and query-side edge generation go through
// `ShardEdge::local_edge(shard_edge.local_sig(sig))`, which provides
// fuse-graph structured base positions in `[0, num_vertices)`. The
// multi-edge layout then adds `(w − 1 − l)` per codeword bit to each
// base position.

impl<K, W, S, E> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe + Hash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, W>: RadixKey,
{
    /// Builds a [`CompVFunc`] from keys and values using default [`VBuilder`]
    /// and [`Huffman`] codec settings.
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
        Self::try_new_with_builder(keys, values, Huffman::new(), VBuilder::default(), pl)
    }

    /// Builds a [`CompVFunc`] from lenders of keys and values using
    /// the given [`Huffman`] codec and [`VBuilder`] configuration.
    ///
    /// See [`try_new`](Self::try_new) for the streaming semantics.
    /// The `builder` argument controls every VBuilder-side
    /// construction knob (offline mode, thread count, sharding ε,
    /// PRNG seed, etc.); the `huffman` argument controls the codec
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
        huffman: Huffman,
        builder: VBuilder<BitVec<Box<[W]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        build_inner_seq::<K, W, B, _, _, _, S, E>(huffman, builder, keys, values, pl)
    }
}

impl<K, W, S, E> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    W: Word + BinSafe + Hash,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, W>: RadixKey,
{
    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using default [`VBuilder`] and [`Huffman`] settings.
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
        Self::try_par_new_with_builder(keys, values, Huffman::new(), VBuilder::default(), pl)
    }

    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using the given [`Huffman`] codec and [`VBuilder`] configuration.
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new_with_builder`] instead.
    ///
    /// [`try_new_with_builder`]: Self::try_new_with_builder
    pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
        keys: &[B],
        values: &[W],
        huffman: Huffman,
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
// Storage during construction is `BitVec<Box<[usize]>>`, which
// implements `SliceByValueMut<Value = bool>` with a word-aligned
// `try_chunks_mut` — exactly what VBuilder's `par_solve` requires.
// The same `BitVec` is handed to `finish_build` unchanged, so the
// query path gets its `get_value_unaligned` / `BitVecU` read
// interface with zero re-wrapping.

/// Builds a [`HuffmanCoder`] from a frequency map. The parallel
/// path populates the map by iterating a value slice; the sequential
/// path populates it by iterating a value lender (see
/// `build_inner_seq`).
fn build_coder_from_frequencies<W: Word + Hash>(
    huffman: Huffman,
    frequencies: HashMap<W, usize>,
) -> HuffmanCoder<W> {
    let coder = huffman.build_coder(&frequencies);
    // The codeword is read via get_unaligned_unchecked, which reads a
    // full W and shifts right by up to 7 bits.
    let w = coder.max_codeword_length();
    let max = W::BITS - 7;
    assert!(
        w <= max,
        "max codeword length {w} exceeds the unaligned-read limit of {max}"
    );
    coder
}

/// Slice fast-path for the frequency map used by `build_inner_par`.
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
    let escape_length = coder.max_codeword_length();
    let escaped_symbol_length = coder.escaped_symbols_length();
    let w = escape_length as usize + escaped_symbol_length as usize;
    let escape_codeword = coder.escape_codeword();

    let encode_val = move |v: W| -> (W, usize, u32) {
        match coder.encode(v) {
            Some(cw) => (W::ZERO, cw, coder.codeword_length(v)),
            None => {
                let lit = if escaped_symbol_length == 0 {
                    W::ZERO
                } else {
                    v.reverse_bits() >> (W::BITS - escaped_symbol_length)
                };
                (lit, escape_codeword, escape_length + escaped_symbol_length)
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
        // ── a) Compute per-shard sums of codeword lengths and
        //      key counts. The shard-edge has to be resized against
        //      the actual edge count and key count, since the
        //      multi-edge expansion is what determines the band
        //      density (edges) while c is set from the key count
        //      (the correlation-aware variant from corr_peel_sweep).
        let mut total_edges: usize = 0;
        let mut max_shard_edges: usize = 0;
        let mut max_shard_keys: usize = 0;
        for shard in store.iter() {
            let mut edges_in_shard: usize = 0;
            let keys_in_shard = shard.len();
            for sv in shard.iter() {
                edges_in_shard += coder.encoded_length(sv.val) as usize;
            }
            total_edges += edges_in_shard;
            max_shard_edges = max_shard_edges.max(edges_in_shard);
            max_shard_keys = max_shard_keys.max(keys_in_shard);
        }

        // ── b) Call the correlated-graph setup on the shard edge.
        // `c` is keyed at `max_shard_keys`, `log2_seg_size` at
        // `max_shard_edges`.
        let (c, lge) =
            vb.shard_edge
                .set_up_corr_graphs(total_edges, max_shard_keys, max_shard_edges);
        // This should never really happen--we have static checks
        assert!(!lge, "CompVFunc does not support LGE");
        vb.c = c;
        vb.lge = false;

        // ── c) Compute the per-shard stride and allocate. ──
        let num_vertices_per_shard = vb.shard_edge.num_vertices();
        let num_shards = vb.shard_edge.num_shards();
        // `par_solve` derives `num_threads.ilog2()` for its internal
        // buffer size; ensure it's initialized to a positive value.
        // VFunc sets this inside `try_build_from_shard_iter`, which
        // we're side-stepping, so we must set it ourselves.
        vb.num_threads = num_shards.min(vb.max_num_threads).max(1);

        // ── Progress-log info mirroring VBuilder's
        // `try_build_from_shard_iter` (vbuilder.rs:1202–1218) plus
        // CompVFunc-specific entropy metrics: average codeword
        // length (≈ H(values) in bits, the information-theoretic
        // optimum) and the actual bits/key ratio.
        pl.info(format_args!(
            "{} with {} signatures",
            vb.shard_edge,
            core::any::type_name::<S>()
        ));
        let entropy = total_edges as f64 / num_keys as f64;
        pl.info(format_args!(
            "Huffman: max codeword length {}, escaped symbol length {}",
            coder.max_codeword_length(),
            coder.escaped_symbols_length()
        ));
        pl.info(format_args!(
            "Average codeword length (entropy): {:.4} bits/key (total edges: {}, shards: {}, max shard edges: {})",
            entropy, total_edges, num_shards, max_shard_edges
        ));
        let raw_stride = num_vertices_per_shard + w;
        // `par_solve` chunks the data via `BitVec::try_chunks_mut`,
        // which requires `chunk_size` to be a multiple of `W::BITS`.
        let stride = raw_stride.next_multiple_of(W::BITS as usize);

        pl.info(format_args!(
            "c: {}, Overhead: {:+.4}% Number of threads: {}",
            c,
            100.0 * ((stride * num_shards) as f64 / (total_edges as f64) - 1.),
            vb.num_threads
        ));
        let padding = stride - num_vertices_per_shard;
        let total_bits = num_shards
            .checked_mul(stride)
            .ok_or_else(|| anyhow!("data size overflow"))?;

        let mut data = BitVec::<Box<[W]>>::new_padded(total_bits);

        // ── d) Call `par_solve` with the multi-edge closure. ──
        //
        // Two peeler families:
        //
        // * **Index peeler** (`solve_system` via `peel_by_index`):
        //   materialises the full `edges: Vec<[u32; 3]>` and
        //   `rhs: BitVec` upfront, then peels by edge index. Used
        //   when per-shard vertex count exceeds
        //   `PackedEdge::MAX_VERTEX` (the streamed peelers steal the
        //   top bit of `v2` for the rhs flag and cannot represent
        //   vertex indices ≥ 2³¹). This only triggers for very large
        //   single-shard builds (typically `Fuse3NoShards` with
        //   >800M keys).
        //
        // * **Data peelers** (`peel_by_data_high_mem` /
        //   `peel_by_data_low_mem`): stream edges directly into an
        //   `XorGraph<PackedEdge>`. The low-mem heuristic mirrors
        //   VBuilder: pick low-mem when explicitly requested, or by
        //   default when `num_threads > 3 && num_shards > 2` (i.e.
        //   the parallel build is memory-pressured).
        //
        // None of the peelers use a Gaussian elimination fallback:
        // on any unpeeled remainder the shard is retried with a new
        // seed.
        let packed_edge_safe = raw_stride <= PackedEdge::MAX_VERTEX as usize;
        let force_index_peeler = !packed_edge_safe;
        let use_low_mem = vb.low_mem == Some(true)
            || (vb.low_mem.is_none() && vb.num_threads > 3 && num_shards > 2);
        let solve_shard = |this: &VBuilder<BitVec<Box<[W]>>, S, E>,
                           _shard_index: usize,
                           shard: std::sync::Arc<Vec<SigVal<S, W>>>,
                           mut shard_data: BitVec<&mut [W]>,
                           _pl: &mut _|
         -> Result<(), ()> {
            if this.failed.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(());
            }
            let shard_edge = &this.shard_edge;

            // Each peeler writes its solution directly into
            // `shard_data` (the per-shard slice of the global
            // `data` array, supplied to us by `par_solve`). No
            // intermediate `BitVec` allocation, no copy. The slice
            // is zero-initialised (the global `data` is
            // `BitVec::new_padded`-built, all zero), and the
            // peelers only call `set_unchecked` on pivots — bits
            // we never touch stay zero.
            if force_index_peeler {
                // Materialise edges/rhs upfront: needed because
                // PackedEdge cannot represent vertex indices ≥ 2³¹.
                let mut edges: Vec<[u32; 3]> = Vec::with_capacity(max_shard_edges);
                let mut rhs: BitVec = BitVec::with_capacity(max_shard_edges);
                for sv in shard.iter() {
                    let (lit, cw, len) = encode_val(sv.val);
                    let local_sig = shard_edge.local_sig(sv.sig);
                    let base = shard_edge.local_edge(local_sig);
                    let cw_bits = len.min(escape_length) as usize;
                    for l in 0..cw_bits {
                        let off = w - 1 - l;
                        edges.push([
                            (base[0] + off) as u32,
                            (base[1] + off) as u32,
                            (base[2] + off) as u32,
                        ]);
                        rhs.push(((cw >> l) & 1) != 0);
                    }
                    let lit_bits = (len as usize).saturating_sub(escape_length as usize);
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
                if this.failed.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(());
                }
                solve_system(raw_stride, &edges, rhs, &mut shard_data).map_err(|_| ())?;
            } else if use_low_mem {
                peel_by_data_low_mem(
                    shard,
                    shard_edge,
                    &encode_val,
                    raw_stride,
                    w,
                    escape_length as usize,
                    escaped_symbol_length as usize,
                    &mut shard_data,
                )?;
            } else {
                peel_by_data_high_mem(
                    shard,
                    shard_edge,
                    &encode_val,
                    raw_stride,
                    w,
                    escape_length as usize,
                    escaped_symbol_length as usize,
                    &mut shard_data,
                )?;
            }
            Ok(())
        };

        // Propagate the SolveError *unwrapped* so that
        // VBuilder's retry loop can downcast it and re-roll the
        // seed on UnsolvableShard / MaxShardTooBig / duplicate-sig
        // outcomes.
        vb.par_solve(
            store.drain(),
            &mut data,
            padding,
            solve_shard,
            &mut pl.concurrent(),
            pl,
        )
        .map_err(anyhow::Error::from)?;

        pl.info(format_args!(
            "Bits/keys: {} ({:+.4}%)",
            data.len() as f64 / num_keys as f64,
            100.0 * (data.len() as f64 / total_edges as f64 - 1.),
        ));

        Ok((data, attempt_seed, stride))
    }
}

/// Finalizes a successful build by packing the construction-side
/// [`BitVec`] into a [`CompVFunc`].
fn finish_build<K, W, S, E>(
    shard_edge: E,
    coder: HuffmanCoder<W>,
    data: BitVec<Box<[W]>>,
    seed_used: u64,
    num_keys: usize,
    shard_size: usize,
) -> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized,
    W: Word,
{
    let decoder = coder.into_decoder();
    let w = decoder.max_codeword_length();
    let codeword_mask = if w == 0 {
        W::ZERO
    } else {
        W::MAX >> (W::BITS - w)
    };
    CompVFunc {
        shard_edge,
        seed: seed_used,
        num_keys,
        shard_size,
        codeword_mask,
        data,
        decoder,
        _marker: PhantomData,
    }
}

/// Empty-function short-circuit shared by both entry points.
fn empty_comp_vfunc<K, W, S, E>(
    coder: HuffmanCoder<W>,
    eps: f64,
) -> CompVFunc<K, BitVec<Box<[W]>>, S, E>
where
    K: ?Sized,
    W: Word,
    E: ShardEdge<S, 3>,
{
    let mut shard_edge = E::default();
    shard_edge.set_up_shards(0, eps);
    let decoder = coder.into_decoder();
    let w = decoder.max_codeword_length();
    let codeword_mask = if w == 0 {
        W::ZERO
    } else {
        W::MAX >> (W::BITS - w)
    };
    CompVFunc {
        shard_edge,
        seed: 0,
        num_keys: 0,
        shard_size: 0,
        codeword_mask,
        data: BitVec::<Box<[W]>>::new_padded(0),
        decoder,
        _marker: PhantomData,
    }
}

/// Lender-based sequential path.
fn build_inner_seq<
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe + Hash,
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
    E: ShardEdge<S, 3>,
>(
    huffman: Huffman,
    builder: VBuilder<BitVec<Box<[W]>>, S, E>,
    keys: L,
    mut values: V,
    pl: &mut P,
) -> Result<CompVFunc<K, BitVec<Box<[W]>>, S, E>>
where
    SigVal<S, W>: RadixKey,
{
    let mut frequencies: HashMap<W, usize> = HashMap::new();
    while let Some(&v) = values.next()? {
        *frequencies.entry(v).or_insert(0) += 1;
    }
    let n: usize = frequencies.values().sum();
    let coder = huffman.build_coder(&frequencies);
    let w = coder.max_codeword_length();
    let max = W::BITS - 7;
    assert!(
        w <= max,
        "max codeword length {w} exceeds the unaligned-read limit of {max}"
    );
    let total_edges: usize = frequencies
        .iter()
        .map(|(&v, &count)| coder.encoded_length(v) as usize * count)
        .sum();
    values = values.rewind()?;

    if n == 0 {
        return Ok(empty_comp_vfunc::<K, W, S, E>(coder, builder.eps));
    }

    let mut builder = builder.shard_size_hint(total_edges);
    let mut build_fn = make_build_fn::<W, S, E, P>(&coder);
    let ((data, seed_used, shard_size), _keys) =
        builder.try_populate_and_build(keys, values, &mut build_fn, pl, ())?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build::<K, W, S, E>(
        shard_edge, coder, data, seed_used, n, shard_size,
    ))
}

/// Slice-based parallel path.
fn build_inner_par<
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    W: Word + BinSafe + Hash,
    B: Borrow<K> + Sync,
    P: ProgressLog + Clone + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    huffman: Huffman,
    builder: VBuilder<BitVec<Box<[W]>>, S, E>,
    keys: &[B],
    values: &[W],
    pl: &mut P,
) -> Result<CompVFunc<K, BitVec<Box<[W]>>, S, E>>
where
    SigVal<S, W>: RadixKey,
{
    let n = keys.len();
    let coder = build_coder_from_frequencies(huffman, frequencies_from_slice(values));

    if n == 0 {
        return Ok(empty_comp_vfunc::<K, W, S, E>(coder, builder.eps));
    }

    let total_edges: usize = values
        .iter()
        .map(|v| coder.encoded_length(*v) as usize)
        .sum();
    let mut builder = builder.expected_num_keys(n).shard_size_hint(total_edges);
    let mut build_fn = make_build_fn::<W, S, E, P>(&coder);
    let (data, seed_used, shard_size) =
        builder.try_par_populate_and_build(keys, &|i: usize| values[i], &mut build_fn, pl, ())?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build::<K, W, S, E>(
        shard_edge, coder, data, seed_used, n, shard_size,
    ))
}

// ── Index-based peeler + solver ────────────────────────────────────

/// Output of [`peel_by_index`]: the peel-order data needed to drive
/// the reverse-peel assignment. The `XorGraph` itself is dropped
/// when peeling returns — callers only need the peeled edge indices
/// and their pivot sides.
struct PeelByIndexOutput {
    /// Upper half holds peeled edge indices in peel order (oldest at
    /// the bottom, newest at the top). Lower half is empty once
    /// peeling terminates.
    double_stack: DoubleStack<u32>,
    /// Pivot side (0, 1, or 2) for each peeled edge, in the same
    /// order as `double_stack.iter_upper()`.
    sides_stack: Vec<u8>,
}

impl PeelByIndexOutput {
    #[inline]
    fn n_peeled(&self) -> usize {
        self.double_stack.upper_len()
    }
}

/// Peels a 3-uniform **F**₂ system and performs reverse-peel
/// assignment.
///
/// If peeling does not succeed completely, returns an error so the
/// caller can retry the shard with a fresh seed.
fn solve_system<W: Word>(
    num_variables: usize,
    edges: &[[u32; 3]],
    rhs: BitVec,
    solution: &mut BitVec<&mut [W]>,
) -> Result<()> {
    let out = peel_by_index(num_variables, edges);
    let n_peeled = out.n_peeled();
    let n_total = edges.len();

    if n_peeled < n_total {
        bail!(
            "peeling failed ({} unpeeled out of {})",
            n_total - n_peeled,
            n_total
        );
    }

    // Reverse-peel assignment. `DoubleStack::iter_upper()` yields
    // items in newest-first order (the upper half grows downward),
    // which is exactly the order we need: assign the latest-peeled
    // pivot first, so that by the time a pivot's value is set, the
    // other two vertices of its edge were pivots of edges peeled
    // *later* (already set earlier in this loop). `sides_stack` is a
    // regular Vec pushed in peel order, so we reverse it explicitly.
    for (&edge_idx, &side) in out
        .double_stack
        .iter_upper()
        .zip(out.sides_stack.iter().rev())
    {
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

/// Peels a 3-uniform **F**₂ hypergraph by edge index, mirroring
/// [`VBuilder::peel_by_index`](crate::func::vbuilder). The payload
/// stored in the [`XorGraph`] is the edge index itself (`u32`), and
/// the original `edges` slice is kept alive by the caller so the
/// reverse-peel assignment can reach back into it.
///
/// The caller checks [`PeelByIndexOutput::n_peeled`] against
/// `edges.len()` to decide whether peeling succeeded.
///
/// Degenerate edges (two or three repeated vertices) are handled
/// naturally by [`XorGraph::add`]: each slot independently bumps the
/// packed `(degree, side)` byte and XORs the edge index into
/// `edges[v]`, matching **F**₂ semantics (`x + x = 0`). The reverse-peel
/// pass XORs `solution[b]` twice for `[a, b, b]`, which cancels.
fn peel_by_index(num_variables: usize, edges: &[[u32; 3]]) -> PeelByIndexOutput {
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

    if xor_graph.overflow {
        // Some vertex has degree ≥ 64 (the 6-bit degree field
        // overflowed). Peeling cannot succeed.
        drop(xor_graph);
        return PeelByIndexOutput {
            double_stack,
            sides_stack,
        };
    }

    // Seed the visit frontier with every degree-1 vertex.
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
        let (edge_idx, side) = xor_graph.edge_and_side(vu);
        xor_graph.zero(vu);
        double_stack.push_upper(edge_idx);
        sides_stack.push(side as u8);

        let e_u32 = edges[edge_idx as usize];
        // `remove_edge!` indexes `$e[..]` directly with usize, so we
        // convert the vertex triple here.
        let e: [usize; 3] = [e_u32[0] as usize, e_u32[1] as usize, e_u32[2] as usize];
        remove_edge!(
            xor_graph,
            e,
            side,
            edge_idx,
            double_stack,
            push_lower,
            |x: usize| x as u32
        );
    }

    drop(xor_graph);
    PeelByIndexOutput {
        double_stack,
        sides_stack,
    }
}

// ── Streamed (data-payload) peelers ────────────────────────────────

/// A 3-uniform hyperedge payload that lives inside
/// [`XorGraph<PackedEdge>`]. Stores the three vertex indices and the
/// right-hand-side bit inline, so the peeler can recover everything
/// it needs from `xor_graph.edges[v]` at degree-1 vertices — no
/// secondary lookup into a caller-owned `Vec<[u32; 3]>` or `BitVec`
/// of rhs values.
///
/// The layout is 12 bytes: three `u32` fields, with the `rhs` bit
/// packed into the **high bit of `v2`**. This buys us 25% better
/// cache density compared to a 16-byte layout (5.33 entries per
/// 64-byte cache line instead of 4), at the cost of a vertex-index
/// limit of `2^31 - 1` ≈ 2.1 billion per shard.
///
/// # Vertex limit
///
/// By stealing the top bit of `v2` for `rhs`, we halve the max per-
/// shard vertex count from `u32::MAX` to `2^31 - 1`. For CompVFunc
/// this is a hard constraint whenever num_vertices_per_shard crosses
/// `1 << 31`; the dispatcher in `make_build_fn` checks
/// `raw_stride <= PackedEdge::MAX_VERTEX` before routing to these
/// peelers and falls back to [`peel_by_index`] (which uses
/// `XorGraph<u32>` with no such restriction) when the limit is
/// exceeded.
///
/// In practice, the default `Fuse3Shards` strategy picks enough
/// shards to keep per-shard vertex counts well under `1 << 31` for
/// any reasonable input. The limit only bites for `Fuse3NoShards`
/// builds with very large key sets.
#[derive(Clone, Copy, Default, Debug)]
struct PackedEdge {
    v0: u32,
    v1: u32,
    /// Low 31 bits: actual v2 vertex index.
    /// Top bit: rhs bit for this edge.
    v2_rhs: u32,
}

// Pin the layout at compile time: the whole point of this design
// is the 12-byte footprint, which gives us 5.33 entries per 64-byte
// cache line instead of the 4 we'd get at 16 bytes. If a future
// refactor accidentally bloats the struct (e.g. by adding a field
// or an alignment attribute), this assertion catches it at build
// time.
const _: () = assert!(std::mem::size_of::<PackedEdge>() == 12);

impl PackedEdge {
    /// Max vertex index that can be stored in `PackedEdge` (since
    /// the top bit of `v2` is stolen for the rhs flag).
    const MAX_VERTEX: u32 = (1 << 31) - 1;

    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, rhs: bool) -> Self {
        debug_assert!(
            v0 <= Self::MAX_VERTEX && v1 <= Self::MAX_VERTEX && v2 <= Self::MAX_VERTEX,
            "PackedEdge vertex index exceeds 2^31 - 1"
        );
        Self {
            v0,
            v1,
            v2_rhs: v2 | ((rhs as u32) << 31),
        }
    }

    /// Returns the three vertex indices and the rhs bit.
    #[inline(always)]
    fn unpack(self) -> ([u32; 3], bool) {
        let v2 = self.v2_rhs & Self::MAX_VERTEX;
        let rhs = (self.v2_rhs >> 31) != 0;
        ([self.v0, self.v1, v2], rhs)
    }
}

impl std::ops::BitXorAssign for PackedEdge {
    #[inline(always)]
    fn bitxor_assign(&mut self, other: Self) {
        self.v0 ^= other.v0;
        self.v1 ^= other.v1;
        // XORing `v2_rhs` XORs both the low-31-bit v2 and the high-
        // bit rhs in parallel — the two halves live in the same u32
        // and the XOR distributes correctly because v2's encoding
        // never touches bit 31 and rhs's encoding is always bit 31.
        self.v2_rhs ^= other.v2_rhs;
    }
}

/// Populates an [`XorGraph<PackedEdge>`] by streaming over the shard
/// once, calling `encode_val` to produce the codeword and optional
/// escaped literal for each `SigVal`, and deriving the hyperedges
/// (with shifted vertex triples) inline — no intermediate
/// `edges: Vec<[u32; 3]>` or `rhs: BitVec` materialization.
///
/// Returns the populated graph and the total number of edges inserted
/// (which is the sum of codeword lengths across the shard and is also
/// the upper bound for the reverse-peel stack).
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
    (xor_graph, n_edges)
}

/// Assigns a pivot's bit value in `solution` from the packed edge
/// and its side. `side` is a `usize` because that's what
/// `XorGraph::edge_and_side` returns; the match arms cover 0, 1, 2
/// with an `unreachable_unchecked` guard for anything else (side is
/// a 2-bit field by construction).
///
/// # Safety
///
/// All three vertex indices must be in-range for `solution`. This
/// holds by construction: the edge was built from vertex indices
/// derived from `base[i] + off` with `base[i] + off < num_variables`,
/// and `solution.len() >= num_variables` by caller contract.
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

/// Streamed peeler for CompVFunc — high-memory variant. Mirrors
/// [`VBuilder::peel_by_sig_vals_high_mem`]: after the graph is
/// populated, peeled-edge metadata lands in a
/// [`FastStack<PackedEdge>`] for cheap reverse-peel access. Faster
/// than [`peel_by_data_low_mem`] at the cost of an extra
/// `n_edges × 16` bytes of peel-time memory.
///
/// Any unpeeled remainder returns `Err(())`, which `par_solve`
/// turns into a seed-retry.
fn peel_by_data_high_mem<W: Word + BinSafe, S, E>(
    shard: std::sync::Arc<Vec<SigVal<S, W>>>,
    shard_edge: &E,
    encode_val: &impl Fn(W) -> (W, usize, u32),
    num_variables: usize,
    w: usize,
    escape_length: usize,
    escaped_symbol_length: usize,
    solution: &mut BitVec<&mut [W]>,
) -> Result<(), ()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let (mut xor_graph, n_edges) = populate_data_graph(
        &shard,
        shard_edge,
        encode_val,
        num_variables,
        w,
        escape_length,
        escaped_symbol_length,
    );

    // Nothing else needs the shard — drop it to free ~shard_len ×
    // size_of::<SigVal>() bytes before the peel walk starts.
    drop(shard);

    if xor_graph.overflow {
        return Err(());
    }

    let mut peeled_edges_stack: FastStack<PackedEdge> = FastStack::new(n_edges);
    let mut sides_stack: FastStack<u8> = FastStack::new(n_edges);
    // VBuilder's heuristic: the visit stack never needs more than
    // ~num_vertices / 3 slots in steady state. Matching
    // `peel_by_sig_vals_high_mem` at vbuilder.rs:1788 exactly.
    let mut visit_stack: Vec<u32> = Vec::with_capacity(num_variables / 3 + 1);

    // Seed the visit frontier with every degree-1 vertex.
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

        let ([v0, v1, v2], _rhs) = pe.unpack();
        let e: [usize; 3] = [v0 as usize, v1 as usize, v2 as usize];
        remove_edge!(xor_graph, e, side, pe, visit_stack, push, |x: usize| x
            as u32);
    }

    if peeled_edges_stack.len() != n_edges {
        log::info!(
            "Peeling failed (peeled {} out of {} edges, {:.2}% unpeeled)",
            peeled_edges_stack.len(),
            n_edges,
            100.0 * (n_edges - peeled_edges_stack.len()) as f64 / n_edges as f64,
        );
        return Err(());
    }

    drop(xor_graph);

    // Reverse-peel assignment writes pivots directly into `solution`
    // (the per-shard slice of the global `data` array, supplied by
    // the caller). Iterate newest-first over both stacks (they were
    // pushed in peel order, oldest first). No intermediate BitVec
    // allocation, no copy.
    for (&pe, &side) in peeled_edges_stack
        .iter()
        .rev()
        .zip(sides_stack.iter().rev())
    {
        // SAFETY: all vertex indices in `pe` are bounded by
        // `num_variables`, and `solution.len() >= num_variables` by
        // caller contract.
        unsafe {
            reverse_peel_assign(solution, pe, side as usize);
        }
    }

    Ok(())
}

/// Streamed peeler for CompVFunc — low-memory variant. Mirrors
/// [`VBuilder::peel_by_sig_vals_low_mem`]: a single
/// [`DoubleStack<u32>`] holds the visit queue in its lower half and
/// peeled pivot vertices in its upper half. At reverse-peel time we
/// read back the packed edge via [`XorGraph::edge_and_side`], which
/// works because [`XorGraph::zero`] only masks the degree bits —
/// `xor_graph.edges[v]` is preserved.
///
/// Uses roughly half the memory of [`peel_by_data_high_mem`] (no
/// separate `FastStack<PackedEdge>`) at the cost of one extra
/// `edge_and_side` call per assignment. Preferred when the build is
/// parallel (high memory pressure) or `low_mem` is explicitly set.
///
/// Any unpeeled remainder returns `Err(())`.
fn peel_by_data_low_mem<W: Word + BinSafe, S, E>(
    shard: std::sync::Arc<Vec<SigVal<S, W>>>,
    shard_edge: &E,
    encode_val: &impl Fn(W) -> (W, usize, u32),
    num_variables: usize,
    w: usize,
    escape_length: usize,
    escaped_symbol_length: usize,
    solution: &mut BitVec<&mut [W]>,
) -> Result<(), ()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let (mut xor_graph, n_edges) = populate_data_graph(
        &shard,
        shard_edge,
        encode_val,
        num_variables,
        w,
        escape_length,
        escaped_symbol_length,
    );

    drop(shard);

    if xor_graph.overflow {
        return Err(());
    }

    let mut double_stack: DoubleStack<u32> = DoubleStack::new(num_variables.max(1));

    // Seed the visit frontier with every degree-1 vertex.
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
        // Store the pivot vertex itself; at assign time we'll read
        // back `(pe, side)` via `edge_and_side` since `zero` only
        // masked the degree bits.
        double_stack.push_upper(v);

        let ([v0, v1, v2], _rhs) = pe.unpack();
        let e: [usize; 3] = [v0 as usize, v1 as usize, v2 as usize];
        remove_edge!(
            xor_graph,
            e,
            side,
            pe,
            double_stack,
            push_lower,
            |x: usize| x as u32
        );
    }

    if double_stack.upper_len() != n_edges {
        log::info!(
            "Peeling failed (peeled {} out of {} edges, {:.2}% unpeeled)",
            double_stack.upper_len(),
            n_edges,
            100.0 * (n_edges - double_stack.upper_len()) as f64 / n_edges as f64,
        );
        return Err(());
    }

    // Reverse-peel assignment writes pivots directly into `solution`
    // (the per-shard slice supplied by the caller). `iter_upper()`
    // yields pivots in newest-first order, exactly reverse peel
    // order. No intermediate BitVec, no copy.
    for &pivot_v in double_stack.iter_upper() {
        let (pe, side) = xor_graph.edge_and_side(pivot_v as usize);
        // SAFETY: all vertex indices in `pe` are bounded by
        // `num_variables`, and `solution.len() >= num_variables`
        // by caller contract.
        unsafe {
            reverse_peel_assign(solution, pe, side);
        }
    }

    Ok(())
}
