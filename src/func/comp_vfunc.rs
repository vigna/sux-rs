/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A compressed static function.
//!
//! [`CompVFunc`] is a sux-rs port of Sux4J's `GV3CompressedFunction`.
//! It maps `n` keys to `u64` values, but instead of allocating a fixed
//! `bit_width`-bit slot per key it represents each value with a
//! [prefix-free codeword][crate::func::codec::Codec] (by default a
//! length-limited Huffman code) and stores those codewords by solving a
//! random 3-uniform linear system on **F**₂ over the data array.
//!
//! When the value distribution is skewed, this uses much less space
//! than [`VFunc`]: roughly the empirical entropy of the value list plus
//! ~10 % overhead. Sharding follows the same ε-cost approach as the rest
//! of sux-rs (via the [`ShardEdge`] trait): keys are partitioned by the
//! high bits of the signature into a small number of large shards, and
//! each shard is solved independently. The single global seed is shared
//! with [`ToSig`] just like [`VFunc`]; on a peeling failure (rare for
//! the large shards used here) the whole build retries with a new seed.
//!
//! Within each shard, every key contributes `L` (= codeword length)
//! linear equations, all sharing the same three base vertex positions
//! and shifted by `l = 0..L−1`. The solver peels the resulting graph
//! and falls back to lazy Gaussian elimination on the unpeeled
//! remainder, mirroring [`VBuilder`]'s `lge_shard`. At query time we
//! read three `w`-bit windows (with `w` = `global_max_codeword_length`)
//! at the per-shard base positions, XOR them, and decode.
//!
//! [`VFunc`]: crate::func::VFunc
//! [`VBuilder`]: crate::func::VBuilder
//! [`ShardEdge`]: crate::func::shard_edge::ShardEdge
//! [`ToSig`]: crate::utils::ToSig

use crate::bits::{BitVec, BitVecU, test_unaligned_any_pos};
use crate::func::VBuilder;
use crate::func::codec::{Codec, Coder, Decoder, ESCAPE, Huffman, HuffmanCoder, HuffmanDecoder};
use crate::func::peeling::{DoubleStack, FastStack, XorGraph, remove_edge};
use crate::func::shard_edge::{FuseLge3Shards, ShardEdge};
use crate::traits::bit_vec_ops::{BitVecOps, BitVecOpsMut, BitVecValueOps};
use crate::traits::{TryIntoUnaligned, UnalignedConversionError};
use crate::utils::mod2_sys::{Modulo2Equation, Modulo2System};
use crate::utils::sig_store::ShardStore;
use crate::utils::{FallibleRewindableLender, Sig, SigVal, ToSig};
use anyhow::{Result, anyhow, bail};
use core::error::Error;
use dsi_progress_logger::ProgressLog;
use lender::FallibleLending;
use mem_dbg::{MemDbg, MemSize};
use rdst::RadixKey;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;

// ── CompVFunc struct ────────────────────────────────────────────────

/// A static function whose values are stored in a prefix-free
/// compressed form.
///
/// See the [module documentation](crate::func::comp_vfunc) for an
/// overview. Build with [`CompVFunc::try_new`] /
/// [`CompVFunc::try_new_with_builder`].
///
/// # Generics
///
/// * `K`: the key type.
/// * `D`: the data backend; defaults to [`BitVec<Box<[usize]>>`].
///   Construction always produces a [`BitVec`]-backed function;
///   [`TryIntoUnaligned`] converts it into a `BitVecU`-backed variant
///   that uses (faster, branchless) unaligned reads.
/// * `S`: the signature type; defaults to `[u64; 2]`.
/// * `E`: the [`ShardEdge`] used for *sharding* (and the local hash).
///   Defaults to [`FuseLge3Shards`]. Note that `CompVFunc` only uses
///   `ShardEdge` for **sharding** and as the source of the local hash
///   (`local_sig` + `edge_hash`); the per-shard edge generation is
///   *not* the one in `ShardEdge::local_edge` because each compressed
///   key needs `L` related equations whose vertex positions don't fit
///   the one-edge-per-key contract.
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompVFunc<K: ?Sized, D = BitVec<Box<[usize]>>, S = [u64; 2], E = FuseLge3Shards> {
    /// The shard/local-hash logic shared with [`VFunc`].
    ///
    /// [`VFunc`]: crate::func::VFunc
    pub(crate) shard_edge: E,
    /// Global hash seed; identical role to [`VFunc::seed`](crate::func::VFunc).
    pub(crate) seed: u64,
    /// Total number of keys.
    pub(crate) num_keys: usize,
    /// Number of bits assigned to *every* shard. Shards are uniform
    /// because we size them once globally as
    /// `max_per_shard_codeword_bits × δ + w`. The bit position of
    /// shard `s` inside [`Self::data`] is `s × shard_size`. No
    /// cumulative offset table.
    pub(crate) shard_size: usize,
    /// `w` — the maximum codeword length, including escaped symbols.
    pub(crate) global_max_codeword_length: u32,
    pub(crate) escape_length: u32,
    pub(crate) escaped_symbol_length: u32,
    /// All shards concatenated into one bit vector. Each shard owns
    /// `shard_size` bits at offset `s × shard_size`.
    pub(crate) data: D,
    /// Canonical-Huffman decoder.
    pub(crate) decoder: HuffmanDecoder,
    #[doc(hidden)]
    pub(crate) _marker: PhantomData<(*const K, S)>,
}

// ── Query path ──────────────────────────────────────────────────────

impl<K: ?Sized + ToSig<S>, D: BitVecValueOps<usize>, S: Sig, E: ShardEdge<S, 3>>
    CompVFunc<K, D, S, E>
{
    /// Returns the value associated with `key`, or an arbitrary value
    /// if `key` is not in the original key set.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> u64 {
        self.get_by_sig(K::to_sig(key.borrow(), self.seed))
    }

    /// Returns the value associated with the given signature, or an
    /// arbitrary value if no key has that signature.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: S) -> u64 {
        if self.num_keys == 0 {
            return 0;
        }
        let shard = self.shard_edge.shard(sig);
        let bucket_offset = shard * self.shard_size;
        let w = self.global_max_codeword_length as usize;
        // Reuse the fuse-graph `local_edge` to derive the three base
        // positions inside the shard. For the multi-edge layout, bit
        // `l` of the codeword is stored at offset `w − 1 − l` above
        // each base vertex — the band structure of fuse graphs is
        // preserved because the offset is vanishingly small relative
        // to the segment size.
        let local_sig = self.shard_edge.local_sig(sig);
        let local_edge = self.shard_edge.local_edge(local_sig);

        let v0 = bucket_offset + local_edge[0];
        let v1 = bucket_offset + local_edge[1];
        let v2 = bucket_offset + local_edge[2];
        // SAFETY: by construction `v_i + w <= bucket_offset + m + w =
        // bucket_offset + shard_size`, which is within `data.len()`.
        let value = unsafe {
            self.data.get_value_unchecked(v0, w) as u64
                ^ self.data.get_value_unchecked(v1, w) as u64
                ^ self.data.get_value_unchecked(v2, w) as u64
        };
        let decoded = self.decoder.decode(value);
        if decoded != ESCAPE {
            return decoded;
        }
        // The escape codeword occupies the top `escape_length` bits of
        // the read window; the literal sits immediately below it,
        // `escaped_symbol_length` bits wide.
        let esc_len = self.escape_length as usize;
        let esym_len = self.escaped_symbol_length as usize;
        if esym_len == 0 {
            return 0;
        }
        let start = w - esc_len - esym_len;
        // SAFETY: same reasoning as above; `start + esym_len <= w`.
        unsafe {
            self.data.get_value_unchecked(v0 + start, esym_len) as u64
                ^ self.data.get_value_unchecked(v1 + start, esym_len) as u64
                ^ self.data.get_value_unchecked(v2 + start, esym_len) as u64
        }
    }
}

impl<K: ?Sized, D, S, E> CompVFunc<K, D, S, E> {
    /// Number of keys in the function.
    pub const fn len(&self) -> usize {
        self.num_keys
    }

    /// Whether the function has no keys.
    pub const fn is_empty(&self) -> bool {
        self.num_keys == 0
    }

    /// The maximum codeword length used by the underlying code (`w`).
    pub const fn global_max_codeword_length(&self) -> u32 {
        self.global_max_codeword_length
    }

    /// Length of the escape codeword (0 when there are no escaped
    /// symbols).
    pub const fn escape_length(&self) -> u32 {
        self.escape_length
    }

    /// Width in bits of the literal field used to encode escaped
    /// symbols (0 when there are no escaped symbols).
    pub const fn escaped_symbol_length(&self) -> u32 {
        self.escaped_symbol_length
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
    pub fn set_decoder_branchless(&mut self, branchless: bool) -> &mut Self {
        self.decoder.branchless(branchless);
        self
    }
}

// ── Aligned ↔ Unaligned conversions ────────────────────────────────

impl<K: ?Sized, S, E> TryIntoUnaligned for CompVFunc<K, BitVec<Box<[usize]>>, S, E> {
    type Unaligned = CompVFunc<K, BitVecU<Box<[usize]>>, S, E>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        // The query path issues two distinct unaligned reads at
        // arbitrary shard-relative positions: a `w`-bit one for the
        // codeword window, and (for escaped keys) an
        // `escaped_symbol_length`-bit one for the literal field.
        // Both positions come from `ShardEdge::local_edge` and can
        // land at any `pos % 8`, so we need the strict
        // arbitrary-position bound (`width <= W::BITS - 7`) — not the
        // looser `BitFieldVec`-style bound used by
        // `BitVec::try_into_unaligned`.
        let w = self.global_max_codeword_length as usize;
        let esym = self.escaped_symbol_length as usize;
        let unaligned_max = usize::BITS as usize - 7;
        let unaligned_err = |bw: usize| {
            UnalignedConversionError(format!(
                "bit width {bw} does not satisfy the constraints for arbitrary-position unaligned reads on usize (must be <= {unaligned_max})"
            ))
        };
        if !test_unaligned_any_pos!(usize, w) {
            return Err(unaligned_err(w));
        }
        if esym > 0 && !test_unaligned_any_pos!(usize, esym) {
            return Err(unaligned_err(esym));
        }
        Ok(CompVFunc {
            shard_edge: self.shard_edge,
            seed: self.seed,
            num_keys: self.num_keys,
            shard_size: self.shard_size,
            global_max_codeword_length: self.global_max_codeword_length,
            escape_length: self.escape_length,
            escaped_symbol_length: self.escaped_symbol_length,
            data: self.data.try_into_unaligned()?,
            decoder: self.decoder,
            _marker: PhantomData,
        })
    }
}

impl<K: ?Sized, S, E> From<CompVFunc<K, BitVecU<Box<[usize]>>, S, E>>
    for CompVFunc<K, BitVec<Box<[usize]>>, S, E>
{
    fn from(u: CompVFunc<K, BitVecU<Box<[usize]>>, S, E>) -> Self {
        CompVFunc {
            shard_edge: u.shard_edge,
            seed: u.seed,
            num_keys: u.num_keys,
            shard_size: u.shard_size,
            global_max_codeword_length: u.global_max_codeword_length,
            escape_length: u.escape_length,
            escaped_symbol_length: u.escaped_symbol_length,
            data: u.data.into(),
            decoder: u.decoder,
            _marker: PhantomData,
        }
    }
}

// ── Hashing helpers ─────────────────────────────────────────────────
//
// Both build- and query-side edge generation go through
// `ShardEdge::local_edge(shard_edge.local_sig(sig))`, which provides
// fuse-graph structured base positions in `[0, num_vertices)`. The
// multi-edge layout then adds `(w − 1 − l)` per codeword bit to each
// base position.

// ── Entry points ────────────────────────────────────────────────────
//
// CompVFunc shares its parallel infrastructure with
// [`VBuilder`](crate::func::VBuilder): callers pass a
// `VBuilder<BitVec<Box<[usize]>>, S, E>` configured with the
// usual VBuilder knobs (offline, check-dups, low-mem, threads, eps,
// seed). The only CompVFunc-specific configuration is the
// [`Huffman`] codec used for values; the default is unlimited-length
// Huffman.

impl<K, S, E> CompVFunc<K, BitVec<Box<[usize]>>, S, E>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, u64>: RadixKey,
{
    /// Builds a [`CompVFunc`] from lender-based streams of keys and
    /// values using default [`VBuilder`] and [`Huffman`] settings.
    ///
    /// Keys and values are consumed one at a time through their
    /// respective lenders; this path is the right choice for input
    /// coming from disk
    /// ([`DekoBufLineLender`](crate::utils::DekoBufLineLender)) or
    /// synthetic ranges. Neither the key set nor the value set needs
    /// to live in memory at once: the values lender is rewound once
    /// during construction (first pass for the Huffman frequency
    /// analysis, second pass to populate the sig-store).
    ///
    /// `n` is the expected number of keys. If it is significantly
    /// wrong, construction still works but may do extra retries.
    ///
    /// If keys and values are available as slices and you want to
    /// parallelize the hashing phase, use
    /// [`try_par_new`](Self::try_par_new) instead.
    ///
    /// See also [`try_new_with_builder`](Self::try_new_with_builder)
    /// for the full configuration surface.
    pub fn try_new<B: ?Sized + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend u64>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_builder(keys, values, n, Huffman::new(), VBuilder::default(), pl)
    }

    /// Builds a [`CompVFunc`] from lenders of keys and values using
    /// the given [`Huffman`] codec and [`VBuilder`] configuration.
    ///
    /// See [`try_new`](Self::try_new) for the streaming semantics.
    /// The `builder` argument controls every VBuilder-side
    /// construction knob (offline mode, thread count, sharding ε,
    /// PRNG seed, etc.); the `huffman` argument controls the codec
    /// used for the values. The data backend is pinned internally to
    /// [`BitVec<Box<[usize]>>`] — the same type used at query time.
    pub fn try_new_with_builder<B: ?Sized + Borrow<K>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend u64>,
        n: usize,
        huffman: Huffman,
        builder: VBuilder<BitVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        build_inner_seq::<K, B, _, _, _, S, E>(huffman, builder, keys, values, n, pl)
    }
}

impl<K, S, E> CompVFunc<K, BitVec<Box<[usize]>>, S, E>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, u64>: RadixKey,
{
    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using default [`VBuilder`] and [`Huffman`] settings.
    ///
    /// This is the parallel counterpart of [`try_new`](Self::try_new):
    /// hashes are computed on a rayon worker pool and deposited
    /// directly into their sig-store buckets. Faster than the
    /// lender-based path for large in-memory key sets, but requires
    /// the whole key set to be addressable as a slice.
    ///
    /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
    /// for the full configuration surface.
    pub fn try_par_new<B: Borrow<K> + Sync>(
        keys: &[B],
        values: &[u64],
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_par_new_with_builder(keys, values, Huffman::new(), VBuilder::default(), pl)
    }

    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using the given [`Huffman`] codec and [`VBuilder`] configuration.
    ///
    /// See [`try_par_new`](Self::try_par_new) for the parallel
    /// semantics and [`try_new_with_builder`](Self::try_new_with_builder)
    /// for the lender-based variant.
    pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
        keys: &[B],
        values: &[u64],
        huffman: Huffman,
        builder: VBuilder<BitVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if keys.len() != values.len() {
            bail!(
                "keys and values must have the same length ({} vs {})",
                keys.len(),
                values.len()
            );
        }
        build_inner_par::<K, B, _, S, E>(huffman, builder, keys, values, pl)
    }
}

// ── Builder core ───────────────────────────────────────────────────
//
// Both `build_inner_seq` and `build_inner_par` are thin adapters
// around VBuilder's retry loop: they delegate the sig-store population,
// duplicate check, and shard solving to VBuilder, and contribute a
// CompVFunc-specific `build_fn` closure that (a) calls `set_up_graphs`
// on the shard edge with *equation* counts rather than key counts and
// (b) calls [`VBuilder::par_solve`] with a per-shard closure doing the
// multi-edge expansion, peel + LGE on remainder, and assignment.
//
// The only difference between the two is the VBuilder entry point:
// `try_populate_and_build` for the lender-based sequential path,
// `try_par_populate_and_build` for the slice-based parallel path.
// Everything else — the Huffman setup, the build_fn closure, the
// output re-wrapping — is shared via `codec_setup` and
// `make_build_fn` below.
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
fn build_coder_from_frequencies(huffman: Huffman, frequencies: HashMap<u64, u64>) -> HuffmanCoder {
    huffman.build_coder(&frequencies)
}

/// Slice fast-path for the frequency map used by `build_inner_par`.
fn frequencies_from_slice(values: &[u64]) -> HashMap<u64, u64> {
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    for &v in values {
        *frequencies.entry(v).or_insert(0) += 1;
    }
    frequencies
}

/// Returns the `build_fn` closure shared by both entry points.
///
/// The closure is called once per retry by VBuilder's populate-and-
/// build loop: it (a) recomputes the per-shard total codeword length
/// and re-sizes the fuse graph via `set_up_graphs`, (b) allocates the
/// data array with the correct per-shard stride, and (c) dispatches
/// `par_solve` with the multi-edge per-shard solver.
#[allow(clippy::type_complexity)]
fn make_build_fn<'c, S, E, P>(
    coder: &'c HuffmanCoder,
) -> impl FnMut(
    &mut VBuilder<BitVec<Box<[usize]>>, S, E>,
    u64,
    Box<dyn ShardStore<S, u64> + Send + Sync>,
    u64,
    usize,
    &mut P,
    &mut (),
) -> Result<(BitVec<Box<[usize]>>, u64, usize)>
+ 'c
where
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, u64>: RadixKey,
    P: dsi_progress_logger::ProgressLog + Clone + Send + Sync,
{
    // All scalars derived from `coder` — pulled here once so the inner
    // closure body doesn't need to carry them through the layered
    // closures by hand.
    let w = coder.max_codeword_length() as usize;
    let escape_length = coder.escape_length();
    let escaped_symbol_length = coder.escaped_symbol_length();
    let escape_codeword = coder.escape();

    // Encode a single `(bits, len)` pair for a value. Used by both
    // the initial edge-count pass and the per-shard solver.
    //
    // The `escaped_symbol_length == 0` guard handles a degenerate
    // corner: the only escaped symbol is the value `0`, whose bit
    // length is 0. Without the guard, `64 - escaped_symbol_length`
    // would be `64` and the shift on the `else` branch would be
    // undefined (Rust panics in debug, unspecified in release). When
    // we hit this case, the literal field is zero bits wide and the
    // escape codeword alone identifies the symbol.
    let encode_val = move |v: u64| -> (u64, u32) {
        let len = coder.codeword_length(v);
        let bits = match coder.encode(v) {
            Some(cw) => cw,
            None => {
                let lit = if escaped_symbol_length == 0 {
                    0
                } else {
                    v.reverse_bits() >> (64 - escaped_symbol_length)
                };
                escape_codeword | (lit << escape_length)
            }
        };
        (bits, len)
    };

    move |vb: &mut VBuilder<BitVec<Box<[usize]>>, S, E>,
          attempt_seed: u64,
          mut store: Box<dyn ShardStore<S, u64> + Send + Sync>,
          _max_val: u64,
          num_keys: usize,
          pl: &mut P,
          _state: &mut ()| {
        // ── a) Compute per-shard sum of codeword lengths. ──
        // VBuilder's set_up_graphs was sized for the *key* count;
        // our graph has ~avg_codeword_length × more equations, so
        // we re-size here.
        let mut total_edges: usize = 0;
        let mut max_shard_edges: usize = 0;
        for shard in store.iter() {
            let mut sum: usize = 0;
            for sv in shard.iter() {
                sum += encode_val(sv.val).1 as usize;
            }
            total_edges += sum;
            max_shard_edges = max_shard_edges.max(sum);
        }

        // ── b) Re-call `set_up_graphs` with edge counts. ──
        // VBuilder already called it with the *key* count, but our
        // multi-edge construction has avg ≈ entropy more equations
        // than keys, so the shard-edge has to be resized against the
        // actual edge count. The returned `lge` flag tells us whether
        // `set_up_graphs` picked a *dense* layout (small expected core
        // → cheap LGE fallback) or a *sparse* one (pure peel expected
        // to succeed; a failing peel leaves a large core that LGE
        // cannot chew through, so we must retry the shard instead).
        let (c, lge) = vb.shard_edge.set_up_graphs(total_edges, max_shard_edges);
        vb.c = c;
        vb.lge = lge;

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
        pl.info(format_args!("{}", vb.shard_edge));
        let entropy = total_edges as f64 / num_keys as f64;
        pl.info(format_args!(
            "Huffman: max codeword length {}, escape length {}, escaped symbol length {}",
            coder.max_codeword_length(),
            coder.escape_length(),
            coder.escaped_symbol_length()
        ));
        pl.info(format_args!(
            "Average codeword length (entropy): {:.4} bits/key (total edges: {}, shards: {}, max shard edges: {})",
            entropy, total_edges, num_shards, max_shard_edges
        ));
        pl.info(format_args!(
            "c: {}, Overhead: {:+.4}% Number of threads: {}",
            c,
            100.0 * ((num_vertices_per_shard * num_shards) as f64 / (total_edges as f64) - 1.),
            vb.num_threads
        ));
        if lge {
            pl.info(format_args!(
                "Peeling with lazy Gaussian elimination fallback"
            ));
        }

        let raw_stride = num_vertices_per_shard + w;
        // `par_solve` chunks the data via `BitVec::try_chunks_mut`,
        // which requires `chunk_size` to be a multiple of `usize::BITS`.
        let stride = raw_stride.next_multiple_of(usize::BITS as usize);
        let padding = stride - num_vertices_per_shard;
        let total_bits = num_shards
            .checked_mul(stride)
            .ok_or_else(|| anyhow!("data size overflow"))?;

        // `new_padded` is defined on `BitVec<Vec<W>>` but returns a
        // `BitVec<Box<[W]>>`, matching the query-side storage type.
        let mut data = BitVec::<Box<[usize]>>::new_padded(total_bits);

        // ── d) Call `par_solve` with the multi-edge closure. ──
        // Dispatch on `lge` and `low_mem` exactly like
        // [`VBuilder::try_build_from_shard_iter`] does for VFunc
        // (vbuilder.rs:1184):
        //
        // * `lge == true`  → materialise the edges/rhs and drive
        //   [`solve_system`] (peel_by_index + LGE fallback). LGE
        //   needs the original edge list alive.
        //
        // * `lge == false` → stream edges directly into
        //   [`XorGraph<PackedEdge>`] via either
        //   [`peel_by_data_low_mem`] or [`peel_by_data_high_mem`],
        //   no LGE fallback. The low-mem heuristic mirrors VBuilder
        //   exactly: pick low-mem when explicitly requested, or by
        //   default when `num_threads > 3 && num_shards > 2` (i.e.
        //   the parallel build is memory-pressured).
        // Force the `peel_by_index` + LGE path when per-shard
        // vertex count exceeds `PackedEdge::MAX_VERTEX`: the
        // streamed peelers steal the top bit of `v2` for the rhs
        // flag and cannot represent vertex indices ≥ 2^31. This
        // only triggers for very large single-shard builds
        // (typically `FuseLge3NoShards` with >800M keys).
        let packed_edge_safe = raw_stride <= PackedEdge::MAX_VERTEX as usize;
        let force_index_peeler = lge || !packed_edge_safe;
        let use_low_mem = vb.low_mem == Some(true)
            || (vb.low_mem.is_none() && vb.num_threads > 3 && num_shards > 2);
        let solve_shard = |this: &VBuilder<BitVec<Box<[usize]>>, S, E>,
                           _shard_index: usize,
                           shard: std::sync::Arc<Vec<SigVal<S, u64>>>,
                           mut shard_data: BitVec<&mut [usize]>,
                           _pl: &mut _|
         -> Result<(), ()> {
            let shard_edge = &this.shard_edge;

            let solution: BitVec = if force_index_peeler {
                // Materialise edges/rhs so LGE (or the fallback
                // index peeler) can iterate the unpeeled remainder.
                // Same as the pre-streaming codepath.
                let mut edges: Vec<[u32; 3]> = Vec::new();
                let mut rhs: BitVec = BitVec::with_capacity(total_edges);
                for sv in shard.iter() {
                    let (bits, len) = encode_val(sv.val);
                    let local_sig = shard_edge.local_sig(sv.sig);
                    let base = shard_edge.local_edge(local_sig);
                    for l in 0..len as usize {
                        let off = w - 1 - l;
                        edges.push([
                            (base[0] + off) as u32,
                            (base[1] + off) as u32,
                            (base[2] + off) as u32,
                        ]);
                        rhs.push(((bits >> l) & 1) == 1);
                    }
                }
                solve_system(raw_stride, &edges, rhs, lge).map_err(|_| ())?
            } else if use_low_mem {
                peel_by_data_low_mem(shard, shard_edge, &encode_val, raw_stride, w)?
            } else {
                peel_by_data_high_mem(shard, shard_edge, &encode_val, raw_stride, w)?
            };

            // Copy the local `BitVec` solution into `shard_data` via
            // a word-level `copy_from_slice` instead of a bit-by-bit
            // loop. Both vectors are `usize`-backed with bit 0 at
            // word 0 / bit 0, so the layout matches exactly. We copy
            // `raw_stride.div_ceil(usize::BITS)` words, which may
            // include a few padding bits past `raw_stride` — those
            // are in the same shard's reserved region in `shard_data`
            // (length `stride = raw_stride.next_multiple_of(...)`),
            // so the write stays in-bounds.
            let n_words = raw_stride.div_ceil(usize::BITS as usize);
            let src: &[usize] = solution.as_ref();
            let dst: &mut [usize] = shard_data.as_mut();
            dst[..n_words].copy_from_slice(&src[..n_words]);
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
fn finish_build<K, S, E>(
    shard_edge: E,
    coder: HuffmanCoder,
    data: BitVec<Box<[usize]>>,
    seed_used: u64,
    num_keys: usize,
    shard_size: usize,
) -> CompVFunc<K, BitVec<Box<[usize]>>, S, E>
where
    K: ?Sized,
{
    CompVFunc {
        shard_edge,
        seed: seed_used,
        num_keys,
        shard_size,
        global_max_codeword_length: coder.max_codeword_length(),
        escape_length: coder.escape_length(),
        escaped_symbol_length: coder.escaped_symbol_length(),
        data,
        decoder: coder.into_decoder(),
        _marker: PhantomData,
    }
}

/// Empty-function short-circuit shared by both entry points.
fn empty_comp_vfunc<K, S, E>(
    coder: HuffmanCoder,
    eps: f64,
) -> CompVFunc<K, BitVec<Box<[usize]>>, S, E>
where
    K: ?Sized,
    E: ShardEdge<S, 3>,
{
    let mut shard_edge = E::default();
    shard_edge.set_up_shards(0, eps);
    CompVFunc {
        shard_edge,
        seed: 0,
        num_keys: 0,
        shard_size: 0,
        global_max_codeword_length: coder.max_codeword_length(),
        escape_length: coder.escape_length(),
        escaped_symbol_length: coder.escaped_symbol_length(),
        data: BitVec::<Box<[usize]>>::new_padded(0),
        decoder: coder.into_decoder(),
        _marker: PhantomData,
    }
}

/// Lender-based sequential path.
fn build_inner_seq<
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    B: ?Sized + Borrow<K>,
    V: FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend u64>,
    L: FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    P: ProgressLog + Clone + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    huffman: Huffman,
    builder: VBuilder<BitVec<Box<[usize]>>, S, E>,
    keys: L,
    mut values: V,
    n: usize,
    pl: &mut P,
) -> Result<CompVFunc<K, BitVec<Box<[usize]>>, S, E>>
where
    SigVal<S, u64>: RadixKey,
{
    // First pass: stream the values lender for the frequency
    // histogram, then rewind so `try_populate_and_build` can consume
    // the same lender to populate the sig-store.
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    while let Some(&v) = values.next()? {
        *frequencies.entry(v).or_insert(0) += 1;
    }
    let coder = build_coder_from_frequencies(huffman, frequencies);
    values = values.rewind()?;

    if n == 0 {
        return Ok(empty_comp_vfunc::<K, S, E>(coder, builder.eps));
    }

    // See `build_inner_par` for why we do *not* pass
    // `shard_size_hint(total_edges)`: it drops CompVFunc into a
    // per-shard formula regime where `peel_by_data_*` fail with
    // ~50% probability and force a retry.
    let mut builder = builder.expected_num_keys(n);
    let mut build_fn = make_build_fn::<S, E, P>(&coder);
    let ((data, seed_used, shard_size), _keys) =
        builder.try_populate_and_build(keys, values, &mut build_fn, pl, ())?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build::<K, S, E>(
        shard_edge, coder, data, seed_used, n, shard_size,
    ))
}

/// Slice-based parallel path.
fn build_inner_par<
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    B: Borrow<K> + Sync,
    P: ProgressLog + Clone + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
>(
    huffman: Huffman,
    builder: VBuilder<BitVec<Box<[usize]>>, S, E>,
    keys: &[B],
    values: &[u64],
    pl: &mut P,
) -> Result<CompVFunc<K, BitVec<Box<[usize]>>, S, E>>
where
    SigVal<S, u64>: RadixKey,
{
    let n = keys.len();
    let coder = build_coder_from_frequencies(huffman, frequencies_from_slice(values));

    if n == 0 {
        return Ok(empty_comp_vfunc::<K, S, E>(coder, builder.eps));
    }

    // NOTE: we do *not* pass `shard_size_hint(total_edges)` to the
    // VBuilder here. Empirically, sharding on the edge count drops
    // CompVFunc into a per-shard formula regime (~12M edges/shard,
    // c=1.11, small-formula log2_seg_size in `shard_edge.rs`)
    // where peeling fails ~50% of the time for CompVFunc's
    // correlated multi-edge hypergraphs. The old, key-based
    // sharding (8 shards × ~25M edges each) lands in the large-
    // formula regime (c=1.105, larger seg_size) where peeling is
    // reliable. Fixing this properly requires tuning the fuse
    // filter thresholds in `shard_edge.rs` for correlated edge
    // densities; until then, key-based sharding wins.
    let mut builder = builder.expected_num_keys(n);
    let mut build_fn = make_build_fn::<S, E, P>(&coder);
    let (data, seed_used, shard_size) =
        builder.try_par_populate_and_build(keys, &|i: usize| values[i], &mut build_fn, pl, ())?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build::<K, S, E>(
        shard_edge, coder, data, seed_used, n, shard_size,
    ))
}

// ── Linear-system solver ───────────────────────────────────────────

/// Output of [`peel_by_index`]: the peel-order data needed to drive
/// the reverse-peel assignment (and, on partial peels, the LGE
/// fallback). Mirrors the `Partial` variant of
/// [`crate::func::VBuilder`]'s `PeelResult`: the `XorGraph` itself is
/// dropped when peeling returns — callers only need the peeled edge
/// indices and their pivot sides.
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

/// Solves a 3-uniform F₂ system.
///
/// If `lge` is `false`, the graph was sized for pure peeling: we peel
/// and bail out with an error on any unpeeled remainder, so the caller
/// can retry the shard with a fresh seed. If `lge` is `true`, the graph
/// was sized for a small LGE core: we mirror [`VBuilder`]'s `lge_shard`
/// and run lazy Gaussian elimination on the unpeeled remainder, then
/// complete the peeled edges in reverse order.
///
/// Running LGE on a shard that was sized for pure peeling would be
/// catastrophic: if peeling fails, the unpeeled core can be a large
/// fraction of the shard, and LGE is cubic in the core size.
///
/// [`VBuilder`]: crate::func::VBuilder
fn solve_system(
    num_variables: usize,
    edges: &[[u32; 3]],
    rhs: BitVec,
    lge: bool,
) -> Result<BitVec> {
    let out = peel_by_index(num_variables, edges);
    let n_peeled = out.n_peeled();
    let n_total = edges.len();

    // Solution is a `BitVec` (1 bit per variable) instead of
    // `Vec<bool>` (1 byte). At 100M keys this saves ~220 MB peak.
    let mut solution: BitVec = BitVec::new(num_variables);

    if n_peeled < n_total {
        if !lge {
            // Graph was sized for pure peeling; a non-empty core means
            // we hit a bad seed. Bail out so `par_solve` can retry.
            bail!(
                "peeling failed on a non-LGE graph ({} unpeeled)",
                n_total - n_peeled
            );
        }
        // Mark which edges were peeled so the LGE equation builder
        // can filter them out. VBuilder's `lge_shard` uses the same
        // trick: `peeled_edges: BitVec` built from `iter_upper()`.
        let mut peeled_edges: BitVec = BitVec::new(n_total);
        for &edge_idx in out.double_stack.iter_upper() {
            peeled_edges.set(edge_idx as usize, true);
        }

        // Build LGE system on the non-peeled edges only. Variables that
        // do not appear in any non-peeled equation get value 0 from
        // LGE; the reverse-peel pass will overwrite the peeled-edge
        // pivots in `solution` afterwards.
        let mut equations: Vec<Modulo2Equation<usize>> = Vec::with_capacity(n_total - n_peeled);
        for i in 0..n_total {
            if peeled_edges[i] {
                continue;
            }
            let mut vs = edges[i];
            vs.sort_unstable();
            // F₂ pair cancellation for duplicate vertices in an edge.
            let vars: Vec<u32> = if vs[0] == vs[1] && vs[1] == vs[2] {
                vec![]
            } else if vs[0] == vs[1] {
                vec![vs[2]]
            } else if vs[1] == vs[2] {
                vec![vs[0]]
            } else if vs[0] == vs[2] {
                // Sorted, vs[0]==vs[2] would imply all equal.
                vec![]
            } else {
                vs.to_vec()
            };
            let r = rhs[i] as usize;
            if vars.is_empty() && r != 0 {
                bail!("trivial unsolvable equation");
            }
            // SAFETY: `vars` is sorted; entries come from `[u32; 3]`
            // indices that the caller bounds by `num_variables`.
            equations.push(unsafe { Modulo2Equation::<usize>::from_parts(vars, r) });
        }

        // SAFETY: see comment on the per-equation push above.
        let mut system = unsafe { Modulo2System::<usize>::from_parts(num_variables, equations) };
        let lge_solution = system
            .lazy_gaussian_elimination()
            .map_err(|e| anyhow!("LGE failed: {e}"))?;
        for (v, &val) in lge_solution.iter().enumerate() {
            solution.set(v, (val & 1) != 0);
        }
    }

    // Reverse-peel assignment. `DoubleStack::iter_upper()` already
    // yields items in newest-first order (the upper half grows
    // downward), which is exactly the order we want: assign the
    // latest-peeled pivot first, so that by the time a pivot's value
    // is set, the other two vertices of its edge are either
    // LGE-assigned or were pivots of edges peeled *later* (already
    // set earlier in this loop). `sides_stack` is a regular Vec
    // pushed in peel order, so we reverse it explicitly.
    //
    // Matching on `side` instead of the `if pivot != v` chain lets us
    // name the two non-pivot vertices directly and avoid three
    // branches per peeled edge.
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

    // Debug-only sanity check: every edge must be satisfied after
    // peeling + LGE + reverse-peel assignment.
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

    Ok(solution)
}

/// Peels a 3-uniform F₂ hypergraph by edge index, mirroring
/// [`VBuilder::peel_by_index`](crate::func::vbuilder). The payload
/// stored in the [`XorGraph`] is the edge index itself (`u32`), and
/// the original `edges` slice is kept alive by the caller so the
/// reverse-peel assignment and LGE fallback can reach back into it.
///
/// The returned [`PeelByIndexOutput`] is always non-partial in shape
/// — it contains the full XorGraph and the peel-order stacks. The
/// caller checks [`PeelByIndexOutput::n_peeled`] against `edges.len()`
/// to decide whether peeling succeeded or LGE fallback is needed.
///
/// Degenerate edges (two or three repeated vertices) are handled
/// naturally by [`XorGraph::add`]: each slot independently bumps the
/// packed `(degree, side)` byte and XORs the edge index into
/// `edges[v]`, matching F₂ semantics (`x + x = 0`). The reverse-peel
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
        // overflowed). Hopeless to peel; let LGE handle everything.
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
/// `num_variables < (1 << 31)` before routing to these peelers and
/// falls back to [`peel_by_index`] (which uses `XorGraph<u32>` with
/// no such restriction) when the limit is exceeded.
///
/// In practice, the default `FuseLge3Shards` strategy picks enough
/// shards to keep per-shard vertex counts well under `1 << 31` for
/// any reasonable input. The limit only bites for `FuseLge3NoShards`
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
/// once, calling `encode_val` to produce the `(bits, len)` for each
/// `SigVal`, and deriving the `len` hyperedges (with shifted vertex
/// triples) inline — no intermediate `edges: Vec<[u32; 3]>` or
/// `rhs: BitVec` materialization.
///
/// Returns the populated graph and the total number of edges inserted
/// (which is the sum of codeword lengths across the shard and is also
/// the upper bound for the reverse-peel stack).
fn populate_data_graph<S, E>(
    shard: &[SigVal<S, u64>],
    shard_edge: &E,
    encode_val: &impl Fn(u64) -> (u64, u32),
    num_variables: usize,
    w: usize,
) -> (XorGraph<PackedEdge>, usize)
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let mut xor_graph: XorGraph<PackedEdge> = XorGraph::new(num_variables);
    let mut n_edges: usize = 0;
    for sv in shard.iter() {
        let (bits, len) = encode_val(sv.val);
        let local_sig = shard_edge.local_sig(sv.sig);
        let base = shard_edge.local_edge(local_sig);
        for l in 0..len as usize {
            let off = w - 1 - l;
            let v0 = (base[0] + off) as u32;
            let v1 = (base[1] + off) as u32;
            let v2 = (base[2] + off) as u32;
            let rhs_bit = ((bits >> l) & 1) != 0;
            let pe = PackedEdge::new(v0, v1, v2, rhs_bit);
            xor_graph.add(v0 as usize, pe, 0);
            xor_graph.add(v1 as usize, pe, 1);
            xor_graph.add(v2 as usize, pe, 2);
            n_edges += 1;
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
/// and `solution` has length `num_variables`.
#[inline(always)]
unsafe fn reverse_peel_assign(solution: &mut BitVec, pe: PackedEdge, side: usize) {
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
/// No LGE fallback: any unpeeled remainder returns `Err(())`, which
/// `par_solve` turns into a seed-retry.
fn peel_by_data_high_mem<S, E>(
    shard: std::sync::Arc<Vec<SigVal<S, u64>>>,
    shard_edge: &E,
    encode_val: &impl Fn(u64) -> (u64, u32),
    num_variables: usize,
    w: usize,
) -> Result<BitVec, ()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let (mut xor_graph, n_edges) =
        populate_data_graph(&shard, shard_edge, encode_val, num_variables, w);

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
        return Err(());
    }

    drop(xor_graph);

    // Reverse-peel assignment. Iterate newest-first over both stacks
    // (they were pushed in peel order, oldest first). The intermediate
    // solution is a `BitVec` (1 bit per variable) instead of
    // `Vec<bool>` (1 byte per variable); at 100M keys this cuts the
    // intermediate from ~250 MB to ~30 MB.
    let mut solution: BitVec = BitVec::new(num_variables);
    for (&pe, &side) in peeled_edges_stack
        .iter()
        .rev()
        .zip(sides_stack.iter().rev())
    {
        // SAFETY: all vertex indices in `pe` are bounded by
        // `num_variables = solution.len()` from edge construction.
        unsafe {
            reverse_peel_assign(&mut solution, pe, side as usize);
        }
    }

    Ok(solution)
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
/// No LGE fallback: any unpeeled remainder returns `Err(())`.
fn peel_by_data_low_mem<S, E>(
    shard: std::sync::Arc<Vec<SigVal<S, u64>>>,
    shard_edge: &E,
    encode_val: &impl Fn(u64) -> (u64, u32),
    num_variables: usize,
    w: usize,
) -> Result<BitVec, ()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let (mut xor_graph, n_edges) =
        populate_data_graph(&shard, shard_edge, encode_val, num_variables, w);

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
        return Err(());
    }

    // Reverse-peel assignment. `iter_upper()` yields pivots in
    // newest-first order, which is exactly reverse peel order.
    // Using a `BitVec` for `solution` instead of `Vec<bool>` saves
    // ~220 MB peak memory at 100M keys.
    let mut solution: BitVec = BitVec::new(num_variables);
    for &pivot_v in double_stack.iter_upper() {
        let (pe, side) = xor_graph.edge_and_side(pivot_v as usize);
        // SAFETY: all vertex indices in `pe` are bounded by
        // `num_variables = solution.len()` from edge construction.
        unsafe {
            reverse_peel_assign(&mut solution, pe, side);
        }
    }

    Ok(solution)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dsi_progress_logger::no_logging;

    fn build_and_check(values: &[u64]) {
        let n = values.len();
        let keys: Vec<u64> = (0..n as u64).collect();
        let func = CompVFunc::<u64>::try_par_new(&keys, values, no_logging![]).expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }

    #[test]
    fn test_empty() {
        let values: Vec<u64> = vec![];
        let keys: Vec<u64> = vec![];
        let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
        assert!(func.is_empty());
        assert_eq!(func.len(), 0);
    }

    #[test]
    fn test_streaming_construction() {
        // Exercises the lender-based `try_new` path. Uses
        // `FromCloneableIntoIterator` as the key lender and
        // `FromSlice` to wrap the value vector as a lender (mirrors
        // the `-n` mode of the `comp_vfunc` binary) so that both
        // keys and values are consumed one at a time, not stored as
        // a slice by the constructor.
        use crate::utils::FromCloneableIntoIterator;
        use crate::utils::lenders::FromSlice;
        let n = 1000usize;
        let values: Vec<u64> = (0..n as u64).map(|i| i % 5).collect();
        let func = CompVFunc::<usize>::try_new(
            FromCloneableIntoIterator::from(0_usize..n),
            FromSlice::new(&values),
            n,
            no_logging![],
        )
        .expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(i), v, "mismatch at key {i}");
        }
    }

    #[test]
    fn test_single_value_distribution() {
        // 100 keys, all mapping to value 7. ZeroCodec-like behavior.
        let values: Vec<u64> = vec![7; 100];
        build_and_check(&values);
    }

    #[test]
    fn test_skewed_small() {
        // 200 keys with a 3-symbol skewed distribution.
        let mut values: Vec<u64> = Vec::with_capacity(200);
        for i in 0..200 {
            values.push(match i % 10 {
                0..=6 => 0,
                7 | 8 => 1,
                _ => 2,
            });
        }
        build_and_check(&values);
    }

    #[test]
    fn test_many_keys() {
        // Force a non-trivial number of keys and a moderately skewed
        // distribution. With FuseLge3Shards this is well above the
        // single-shard threshold.
        let n = 5_000usize;
        let values: Vec<u64> = (0..n)
            .map(|i| match i % 16 {
                0..=7 => 0u64,
                8..=11 => 1,
                12..=13 => 2,
                14 => 3,
                _ => 4,
            })
            .collect();
        let keys: Vec<u64> = (0..n as u64).collect();
        let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }

    #[test]
    fn test_string_keys() {
        // Verify the ToSig implementation works with string keys.
        let n = 300usize;
        let keys: Vec<String> = (0..n).map(|i| format!("key-{i:08}")).collect();
        let values: Vec<u64> = (0..n as u64).map(|i| i % 5).collect();
        let func = CompVFunc::<String>::try_par_new(&keys, &values, no_logging![]).expect("build");
        for (k, &v) in keys.iter().zip(values.iter()) {
            assert_eq!(func.get(k), v, "mismatch at {k}");
        }
    }

    #[test]
    fn test_try_into_unaligned() {
        // Build a normal CompVFunc, convert to the unaligned variant,
        // and check that queries still match.
        let n = 1500usize;
        let keys: Vec<u64> = (0..n as u64).collect();
        let values: Vec<u64> = (0..n as u64).map(|i| i % 7).collect();
        let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
        let unaligned = func.try_into_unaligned().expect("convert");
        for (k, &v) in keys.iter().zip(values.iter()) {
            assert_eq!(unaligned.get(*k), v, "mismatch at {k}");
        }
        // Round-trip back.
        let back: CompVFunc<u64> = unaligned.into();
        for (k, &v) in keys.iter().zip(values.iter()) {
            assert_eq!(back.get(*k), v, "mismatch at {k} after round-trip");
        }
    }

    #[test]
    fn test_with_escapes() {
        // 16 distinct values with very skewed distribution; the codec
        // should escape the rare ones.
        let mut values: Vec<u64> = Vec::with_capacity(2000);
        for i in 0..2000 {
            // Pareto-ish: most are 0, exponential tail.
            let v = if i % 100 < 50 {
                0
            } else if i % 100 < 75 {
                1
            } else if i % 100 < 88 {
                2
            } else if i % 100 < 94 {
                3
            } else if i % 100 < 97 {
                4
            } else {
                (5 + (i % 11)) as u64
            };
            values.push(v);
        }
        let keys: Vec<u64> = (0..values.len() as u64).collect();
        let func = CompVFunc::<u64>::try_par_new_with_builder(
            &keys,
            &values,
            Huffman::length_limited(8, 0.95),
            VBuilder::default(),
            no_logging![],
        )
        .expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }
}
