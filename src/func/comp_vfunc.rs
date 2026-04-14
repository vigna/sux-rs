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

use crate::bits::{BitFieldVec, BitVec, BitVecU, test_unaligned_any_pos};
use crate::func::VBuilder;
use crate::func::codec::{Codec, Coder, Decoder, ESCAPE, Huffman, HuffmanCoder, HuffmanDecoder};
use crate::func::shard_edge::{FuseLge3Shards, ShardEdge};
use crate::traits::bit_vec_ops::BitVecValueOps;
use crate::traits::{TryIntoUnaligned, UnalignedConversionError};
use crate::utils::lenders::FromSlice;
use crate::utils::mod2_sys::{Modulo2Equation, Modulo2System};
use crate::utils::sig_store::ShardStore;
use crate::utils::{FallibleRewindableLender, Sig, SigVal, ToSig};
use anyhow::{Result, anyhow, bail};
use core::error::Error;
use dsi_progress_logger::no_logging;
use lender::FallibleLending;
use mem_dbg::{MemDbg, MemSize};
use rdst::RadixKey;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;
use value_traits::slices::{SliceByValue, SliceByValueMut};

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
// `VBuilder<BitFieldVec<Box<[usize]>>, S, E>` configured with the
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
    /// Builds a [`CompVFunc`] from lender-based streams of keys and a
    /// slice of values using default [`VBuilder`] and [`Huffman`]
    /// settings.
    ///
    /// Keys are consumed one at a time through the lender; this path
    /// is the right choice for input coming from disk
    /// ([`DekoBufLineLender`](crate::utils::DekoBufLineLender)) or
    /// synthetic ranges. The whole key set never needs to live in
    /// memory at once. Values, however, are passed as a slice because
    /// CompVFunc has to iterate them once to build the Huffman codec
    /// before starting construction.
    ///
    /// `n` is the expected number of keys. If it is significantly
    /// wrong, construction still works but may do extra retries.
    ///
    /// If keys are available as a slice and you want to parallelize
    /// the hashing phase, use [`try_par_new`](Self::try_par_new)
    /// instead.
    ///
    /// See also [`try_new_with_builder`](Self::try_new_with_builder)
    /// for the full configuration surface.
    pub fn try_new<L, B>(keys: L, values: &[u64], n: usize) -> Result<Self>
    where
        B: ?Sized + Borrow<K>,
        L: FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    {
        Self::try_new_with_builder(keys, values, n, Huffman::new(), VBuilder::default())
    }

    /// Builds a [`CompVFunc`] from a lender of keys and a slice of
    /// values using the given [`Huffman`] codec and [`VBuilder`]
    /// configuration.
    ///
    /// See [`try_new`](Self::try_new) for the streaming semantics.
    /// The `builder` argument controls every VBuilder-side
    /// construction knob (offline mode, thread count, sharding ε,
    /// PRNG seed, etc.); the `huffman` argument controls the codec
    /// used for the values. The data backend is pinned internally to
    /// `BitFieldVec<Box<[usize]>>` — the query-side [`BitVec`] is
    /// obtained by re-wrapping the raw storage at the end.
    pub fn try_new_with_builder<L, B>(
        keys: L,
        values: &[u64],
        n: usize,
        huffman: Huffman,
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
    ) -> Result<Self>
    where
        B: ?Sized + Borrow<K>,
        L: FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    {
        build_inner_seq::<K, B, S, E, L>(huffman, builder, keys, values, n)
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
    pub fn try_par_new<B: Borrow<K> + Sync>(keys: &[B], values: &[u64]) -> Result<Self> {
        Self::try_par_new_with_builder(keys, values, Huffman::new(), VBuilder::default())
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
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
    ) -> Result<Self> {
        if keys.len() != values.len() {
            bail!(
                "keys and values must have the same length ({} vs {})",
                keys.len(),
                values.len()
            );
        }
        build_inner_par::<K, B, S, E>(huffman, builder, keys, values)
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
// Storage during construction is `BitFieldVec<Box<[usize]>>` with
// `bit_width = 1` so it satisfies VBuilder's `D: BitFieldSlice +
// BitFieldSliceMut` bounds. After the build, we re-wrap the raw
// `Box<[usize]>` as a `BitVec` (same byte layout, different wrapper)
// so the query path keeps its `BitVec::get_value` / `BitVecU`
// unaligned-read interface.

/// Builds a [`HuffmanCoder`] from the value distribution. Used by
/// both `build_inner_seq` and `build_inner_par`.
fn build_coder(huffman: Huffman, values: &[u64]) -> HuffmanCoder {
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    for &v in values {
        *frequencies.entry(v).or_insert(0) += 1;
    }
    huffman.build_coder(&frequencies)
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
    &mut VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
    u64,
    Box<dyn ShardStore<S, u64> + Send + Sync>,
    u64,
    usize,
    &mut P,
    &mut (),
) -> Result<(BitFieldVec<Box<[usize]>>, u64, usize)>
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

    move |vb: &mut VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
          attempt_seed: u64,
          mut store: Box<dyn ShardStore<S, u64> + Send + Sync>,
          _max_val: u64,
          _num_keys: usize,
          _pl: &mut P,
          _state: &mut ()| {
        // ── a) Compute per-shard sum of codeword lengths. ──
        // VBuilder's set_up_graphs was sized for the *key* count;
        // our graph has ~avg_codeword_length × more equations, so
        // we re-size here.
        let mut total_edges: u64 = 0;
        let mut max_shard_edges: u64 = 0;
        for shard in store.iter() {
            let mut sum: u64 = 0;
            for sv in shard.iter() {
                sum += encode_val(sv.val).1 as u64;
            }
            total_edges += sum;
            max_shard_edges = max_shard_edges.max(sum);
        }

        // ── b) Re-call `set_up_graphs` with edge counts. ──
        // Capture `lge`: `true` means the graph's `c` is at/below
        // the peeling threshold and we *must* use LGE as a fallback
        // for the unpeeled core. `false` means `c` is above the
        // peeling threshold and pure peeling is expected to succeed.
        let (_c, require_lge) = vb
            .shard_edge
            .set_up_graphs(total_edges.max(1) as usize, max_shard_edges.max(1) as usize);
        vb.lge = require_lge;

        // ── c) Compute the per-shard stride and allocate. ──
        let num_vertices_per_shard = vb.shard_edge.num_vertices();
        let num_shards = vb.shard_edge.num_shards();
        // `par_solve` derives `num_threads.ilog2()` for its internal
        // buffer size; ensure it's initialized to a positive value.
        // VFunc sets this inside `try_build_from_shard_iter`, which
        // we're side-stepping, so we must set it ourselves.
        vb.num_threads = num_shards.min(vb.max_num_threads).max(1);

        // Peeling-strategy selection mirrors VBuilder's logic in
        // `try_build_func_and_store`:
        //
        //   * `require_lge == true`                      → by-index + LGE
        //   * `low_mem == Some(true)` (or auto: threads>3 and
        //     num_shards>2)                              → data low-mem
        //   * otherwise                                   → data high-mem
        //
        // The two "data" peelers drop the flat `edges`/`rhs`
        // buffers after building the XorGraph, so peak memory is
        // dominated by the XorGraph itself. They do **not** support
        // LGE fallback — if peeling fails we bail with
        // `SolveError::UnsolvableShard` and VBuilder retries with a
        // new seed (the same recovery path used by the two sibling
        // peelers in VFunc).
        let low_mem_auto = vb.num_threads > 3 && num_shards > 2;
        let strategy = if require_lge {
            PeelStrategy::ByIndexLge
        } else if vb.low_mem == Some(true) || (vb.low_mem.is_none() && low_mem_auto) {
            PeelStrategy::DataLowMem
        } else {
            PeelStrategy::DataHighMem
        };
        let raw_stride = num_vertices_per_shard + w;
        // `par_solve` chunks the data via `BitFieldVec::try_chunks_mut`,
        // which requires the product `chunk_size * bit_width` to be
        // a multiple of `usize::BITS`. Since `bit_width = 1`, chunk
        // size must itself be a multiple of `usize::BITS`.
        let stride = raw_stride.next_multiple_of(usize::BITS as usize);
        let padding = stride - num_vertices_per_shard;
        let total_bits = num_shards
            .checked_mul(stride)
            .ok_or_else(|| anyhow!("data size overflow"))?;

        let mut data = BitFieldVec::<Box<[usize]>>::new_padded(1, total_bits);

        // ── d) Call `par_solve` with the multi-edge closure. ──
        //
        // The new design eliminates the previous `Vec<[u32; 3]> +
        // Vec<bool>` intermediate buffers entirely: edges are
        // generated on the fly by `gen_edges` closures handed to the
        // generic peelers in [`crate::func::peeling`]. Each strategy
        // dispatches differently:
        //
        // * **ByIndexLge**: payload = `u32` encoding `key_idx * w +
        //   l`. The peeler borrows the shard via the `gen_edges` and
        //   `verts_of` closures, so the shard stays alive for the
        //   reverse-peel assignment — and for the LGE residual pass
        //   on a partial peel.
        //
        // * **DataHighMem** / **DataLowMem**: payload = `PackedEdge`
        //   (`u128` packing `v0 | v1 | v2 | rhs`). The `gen_edges`
        //   closure takes the shard *by value* and lets it drop at
        //   end-of-closure — the peeler then proceeds with only the
        //   `XorGraph` in memory.
        let solve_shard = move |this: &VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
                                _shard_index: usize,
                                shard: std::sync::Arc<Vec<SigVal<S, u64>>>,
                                mut shard_data: BitFieldVec<&mut [usize]>,
                                _pl: &mut _|
              -> Result<(), ()> {
            let shard_edge = &this.shard_edge;
            let failed = &this.failed;

            // First pass over the shard to compute the exact edge
            // count for this build (sum of codeword lengths). The
            // peelers need it to size their internal stacks and to
            // detect partial peels (`upper_len() == num_edges`).
            let num_edges: usize = shard
                .iter()
                .map(|sv| encode_val(sv.val).1 as usize)
                .sum();

            match strategy {
                PeelStrategy::ByIndexLge => {
                    // Bound check on the `key_idx * w + l` encoding:
                    // we use `u32` for the `XorGraph` payload, so the
                    // id space must fit. For all practical inputs
                    // this is comfortable (e.g. 100 M keys × w=20 =
                    // 2 G).
                    let id_space = shard.len().checked_mul(w).expect("id space overflow");
                    assert!(
                        id_space <= u32::MAX as usize,
                        "ByIndexLge id space ({} keys × w={}) exceeds u32::MAX",
                        shard.len(),
                        w
                    );

                    let output = peel_by_index::<u32, _, _, _, _>(
                        raw_stride,
                        num_edges,
                        failed,
                        |v| v as u32,
                        |i| i as usize,
                        |xg| {
                            // Both gen_edges and verts_of below
                            // borrow `shard` immutably. Shared
                            // borrows compose, so this is fine.
                            for (id, [a, b, c], _rhs) in
                                iter_multi_edges(&shard, shard_edge, &encode_val, w)
                            {
                                xg.add(a, id, 0);
                                xg.add(b, id, 1);
                                xg.add(c, id, 2);
                            }
                        },
                        |id| decode_multi_edge(id, &shard, shard_edge, &encode_val, w).0,
                    )?;

                    match output {
                        IndexPeelOutput::Complete { .. } => {
                            // Direct reverse-peel into `shard_data`.
                            for (id, side) in output.iter_reverse_peel() {
                                let (verts, rhs_bit) =
                                    decode_multi_edge(id, &shard, shard_edge, &encode_val, w);
                                assign_pivot(&mut shard_data, verts, rhs_bit, side);
                            }
                        }
                        IndexPeelOutput::Partial {
                            double_stack,
                            sides_stack,
                        } => {
                            // LGE fallback on the unpeeled core.
                            lge_fallback(
                                &shard,
                                shard_edge,
                                &encode_val,
                                w,
                                raw_stride,
                                &mut shard_data,
                                double_stack,
                                sides_stack,
                            )
                            .map_err(|_| ())?;
                        }
                    }
                }
                PeelStrategy::DataHighMem => {
                    // gen_edges takes `shard` by value (move) so it
                    // drops as soon as the graph is built; verts_of
                    // operates on the self-contained `PackedEdge`
                    // payload alone.
                    let shard_for_gen = shard;
                    let Some(output) = peel_by_data_high_mem::<PackedEdge, _, _>(
                        raw_stride,
                        num_edges,
                        failed,
                        move |xg| {
                            add_packed_edges(xg, &shard_for_gen, shard_edge, &encode_val, w)
                        },
                        |pe: PackedEdge| pe.unpack().0,
                    )?
                    else {
                        return Err(());
                    };
                    assign_packed_edge_record(&mut shard_data, output.iter_reverse_peel());
                }
                PeelStrategy::DataLowMem => {
                    let shard_for_gen = shard;
                    let Some(output) = peel_by_data_low_mem::<PackedEdge, _, _>(
                        raw_stride,
                        num_edges,
                        failed,
                        move |xg| {
                            add_packed_edges(xg, &shard_for_gen, shard_edge, &encode_val, w)
                        },
                        |pe: PackedEdge| pe.unpack().0,
                    )?
                    else {
                        return Err(());
                    };
                    assign_packed_edge_record(&mut shard_data, output.iter_reverse_peel());
                }
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
            no_logging![],
            no_logging![],
        )
        .map_err(anyhow::Error::from)?;

        Ok((data, attempt_seed, stride))
    }
}

/// Finalizes a successful build by re-wrapping the construction-side
/// `BitFieldVec<Box<[usize]>>` (bit width 1) as the query-side
/// [`BitVec`]. Same byte layout, different wrapper.
fn finish_build<K, S, E>(
    shard_edge: E,
    coder: HuffmanCoder,
    data_bfv: BitFieldVec<Box<[usize]>>,
    seed_used: u64,
    num_keys: usize,
    shard_size: usize,
) -> CompVFunc<K, BitVec<Box<[usize]>>, S, E>
where
    K: ?Sized,
{
    let (raw_bits, _bit_width, len_in_elements) = data_bfv.into_raw_parts();
    let data = unsafe { BitVec::<Box<[usize]>>::from_raw_parts(raw_bits, len_in_elements) };
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
        data: BitVec::<Vec<usize>>::new_padded(0),
        decoder: coder.into_decoder(),
        _marker: PhantomData,
    }
}

/// Lender-based sequential path.
fn build_inner_seq<K, B, S, E, L>(
    huffman: Huffman,
    builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
    keys: L,
    values: &[u64],
    n: usize,
) -> Result<CompVFunc<K, BitVec<Box<[usize]>>, S, E>>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug,
    B: ?Sized + Borrow<K>,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, u64>: RadixKey,
    L: FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
{
    let coder = build_coder(huffman, values);

    if n == 0 {
        return Ok(empty_comp_vfunc::<K, S, E>(coder, builder.eps));
    }

    let mut builder = builder.expected_num_keys(n);
    let mut build_fn = make_build_fn::<S, E, _>(&coder);
    let ((data_bfv, seed_used, shard_size), _keys) = builder.try_populate_and_build(
        keys,
        FromSlice::new(values),
        &mut build_fn,
        no_logging![],
        (),
    )?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build::<K, S, E>(
        shard_edge, coder, data_bfv, seed_used, n, shard_size,
    ))
}

/// Slice-based parallel path.
fn build_inner_par<K, B, S, E>(
    huffman: Huffman,
    builder: VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
    keys: &[B],
    values: &[u64],
) -> Result<CompVFunc<K, BitVec<Box<[usize]>>, S, E>>
where
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    B: Borrow<K> + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    SigVal<S, u64>: RadixKey,
{
    let n = keys.len();
    let coder = build_coder(huffman, values);

    if n == 0 {
        return Ok(empty_comp_vfunc::<K, S, E>(coder, builder.eps));
    }

    let mut builder = builder.expected_num_keys(n);
    let mut build_fn = make_build_fn::<S, E, _>(&coder);
    let (data_bfv, seed_used, shard_size) = builder.try_par_populate_and_build(
        keys,
        &|i: usize| values[i],
        &mut build_fn,
        no_logging![],
        (),
    )?;
    drop(build_fn);
    let shard_edge = builder.shard_edge;
    Ok(finish_build::<K, S, E>(
        shard_edge, coder, data_bfv, seed_used, n, shard_size,
    ))
}

// ── Linear-system solver ───────────────────────────────────────────
//
// CompVFunc shares the same three-strategy peeling architecture as
// [`VFunc`]: an index peeler with LGE fallback (the only strategy that
// can recover from a partial peel) plus two data-payload peelers
// (one fast/high-mem, one low-mem) that can drop the shard
// immediately after building the [`XorGraph`].
//
// Where VFunc has **one edge per key** (its `SigVal` is a "natural"
// edge id), CompVFunc has **L ≥ 1 edges per key** — one per codeword
// bit of the encoded value, at vertices `base + (w − 1 − l)`. This
// section threads the multi-edge structure through the **generic**
// peelers in [`crate::func::peeling`] via two closures:
//
// 1. `gen_edges`: iterates the shard, calls `encode_val(sv.val)` to
//    get `(bits, len)`, computes the `len` multi-edge vertices, and
//    adds them to the [`XorGraph`].
// 2. `verts_of`: given the `XorGraph` payload of a peeled edge,
//    recovers the 3 vertices.
//
// For the index peeler (`PeelStrategy::ByIndexLge`), the payload is a
// `u32` encoding `key_idx * w + l`. The closure recovers `(key_idx,
// l)` by integer division and then re-derives the vertices from the
// shard. The LGE fallback iterates the shard a second time, recomputes
// each `(key_idx, l)` edge, and adds the non-peeled ones to a residual
// `Modulo2System`.
//
// For the data peelers (`PeelStrategy::Data{High,Low}Mem`), the
// payload is a [`PackedEdge`] (`u128` packing `(v0, v1, v2, rhs)`) so
// the closure that takes ownership of the shard can drop it right
// after building the `XorGraph` — the payload alone is enough for
// reverse-peel assignment.
//
// `PeelStrategy::Data{High,Low}Mem` do **not** support LGE fallback:
// once the shard is dropped we can't reconstruct the residual system.
// On peeling failure they bail with `SolveError::UnsolvableShard` and
// VBuilder's retry loop re-rolls the seed (the same recovery path
// VFunc uses).

use crate::func::peeling::{
    DoubleStack, IndexPeelOutput, peel_by_data_high_mem, peel_by_data_low_mem, peel_by_index,
};

/// Dispatch selector for the per-shard peeling strategy. Mirrors
/// VBuilder's selection logic in `try_build_func_and_store`, adapted
/// for the multi-edge CompVFunc case.
#[derive(Copy, Clone, Debug)]
enum PeelStrategy {
    /// Index peeler + LGE fallback on the unpeeled core. Required
    /// when `c` sits at or below the fuse-graph peeling threshold;
    /// the shard stays alive so the LGE residual system can be
    /// reconstructed if peeling can't complete on its own.
    ByIndexLge,
    /// Data peeler with packed-edge `XorGraph` payload and a
    /// `FastStack` of payloads for the peel record. Drops the shard
    /// after graph construction. Fastest at assign time, highest
    /// memory.
    DataHighMem,
    /// Data peeler with the same packed-edge payload but a
    /// `DoubleStack<u32>` for peel-order tracking (half the stack
    /// memory of the high-mem variant).
    DataLowMem,
}

/// Packed edge payload for the `by_data` peelers.
///
/// Layout (low → high): `v0` (32b) | `v1` (32b) | `v2` (32b) | `rhs`
/// (32b). The `rhs` slot is 32 bits wide only because `u128` has no
/// finer alignment — it really carries a single bit. The whole struct
/// supports `BitXorAssign + Default + Copy` so it can be stored
/// inside an [`XorGraph`].
///
/// When three incidences of the same edge have been added to an
/// `XorGraph`, each of its 3 vertices holds the same `PackedEdge`. As
/// peeling removes the edge from two of its vertices, the XOR
/// contributions cancel pairwise and the third vertex is left with
/// the edge's full payload — from which we recover its three
/// endpoints and RHS.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
struct PackedEdge(u128);

impl PackedEdge {
    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, rhs: bool) -> Self {
        PackedEdge(
            (v0 as u128)
                | ((v1 as u128) << 32)
                | ((v2 as u128) << 64)
                | ((rhs as u128) << 96),
        )
    }

    #[inline(always)]
    fn unpack(self) -> ([usize; 3], bool) {
        let v0 = self.0 as u32 as usize;
        let v1 = (self.0 >> 32) as u32 as usize;
        let v2 = (self.0 >> 64) as u32 as usize;
        let rhs = ((self.0 >> 96) & 1) != 0;
        ([v0, v1, v2], rhs)
    }
}

impl core::ops::BitXorAssign for PackedEdge {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

/// Adds every multi-edge of `shard` to a `PackedEdge`-backed
/// [`XorGraph`]. Used as the body of the `gen_edges` closure for both
/// `peel_by_data_high_mem` and `peel_by_data_low_mem`.
#[inline]
fn add_packed_edges<S, E, F>(
    xg: &mut crate::func::peeling::XorGraph<PackedEdge>,
    shard: &[SigVal<S, u64>],
    shard_edge: &E,
    encode_val: &F,
    w: usize,
) where
    S: Sig,
    E: ShardEdge<S, 3>,
    F: Fn(u64) -> (u64, u32),
{
    for (_id, [v0, v1, v2], rhs) in iter_multi_edges(shard, shard_edge, encode_val, w) {
        let pe = PackedEdge::new(v0 as u32, v1 as u32, v2 as u32, rhs);
        xg.add(v0, pe, 0);
        xg.add(v1, pe, 1);
        xg.add(v2, pe, 2);
    }
}

/// Drives the reverse-peel assignment for a `PackedEdge`-based peel
/// record (the output of either `peel_by_data_high_mem` or
/// `peel_by_data_low_mem`). Iterates `(payload, side)` in
/// reverse-peel order and writes each pivot into `shard_data`.
#[inline]
fn assign_packed_edge_record(
    shard_data: &mut BitFieldVec<&mut [usize]>,
    iter: impl Iterator<Item = (PackedEdge, u8)>,
) {
    for (pe, side) in iter {
        let (verts, rhs_bit) = pe.unpack();
        assign_pivot(shard_data, verts, rhs_bit, side);
    }
}

/// Reverse-peel pivot assignment: writes `solution[pivot] = rhs ⊕
/// solution[other1] ⊕ solution[other2]` into the shard data, where
/// `pivot` is `verts[side]` and `other1`/`other2` are the other two
/// vertices of the edge.
///
/// `shard_data` is read-then-written: the "other two" vertices are
/// either non-pivots (still 0 in the shard data) or pivots of edges
/// that were peeled later than the current one (already assigned in
/// this reverse-peel loop, since we iterate newest-first). For the
/// LGE-fallback path, the LGE solver writes the residual values into
/// `shard_data` *before* the reverse peel runs, so reads of vertices
/// in the unpeeled core return the LGE-supplied values.
#[inline(always)]
fn assign_pivot(
    shard_data: &mut BitFieldVec<&mut [usize]>,
    verts: [usize; 3],
    rhs_bit: bool,
    side: u8,
) {
    let [a, b, c] = verts;
    unsafe {
        let xor_other = match side {
            0 => shard_data.get_value_unchecked(b) ^ shard_data.get_value_unchecked(c),
            1 => shard_data.get_value_unchecked(a) ^ shard_data.get_value_unchecked(c),
            2 => shard_data.get_value_unchecked(a) ^ shard_data.get_value_unchecked(b),
            _ => core::hint::unreachable_unchecked(),
        };
        let pivot = verts[side as usize];
        shard_data.set_value_unchecked(pivot, (rhs_bit as usize) ^ xor_other);
    }
}

/// Recovers `(verts, rhs_bit)` for a multi-edge id encoded as
/// `key_idx * w + l`.
///
/// Used by the `ByIndexLge` strategy at both peel-time
/// (`peel_by_index`'s `verts_of` closure, where we need a single
/// random-access lookup) and assignment time (reverse-peel).
#[inline(always)]
fn decode_multi_edge<S, E>(
    id: u32,
    shard: &[SigVal<S, u64>],
    shard_edge: &E,
    encode_val: &impl Fn(u64) -> (u64, u32),
    w: usize,
) -> ([usize; 3], bool)
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let key_idx = (id as usize) / w;
    let l = (id as usize) % w;
    let sv = &shard[key_idx];
    let (bits, _len) = encode_val(sv.val);
    let local_sig = shard_edge.local_sig(sv.sig);
    let base = shard_edge.local_edge(local_sig);
    let off = w - 1 - l;
    let verts = [base[0] + off, base[1] + off, base[2] + off];
    let rhs = ((bits >> l) & 1) != 0;
    (verts, rhs)
}

/// Iterates the multi-edges of a shard in `(id, verts, rhs)` order,
/// where `id = key_idx * w + l` is the [`peel_by_index`] payload
/// encoding and `verts` are the 3 vertices of the *l*-th codeword
/// edge for that key.
///
/// This is the **single source of truth** for the multi-edge
/// generation logic — both the [`peel_by_index`] `gen_edges` closure
/// and the [`lge_fallback`] residual-system loop iterate via this
/// function. It avoids inlining the same `for sv in shard { for l
/// in 0..len { ... } }` body in three different places.
#[inline]
fn iter_multi_edges<'a, S, E, F>(
    shard: &'a [SigVal<S, u64>],
    shard_edge: &'a E,
    encode_val: &'a F,
    w: usize,
) -> impl Iterator<Item = (u32, [usize; 3], bool)> + 'a
where
    S: Sig + 'a,
    E: ShardEdge<S, 3> + 'a,
    F: Fn(u64) -> (u64, u32) + 'a,
{
    let w_u32 = w as u32;
    shard.iter().enumerate().flat_map(move |(key_idx, sv)| {
        let (bits, len) = encode_val(sv.val);
        let local_sig = shard_edge.local_sig(sv.sig);
        let base = shard_edge.local_edge(local_sig);
        let key_idx_u32 = key_idx as u32;
        (0..len).map(move |l| {
            let off = w - 1 - l as usize;
            let id = key_idx_u32 * w_u32 + l;
            let verts = [base[0] + off, base[1] + off, base[2] + off];
            let rhs = ((bits >> l) & 1) != 0;
            (id, verts, rhs)
        })
    })
}

/// LGE fallback for the `ByIndexLge` strategy: builds the residual
/// system from the non-peeled multi-edges of the shard, solves it
/// with lazy Gaussian elimination, writes the solution into
/// `shard_data`, then drives the reverse peel from `double_stack` /
/// `sides_stack`.
fn lge_fallback<S, E>(
    shard: &[SigVal<S, u64>],
    shard_edge: &E,
    encode_val: &impl Fn(u64) -> (u64, u32),
    w: usize,
    num_vertices: usize,
    shard_data: &mut BitFieldVec<&mut [usize]>,
    double_stack: DoubleStack<u32>,
    sides_stack: Vec<u8>,
) -> Result<()>
where
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let id_space = shard.len() * w;
    let mut peeled_mask = vec![false; id_space];
    for &id in double_stack.iter_upper() {
        peeled_mask[id as usize] = true;
    }

    // Build LGE system from non-peeled multi-edges. Reuses
    // `iter_multi_edges` so the encoding stays in lockstep with the
    // peeler's `gen_edges` closure.
    let mut equations: Vec<Modulo2Equation<usize>> = Vec::new();
    for (id, verts, rhs_bit) in iter_multi_edges(shard, shard_edge, encode_val, w) {
        if peeled_mask[id as usize] {
            continue;
        }
        let mut vs = verts;
        vs.sort_unstable();
        // F₂ pair cancellation for degenerate edges (repeated
        // vertices). Note that all three vertices are derived from
        // `base + off`, so the only way to get a repeated vertex is
        // for the underlying ShardEdge to produce one.
        let vars: Vec<u32> = if vs[0] == vs[1] && vs[1] == vs[2] {
            vec![]
        } else if vs[0] == vs[1] {
            vec![vs[2] as u32]
        } else if vs[1] == vs[2] {
            vec![vs[0] as u32]
        } else if vs[0] == vs[2] {
            vec![]
        } else {
            vec![vs[0] as u32, vs[1] as u32, vs[2] as u32]
        };
        let r = rhs_bit as usize;
        if vars.is_empty() && r != 0 {
            bail!("trivial unsolvable equation");
        }
        // SAFETY: `vars` is sorted; entries come from
        // `local_edge`-derived offsets bounded by `num_vertices`.
        equations.push(unsafe { Modulo2Equation::<usize>::from_parts(vars, r) });
    }

    let mut system = unsafe { Modulo2System::<usize>::from_parts(num_vertices, equations) };
    let lge_solution = system
        .lazy_gaussian_elimination()
        .map_err(|e| anyhow!("LGE failed: {e}"))?;

    // Write LGE values into shard_data BEFORE the reverse peel, so
    // that subsequent `get_value_unchecked` calls in `assign_pivot`
    // see the LGE-supplied values for vertices in the unpeeled core.
    for (v, &val) in lge_solution.iter().enumerate() {
        if (val & 1) != 0 {
            // SAFETY: `v < num_vertices ≤ shard_data.len()`.
            unsafe {
                shard_data.set_value_unchecked(v, 1);
            }
        }
    }

    // Reverse-peel. `iter_upper()` yields peeled edge ids
    // newest-first; `sides_stack.iter().rev()` matches.
    for (&id, &side) in double_stack.iter_upper().zip(sides_stack.iter().rev()) {
        let (verts, rhs_bit) = decode_multi_edge(id, shard, shard_edge, encode_val, w);
        assign_pivot(shard_data, verts, rhs_bit, side);
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_and_check(values: &[u64]) {
        let n = values.len();
        let keys: Vec<u64> = (0..n as u64).collect();
        let func = CompVFunc::<u64>::try_par_new(&keys, values).expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }

    #[test]
    fn test_empty() {
        let values: Vec<u64> = vec![];
        let keys: Vec<u64> = vec![];
        let func = CompVFunc::<u64>::try_par_new(&keys, &values).expect("build");
        assert!(func.is_empty());
        assert_eq!(func.len(), 0);
    }

    #[test]
    fn test_streaming_construction() {
        // Exercises the lender-based `try_new` path. Uses
        // `FromCloneableIntoIterator` as the key lender (mirrors the
        // `-n` mode of the `comp_vfunc` binary) so that keys are
        // consumed one at a time, not materialized as a slice.
        use crate::utils::FromCloneableIntoIterator;
        let n = 1000usize;
        let values: Vec<u64> = (0..n as u64).map(|i| i % 5).collect();
        let func = CompVFunc::<usize>::try_new(
            FromCloneableIntoIterator::from(0_usize..n),
            &values,
            n,
        )
        .expect("build");
        for i in 0..n {
            assert_eq!(func.get(i), values[i], "mismatch at key {i}");
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
        let func = CompVFunc::<u64>::try_par_new(&keys, &values).expect("build");
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
        let func = CompVFunc::<String>::try_par_new(&keys, &values).expect("build");
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
        let func = CompVFunc::<u64>::try_par_new(&keys, &values).expect("build");
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
        )
        .expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }
}
