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

use crate::bits::{BitFieldVec, BitVec, BitVecU, ensure_unaligned_any_pos, test_unaligned_any_pos};
use crate::func::VBuilder;
use crate::func::codec::{Codec, Coder, Decoder, ESCAPE, Huffman, HuffmanCoder, HuffmanDecoder};
use crate::func::shard_edge::{FuseLge3Shards, ShardEdge};
use crate::traits::bit_vec_ops::BitVecValueOps;
use crate::traits::{TryIntoUnaligned, UnalignedConversionError};
use crate::utils::mod2_sys::{Modulo2Equation, Modulo2System};
use crate::utils::sig_store::ShardStore;
use crate::utils::{Sig, ToSig};
use anyhow::{Result, anyhow, bail};
use dsi_progress_logger::no_logging;
use mem_dbg::{MemDbg, MemSize};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;
use value_traits::slices::SliceByValueMut;

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
        ensure_unaligned_any_pos!(usize, w);
        if esym > 0 {
            ensure_unaligned_any_pos!(usize, esym);
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
    K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    crate::utils::SigVal<S, u64>: rdst::RadixKey,
{
    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using default [`VBuilder`] and [`Huffman`] settings.
    ///
    /// See also [`try_new_with_builder`](Self::try_new_with_builder) for
    /// the full configuration surface.
    pub fn try_new<B: Borrow<K> + Sync>(keys: &[B], values: &[u64]) -> Result<Self> {
        Self::try_new_with_builder(keys, values, Huffman::new(), VBuilder::default())
    }

    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using the given [`Huffman`] codec and [`VBuilder`] configuration.
    ///
    /// The `builder` argument controls every VBuilder-side construction
    /// knob (offline mode, thread count, sharding ε, PRNG seed, etc.);
    /// the `huffman` argument controls the codec used for the values.
    /// The data backend is pinned internally to
    /// `BitFieldVec<Box<[usize]>>` — the query-side [`BitVec`] is
    /// obtained by re-wrapping the raw storage at the end.
    pub fn try_new_with_builder<B: Borrow<K> + Sync>(
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
        build_inner::<K, B, S, E>(huffman, builder, keys, values)
    }
}

// ── Builder core ───────────────────────────────────────────────────
//
// `build_inner` is a thin adapter around
// [`VBuilder::try_par_populate_and_build`]: it delegates the sig-store
// population, retry loop, duplicate check, and parallel shard solving
// to VBuilder, and only contributes a CompVFunc-specific `build_fn`
// closure that (a) calls `set_up_graphs` on the shard edge with
// *equation* counts rather than key counts and (b) calls
// [`VBuilder::par_solve`] with a per-shard closure doing the
// multi-edge expansion, peel + LGE on remainder, and assignment.
//
// Storage during construction is `BitFieldVec<Box<[usize]>>` with
// `bit_width = 1` so it satisfies VBuilder's `D: BitFieldSlice +
// BitFieldSliceMut` bounds. After the build, we re-wrap the raw
// `Box<[usize]>` as a `BitVec` (same byte layout, different wrapper)
// so the query path keeps its `BitVec::get_value` / `BitVecU`
// unaligned-read interface.

fn build_inner<K, B, S, E>(
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
    crate::utils::SigVal<S, u64>: rdst::RadixKey,
{
    let n = keys.len();

    // ── Phase 1: frequencies + codec (independent of seed) ────────
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    for &v in values {
        *frequencies.entry(v).or_insert(0) += 1;
    }

    let coder: HuffmanCoder = huffman.build_coder(&frequencies);
    let global_max_codeword_length = coder.max_codeword_length();
    let escape_length = coder.escape_length();
    let escaped_symbol_length = coder.escaped_symbol_length();
    let w = global_max_codeword_length as usize;
    let escape_codeword = coder.escape();

    if n == 0 {
        let mut shard_edge = E::default();
        shard_edge.set_up_shards(0, builder.eps);
        return Ok(CompVFunc {
            shard_edge,
            seed: 0,
            num_keys: 0,
            shard_size: 0,
            global_max_codeword_length,
            escape_length,
            escaped_symbol_length,
            data: BitVec::<Vec<usize>>::new_padded(0),
            decoder: coder.into_decoder(),
            _marker: PhantomData,
        });
    }

    // Encode a single `(bits, len)` pair for a value. Used by both
    // the sum-computation pass and the per-shard closure.
    let coder_ref = &coder;
    let encode_val = move |v: u64| -> (u64, u32) {
        let len = coder_ref.codeword_length(v);
        let bits = match coder_ref.encode(v) {
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

    // Caller-supplied VBuilder is our orchestrator. We override
    // `expected_num_keys` so sharding is set up against the actual
    // key count, independently of anything the caller might have
    // set; all other VBuilder-side knobs (offline, check-dups,
    // low-mem, threads, eps, seed) are preserved.
    let mut builder = builder.expected_num_keys(n);
    let values_ref = values;

    let build_result: Result<(BitFieldVec<Box<[usize]>>, u64, usize)> = builder
        .try_par_populate_and_build(
            keys,
            &move |i: usize| values_ref[i],
            &mut |vb: &mut VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
                  attempt_seed: u64,
                  mut store: Box<dyn ShardStore<S, u64> + Send + Sync>,
                  _max_val: u64,
                  _num_keys: usize,
                  _pl: &mut _,
                  _state: &mut ()| {
                // ── a) Compute per-shard sum of codeword lengths. ──
                // VBuilder's set_up_graphs was sized for the *key*
                // count; our graph has ~`avg_codeword_length` ×
                // more equations, so we re-size here.
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
                // VBuilder already called it with the *key* count, but
                // our multi-edge construction has avg ≈ entropy more
                // equations than keys, so the shard-edge has to be
                // resized against the actual edge count.
                vb.shard_edge
                    .set_up_graphs(total_edges.max(1) as usize, max_shard_edges.max(1) as usize);

                // ── c) Compute the per-shard stride and allocate. ──
                let num_vertices_per_shard = vb.shard_edge.num_vertices();
                let num_shards = vb.shard_edge.num_shards();
                // `par_solve` derives `num_threads.ilog2()` for its
                // internal buffer size; ensure it's initialized to a
                // positive value. VFunc sets this inside
                // `try_build_from_shard_iter`, which we're
                // side-stepping, so we must set it ourselves.
                vb.num_threads = num_shards.min(vb.max_num_threads).max(1);
                let raw_stride = num_vertices_per_shard + w;
                // `par_solve` chunks the data via
                // `BitFieldVec::try_chunks_mut`, which requires the
                // product `chunk_size * bit_width` to be a multiple
                // of `usize::BITS`. Since `bit_width = 1`, chunk
                // size must itself be a multiple of `usize::BITS`.
                let stride = raw_stride.next_multiple_of(usize::BITS as usize);
                let padding = stride - num_vertices_per_shard;
                let total_bits = num_shards
                    .checked_mul(stride)
                    .ok_or_else(|| anyhow!("data size overflow"))?;

                let mut data = BitFieldVec::<Box<[usize]>>::new_padded(1, total_bits);

                // ── d) Call `par_solve` with the multi-edge closure. ──
                // The closure captures `coder_ref` (an `&HuffmanCoder`
                // shared across threads — HashMap<u64, usize> is
                // Sync) and the three escape-related constants.
                // `raw_stride` is the real variable count for the
                // solver; everything above it up to `stride` is word
                // alignment padding and stays zero.
                let solve_shard = |this: &VBuilder<BitFieldVec<Box<[usize]>>, S, E>,
                                   _shard_index: usize,
                                   shard: std::sync::Arc<Vec<crate::utils::SigVal<S, u64>>>,
                                   mut shard_data: BitFieldVec<&mut [usize]>,
                                   _pl: &mut _|
                 -> Result<(), ()> {
                    let shard_edge = &this.shard_edge;
                    let mut edges: Vec<[u32; 3]> = Vec::new();
                    let mut rhs: Vec<bool> = Vec::new();
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

                    let solution = solve_system(raw_stride, &edges, &rhs).map_err(|_| ())?;
                    for (i, &bit) in solution.iter().enumerate() {
                        // SAFETY: `i < raw_stride <= stride`, and
                        // shard_data has length `stride`.
                        unsafe {
                            shard_data.set_value_unchecked(i, bit as usize);
                        }
                    }
                    Ok(())
                };

                // Propagate the SolveError *unwrapped* so that
                // `try_par_populate_and_build`'s retry loop can
                // downcast it and re-roll the seed on
                // UnsolvableShard / MaxShardTooBig / duplicate-sig
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
            },
            no_logging![],
            (),
        );

    let (data_bfv, seed_used, shard_size) = build_result?;
    let shard_edge = builder.shard_edge;

    // Re-wrap the raw `Box<[usize]>` as a `BitVec` — same byte
    // layout, different meta. The padding word from `new_padded`
    // carries over.
    let (raw_bits, _bit_width, len_in_elements) = data_bfv.into_raw_parts();
    let data = unsafe { BitVec::<Box<[usize]>>::from_raw_parts(raw_bits, len_in_elements) };

    Ok(CompVFunc {
        shard_edge,
        seed: seed_used,
        num_keys: n,
        shard_size,
        global_max_codeword_length,
        escape_length,
        escaped_symbol_length,
        data,
        decoder: coder.into_decoder(),
        _marker: PhantomData,
    })
}

// ── Linear-system solver ───────────────────────────────────────────

/// Result of a partial peeling attempt.
struct PartialPeel {
    /// `peeled[i]` is `true` iff edge `i` was successfully peeled.
    peeled: Vec<bool>,
    /// `(pivot, edge_index)` pairs in **peel order**: the first push
    /// is the deepest peel, the last push is the most recent. The
    /// reverse-peel assignment iterates this in reverse.
    stack: Vec<(u32, u32)>,
}

/// Solves a 3-uniform F₂ system mirroring [`VBuilder`]'s `lge_shard`:
/// peel as much as possible, then run lazy Gaussian elimination on the
/// unpeeled remainder, then complete the peeled edges in reverse order.
///
/// [`VBuilder`]: crate::func::VBuilder
fn solve_system(num_variables: usize, edges: &[[u32; 3]], rhs: &[bool]) -> Result<Vec<bool>> {
    let peel = peel_partial(num_variables, edges);
    let n_peeled = peel.stack.len();
    let n_total = edges.len();

    let mut solution = vec![false; num_variables];

    if n_peeled < n_total {
        // Build LGE system on the non-peeled edges only. Variables that
        // do not appear in any non-peeled equation get value 0 from
        // LGE; the reverse-peel pass will overwrite the peeled-edge
        // pivots in `solution` afterwards.
        let mut equations: Vec<Modulo2Equation<usize>> = Vec::with_capacity(n_total - n_peeled);
        for (i, &was_peeled) in peel.peeled.iter().enumerate() {
            if was_peeled {
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
            solution[v] = (val & 1) != 0;
        }
    }

    // Reverse-peel assignment. By the peeling-order argument, when we
    // process stack[i] in reverse the *other* two vertices are either
    // (a) variables that never became a pivot (left at their LGE/zero
    // value) or (b) pivots of edges peeled *later* (i.e. processed
    // earlier in this reverse loop, hence already assigned).
    for &(pivot, edge_idx) in peel.stack.iter().rev() {
        let [a, b, c] = edges[edge_idx as usize];
        let mut val = rhs[edge_idx as usize];
        if a != pivot {
            val ^= solution[a as usize];
        }
        if b != pivot {
            val ^= solution[b as usize];
        }
        if c != pivot {
            val ^= solution[c as usize];
        }
        solution[pivot as usize] = val;
    }

    // Debug-only sanity check: every edge must be satisfied after
    // peeling + LGE + reverse-peel assignment.
    #[cfg(debug_assertions)]
    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let val = solution[a as usize] ^ solution[b as usize] ^ solution[c as usize];
        debug_assert_eq!(
            val, rhs[i],
            "edge {i} not satisfied: vars=[{a},{b},{c}] rhs={} got={}",
            rhs[i] as u8, val as u8,
        );
    }

    Ok(solution)
}

/// Greedy 3-uniform peeling. Returns the set of peeled edges and the
/// peel-order `(pivot, edge_index)` stack.
///
/// Degenerate edges (with two or three repeated vertices) are handled
/// naturally: the double increment of the repeated vertex's degree and
/// the double XOR of the edge index into its `edge_xor` both match the
/// F₂ semantics (`x + x = 0`), and the reverse-peel assignment XORs
/// `solution[b]` twice for `[a, b, b]` which cancels. Skipping
/// degenerate edges at insert time would leave the repeated vertex
/// eligible to be peeled as the pivot of some *other* edge, and the
/// reverse-peel would then overwrite the LGE-imposed constraint on
/// that vertex — violating the degenerate edge.
fn peel_partial(num_variables: usize, edges: &[[u32; 3]]) -> PartialPeel {
    let n_edges = edges.len();
    let mut peeled = vec![false; n_edges];
    let mut stack: Vec<(u32, u32)> = Vec::with_capacity(n_edges);

    let mut edge_xor: Vec<u32> = vec![0; num_variables];
    let mut degree: Vec<u8> = vec![0; num_variables];

    // Store `idx + 1` in edge_xor so that the default value `0`
    // unambiguously means "no edge incident on this vertex". Storing
    // the raw `idx` would make edge 0 collide with the "no edge"
    // sentinel because `0 ^ x = x` is the identity.
    //
    // Slots are processed *sequentially* so that a degenerate edge
    // `[v, v, u]` increments `degree[v]` twice (once per slot). A
    // naive parallel increment would read the same old `degree[v]`
    // for both slots and overwrite itself, producing `+1` instead of
    // `+2` and breaking the `degree == sum_of_slots` invariant.
    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let stored = (i + 1) as u32;
        for &v in &[a, b, c] {
            let v = v as usize;
            edge_xor[v] ^= stored;
            let (new_deg, overflow) = degree[v].overflowing_add(1);
            if overflow {
                // Degree overflow: any vertex with ≥256 incident
                // slots is hopeless to peel. Bail; LGE will handle
                // the whole remainder.
                return PartialPeel { peeled, stack };
            }
            degree[v] = new_deg;
        }
    }

    let mut to_visit: Vec<u32> = (0..num_variables as u32)
        .filter(|&v| degree[v as usize] == 1)
        .collect();

    while let Some(v) = to_visit.pop() {
        let vu = v as usize;
        if degree[vu] != 1 {
            continue;
        }
        // Decode 1-based stored value back to a 0-based edge index.
        let stored = edge_xor[vu];
        debug_assert_ne!(
            stored, 0,
            "degree[v]=1 but edge_xor[v]=0 (invariant broken)"
        );
        let edge_idx = (stored - 1) as u32;
        peeled[edge_idx as usize] = true;
        stack.push((v, edge_idx));
        let [a, b, c] = edges[edge_idx as usize];
        for &u in &[a, b, c] {
            let uu = u as usize;
            degree[uu] -= 1;
            edge_xor[uu] ^= stored;
            if degree[uu] == 1 {
                to_visit.push(u);
            }
        }
    }

    PartialPeel { peeled, stack }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_and_check(values: &[u64]) {
        let n = values.len();
        let keys: Vec<u64> = (0..n as u64).collect();
        let func = CompVFunc::<u64>::try_new(&keys, values).expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }

    #[test]
    fn test_empty() {
        let values: Vec<u64> = vec![];
        let keys: Vec<u64> = vec![];
        let func = CompVFunc::<u64>::try_new(&keys, &values).expect("build");
        assert!(func.is_empty());
        assert_eq!(func.len(), 0);
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
        let func = CompVFunc::<u64>::try_new(&keys, &values).expect("build");
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
        let func = CompVFunc::<String>::try_new(&keys, &values).expect("build");
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
        let func = CompVFunc::<u64>::try_new(&keys, &values).expect("build");
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
        let func = CompVFunc::<u64>::try_new_with_builder(
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
