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

use crate::bits::{BitVec, BitVecU};
use crate::func::codec::{Codec, Coder, Decoder, ESCAPE, Huffman, HuffmanCoder, HuffmanDecoder};
use crate::func::mix64;
use crate::func::shard_edge::{FuseLge3Shards, ShardEdge};
use crate::traits::bit_vec_ops::{BitVecOpsMut, BitVecValueOps};
use crate::traits::{TryIntoUnaligned, UnalignedConversionError};
use crate::utils::{Sig, ToSig};
use crate::utils::mod2_sys::{Modulo2Equation, Modulo2System};
use anyhow::{Result, anyhow, bail};
use mem_dbg::{MemDbg, MemSize};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::RngExt;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;

// ── Layout constants ────────────────────────────────────────────────

/// Per-shard variable count overhead factor for the peel-only path,
/// times 256. `floor(1.23 * 256) = 314`.
const DELTA_PEEL_TIMES_256: u64 = 314;
/// Same for peeling + lazy Gaussian elimination. `floor(1.10 * 256) =
/// 281`.
const DELTA_GAUSSIAN_TIMES_256: u64 = 281;

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

impl<
    K: ?Sized + ToSig<S>,
    D: BitVecValueOps<usize>,
    S: Sig,
    E: ShardEdge<S, 3>,
> CompVFunc<K, D, S, E>
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
        let m = self.shard_size - w;
        // Local hash: ShardEdge tells us which bits don't carry the
        // shard index (`local_sig`) and how to mix them down to a
        // 64-bit value (`edge_hash`). The custom 3-position
        // derivation lives in `equation_from_hash`.
        let local_hash = self
            .shard_edge
            .edge_hash(self.shard_edge.local_sig(sig));
        let e = equation_from_hash(local_hash, m);

        let v0 = bucket_offset + e[0] as usize;
        let v1 = bucket_offset + e[1] as usize;
        let v2 = bucket_offset + e[2] as usize;
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
        self.decoder.set_branchless(branchless);
        self
    }
}

// ── Aligned ↔ Unaligned conversions ────────────────────────────────

impl<K: ?Sized, S, E> TryIntoUnaligned for CompVFunc<K, BitVec<Box<[usize]>>, S, E> {
    type Unaligned = CompVFunc<K, BitVecU<Box<[usize]>>, S, E>;

    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError> {
        // The query path issues two distinct unaligned reads: a
        // `w`-bit one for the codeword window, and (for escaped keys)
        // an `escaped_symbol_length`-bit one for the literal field.
        // Both must satisfy the unaligned constraints on `usize`.
        let w = self.global_max_codeword_length as usize;
        let esym = self.escaped_symbol_length as usize;
        check_unaligned_usize(w)?;
        if esym > 0 {
            check_unaligned_usize(esym)?;
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

/// Mirrors the `test_unaligned!` macro from `bits/mod.rs` for `usize`,
/// returning a structured error instead of asserting.
fn check_unaligned_usize(width: usize) -> Result<(), UnalignedConversionError> {
    let bits = usize::BITS as usize;
    let max_small = bits - 6;
    let mid = bits - 4;
    if width <= max_small || width == mid || width == bits {
        Ok(())
    } else {
        Err(UnalignedConversionError(format!(
            "bit width {width} does not satisfy the constraints for unaligned reads on usize (must be <= {max_small}, or == {mid}, or == {bits})"
        )))
    }
}

// ── Hashing helpers ─────────────────────────────────────────────────

/// Derives three vertex positions in `[0, m)` from a single 64-bit
/// local hash.
///
/// We don't reuse `ShardEdge::local_edge` because the multi-edge layout
/// of `CompVFunc` requires that the three positions are *base*
/// positions to which `(w − 1 − l)` is later added per codeword bit;
/// `local_edge` produces positions sized for one edge per key with the
/// `ShardEdge`'s own per-shard vertex count, which is unrelated to our
/// per-shard `m`.
///
/// Three [`mix64`] perturbations of the same input give three
/// uncorrelated hashes; the [`mul_high`](u128) reduction maps each into
/// `[0, m)` without modulo bias.
#[inline(always)]
fn equation_from_hash(h: u64, m: usize) -> [u32; 3] {
    let h0 = mix64(h);
    let h1 = mix64(h.wrapping_add(0x9e37_79b9_7f4a_7c15));
    let h2 = mix64(
        h.rotate_left(31)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9),
    );
    let m_u128 = m as u128;
    [
        ((h0 as u128 * m_u128) >> 64) as u32,
        ((h1 as u128 * m_u128) >> 64) as u32,
        ((h2 as u128 * m_u128) >> 64) as u32,
    ]
}

// ── Builder ─────────────────────────────────────────────────────────

/// Builder for [`CompVFunc`].
///
/// Mirrors the configuration surface of [`VBuilder`](crate::func::VBuilder)
/// in spirit. Construction is currently single-threaded; on a peeling
/// failure the whole build retries with a new global seed (just like
/// [`VFunc`](crate::func::VFunc)).
#[derive(Debug, Clone)]
pub struct CompVBuilder {
    /// Initial PRNG seed; subsequent build attempts (if any) draw
    /// fresh seeds from this PRNG.
    pub seed: u64,
    /// Use peel-only (`δ = 1.23`, +12 % space) instead of peel + LGE
    /// fallback (`δ = 1.10`). Peel-only is faster to construct (no
    /// LGE on the unpeeled remainder) but uses ~12 % more space.
    pub peel_only: bool,
    /// Length-limited Huffman parameters; the default is unlimited.
    pub huffman: Huffman,
    /// Target relative space loss due to ε-cost sharding. Same role
    /// as [`VBuilder::eps`](crate::func::VBuilder); 0.001 is the
    /// usual default.
    pub eps: f64,
    /// Maximum number of build attempts before giving up.
    pub max_attempts: u32,
}

impl Default for CompVBuilder {
    fn default() -> Self {
        Self {
            seed: 0,
            // Peel-only is the safe default: our custom 3-uniform
            // random hypergraph has a peeling threshold of ~0.818,
            // so any ratio of edges to variables above that will
            // make pure peeling fail and force the LGE fallback.
            // LGE on a multi-million-equation system is effectively
            // unbounded, so for typical ShardEdge-sized shards the
            // peel-only (δ = 1.23) mode is the only one that makes
            // sense. For tiny key sets the +12 % space overhead is
            // negligible.
            peel_only: true,
            huffman: Huffman::new(),
            eps: 0.001,
            max_attempts: 16,
        }
    }
}

impl CompVBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn peel_only(mut self, peel_only: bool) -> Self {
        self.peel_only = peel_only;
        self
    }

    pub fn huffman(mut self, huffman: Huffman) -> Self {
        self.huffman = huffman;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Builds a [`CompVFunc`] from parallel slices of keys and values
    /// using the default [`ShardEdge`] (`FuseLge3Shards`).
    pub fn try_build<K: ?Sized + ToSig<[u64; 2]>, B: Borrow<K>>(
        self,
        keys: &[B],
        values: &[u64],
    ) -> Result<CompVFunc<K>> {
        self.try_build_with_shard_edge::<K, B, [u64; 2], FuseLge3Shards>(keys, values)
    }

    /// Builds a [`CompVFunc`] using a caller-chosen [`ShardEdge`] /
    /// signature combination.
    pub fn try_build_with_shard_edge<K, B, S, E>(
        self,
        keys: &[B],
        values: &[u64],
    ) -> Result<CompVFunc<K, BitVec<Box<[usize]>>, S, E>>
    where
        K: ?Sized + ToSig<S>,
        B: Borrow<K>,
        S: Sig,
        E: ShardEdge<S, 3>,
    {
        if keys.len() != values.len() {
            bail!(
                "keys and values must have the same length ({} vs {})",
                keys.len(),
                values.len()
            );
        }
        build_inner::<K, B, S, E>(self, keys, values)
    }
}

/// Convenience entry points analogous to [`VFunc::try_new`](crate::func::VFunc).
impl<K: ?Sized + ToSig<[u64; 2]>> CompVFunc<K> {
    pub fn try_new<B: Borrow<K>>(keys: &[B], values: &[u64]) -> Result<Self> {
        CompVBuilder::default().try_build::<K, B>(keys, values)
    }

    pub fn try_new_with_builder<B: Borrow<K>>(
        keys: &[B],
        values: &[u64],
        builder: CompVBuilder,
    ) -> Result<Self> {
        builder.try_build::<K, B>(keys, values)
    }
}

// ── Builder core ───────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct EntryShim<S> {
    sig: S,
    bits: u64,
    len: u32,
}

fn build_inner<K, B, S, E>(
    cfg: CompVBuilder,
    keys: &[B],
    values: &[u64],
) -> Result<CompVFunc<K, BitVec<Box<[usize]>>, S, E>>
where
    K: ?Sized + ToSig<S>,
    B: Borrow<K>,
    S: Sig,
    E: ShardEdge<S, 3>,
{
    let n = keys.len();

    // ── Phase 1: frequencies + codec (independent of seed) ────────
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    for &v in values {
        *frequencies.entry(v).or_insert(0) += 1;
    }

    let coder: HuffmanCoder = cfg.huffman.build_coder(&frequencies);
    let global_max_codeword_length = coder.max_codeword_length();
    let escape_length = coder.escape_length();
    let escaped_symbol_length = coder.escaped_symbol_length();
    let w = global_max_codeword_length as usize;
    let escape_codeword = coder.escape();

    if n == 0 {
        let mut shard_edge = E::default();
        shard_edge.set_up_shards(0, cfg.eps);
        return Ok(CompVFunc {
            shard_edge,
            seed: cfg.seed,
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

    // ── Phase 2: set up sharding ──────────────────────────────────
    let mut shard_edge = E::default();
    shard_edge.set_up_shards(n, cfg.eps);
    let num_shards = shard_edge.num_shards();

    let delta = if cfg.peel_only {
        DELTA_PEEL_TIMES_256
    } else {
        DELTA_GAUSSIAN_TIMES_256
    };

    // ── Retry loop with whole-build seed re-roll ──────────────────
    let mut prng = SmallRng::seed_from_u64(cfg.seed);

    let mut last_err: Option<anyhow::Error> = None;

    for _attempt in 0..cfg.max_attempts.max(1) {
        let attempt_seed: u64 = prng.random();

        // ── Phase 3: hash all keys with this attempt's seed ────────
        let mut entries: Vec<EntryShim<S>> = Vec::with_capacity(n);
        for (k, &v) in keys.iter().zip(values.iter()) {
            let sig = K::to_sig(k.borrow(), attempt_seed);
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
            entries.push(EntryShim { sig, bits, len });
        }

        // ── Phase 4: group by shard ────────────────────────────────
        entries.sort_unstable_by_key(|e| shard_edge.shard(e.sig));

        let mut shard_starts: Vec<usize> = vec![0; num_shards + 1];
        let mut idx = 0usize;
        for (s, start) in shard_starts.iter_mut().take(num_shards).enumerate() {
            *start = idx;
            while idx < n && shard_edge.shard(entries[idx].sig) == s {
                idx += 1;
            }
        }
        shard_starts[num_shards] = n;

        // ── Phase 5: per-shard codeword sums + uniform shard size ──
        let mut max_sum: u64 = 0;
        for s in 0..num_shards {
            let sum: u64 = entries[shard_starts[s]..shard_starts[s + 1]]
                .iter()
                .map(|e| e.len as u64)
                .sum();
            if sum > max_sum {
                max_sum = sum;
            }
        }
        // Uniform shard size = δ × max_sum + w. The `+w` accommodates
        // the maximum offset introduced by the multi-edge expansion
        // (vertex `e_i + w − 1 − l` for `l = 0..L−1`).
        let shard_size = std::cmp::max(3, ((max_sum * delta) >> 8) + w as u64) as usize;
        let total_bits = num_shards
            .checked_mul(shard_size)
            .ok_or_else(|| anyhow!("data size overflow"))?;

        let mut data = BitVec::<Vec<usize>>::new_padded(total_bits);

        // ── Phase 6: solve each shard ───────────────────────────────
        let m = shard_size - w;
        let mut all_solved = true;
        let mut attempt_err: Option<anyhow::Error> = None;
        for s in 0..num_shards {
            let bucket_offset = s * shard_size;
            let shard_entries = &entries[shard_starts[s]..shard_starts[s + 1]];

            // Generate the multi-edges for this shard.
            let mut edges: Vec<[u32; 3]> = Vec::new();
            let mut rhs: Vec<bool> = Vec::new();
            for entry in shard_entries {
                let local_hash = shard_edge.edge_hash(shard_edge.local_sig(entry.sig));
                let e = equation_from_hash(local_hash, m);
                let len = entry.len as usize;
                for l in 0..len {
                    let off = (w - 1 - l) as u32;
                    edges.push([e[0] + off, e[1] + off, e[2] + off]);
                    rhs.push(((entry.bits >> l) & 1) == 1);
                }
            }

            match solve_system(shard_size, &edges, &rhs, cfg.peel_only) {
                Ok(solution) => {
                    for (i, &b) in solution.iter().enumerate() {
                        data.set(bucket_offset + i, b);
                    }
                }
                Err(e) => {
                    all_solved = false;
                    attempt_err = Some(e);
                    break;
                }
            }
        }

        if all_solved {
            return Ok(CompVFunc {
                shard_edge,
                seed: attempt_seed,
                num_keys: n,
                shard_size,
                global_max_codeword_length,
                escape_length,
                escaped_symbol_length,
                data,
                decoder: coder.into_decoder(),
                _marker: PhantomData,
            });
        }
        last_err = attempt_err;
    }

    Err(anyhow!(
        "Failed to build CompVFunc after {} attempts: {}",
        cfg.max_attempts.max(1),
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown".into())
    ))
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
fn solve_system(
    num_variables: usize,
    edges: &[[u32; 3]],
    rhs: &[bool],
    peel_only: bool,
) -> Result<Vec<bool>> {
    let peel = peel_partial(num_variables, edges);
    let n_peeled = peel.stack.len();
    let n_total = edges.len();

    let mut solution = vec![false; num_variables];

    if n_peeled < n_total {
        if peel_only {
            bail!(
                "peel-only mode: {} of {} edges remain unpeeled",
                n_total - n_peeled,
                n_total
            );
        }

        // Build LGE system on the non-peeled edges only. Variables that
        // do not appear in any non-peeled equation get value 0 from
        // LGE; the reverse-peel pass will overwrite the peeled-edge
        // pivots in `solution` afterwards.
        let mut equations: Vec<Modulo2Equation<usize>> =
            Vec::with_capacity(n_total - n_peeled);
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
        let mut system =
            unsafe { Modulo2System::<usize>::from_parts(num_variables, equations) };
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

    fn build_and_check(values: &[u64], peel_only: bool) {
        let n = values.len();
        let keys: Vec<u64> = (0..n as u64).collect();
        let func = CompVFunc::<u64>::try_new_with_builder(
            &keys,
            values,
            CompVBuilder::default().peel_only(peel_only),
        )
        .expect("build");
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
        build_and_check(&values, false);
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
        build_and_check(&values, false);
    }

    #[test]
    fn test_skewed_small_peel_only() {
        let mut values: Vec<u64> = Vec::with_capacity(200);
        for i in 0..200 {
            values.push(match i % 10 {
                0..=6 => 0,
                7 | 8 => 1,
                _ => 2,
            });
        }
        build_and_check(&values, true);
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
            CompVBuilder::default().huffman(Huffman::length_limited(8, 0.95)),
        )
        .expect("build");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(func.get(keys[i]), v, "mismatch at key {}", keys[i]);
        }
    }
}
