/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! Hollow-trie-based monotone minimal perfect hash functions.
//!
//! Given *n* keys in sorted order, a monotone minimal perfect hash function
//! maps each key to its rank (0 to *n* − 1). Querying a key not in the
//! original set returns an arbitrary value (same contract as
//! [`VFunc::get`]).
//!
//! [`HtDistMmphfInt`] works with any primitive integer type, whereas
//! [`HtDistMmphf`] works with any byte-sequence key type (`K: AsRef<[u8]>`).
//! Type aliases [`HtDistMmphfStr`] and [`HtDistMmphfSliceU8`] are provided for
//! convenience. For the byte-sequence variant, keys must not contain zeros, as
//! a virtual zero byte is appended internally to ensure prefix-freeness.
//! Alternatively, they must be prefix-free, in which case they can contain
//! zeros.
//!
//! These type of monotone minimal perfect hash functions are extremely compact
//! (in fact, provably optimal, as the use log log log *u* bits per key), but
//! they are very slow. Usually, an LCP-based solution
//! ([`lcp_mmphf`](crate::func::lcp_mmphf) or
//! [`lcp2_mmphf`](crate::func::lcp2_mmphf)) is more practical.
//!
//! These structures implement the [`TryIntoUnaligned`] trait, allowing them
//! to be converted into (usually faster) structures using unaligned access.
//!
//! # Implementation details
//!
//! The keys are divided into equal-size buckets. A hollow trie (a compacted
//! trie whose edge labels have been replaced with their lengths) built on the
//! bucket delimiters (last key of each full bucket) distributes each key to its
//! bucket in +O*(log *u*) time, where *u* is the universe size. Then a per-key
//! offset within the bucket is stored in a [`VFunc`]. The rank of a key is
//! `bucket * bucket_size + offset`.
//!
//! The trie walk uses behaviour functions (stored as [`VFunc`]s) to determine
//! the exit direction and detect false follows.
//!
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
//! [Theory and practice of monotone minimal perfect
//! hashing](https://doi.org/10.1145/2775054). *Journal of Experimental
//! Algorithmics* 20(3):1−26, 2016.

use {
    crate::{
        bal_paren::{BalParen, JacobsonBalParen},
        bits::BitFieldVec,
        func::VFunc,
        list::PrefixSumIntList,
        traits::{TryIntoUnaligned, Unaligned},
        utils::*,
    },
    mem_dbg::*,
    num_primitive::PrimitiveInteger,
    std::{borrow::Borrow, ops::Index},
    value_traits::slices::SliceByValue,
};

// ═══════════════════════════════════════════════════════════════════
// Bit manipulation helpers
// ═══════════════════════════════════════════════════════════════════

/// Read bit `i` from a byte slice (MSB-first within each byte) with
/// a prefix-free virtual terminator.
///
/// For positions within the key: returns the actual bit.
/// For positions beyond the key: returns `false` (virtual NUL byte).
///
/// This makes the bit representation prefix-free: since actual bytes
/// are assumed nonzero, the all-zero virtual NUL byte differs from
/// any continuation byte. The NUL sorts before any real byte,
/// preserving lexicographic order.
#[inline]
fn get_key_bit(key: &[u8], i: usize) -> bool {
    let byte_idx = i / 8;
    let bit_idx = 7 - (i % 8);
    if byte_idx < key.len() {
        (key[byte_idx] >> bit_idx) & 1 != 0
    } else {
        false // virtual NUL
    }
}

/// Read bit `i` (MSB-first) from an integer.
#[inline]
fn get_int_bit<K: PrimitiveInteger>(key: K, i: usize) -> bool {
    (key >> (K::BITS as usize - 1 - i)) & K::from(true) != K::default()
}

// ═══════════════════════════════════════════════════════════════════
// Behaviour key encoding
// ═══════════════════════════════════════════════════════════════════

/// Header size in bytes for behaviour key encoding: two `usize` values
/// (node position and path length).
const BEHAVIOUR_KEY_HEADER: usize = 2 * (usize::BITS as usize / 8);

/// Encodes a (node position, path fragment) pair as a byte vector for
/// use as a VFunc key. The encoding is:
///   - `usize::BITS / 8` bytes: node position
///   - `usize::BITS / 8` bytes: path length in bits
///   - `ceil((bit_end - bit_start) / 8)` bytes: path bits packed MSB-first
///
/// The encoding must be injective for correctness.
///
/// Returns the number of bytes written into `buf`.
#[inline]
fn encode_behaviour_key_into(
    buf: &mut [u8],
    node_pos: usize,
    key_bytes: &[u8],
    bit_start: usize,
    bit_end: usize,
) -> usize {
    let path_len = bit_end - bit_start;
    let packed_bytes = path_len.div_ceil(8);
    let total = BEHAVIOUR_KEY_HEADER + packed_bytes;
    debug_assert!(buf.len() >= total);
    let w = usize::BITS as usize / 8;
    buf[..w].copy_from_slice(&node_pos.to_ne_bytes());
    buf[w..BEHAVIOUR_KEY_HEADER].copy_from_slice(&path_len.to_ne_bytes());

    // Pack path bits MSB-first into bytes. Since key_bytes is MSB-first,
    // we can use byte-level operations when bit_start is byte-aligned,
    // and shift operations otherwise.
    let start_byte = bit_start / 8;
    let start_bit_offset = bit_start % 8;

    if start_bit_offset == 0 {
        // Byte-aligned: copy whole bytes, mask the last one.
        let copy_bytes = packed_bytes.min(key_bytes.len().saturating_sub(start_byte));
        buf[BEHAVIOUR_KEY_HEADER..BEHAVIOUR_KEY_HEADER + copy_bytes]
            .copy_from_slice(&key_bytes[start_byte..start_byte + copy_bytes]);
        // Zero any bytes beyond the key (virtual NUL).
        for i in copy_bytes..packed_bytes {
            buf[BEHAVIOUR_KEY_HEADER + i] = 0;
        }
    } else {
        // Unaligned: shift pairs of source bytes.
        let shift = start_bit_offset;
        for b in 0..packed_bytes {
            let src_idx = start_byte + b;
            let hi = if src_idx < key_bytes.len() {
                key_bytes[src_idx]
            } else {
                0
            };
            let lo = if src_idx + 1 < key_bytes.len() {
                key_bytes[src_idx + 1]
            } else {
                0
            };
            buf[BEHAVIOUR_KEY_HEADER + b] = (hi << shift) | (lo >> (8 - shift));
        }
    }

    // Mask off trailing bits in the last byte.
    let trail = path_len % 8;
    if trail != 0 && packed_bytes > 0 {
        buf[BEHAVIOUR_KEY_HEADER + packed_bytes - 1] &= !((1u8 << (8 - trail)) - 1);
    }

    total
}

/// Header size for integer behaviour keys: `node_pos`, `path_len`,
/// and the extracted path bits as an integer of type `K`.
const fn int_behaviour_key_size<K>() -> usize {
    2 * (usize::BITS as usize / 8) + std::mem::size_of::<K>()
}

/// Encode a behaviour key for an integer key into a pre-allocated
/// buffer by extracting the path bits directly with shifts.
///
/// The encoding packs `(node_pos, path_len, extracted_bits)` as
/// native-endian bytes, where `extracted_bits` is the contiguous
/// range `[bit_start . . bit_end)` of the XOR-mapped key,
/// right-aligned.
///
/// Returns the number of bytes written.
#[inline]
fn encode_int_behaviour_key_into<K: PrimitiveInteger>(
    buf: &mut [u8],
    node_pos: usize,
    key: K,
    bit_start: usize,
    bit_end: usize,
) -> usize {
    let path_len = bit_end - bit_start;
    let w = usize::BITS as usize / 8;
    buf[..w].copy_from_slice(&node_pos.to_ne_bytes());
    buf[w..2 * w].copy_from_slice(&path_len.to_ne_bytes());
    // Extract bits [bit_start . . bit_end) right-aligned.
    let extracted = if path_len == 0 {
        K::default()
    } else {
        (key << bit_start) >> (K::BITS as usize - path_len)
    };
    let ext_bytes: K::Bytes = extracted.to_ne_bytes();
    let ext_bytes: &[u8] = ext_bytes.borrow();
    buf[2 * w..2 * w + ext_bytes.len()].copy_from_slice(ext_bytes);
    int_behaviour_key_size::<K>()
}

// ═══════════════════════════════════════════════════════════════════
// Behaviour enum
// ═══════════════════════════════════════════════════════════════════

/// The behaviour of a key at a trie node: exit left, exit right, or
/// follow the trie edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    /// Exit on the left (closer to left delimiter).
    Left = 0,
    /// Exit on the right (closer to right delimiter).
    Right = 1,
    /// Follow the trie edge (true follow).
    Follow = 2,
}

impl Behaviour {
    /// Convert a `usize` value (from a VFunc lookup) to a `Behaviour`.
    #[inline]
    fn from_usize(val: usize) -> Self {
        match val {
            0 => Behaviour::Left,
            1 => Behaviour::Right,
            _ => Behaviour::Follow,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Struct definitions (always available)
// ═══════════════════════════════════════════════════════════════════

/// A distributor that assigns sorted byte-sequence keys to bucket indices using
/// a hollow trie.
///
/// Built from sorted keys and a bucket size. Uses a hollow trie on
/// the bucket delimiters combined with behaviour functions stored as
/// [`VFunc`]s.
#[derive(Debug)]
pub struct HtDist<
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    /// Balanced-parentheses support structure for the trie.
    bal_paren: B,
    /// Skip values stored as a prefix-sum list over Elias-Fano.
    skips: S,
    /// Number of internal nodes (= number of delimiters - 1).
    #[allow(dead_code)]
    num_nodes: usize,
    /// Number of delimiters.
    num_delimiters: usize,
    /// External behaviour: maps (node, path) -> LEFT (0) or RIGHT (1).
    external_behaviour: E,
    /// Detects false follows: maps (node, path) -> 0 (true follow) or
    /// 1 (false follow).
    false_follows_detector: F,
}

/// A monotone minimal perfect hash function for sorted byte-sequence keys,
/// based on a hollow trie distributor ([`HtDist`]) and per-bucket offsets
/// stored in a [`VFunc`].
///
/// See the [module documentation](self) for the algorithmic description.
/// See [`HtDistMmphfStr`] and [`HtDistMmphfSliceU8`] for common
/// instantiations, and [`HtDistMmphfInt`] for integer keys.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to
/// be converted into (usually faster) structures using unaligned access.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::hollow_trie::HtDistMmphfStr;
/// # use sux::utils::FromSlice;
/// let keys: Vec<&str> = vec!["alpha", "beta", "delta", "gamma"];
/// let func: HtDistMmphfStr =
///     HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
#[derive(Debug)]
pub struct HtDistMmphf<
    K: ?Sized,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    O = VFunc<K, BitFieldVec<Box<[usize]>>>,
    B: BalParen = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    /// The hollow trie distributor.
    distributor: HtDist<E, F, B, S>,
    /// Per-key offset within the bucket.
    offset: O,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
    /// Number of keys.
    n: usize,
    /// Phantom data for `K`.
    _marker: std::marker::PhantomData<*const K>,
}

/// An [`HtDistMmphf`] for `str` keys.
///
/// This structure implements the
/// [`TryIntoUnaligned`] trait, allowing
/// it to be converted into (usually faster) structures using unaligned
/// access.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::hollow_trie::HtDistMmphfStr;
/// # use sux::utils::FromSlice;
/// let keys: Vec<&str> = vec!["alpha", "beta", "delta", "gamma"];
/// let func: HtDistMmphfStr =
///     HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type HtDistMmphfStr = HtDistMmphf<str>;

/// An [`HtDistMmphf`] for `[u8]` keys.
///
/// This structure implements the
/// [`TryIntoUnaligned`] trait, allowing
/// it to be converted into (usually faster) structures using unaligned
/// access.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::hollow_trie::HtDistMmphfSliceU8;
/// # use sux::utils::FromSlice;
/// let keys: Vec<Vec<u8>> = vec![
///     b"alpha".to_vec(),
///     b"beta".to_vec(),
///     b"delta".to_vec(),
///     b"gamma".to_vec(),
/// ];
///
/// let func: HtDistMmphfSliceU8 = HtDistMmphfSliceU8::try_new(
///     FromSlice::new(&keys),
///     keys.len(),
///     no_logging![],
/// )?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_slice()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type HtDistMmphfSliceU8 = HtDistMmphf<[u8]>;

/// A hollow trie distributor for sorted integer keys.
///
/// This is the integer-key analogue of [`HtDist`]; see the
/// [module documentation](self) for the algorithmic description.
#[derive(Debug)]
pub struct HtDistInt<
    K,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    B = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    /// Balanced-parentheses support structure for the trie.
    bal_paren: B,
    /// Skip values stored as a prefix-sum list over Elias-Fano.
    skips: S,
    /// Number of internal nodes (= number of delimiters - 1).
    #[allow(dead_code)]
    num_nodes: usize,
    /// Number of delimiters.
    num_delimiters: usize,
    /// External behaviour: maps (node, path) -> LEFT (0) or RIGHT (1).
    external_behaviour: E,
    /// Detects false follows: maps (node, path) -> 0 (true follow) or
    /// 1 (false follow).
    false_follows_detector: F,
    /// Phantom data for `K`.
    _marker: std::marker::PhantomData<K>,
}

/// A monotone minimal perfect hash function for sorted integer keys,
/// based on a hollow trie distributor ([`HtDistInt`]) and per-bucket
/// offsets stored in a [`VFunc`].
///
/// See the [module documentation](self) for the algorithmic description.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to
/// be converted into (usually faster) structures using unaligned access.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::hollow_trie::HtDistMmphfInt;
/// # use sux::utils::FromSlice;
/// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
///
/// let func: HtDistMmphfInt<u64> =
///     HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
///
/// for (i, &key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
#[derive(Debug)]
pub struct HtDistMmphfInt<
    K,
    E = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    F = VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    O = VFunc<K, BitFieldVec<Box<[usize]>>>,
    B = JacobsonBalParen,
    S = PrefixSumIntList,
> {
    /// The hollow trie distributor.
    distributor: HtDistInt<K, E, F, B, S>,
    /// Per-key offset within the bucket.
    offset: O,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
    /// Number of keys.
    n: usize,
}

// ═══════════════════════════════════════════════════════════════════
// Query methods
// ═══════════════════════════════════════════════════════════════════

impl<
    D: SliceByValue<Value = usize>,
    B: BalParen + AsRef<[usize]> + Index<usize, Output = bool>,
    S: SliceByValue<Value = usize>,
> HtDist<VFunc<[u8], D>, VFunc<[u8], D>, B, S>
{
    /// Returns the bucket index for the given key.
    ///
    /// The key is navigated through the hollow trie using the balanced
    /// parentheses structure and skip values. At each internal node,
    /// the behaviour functions determine whether to follow the trie
    /// edge or exit left/right.
    pub fn get(&self, key: &[u8]) -> usize {
        if self.num_delimiters == 0 {
            return 0;
        }

        let trie_words = &self.bal_paren;
        let length = key.len() * 8 + 8; // including virtual NUL terminator
        let mut p: usize = 1;
        let mut index: usize = 0;
        let mut r: usize = 0;
        let mut s: usize = 0;
        let mut last_left_turn: usize = 0;
        let mut last_left_turn_index: usize = 0;
        // Buffer for behaviour key encoding. Reused across loop iterations
        // to avoid per-node allocation. Max size: 16 header + key path bytes.
        let buf_size = BEHAVIOUR_KEY_HEADER + key.len() + 1;
        let mut key_buf_storage;
        let mut key_buf_vec;
        let key_buf: &mut [u8] = if buf_size <= 528 {
            key_buf_storage = [0u8; 528];
            &mut key_buf_storage[..buf_size]
        } else {
            key_buf_vec = vec![0u8; buf_size];
            &mut key_buf_vec
        };

        loop {
            let is_internal = trie_words[p];
            let skip: usize = if is_internal {
                self.skips.index_value(r)
            } else {
                0
            };

            let behaviour = if is_internal {
                let n = encode_behaviour_key_into(key_buf, p - 1, key, s, (s + skip).min(length));
                if self.false_follows_detector.get(&key_buf[..n]) == 0 {
                    Behaviour::Follow
                } else {
                    Behaviour::from_usize(self.external_behaviour.get(&key_buf[..n]))
                }
            } else {
                let n = encode_behaviour_key_into(key_buf, p - 1, key, s, length);
                Behaviour::from_usize(self.external_behaviour.get(&key_buf[..n]))
            };

            // Exit condition: behaviour is not FOLLOW, or we reached a
            // leaf, or the key's bits are exhausted after consuming the
            // skip. The `{s += skip; s >= length}` block both advances
            // `s` and checks the stop condition (mirrors the Java's
            // `(s += skip) >= maxDescentLength`).
            if behaviour != Behaviour::Follow || !is_internal || {
                s += skip;
                s >= length
            } {
                // Compute the bucket index from the exit point.
                if behaviour == Behaviour::Left {
                    // All leaves to the left are counted in `index`.
                    return index;
                } else if is_internal {
                    // EXIT RIGHT at an internal node: the bucket index
                    // is the total number of leaves in the subtree
                    // rooted at the last left turn plus the leaves
                    // already counted.
                    let q = self
                        .bal_paren
                        .find_close(last_left_turn)
                        .expect("balanced parentheses broken");
                    #[allow(clippy::manual_div_ceil)]
                    return ((q - last_left_turn + 1) / 2) + last_left_turn_index;
                } else {
                    // EXIT RIGHT at a leaf: one more than the current
                    // leaf count.
                    return index + 1;
                }
            }

            // FOLLOW: continue descent. Turn left or right based on
            // the key's bit at position `s` (the branching bit after
            // the compacted path).
            if get_key_bit(key, s) {
                // RIGHT: skip over the entire left subtree.
                // `findClose(p)` gives the matching `)` of the left
                // subtree; the right child starts at `q = findClose(p) + 1`.
                // The number of leaves in the left subtree is `(q - p) / 2`.
                let q = self
                    .bal_paren
                    .find_close(p)
                    .expect("balanced parentheses broken")
                    + 1;
                index += (q - p) / 2;
                r += (q - p) / 2;
                p = q;
            } else {
                // LEFT: the left child is at `p + 1`. Record this as
                // the last left turn (needed for RIGHT exits later).
                last_left_turn = p;
                last_left_turn_index = index;
                p += 1;
                r += 1;
            }

            s += 1; // consume the branching bit
        }
    }
}

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<[u64; 2]>,
    D: SliceByValue<Value = usize> + MemSize,
    B: BalParen + AsRef<[usize]> + Index<usize, Output = bool>,
    S: SliceByValue<Value = usize>,
> HtDistMmphf<K, VFunc<[u8], D>, VFunc<[u8], D>, VFunc<K, D>, B, S>
{
    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary.
    #[inline]
    pub fn get(&self, key: &K) -> usize {
        if self.n <= 1 {
            return 0;
        }
        let bucket = self.distributor.get(key.as_ref());
        (bucket << self.log2_bucket_size) + self.offset.get(key)
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl<
    K: PrimitiveInteger,
    D: SliceByValue<Value = usize>,
    B: BalParen + AsRef<[usize]> + Index<usize, Output = bool>,
    S: SliceByValue<Value = usize>,
> HtDistInt<K, VFunc<[u8], D>, VFunc<[u8], D>, B, S>
{
    /// Returns the bucket index for the given integer key.
    ///
    /// The key is XOR-mapped with `K::MIN` and navigated through the
    /// hollow trie using the balanced parentheses structure and skip
    /// values. At each internal node, the behaviour functions determine
    /// whether to follow the trie edge or exit left/right.
    pub fn get(&self, key: K) -> usize {
        if self.num_delimiters == 0 {
            return 0;
        }

        let mapped = key ^ K::MIN;
        let length = K::BITS as usize;

        let bal_paren = &self.bal_paren;
        let mut p: usize = 1;
        let mut index: usize = 0;
        let mut r: usize = 0;
        let mut s: usize = 0;
        let mut last_left_turn: usize = 0;
        let mut last_left_turn_index: usize = 0;
        // Max: 2 * 8 (header) + 16 (u128) = 32 bytes.
        let mut key_buf = [0u8; BEHAVIOUR_KEY_HEADER + 16];

        loop {
            let is_internal = bal_paren[p];
            let skip: usize = if is_internal {
                self.skips.index_value(r)
            } else {
                0
            };

            let behaviour = if is_internal {
                let n = encode_int_behaviour_key_into(
                    &mut key_buf,
                    p - 1,
                    mapped,
                    s,
                    (s + skip).min(length),
                );
                if self.false_follows_detector.get(&key_buf[..n]) == 0 {
                    Behaviour::Follow
                } else {
                    Behaviour::from_usize(self.external_behaviour.get(&key_buf[..n]))
                }
            } else {
                let n = encode_int_behaviour_key_into(&mut key_buf, p - 1, mapped, s, length);
                Behaviour::from_usize(self.external_behaviour.get(&key_buf[..n]))
            };

            if behaviour != Behaviour::Follow || !is_internal || {
                s += skip;
                s >= length
            } {
                if behaviour == Behaviour::Left {
                    return index;
                } else if is_internal {
                    let q = self
                        .bal_paren
                        .find_close(last_left_turn)
                        .expect("balanced parentheses broken");
                    #[allow(clippy::manual_div_ceil)]
                    return ((q - last_left_turn + 1) / 2) + last_left_turn_index;
                } else {
                    return index + 1;
                }
            }

            if get_int_bit(mapped, s) {
                let q = self
                    .bal_paren
                    .find_close(p)
                    .expect("balanced parentheses broken")
                    + 1;
                index += (q - p) / 2;
                r += (q - p) / 2;
                p = q;
            } else {
                last_left_turn = p;
                last_left_turn_index = index;
                p += 1;
                r += 1;
            }

            s += 1;
        }
    }
}

impl<
    K: PrimitiveInteger + ToSig<[u64; 2]>,
    D: SliceByValue<Value = usize> + MemSize + mem_dbg::FlatType,
    B: BalParen + AsRef<[usize]> + Index<usize, Output = bool>,
    S: SliceByValue<Value = usize>,
> HtDistMmphfInt<K, VFunc<[u8], D>, VFunc<[u8], D>, VFunc<K, D>, B, S>
{
    /// Returns the rank (0-based position) of the given key.
    ///
    /// If the key was not in the original set, the result is arbitrary.
    #[inline]
    pub fn get(&self, key: K) -> usize {
        if self.n <= 1 {
            return 0;
        }
        let bucket = self.distributor.get(key);
        (bucket << self.log2_bucket_size) + self.offset.get(&key)
    }

    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

// ═══════════════════════════════════════════════════════════════════
// Aligned ↔ Unaligned conversions
// ═══════════════════════════════════════════════════════════════════

impl<E: TryIntoUnaligned, F: TryIntoUnaligned, B: BalParen + TryIntoUnaligned, S> TryIntoUnaligned
    for HtDist<E, F, B, S>
where
    Unaligned<B>: BalParen,
{
    type Unaligned = HtDist<Unaligned<E>, Unaligned<F>, Unaligned<B>, S>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(HtDist {
            bal_paren: self.bal_paren.try_into_unaligned()?,
            skips: self.skips,
            num_nodes: self.num_nodes,
            num_delimiters: self.num_delimiters,
            false_follows_detector: self.false_follows_detector.try_into_unaligned()?,
            external_behaviour: self.external_behaviour.try_into_unaligned()?,
        })
    }
}

impl<
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemSize for HtDist<E, F, B, S>
{
    fn mem_size_rec(&self, flags: SizeFlags, refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        let mut size = core::mem::size_of::<Self>();
        size += self.bal_paren.mem_size_rec(flags, refs);
        size += self.skips.mem_size_rec(flags, refs);
        size += self.false_follows_detector.mem_size_rec(flags, refs);
        size += self.external_behaviour.mem_size_rec(flags, refs);
        size
    }
}

impl<
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemDbgImpl for HtDist<E, F, B, S>
{
}

impl
    From<
        HtDist<
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    > for HtDist<VFunc<[u8], BitFieldVec<Box<[usize]>>>, VFunc<[u8], BitFieldVec<Box<[usize]>>>>
{
    fn from(
        f: HtDist<
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    ) -> Self {
        // SAFETY: Into::into preserves the semantics of the pioneer
        // position and offset structures.
        let bal_paren = unsafe {
            f.bal_paren
                .map_pioneer_positions(Into::into)
                .map_pioneer_match_offsets(Into::into)
        };
        Self {
            bal_paren,
            skips: f.skips,
            num_nodes: f.num_nodes,
            num_delimiters: f.num_delimiters,
            false_follows_detector: f.false_follows_detector.into(),
            external_behaviour: f.external_behaviour.into(),
        }
    }
}

impl<
    K: ?Sized,
    E: TryIntoUnaligned,
    F: TryIntoUnaligned,
    O: TryIntoUnaligned,
    B: BalParen + TryIntoUnaligned,
    S,
> TryIntoUnaligned for HtDistMmphf<K, E, F, O, B, S>
where
    Unaligned<B>: BalParen,
{
    type Unaligned = HtDistMmphf<K, Unaligned<E>, Unaligned<F>, Unaligned<O>, Unaligned<B>, S>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(HtDistMmphf {
            distributor: self.distributor.try_into_unaligned()?,
            offset: self.offset.try_into_unaligned()?,
            log2_bucket_size: self.log2_bucket_size,
            n: self.n,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<
    K: ?Sized,
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    O: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemSize for HtDistMmphf<K, E, F, O, B, S>
{
    fn mem_size_rec(&self, flags: SizeFlags, refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        let mut size = core::mem::size_of::<Self>();
        size += self.distributor.mem_size_rec(flags, refs);
        size += self.offset.mem_size_rec(flags, refs);
        size
    }
}

impl<
    K: ?Sized,
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    O: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemDbgImpl for HtDistMmphf<K, E, F, O, B, S>
{
}

impl<K: ?Sized>
    From<
        HtDistMmphf<
            K,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<K, Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    >
    for HtDistMmphf<
        K,
        VFunc<[u8], BitFieldVec<Box<[usize]>>>,
        VFunc<[u8], BitFieldVec<Box<[usize]>>>,
        VFunc<K, BitFieldVec<Box<[usize]>>>,
    >
{
    fn from(
        f: HtDistMmphf<
            K,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<K, Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    ) -> Self {
        Self {
            distributor: f.distributor.into(),
            offset: f.offset.into(),
            log2_bucket_size: f.log2_bucket_size,
            n: f.n,
            _marker: std::marker::PhantomData,
        }
    }
}

// ── Integer TryIntoUnaligned conversions ──────────────────────────

impl<K, E: TryIntoUnaligned, F: TryIntoUnaligned, B: BalParen + TryIntoUnaligned, S>
    TryIntoUnaligned for HtDistInt<K, E, F, B, S>
where
    Unaligned<B>: BalParen,
{
    type Unaligned = HtDistInt<K, Unaligned<E>, Unaligned<F>, Unaligned<B>, S>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(HtDistInt {
            bal_paren: self.bal_paren.try_into_unaligned()?,
            skips: self.skips,
            num_nodes: self.num_nodes,
            num_delimiters: self.num_delimiters,
            false_follows_detector: self.false_follows_detector.try_into_unaligned()?,
            external_behaviour: self.external_behaviour.try_into_unaligned()?,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<
    K,
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemSize for HtDistInt<K, E, F, B, S>
{
    fn mem_size_rec(&self, flags: SizeFlags, refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        let mut size = core::mem::size_of::<Self>();
        size += self.bal_paren.mem_size_rec(flags, refs);
        size += self.skips.mem_size_rec(flags, refs);
        size += self.false_follows_detector.mem_size_rec(flags, refs);
        size += self.external_behaviour.mem_size_rec(flags, refs);
        size
    }
}

impl<
    K,
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemDbgImpl for HtDistInt<K, E, F, B, S>
{
}

impl<K: PrimitiveInteger>
    From<
        HtDistInt<
            K,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    >
    for HtDistInt<K, VFunc<[u8], BitFieldVec<Box<[usize]>>>, VFunc<[u8], BitFieldVec<Box<[usize]>>>>
{
    fn from(
        f: HtDistInt<
            K,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    ) -> Self {
        // SAFETY: Into::into preserves the semantics of the pioneer
        // position and offset structures.
        let bal_paren = unsafe {
            f.bal_paren
                .map_pioneer_positions(Into::into)
                .map_pioneer_match_offsets(Into::into)
        };
        Self {
            bal_paren,
            skips: f.skips,
            num_nodes: f.num_nodes,
            num_delimiters: f.num_delimiters,
            false_follows_detector: f.false_follows_detector.into(),
            external_behaviour: f.external_behaviour.into(),
            _marker: std::marker::PhantomData,
        }
    }
}

// ── Integer MMPHF TryIntoUnaligned conversions ────────────────────

impl<
    K,
    E: TryIntoUnaligned,
    F: TryIntoUnaligned,
    O: TryIntoUnaligned,
    B: BalParen + TryIntoUnaligned,
    S,
> TryIntoUnaligned for HtDistMmphfInt<K, E, F, O, B, S>
where
    Unaligned<B>: BalParen,
{
    type Unaligned = HtDistMmphfInt<K, Unaligned<E>, Unaligned<F>, Unaligned<O>, Unaligned<B>, S>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(HtDistMmphfInt {
            distributor: self.distributor.try_into_unaligned()?,
            offset: self.offset.try_into_unaligned()?,
            log2_bucket_size: self.log2_bucket_size,
            n: self.n,
        })
    }
}

impl<
    K,
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    O: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemSize for HtDistMmphfInt<K, E, F, O, B, S>
{
    fn mem_size_rec(&self, flags: SizeFlags, refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        let mut size = core::mem::size_of::<Self>();
        size += self.distributor.mem_size_rec(flags, refs);
        size += self.offset.mem_size_rec(flags, refs);
        size
    }
}

impl<
    K,
    E: MemSize + mem_dbg::FlatType,
    F: MemSize + mem_dbg::FlatType,
    O: MemSize + mem_dbg::FlatType,
    B: BalParen + MemSize + mem_dbg::FlatType,
    S: MemSize + mem_dbg::FlatType,
> MemDbgImpl for HtDistMmphfInt<K, E, F, O, B, S>
{
}

impl<K: PrimitiveInteger>
    From<
        HtDistMmphfInt<
            K,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<K, Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    >
    for HtDistMmphfInt<
        K,
        VFunc<[u8], BitFieldVec<Box<[usize]>>>,
        VFunc<[u8], BitFieldVec<Box<[usize]>>>,
        VFunc<K, BitFieldVec<Box<[usize]>>>,
    >
{
    fn from(
        f: HtDistMmphfInt<
            K,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<[u8], Unaligned<BitFieldVec<Box<[usize]>>>>,
            VFunc<K, Unaligned<BitFieldVec<Box<[usize]>>>>,
            Unaligned<JacobsonBalParen>,
        >,
    ) -> Self {
        Self {
            distributor: f.distributor.into(),
            offset: f.offset.into(),
            log2_bucket_size: f.log2_bucket_size,
            n: f.n,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Construction (requires rayon)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::{
        bits::BitVec,
        func::lcp_mmphf::{lcp_bits, lcp_bits_nul},
    };
    use anyhow::Result;
    use dsi_progress_logger::ProgressLog;
    use lender::FallibleLending;
    use log::info;

    // ── Trie construction (online, stack-based) ──────────────────

    /// A node on the right spine during incremental trie construction.
    ///
    /// Each node represents an internal node of the compacted trie.
    /// `repr` and `repr_skips` hold the serialized representation of
    /// the node's left subtree (already finalized). The right child
    /// is the next node on the spine (tracked by the stack).
    struct SpineNode {
        /// Skip value (compacted path length in bits).
        skip: usize,
        /// Balanced-parentheses representation of the left subtree.
        repr: BitVec,
        /// Skip values for internal nodes in the left subtree (DFS order).
        repr_skips: Vec<usize>,
    }

    impl SpineNode {
        fn new(skip: usize) -> Self {
            Self {
                skip,
                repr: BitVec::new(0),
                repr_skips: Vec::new(),
            }
        }
    }

    /// Builds the hollow trie from sorted, prefix-free byte sequences.
    ///
    /// The builder produces a balanced-parentheses bit vector and a skip
    /// sequence. Call [`push`](Self::push) for each delimiter in sorted
    /// order, then [`finish`](Self::finish).
    pub(crate) struct HollowTrieBuilder {
        /// Stack representing the right spine. Index 0 = root.
        stack: Vec<SpineNode>,
        /// Cumulative path length (in bits) for each stack entry.
        /// `lens[i]` = total bits consumed from the root to reach `stack[i]`.
        lens: Vec<usize>,
        /// Previous key (for LCP computation).
        prev: Vec<u8>,
        /// Number of keys pushed (= number of leaves).
        count: usize,
        /// Number of internal nodes created.
        num_nodes: usize,
    }

    impl HollowTrieBuilder {
        pub fn new() -> Self {
            Self {
                stack: Vec::new(),
                lens: Vec::new(),
                prev: Vec::new(),
                count: 0,
                num_nodes: 0,
            }
        }

        /// Serializes a chain of spine nodes (from `nodes[0]` to
        /// `nodes[last]`) into `repr` and `repr_skips`.
        ///
        /// Each node is wrapped in `1 [node.repr] 0` and its skip is
        /// prepended to the skip list.
        fn serialize_chain(nodes: &[SpineNode]) -> (BitVec, Vec<usize>) {
            let mut repr: BitVec = BitVec::new(0);
            let mut skips = Vec::new();

            for node in nodes {
                repr.push(true); // (
                repr.append(&node.repr);
                repr.push(false); // )
                skips.push(node.skip);
                skips.extend_from_slice(&node.repr_skips);
            }

            (repr, skips)
        }

        /// Push a new key (delimiter). Keys must be in strictly increasing
        /// lexicographic order.
        pub fn push(&mut self, key: &[u8]) {
            if self.count == 0 {
                // First key: just record it; the trie has no nodes yet.
                self.prev = key.to_vec();
                self.count = 1;
                return;
            }

            let lcp = lcp_bits_nul::<true>(&self.prev, key);

            // Pop nodes whose cumulative path length exceeds the LCP.
            let mut last = self.stack.len() as isize - 1;
            while last >= 0 && self.lens[last as usize] > lcp {
                last -= 1;
            }

            // Nodes from (last+1).. are being "closed off" — they form
            // a right-spine chain that must be serialized.
            let pop_start = (last + 1) as usize;
            let popped: Vec<SpineNode> = self.stack.drain(pop_start..).collect();
            self.lens.truncate(pop_start);

            let prefix = if last >= 0 {
                lcp - self.lens[last as usize]
            } else {
                lcp
            };

            if !popped.is_empty() {
                // The new internal node absorbs `prefix` bits of path
                // plus 1 branching bit from the topmost popped node.
                let mut adjusted = popped;
                adjusted[0].skip -= prefix + 1;
                debug_assert!(
                    adjusted[0].skip < usize::MAX / 2,
                    "skip underflow: prefix={prefix}, original skip would give negative"
                );

                // Serialize the adjusted chain into a new node's left subtree.
                let (repr, repr_skips) = Self::serialize_chain(&adjusted);

                let mut new_node = SpineNode::new(prefix);
                new_node.repr = repr;
                new_node.repr_skips = repr_skips;

                // Push the new internal node.
                let new_len = if last >= 0 {
                    self.lens[last as usize] + prefix + 1
                } else {
                    prefix + 1
                };
                self.stack.push(new_node);
                self.lens.push(new_len);
                self.num_nodes += 1;
            } else {
                // No nodes popped: the LCP falls exactly on the current
                // top of the stack (or the stack is empty). Create a new
                // internal node with empty left subtree.
                let new_node = SpineNode::new(prefix);
                let new_len = if last >= 0 {
                    self.lens[last as usize] + prefix + 1
                } else {
                    prefix + 1
                };
                self.stack.push(new_node);
                self.lens.push(new_len);
                self.num_nodes += 1;
            }

            self.prev.clear();
            self.prev.extend_from_slice(key);
            self.count += 1;
        }

        /// Finalize the trie and return:
        /// - balanced-parentheses bit vector as `BitVec<Box<[usize]>>`
        /// - `skips`: skip values in DFS preorder
        /// - `num_nodes`: number of internal nodes
        pub fn finish(self) -> (BitVec<Box<[usize]>>, Vec<usize>, usize) {
            if self.count <= 1 {
                let mut trie: BitVec = BitVec::new(0);
                trie.push(true);
                trie.push(false);
                return (trie.into(), Vec::new(), 0);
            }

            // Serialize the remaining right spine.
            let (chain_repr, chain_skips) = Self::serialize_chain(&self.stack);

            // Wrap in fake root brackets: 1 [chain] 0
            let mut trie: BitVec = BitVec::new(0);
            trie.push(true);
            trie.append(&chain_repr);
            trie.push(false);

            debug_assert_eq!(
                trie.len(),
                2 * self.num_nodes + 2,
                "trie length mismatch: expected {}, got {}",
                2 * self.num_nodes + 2,
                trie.len()
            );

            (trie.into(), chain_skips, self.num_nodes)
        }
    }

    /// Convenience wrapper that allocates a `Vec` — used during construction
    /// where keys need to be stored.
    fn encode_behaviour_key(
        node_pos: usize,
        key_bytes: &[u8],
        bit_start: usize,
        bit_end: usize,
    ) -> Vec<u8> {
        let path_len = bit_end - bit_start;
        let packed_bytes = path_len.div_ceil(8);
        let mut buf = vec![0u8; BEHAVIOUR_KEY_HEADER + packed_bytes];
        encode_behaviour_key_into(&mut buf, node_pos, key_bytes, bit_start, bit_end);
        buf
    }

    /// Convenience wrapper that allocates a `Vec`.
    fn encode_int_behaviour_key<K: PrimitiveInteger>(
        node_pos: usize,
        key: K,
        bit_start: usize,
        bit_end: usize,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; int_behaviour_key_size::<K>()];
        encode_int_behaviour_key_into(&mut buf, node_pos, key, bit_start, bit_end);
        buf
    }

    /// Builds the hollow trie from sorted integers (XOR-mapped with
    /// `K::MIN` so that numeric order matches bit-lexicographic order).
    struct HollowTrieBuilderInt<K: PrimitiveInteger> {
        stack: Vec<SpineNode>,
        lens: Vec<usize>,
        prev: K,
        count: usize,
        num_nodes: usize,
    }

    impl<K: PrimitiveInteger> HollowTrieBuilderInt<K> {
        fn new() -> Self {
            Self {
                stack: Vec::new(),
                lens: Vec::new(),
                prev: K::default(),
                count: 0,
                num_nodes: 0,
            }
        }

        fn serialize_chain(nodes: &[SpineNode]) -> (BitVec, Vec<usize>) {
            let mut repr: BitVec = BitVec::new(0);
            let mut skips = Vec::new();
            for node in nodes {
                repr.push(true);
                repr.append(&node.repr);
                repr.push(false);
                skips.push(node.skip);
                skips.extend_from_slice(&node.repr_skips);
            }
            (repr, skips)
        }

        /// Push a key (already XOR-mapped). Keys must be in strictly
        /// increasing order.
        fn push(&mut self, key: K) {
            if self.count == 0 {
                self.prev = key;
                self.count = 1;
                return;
            }

            let lcp = lcp_bits(self.prev, key);

            let mut last = self.stack.len() as isize - 1;
            while last >= 0 && self.lens[last as usize] > lcp {
                last -= 1;
            }

            let pop_start = (last + 1) as usize;
            let popped: Vec<SpineNode> = self.stack.drain(pop_start..).collect();
            self.lens.truncate(pop_start);

            let prefix = if last >= 0 {
                lcp - self.lens[last as usize]
            } else {
                lcp
            };

            if !popped.is_empty() {
                let mut adjusted = popped;
                adjusted[0].skip -= prefix + 1;
                debug_assert!(
                    adjusted[0].skip < usize::MAX / 2,
                    "skip underflow: prefix={prefix}, original skip would give negative"
                );

                let (repr, repr_skips) = Self::serialize_chain(&adjusted);
                let mut new_node = SpineNode::new(prefix);
                new_node.repr = repr;
                new_node.repr_skips = repr_skips;

                let new_len = if last >= 0 {
                    self.lens[last as usize] + prefix + 1
                } else {
                    prefix + 1
                };
                self.stack.push(new_node);
                self.lens.push(new_len);
                self.num_nodes += 1;
            } else {
                let new_node = SpineNode::new(prefix);
                let new_len = if last >= 0 {
                    self.lens[last as usize] + prefix + 1
                } else {
                    prefix + 1
                };
                self.stack.push(new_node);
                self.lens.push(new_len);
                self.num_nodes += 1;
            }

            self.prev = key;
            self.count += 1;
        }

        fn finish(self) -> (BitVec<Box<[usize]>>, Vec<usize>, usize) {
            if self.count <= 1 {
                let mut trie: BitVec = BitVec::new(0);
                trie.push(true);
                trie.push(false);
                return (trie.into(), Vec::new(), 0);
            }

            let (chain_repr, chain_skips) = Self::serialize_chain(&self.stack);
            let mut trie: BitVec = BitVec::new(0);
            trie.push(true);
            trie.append(&chain_repr);
            trie.push(false);

            debug_assert_eq!(
                trie.len(),
                2 * self.num_nodes + 2,
                "trie length mismatch: expected {}, got {}",
                2 * self.num_nodes + 2,
                trie.len()
            );

            (trie.into(), chain_skips, self.num_nodes)
        }
    }

    // ── HtDist constructor ───────────────────────────────────────

    impl HtDist<VFunc<[u8], BitFieldVec<Box<[usize]>>>, VFunc<[u8], BitFieldVec<Box<[usize]>>>> {
        /// Builds a hollow trie distributor from sorted keys.
        ///
        /// `keys` must be in strictly increasing lexicographic order.
        /// `n` is the number of keys and `log2_bucket_size` the base-2 log
        /// of the bucket size.
        pub fn try_new(
            keys: &[impl AsRef<[u8]>],
            log2_bucket_size: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let n = keys.len();
            let bucket_size = 1usize << log2_bucket_size;

            // ── Phase 1: Build trie on delimiters ──────────────────────
            // Delimiters are the last key of each FULL bucket. A partial
            // last bucket does not contribute a delimiter.
            let mut builder = HollowTrieBuilder::new();
            let num_full_buckets = n / bucket_size;
            let num_delimiters = num_full_buckets;

            for b in 0..num_delimiters {
                let last_idx = (b + 1) * bucket_size - 1;
                builder.push(keys[last_idx].as_ref());
            }

            let (trie, raw_skips, num_nodes) = builder.finish();
            let bal_paren = JacobsonBalParen::new(trie);

            // Store skips as a prefix-sum list over Elias-Fano.
            let skips = crate::list::PrefixSumIntList::new(&raw_skips);

            if num_delimiters == 0 {
                return Ok(Self {
                    bal_paren,
                    skips,
                    num_nodes,
                    num_delimiters,
                    false_follows_detector: VFunc::empty(),
                    external_behaviour: VFunc::empty(),
                });
            }

            // ── Phase 2: Compute behaviours ────────────────────────────
            //
            // For each key, descend the trie recording:
            //   - True follows at internal nodes visited for the first time
            //   - False follows + exit behaviour at the stopping point

            // Track which internal nodes (by rank r) have been emitted as
            // true follows.
            let mut emitted = vec![false; num_nodes];

            // Collected keys and values for the two VFuncs.
            let mut ff_keys: Vec<Vec<u8>> = Vec::new();
            let mut ff_values: Vec<usize> = Vec::new();
            let mut ext_keys: Vec<Vec<u8>> = Vec::new();
            let mut ext_values: Vec<usize> = Vec::new();

            pl.info(format_args!(
                "Computing behaviour keys ({n} keys, {num_delimiters} delimiters, {num_nodes} internal nodes)..."
            ));

            // Process all buckets (including any partial last bucket)
            let mut left_delimiter: Option<&[u8]> = None;
            let mut delimiter_lcp: Option<usize>;

            // Create an iterator of right delimiters: Some for each full bucket,
            // then None for the partial bucket (if any).
            let right_delimiters: Vec<Option<&[u8]>> = keys
                .chunks(bucket_size)
                .map(|chunk| {
                    if chunk.len() == bucket_size {
                        Some(chunk[bucket_size - 1].as_ref())
                    } else {
                        None
                    }
                })
                .collect();

            for (b, right_delimiter) in right_delimiters.into_iter().enumerate() {
                let bucket_start = b * bucket_size;
                let bucket_end = ((b + 1) * bucket_size).min(n);

                delimiter_lcp = match (left_delimiter, right_delimiter) {
                    (Some(l), Some(r)) => Some(lcp_bits_nul::<true>(l, r)),
                    _ => None,
                };

                // Stack for resuming the trie walk across keys in the same
                // bucket (keys are sorted, so we can skip the common prefix).
                // Stack for resuming the trie walk across keys in the same
                // bucket. Each entry is (p, r, s, index).
                let mut stack: Vec<(usize, usize, usize, usize)> = vec![(1, 0, 0, 0)];
                let mut depth: usize = 0;

                let mut last_node: Option<usize> = None;
                let mut last_path: Option<Vec<u8>> = None;
                let mut prev_key: Option<&[u8]> = None;

                for key_item in &keys[bucket_start..bucket_end] {
                    let curr = key_item.as_ref();
                    let length = curr.len() * 8 + 8; // bit length including prefix-free terminator

                    // Adjust stack using LCP with previous key in bucket
                    if let Some(prev) = prev_key {
                        let prefix = lcp_bits_nul::<true>(prev, curr);
                        while depth > 0 && stack[depth].2 > prefix {
                            depth -= 1;
                        }
                    }

                    let (mut p, mut r, mut s, mut index) = stack[depth];

                    // Determine exit direction and max descent length.
                    //
                    // `exit_left`: whether the key exits to the left of its
                    // closer delimiter. At the delimiter LCP position, the
                    // left delimiter has bit 0 and the right has bit 1. If
                    // the key's bit matches the right delimiter (bit 1), the
                    // key is in the right half → it exits LEFT of the right
                    // delimiter. This matches the Java's `exitLeft =
                    // curr.getBoolean(delimiterLcp)`.
                    //
                    // `max_descent_length`: the trie walk continues until
                    // `s >= max_descent_length`. Set to `lcp(key, closer
                    // delimiter) + 1`, so the walk stops just past the point
                    // where key and its delimiter diverge.
                    let (exit_left, max_descent_length) = match (left_delimiter, right_delimiter) {
                        (None, Some(rd)) => {
                            // First bucket: no left delimiter; key exits left
                            // of the right delimiter.
                            (true, lcp_bits_nul::<false>(curr, rd) + 1)
                        }
                        (Some(ld), None) => {
                            // Last (partial) bucket: no right delimiter; key
                            // exits right of the left delimiter.
                            (false, lcp_bits_nul::<false>(curr, ld) + 1)
                        }
                        (Some(ld), Some(rd)) => {
                            // Key falls between two delimiters. Check the bit
                            // at the delimiter LCP position to determine which
                            // delimiter is closer.
                            let dlcp = delimiter_lcp.unwrap();
                            if get_key_bit(curr, dlcp) {
                                // Key bit = 1 (matches right delimiter) → exit
                                // left of right delimiter.
                                (true, lcp_bits_nul::<false>(curr, rd) + 1)
                            } else {
                                // Key bit = 0 (matches left delimiter) → exit
                                // right of left delimiter.
                                (false, lcp_bits_nul::<false>(curr, ld) + 1)
                            }
                        }
                        (None, None) => {
                            // Single bucket (no delimiters at all).
                            (true, length + 1)
                        }
                    };

                    // Walk the trie
                    let mut is_internal;
                    let mut skip = 0usize;

                    loop {
                        is_internal = bal_paren[p];
                        if is_internal {
                            skip = skips.index_value(r);
                        }

                        // If this is an internal node, first-time visit, and
                        // within the descent range: record true follow.
                        if is_internal && s + skip < max_descent_length && !emitted[r] {
                            emitted[r] = true;
                            let key = encode_behaviour_key(p - 1, curr, s, (s + skip).min(length));
                            ff_keys.push(key);
                            ff_values.push(0); // true follow
                        }

                        // Stop condition: mirror Java's `(s += skip) >= maxDescentLength`
                        if !is_internal {
                            break;
                        }
                        s += skip;
                        if s >= max_descent_length {
                            break;
                        }

                        // Turn left or right based on key bit at position s
                        if get_key_bit(curr, s) {
                            // Turn right
                            let q = bal_paren
                                .find_close(p)
                                .expect("balanced parentheses broken")
                                + 1;
                            index += (q - p) / 2;
                            r += (q - p) / 2;
                            p = q;
                        } else {
                            // Turn left
                            p += 1;
                            r += 1;
                        }

                        s += 1;

                        // Push to stack
                        depth += 1;
                        if depth >= stack.len() {
                            stack.resize(depth + 1, (0, 0, 0, 0));
                        }
                        stack[depth] = (p, r, s, index);
                    }

                    // Compute path fragment for the exit point
                    let (start_path, end_path) = if is_internal {
                        (s.saturating_sub(skip), s.min(length))
                    } else {
                        (s.min(length), length)
                    };
                    debug_assert!(
                        start_path <= end_path,
                        "bad path range: start={start_path}, end={end_path}, s={s}, skip={skip}, length={length}, is_internal={is_internal}"
                    );

                    // If we exit on a leaf, invalidate last node/path
                    if !is_internal {
                        last_node = None;
                    }

                    let path_key = encode_behaviour_key(p - 1, curr, start_path, end_path);

                    // Deduplicate: only emit if this is a new (node, path)
                    let is_dup = last_node == Some(p - 1)
                        && last_path.as_deref() == Some(path_key.as_slice());

                    if !is_dup {
                        ext_values.push(if exit_left {
                            Behaviour::Left as usize
                        } else {
                            Behaviour::Right as usize
                        });

                        // If exiting at an internal node (false follow), also
                        // record it in the false-follows detector.
                        if is_internal {
                            last_path = Some(path_key.clone());
                            last_node = Some(p - 1);
                            ff_keys.push(path_key.clone());
                            ff_values.push(1); // false follow
                        }
                        ext_keys.push(path_key);
                    }

                    prev_key = Some(key_item.as_ref());
                }

                left_delimiter = right_delimiter;
            }

            pl.info(format_args!(
                "Building false-follows detector ({} keys)...",
                ff_keys.len()
            ));

            let false_follows_detector = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ff_keys),
                FromCloneableIntoIterator::new(ff_values.iter().copied()),
                ff_keys.len(),
                pl,
            )?;

            pl.info(format_args!(
                "Building external behaviour ({} keys)...",
                ext_keys.len()
            ));

            let external_behaviour = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ext_keys),
                FromCloneableIntoIterator::new(ext_values.iter().copied()),
                ext_keys.len(),
                pl,
            )?;

            Ok(Self {
                bal_paren,
                skips,
                num_nodes,
                num_delimiters,
                false_follows_detector,
                external_behaviour,
            })
        }
    }

    // ── HtDistMmphf constructor ──────────────────────────────────

    impl<K: ?Sized + AsRef<[u8]> + ToSig<[u64; 2]> + std::fmt::Debug>
        HtDistMmphf<
            K,
            VFunc<[u8], BitFieldVec<Box<[usize]>>>,
            VFunc<[u8], BitFieldVec<Box<[usize]>>>,
            VFunc<K, BitFieldVec<Box<[usize]>>>,
        >
    {
        /// Builds a new hollow-trie-distributor-based monotone minimal
        /// perfect hash function from sorted byte-sequence keys.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        pub fn try_new<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            mut keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            if n == 0 {
                return Ok(Self {
                    distributor: HtDist {
                        bal_paren: JacobsonBalParen::new({
                            let mut bv: BitVec = BitVec::new(0);
                            bv.push(true);
                            bv.push(false);
                            bv.into()
                        }),
                        skips: crate::list::PrefixSumIntList::new(&Vec::<usize>::new()),
                        num_nodes: 0,
                        num_delimiters: 0,
                        false_follows_detector: VFunc::empty(),
                        external_behaviour: VFunc::empty(),
                    },
                    offset: VFunc::empty(),
                    log2_bucket_size: 0,
                    n: 0,
                    _marker: std::marker::PhantomData,
                });
            }

            // ── Pass 1: compute avg_bits → derive bucket size ─────────
            let mut total_bits: usize = 0;
            let mut count = 0usize;
            while let Some(key) = keys.next()? {
                total_bits += key.as_ref().len() * 8 + 8;
                count += 1;
            }
            anyhow::ensure!(count == n, "Expected {n} keys but got {count}");
            let avg_bits = total_bits as f64 / n as f64;

            let log2_bs = if n <= 1 {
                0
            } else {
                let c = 1.10_f64; // GOV3/VFunc overhead constant
                let val = (avg_bits.ln() + 2.0) * f64::ln(2.0) / c;
                let l = val.max(1.0).round() as usize;
                let l = l.next_power_of_two().ilog2() as usize;
                // Ensure we have at least 2 buckets
                if n / (1usize << l) <= 1 { 0 } else { l }
            };
            let bucket_size = 1usize << log2_bs;
            let bucket_mask = bucket_size - 1;
            let num_full_buckets = n / bucket_size;
            let num_delimiters = num_full_buckets;
            let num_buckets = n.div_ceil(bucket_size);

            pl.info(format_args!(
                "HtDistMmphf: {n} keys, bucket_size=2^{log2_bs}={bucket_size}, avg_key_bits={avg_bits:.0}"
            ));

            // ── Pass 2: collect delimiters, build trie ────────────────
            let mut keys = keys.rewind()?;
            let mut builder = HollowTrieBuilder::new();
            let mut delimiters: Vec<Vec<u8>> = Vec::with_capacity(num_delimiters);
            let mut i = 0usize;
            while let Some(key) = keys.next()? {
                // Delimiter = last key of each full bucket
                if i % bucket_size == bucket_size - 1 && delimiters.len() < num_delimiters {
                    let bytes = key.as_ref();
                    builder.push(bytes);
                    delimiters.push(bytes.to_vec());
                }
                i += 1;
            }
            debug_assert_eq!(delimiters.len(), num_delimiters);

            let (trie, raw_skips, num_nodes) = builder.finish();
            let bal_paren = JacobsonBalParen::new(trie);
            let skips = crate::list::PrefixSumIntList::new(&raw_skips);

            if num_delimiters == 0 {
                return Ok(Self {
                    distributor: HtDist {
                        bal_paren,
                        skips,
                        num_nodes,
                        num_delimiters,
                        false_follows_detector: VFunc::empty(),
                        external_behaviour: VFunc::empty(),
                    },
                    offset: VFunc::empty(),
                    log2_bucket_size: log2_bs,
                    n,
                    _marker: std::marker::PhantomData,
                });
            }

            // ── Pass 3: compute behaviours from lender ────────────────
            let mut keys = keys.rewind()?;

            let mut emitted = vec![false; num_nodes];
            let mut ff_keys: Vec<Vec<u8>> = Vec::new();
            let mut ff_values: Vec<usize> = Vec::new();
            let mut ext_keys: Vec<Vec<u8>> = Vec::new();
            let mut ext_values: Vec<usize> = Vec::new();

            pl.info(format_args!(
                "Computing behaviour keys ({n} keys, {num_delimiters} delimiters, {num_nodes} internal nodes)..."
            ));

            let mut left_delimiter: Option<&[u8]> = None;
            let mut delimiter_lcp: Option<usize>;

            // Create an iterator of right delimiters: Some for each full bucket,
            // then None for the partial bucket (if any).
            let right_delimiters: Vec<Option<&[u8]>> = delimiters
                .iter()
                .map(|d| Some(d.as_slice()))
                .chain(if num_buckets > num_full_buckets {
                    Some(None)
                } else {
                    None
                })
                .collect();

            for (b, right_delimiter) in right_delimiters.into_iter().enumerate() {
                let real_bucket_size = if b < num_full_buckets {
                    bucket_size
                } else {
                    n - b * bucket_size
                };

                delimiter_lcp = match (left_delimiter, right_delimiter) {
                    (Some(l), Some(r)) => Some(lcp_bits_nul::<true>(l, r)),
                    _ => None,
                };

                let mut stack: Vec<(usize, usize, usize, usize)> = vec![(1, 0, 0, 0)];
                let mut depth: usize = 0;

                let mut last_node: Option<usize> = None;
                let mut last_path: Option<Vec<u8>> = None;
                let mut prev_key_buf: Vec<u8> = Vec::new();

                for j in 0..real_bucket_size {
                    let key_ref = keys.next()?.expect("unexpected end of keys");
                    let curr: &[u8] = key_ref.as_ref();
                    let length = curr.len() * 8 + 8;

                    // Adjust stack using LCP with previous key in bucket
                    if j > 0 {
                        let prefix = lcp_bits_nul::<true>(&prev_key_buf, curr);
                        while depth > 0 && stack[depth].2 > prefix {
                            depth -= 1;
                        }
                    }

                    let (mut p, mut r, mut s, mut index) = stack[depth];

                    let (exit_left, max_descent_length) = match (left_delimiter, right_delimiter) {
                        (None, Some(rd)) => (true, lcp_bits_nul::<false>(curr, rd) + 1),
                        (Some(ld), None) => (false, lcp_bits_nul::<false>(curr, ld) + 1),
                        (Some(ld), Some(rd)) => {
                            let dlcp = delimiter_lcp.unwrap();
                            if get_key_bit(curr, dlcp) {
                                (true, lcp_bits_nul::<false>(curr, rd) + 1)
                            } else {
                                (false, lcp_bits_nul::<false>(curr, ld) + 1)
                            }
                        }
                        (None, None) => (true, length + 1),
                    };

                    // Walk the trie
                    let mut is_internal;
                    let mut skip = 0usize;

                    loop {
                        is_internal = bal_paren[p];
                        if is_internal {
                            skip = skips.index_value(r);
                        }

                        if is_internal && s + skip < max_descent_length && !emitted[r] {
                            emitted[r] = true;
                            let key = encode_behaviour_key(p - 1, curr, s, (s + skip).min(length));
                            ff_keys.push(key);
                            ff_values.push(0); // true follow
                        }

                        if !is_internal {
                            break;
                        }
                        s += skip;
                        if s >= max_descent_length {
                            break;
                        }

                        if get_key_bit(curr, s) {
                            let q = bal_paren
                                .find_close(p)
                                .expect("balanced parentheses broken")
                                + 1;
                            index += (q - p) / 2;
                            r += (q - p) / 2;
                            p = q;
                        } else {
                            p += 1;
                            r += 1;
                        }

                        s += 1;

                        depth += 1;
                        if depth >= stack.len() {
                            stack.resize(depth + 1, (0, 0, 0, 0));
                        }
                        stack[depth] = (p, r, s, index);
                    }

                    let (start_path, end_path) = if is_internal {
                        (s.saturating_sub(skip), s.min(length))
                    } else {
                        (s.min(length), length)
                    };
                    debug_assert!(
                        start_path <= end_path,
                        "bad path range: start={start_path}, end={end_path}, s={s}, skip={skip}, length={length}, is_internal={is_internal}"
                    );

                    if !is_internal {
                        last_node = None;
                    }

                    let path_key = encode_behaviour_key(p - 1, curr, start_path, end_path);

                    let is_dup = last_node == Some(p - 1)
                        && last_path.as_deref() == Some(path_key.as_slice());

                    if !is_dup {
                        ext_values.push(if exit_left {
                            Behaviour::Left as usize
                        } else {
                            Behaviour::Right as usize
                        });

                        if is_internal {
                            last_path = Some(path_key.clone());
                            last_node = Some(p - 1);
                            ff_keys.push(path_key.clone());
                            ff_values.push(1); // false follow
                        }
                        ext_keys.push(path_key);
                    }

                    prev_key_buf = curr.to_vec();
                }

                left_delimiter = right_delimiter;
            }

            pl.info(format_args!(
                "Building false-follows detector ({} keys)...",
                ff_keys.len()
            ));

            let false_follows_detector = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ff_keys),
                FromCloneableIntoIterator::new(ff_values.iter().copied()),
                ff_keys.len(),
                pl,
            )?;

            pl.info(format_args!(
                "Building external behaviour ({} keys)...",
                ext_keys.len()
            ));

            let external_behaviour = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ext_keys),
                FromCloneableIntoIterator::new(ext_values.iter().copied()),
                ext_keys.len(),
                pl,
            )?;

            let distributor = HtDist {
                bal_paren,
                skips,
                num_nodes,
                num_delimiters,
                false_follows_detector,
                external_behaviour,
            };

            // ── Pass 4: build offset VFunc ────────────────────────────
            let keys = keys.rewind()?;

            pl.info(format_args!("Building offset VFunc..."));

            let offset = <VFunc<K, BitFieldVec<Box<[usize]>>>>::try_new(
                keys,
                FromCloneableIntoIterator::new((0..n).map(|i| i & bucket_mask)),
                n,
                pl,
            )?;

            let result = Self {
                distributor,
                offset,
                log2_bucket_size: log2_bs,
                n,
                _marker: std::marker::PhantomData,
            };

            let flags = SizeFlags::default();
            let total_bits = result.mem_size(flags) * 8;
            let mut refs = mem_dbg::HashMap::default();
            let dist_bp = result.distributor.bal_paren.mem_size_rec(flags, &mut refs) * 8;
            let dist_bp_words = result.distributor.bal_paren.as_ref().len() * 8 * 8;
            let dist_bp_pioneers = dist_bp - dist_bp_words;
            let dist_skips = result.distributor.skips.mem_size(flags) * 8;
            let dist_ff = result.distributor.false_follows_detector.mem_size(flags) * 8;
            let dist_ext = result.distributor.external_behaviour.mem_size(flags) * 8;
            let offset_bits = result.offset.mem_size(flags) * 8;
            info!(
                "HtDistMmphf: {:.2} bits/key ({total_bits} bits for {n} keys)",
                total_bits as f64 / n as f64
            );
            info!(
                "  Trie BP words: {dist_bp_words} bits ({:.2}/key), pioneers: {dist_bp_pioneers} bits ({:.2}/key)",
                dist_bp_words as f64 / n as f64,
                dist_bp_pioneers as f64 / n as f64,
            );
            info!(
                "  Skips: {dist_skips} bits ({:.2}/key)",
                dist_skips as f64 / n as f64,
            );
            info!(
                "  False-follows VFunc: {dist_ff} bits ({:.2}/key, {} keys)",
                dist_ff as f64 / n as f64,
                result.distributor.false_follows_detector.len(),
            );
            info!(
                "  External behaviour VFunc: {dist_ext} bits ({:.2}/key, {} keys)",
                dist_ext as f64 / n as f64,
                result.distributor.external_behaviour.len(),
            );
            info!(
                "  Offset VFunc: {offset_bits} bits ({:.2}/key)",
                offset_bits as f64 / n as f64,
            );

            Ok(result)
        }
    }

    // ── HtDistInt constructor ────────────────────────────────────

    impl<K> HtDistInt<K, VFunc<[u8], BitFieldVec<Box<[usize]>>>, VFunc<[u8], BitFieldVec<Box<[usize]>>>>
    where
        K: PrimitiveInteger + Copy + Ord + Send + Sync + std::fmt::Debug,
    {
        /// Builds a hollow trie distributor from sorted integer keys.
        ///
        /// `keys` must be in strictly increasing order. `n` is the total
        /// number of keys and `log2_bucket_size` the base-2 log of the
        /// bucket size.
        ///
        /// The constructor iterates over all keys twice: once to collect
        /// delimiters and build the trie, and once to compute behaviour
        /// keys. The lender is rewound between passes.
        pub fn try_new(
            mut keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            log2_bucket_size: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let bit_length = K::BITS as usize;
            let bucket_size = 1usize << log2_bucket_size;
            let num_full_buckets = n / bucket_size;
            let num_delimiters = num_full_buckets;
            let num_buckets = n.div_ceil(bucket_size);

            // ── Pass 1: collect delimiters, build trie ────────────────
            let mut builder = HollowTrieBuilderInt::<K>::new();
            let mut delimiters: Vec<K> = Vec::with_capacity(num_delimiters);
            let mut i = 0usize;
            while let Some(key) = keys.next()? {
                let mapped = *key ^ K::MIN;
                if i % bucket_size == bucket_size - 1 && delimiters.len() < num_delimiters {
                    builder.push(mapped);
                    delimiters.push(mapped);
                }
                i += 1;
            }
            anyhow::ensure!(i == n, "Expected {n} keys but got {i}");
            debug_assert_eq!(delimiters.len(), num_delimiters);

            let (trie, raw_skips, num_nodes) = builder.finish();
            let bal_paren = JacobsonBalParen::new(trie);
            let skips = crate::list::PrefixSumIntList::new(&raw_skips);

            if num_delimiters == 0 {
                return Ok(Self {
                    bal_paren,
                    skips,
                    num_nodes,
                    num_delimiters,
                    false_follows_detector: VFunc::empty(),
                    external_behaviour: VFunc::empty(),
                    _marker: std::marker::PhantomData,
                });
            }

            // ── Pass 2: compute behaviours from lender ────────────────
            let mut keys = keys.rewind()?;

            let mut emitted = vec![false; num_nodes];
            let mut ff_keys: Vec<Vec<u8>> = Vec::new();
            let mut ff_values: Vec<usize> = Vec::new();
            let mut ext_keys: Vec<Vec<u8>> = Vec::new();
            let mut ext_values: Vec<usize> = Vec::new();

            pl.info(format_args!(
                "Computing behaviour keys ({n} keys, {num_delimiters} delimiters, {num_nodes} internal nodes)..."
            ));

            let mut left_delimiter: Option<K> = None;
            let mut delimiter_lcp: Option<usize>;

            // Create an iterator of right delimiters: Some for each full bucket,
            // then None for the partial bucket (if any).
            let right_delimiters: Vec<Option<K>> = delimiters
                .iter()
                .copied()
                .map(Some)
                .chain(if num_buckets > num_full_buckets {
                    Some(None)
                } else {
                    None
                })
                .collect();

            for (b, right_delimiter) in right_delimiters.into_iter().enumerate() {
                let real_bucket_size = if b < num_full_buckets {
                    bucket_size
                } else {
                    n - b * bucket_size
                };

                delimiter_lcp = match (left_delimiter, right_delimiter) {
                    (Some(l), Some(r)) => Some(lcp_bits(l, r)),
                    _ => None,
                };

                let mut stack: Vec<(usize, usize, usize, usize)> = vec![(1, 0, 0, 0)];
                let mut depth: usize = 0;

                let mut last_node: Option<usize> = None;
                let mut last_path: Option<Vec<u8>> = None;
                let mut prev_mapped: K = K::default();

                for j in 0..real_bucket_size {
                    let key_ref = keys.next()?.expect("unexpected end of keys");
                    let mapped = *key_ref ^ K::MIN;

                    if j > 0 {
                        let prefix = lcp_bits(prev_mapped, mapped);
                        while depth > 0 && stack[depth].2 > prefix {
                            depth -= 1;
                        }
                    }

                    let (mut p, mut r, mut s, mut index) = stack[depth];

                    // For integers: identical keys yield lcp == K::BITS,
                    // so max_descent_length == K::BITS + 1 — correct.
                    let (exit_left, max_descent_length) = match (left_delimiter, right_delimiter) {
                        (None, Some(rd)) => (true, lcp_bits(mapped, rd) + 1),
                        (Some(ld), None) => (false, lcp_bits(mapped, ld) + 1),
                        (Some(ld), Some(rd)) => {
                            let dlcp = delimiter_lcp.unwrap();
                            if get_int_bit(mapped, dlcp) {
                                (true, lcp_bits(mapped, rd) + 1)
                            } else {
                                (false, lcp_bits(mapped, ld) + 1)
                            }
                        }
                        (None, None) => (true, bit_length + 1),
                    };

                    let mut is_internal;
                    let mut skip = 0usize;

                    loop {
                        is_internal = bal_paren[p];
                        if is_internal {
                            skip = skips.index_value(r);
                        }

                        if is_internal && s + skip < max_descent_length && !emitted[r] {
                            emitted[r] = true;
                            let key = encode_int_behaviour_key(
                                p - 1,
                                mapped,
                                s,
                                (s + skip).min(bit_length),
                            );
                            ff_keys.push(key);
                            ff_values.push(0);
                        }

                        if !is_internal {
                            break;
                        }
                        s += skip;
                        if s >= max_descent_length {
                            break;
                        }

                        if get_int_bit(mapped, s) {
                            let q = bal_paren
                                .find_close(p)
                                .expect("balanced parentheses broken")
                                + 1;
                            index += (q - p) / 2;
                            r += (q - p) / 2;
                            p = q;
                        } else {
                            p += 1;
                            r += 1;
                        }

                        s += 1;

                        depth += 1;
                        if depth >= stack.len() {
                            stack.resize(depth + 1, (0, 0, 0, 0));
                        }
                        stack[depth] = (p, r, s, index);
                    }

                    let (start_path, end_path) = if is_internal {
                        (s.saturating_sub(skip), s.min(bit_length))
                    } else {
                        (s.min(bit_length), bit_length)
                    };
                    debug_assert!(
                        start_path <= end_path,
                        "bad path range: start={start_path}, end={end_path}, s={s}, skip={skip}, bit_length={bit_length}, is_internal={is_internal}"
                    );

                    if !is_internal {
                        last_node = None;
                    }

                    let path_key = encode_int_behaviour_key(p - 1, mapped, start_path, end_path);

                    let is_dup = last_node == Some(p - 1)
                        && last_path.as_deref() == Some(path_key.as_slice());

                    if !is_dup {
                        ext_values.push(if exit_left {
                            Behaviour::Left as usize
                        } else {
                            Behaviour::Right as usize
                        });
                        if is_internal {
                            last_path = Some(path_key.clone());
                            last_node = Some(p - 1);
                            ff_keys.push(path_key.clone());
                            ff_values.push(1);
                        }
                        ext_keys.push(path_key);
                    }

                    prev_mapped = mapped;
                }

                left_delimiter = right_delimiter;
            }

            pl.info(format_args!(
                "Building false-follows detector ({} keys)...",
                ff_keys.len()
            ));

            let false_follows_detector = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ff_keys),
                FromCloneableIntoIterator::new(ff_values.iter().copied()),
                ff_keys.len(),
                pl,
            )?;

            pl.info(format_args!(
                "Building external behaviour ({} keys)...",
                ext_keys.len()
            ));

            let external_behaviour = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ext_keys),
                FromCloneableIntoIterator::new(ext_values.iter().copied()),
                ext_keys.len(),
                pl,
            )?;

            Ok(Self {
                bal_paren,
                skips,
                num_nodes,
                num_delimiters,
                false_follows_detector,
                external_behaviour,
                _marker: std::marker::PhantomData,
            })
        }
    }

    // ── HtDistMmphfInt constructor ───────────────────────────────

    impl<K> HtDistMmphfInt<K>
    where
        K: PrimitiveInteger + ToSig<[u64; 2]> + Copy + Ord + Send + Sync + std::fmt::Debug,
    {
        /// Builds a new hollow-trie-distributor-based monotone minimal
        /// perfect hash function from sorted integer keys.
        ///
        /// The keys must be in strictly increasing order.
        pub fn try_new(
            mut keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let bit_length = K::BITS as usize;

            if n == 0 {
                return Ok(Self {
                    distributor: HtDistInt {
                        bal_paren: JacobsonBalParen::new({
                            let mut bv: BitVec = BitVec::new(0);
                            bv.push(true);
                            bv.push(false);
                            bv.into()
                        }),
                        skips: crate::list::PrefixSumIntList::new(&Vec::<usize>::new()),
                        num_nodes: 0,
                        num_delimiters: 0,
                        false_follows_detector: VFunc::empty(),
                        external_behaviour: VFunc::empty(),
                        _marker: std::marker::PhantomData,
                    },
                    offset: VFunc::empty(),
                    log2_bucket_size: 0,
                    n: 0,
                });
            }

            // Bucket size from key bit-width and n — no iteration needed.
            let avg_bits = bit_length as f64;
            let log2_bs = if n <= 1 {
                0
            } else {
                let c = 1.10_f64;
                let val = (avg_bits.ln() + 2.0) * f64::ln(2.0) / c;
                let l = val.max(1.0).round() as usize;
                let l = l.next_power_of_two().ilog2() as usize;
                if n / (1usize << l) <= 1 { 0 } else { l }
            };
            let bucket_size = 1usize << log2_bs;
            let bucket_mask = bucket_size - 1;
            let num_full_buckets = n / bucket_size;
            let num_delimiters = num_full_buckets;
            let num_buckets = n.div_ceil(bucket_size);

            pl.info(format_args!(
                "HtDistMmphfInt<{}>: {n} keys, bucket_size=2^{log2_bs}={bucket_size}, key_bits={bit_length}",
                std::any::type_name::<K>()
            ));

            // ── Pass 1: collect delimiters, build trie ────────────────
            let mut builder = HollowTrieBuilderInt::<K>::new();
            let mut delimiters: Vec<K> = Vec::with_capacity(num_delimiters);
            let mut i = 0usize;
            while let Some(key) = keys.next()? {
                let mapped = *key ^ K::MIN;
                if i % bucket_size == bucket_size - 1 && delimiters.len() < num_delimiters {
                    builder.push(mapped);
                    delimiters.push(mapped);
                }
                i += 1;
            }
            anyhow::ensure!(i == n, "Expected {n} keys but got {i}");
            debug_assert_eq!(delimiters.len(), num_delimiters);

            let (trie, raw_skips, num_nodes) = builder.finish();
            let bal_paren = JacobsonBalParen::new(trie);
            let skips = crate::list::PrefixSumIntList::new(&raw_skips);

            if num_delimiters == 0 {
                // Rewind for the offset VFunc pass.
                let keys = keys.rewind()?;
                let offset = <VFunc<K, BitFieldVec<Box<[usize]>>>>::try_new(
                    keys,
                    FromCloneableIntoIterator::new((0..n).map(|i| i & bucket_mask)),
                    n,
                    pl,
                )?;
                return Ok(Self {
                    distributor: HtDistInt {
                        bal_paren,
                        skips,
                        num_nodes,
                        num_delimiters,
                        false_follows_detector: VFunc::empty(),
                        external_behaviour: VFunc::empty(),
                        _marker: std::marker::PhantomData,
                    },
                    offset,
                    log2_bucket_size: log2_bs,
                    n,
                });
            }

            // ── Pass 2: compute behaviours from lender ────────────────
            let mut keys = keys.rewind()?;

            let mut emitted = vec![false; num_nodes];
            let mut ff_keys: Vec<Vec<u8>> = Vec::new();
            let mut ff_values: Vec<usize> = Vec::new();
            let mut ext_keys: Vec<Vec<u8>> = Vec::new();
            let mut ext_values: Vec<usize> = Vec::new();

            pl.info(format_args!(
                "Computing behaviour keys ({n} keys, {num_delimiters} delimiters, {num_nodes} internal nodes)..."
            ));

            let mut left_delimiter: Option<K> = None;
            let mut delimiter_lcp: Option<usize>;

            // Create an iterator of right delimiters: Some for each full bucket,
            // then None for the partial bucket (if any).
            let right_delimiters: Vec<Option<K>> = delimiters
                .iter()
                .copied()
                .map(Some)
                .chain(if num_buckets > num_full_buckets {
                    Some(None)
                } else {
                    None
                })
                .collect();

            for (b, right_delimiter) in right_delimiters.into_iter().enumerate() {
                let real_bucket_size = if b < num_full_buckets {
                    bucket_size
                } else {
                    n - b * bucket_size
                };

                delimiter_lcp = match (left_delimiter, right_delimiter) {
                    (Some(l), Some(r)) => Some(lcp_bits(l, r)),
                    _ => None,
                };

                let mut stack: Vec<(usize, usize, usize, usize)> = vec![(1, 0, 0, 0)];
                let mut depth: usize = 0;

                let mut last_node: Option<usize> = None;
                let mut last_path: Option<Vec<u8>> = None;
                let mut prev_mapped: K = K::default();

                for j in 0..real_bucket_size {
                    let key_ref = keys.next()?.expect("unexpected end of keys");
                    let mapped = *key_ref ^ K::MIN;

                    if j > 0 {
                        let prefix = lcp_bits(prev_mapped, mapped);
                        while depth > 0 && stack[depth].2 > prefix {
                            depth -= 1;
                        }
                    }

                    let (mut p, mut r, mut s, mut index) = stack[depth];

                    // For integers: identical keys yield lcp == K::BITS,
                    // so max_descent_length == K::BITS + 1 — correct.
                    let (exit_left, max_descent_length) = match (left_delimiter, right_delimiter) {
                        (None, Some(rd)) => (true, lcp_bits(mapped, rd) + 1),
                        (Some(ld), None) => (false, lcp_bits(mapped, ld) + 1),
                        (Some(ld), Some(rd)) => {
                            let dlcp = delimiter_lcp.unwrap();
                            if get_int_bit(mapped, dlcp) {
                                (true, lcp_bits(mapped, rd) + 1)
                            } else {
                                (false, lcp_bits(mapped, ld) + 1)
                            }
                        }
                        (None, None) => (true, bit_length + 1),
                    };

                    let mut is_internal;
                    let mut skip = 0usize;

                    loop {
                        is_internal = bal_paren[p];
                        if is_internal {
                            skip = skips.index_value(r);
                        }

                        if is_internal && s + skip < max_descent_length && !emitted[r] {
                            emitted[r] = true;
                            let key = encode_int_behaviour_key(
                                p - 1,
                                mapped,
                                s,
                                (s + skip).min(bit_length),
                            );
                            ff_keys.push(key);
                            ff_values.push(0);
                        }

                        if !is_internal {
                            break;
                        }
                        s += skip;
                        if s >= max_descent_length {
                            break;
                        }

                        if get_int_bit(mapped, s) {
                            let q = bal_paren
                                .find_close(p)
                                .expect("balanced parentheses broken")
                                + 1;
                            index += (q - p) / 2;
                            r += (q - p) / 2;
                            p = q;
                        } else {
                            p += 1;
                            r += 1;
                        }

                        s += 1;

                        depth += 1;
                        if depth >= stack.len() {
                            stack.resize(depth + 1, (0, 0, 0, 0));
                        }
                        stack[depth] = (p, r, s, index);
                    }

                    let (start_path, end_path) = if is_internal {
                        (s.saturating_sub(skip), s.min(bit_length))
                    } else {
                        (s.min(bit_length), bit_length)
                    };
                    debug_assert!(
                        start_path <= end_path,
                        "bad path range: start={start_path}, end={end_path}, s={s}, skip={skip}, bit_length={bit_length}, is_internal={is_internal}"
                    );

                    if !is_internal {
                        last_node = None;
                    }

                    let path_key = encode_int_behaviour_key(p - 1, mapped, start_path, end_path);

                    let is_dup = last_node == Some(p - 1)
                        && last_path.as_deref() == Some(path_key.as_slice());

                    if !is_dup {
                        ext_values.push(if exit_left {
                            Behaviour::Left as usize
                        } else {
                            Behaviour::Right as usize
                        });
                        if is_internal {
                            last_path = Some(path_key.clone());
                            last_node = Some(p - 1);
                            ff_keys.push(path_key.clone());
                            ff_values.push(1);
                        }
                        ext_keys.push(path_key);
                    }

                    prev_mapped = mapped;
                }

                left_delimiter = right_delimiter;
            }

            pl.info(format_args!(
                "Building false-follows detector ({} keys)...",
                ff_keys.len()
            ));

            let false_follows_detector = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ff_keys),
                FromCloneableIntoIterator::new(ff_values.iter().copied()),
                ff_keys.len(),
                pl,
            )?;

            pl.info(format_args!(
                "Building external behaviour ({} keys)...",
                ext_keys.len()
            ));

            let external_behaviour = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ext_keys),
                FromCloneableIntoIterator::new(ext_values.iter().copied()),
                ext_keys.len(),
                pl,
            )?;

            let distributor = HtDistInt {
                bal_paren,
                skips,
                num_nodes,
                num_delimiters,
                false_follows_detector,
                external_behaviour,
                _marker: std::marker::PhantomData,
            };

            // ── Pass 3: build offset VFunc ────────────────────────────
            // Pass the lender directly to VFunc<K> — no collection needed.
            let keys = keys.rewind()?;

            pl.info(format_args!("Building offset VFunc..."));

            let offset = <VFunc<K, BitFieldVec<Box<[usize]>>>>::try_new(
                keys,
                FromCloneableIntoIterator::new((0..n).map(|i| i & bucket_mask)),
                n,
                pl,
            )?;

            let result = Self {
                distributor,
                offset,
                log2_bucket_size: log2_bs,
                n,
            };

            let flags = SizeFlags::default();
            let total_bits = result.mem_size(flags) * 8;
            info!(
                "HtDistMmphfInt: {:.2} bits/key ({total_bits} bits for {n} keys)",
                total_bits as f64 / n as f64
            );

            Ok(result)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests {
    use super::build::*;
    use super::*;

    #[test]
    fn test_trie_builder_empty() {
        let builder = HollowTrieBuilder::new();
        let (trie, skips, num_nodes) = builder.finish();
        assert_eq!(trie.len(), 2); // just ()
        assert_eq!(num_nodes, 0);
        assert!(skips.is_empty());
    }

    #[test]
    fn test_trie_builder_single() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"hello");
        let (trie, _skips, num_nodes) = builder.finish();
        assert_eq!(trie.len(), 2); // just ()
        assert_eq!(num_nodes, 0);
    }

    #[test]
    fn test_trie_builder_two_keys() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"abc");
        builder.push(b"abd");
        let (trie, skips, num_nodes) = builder.finish();
        // Two keys = one internal node = 1()0 + fake = 1 1 0 0 = 4 bits
        assert_eq!(num_nodes, 1);
        assert_eq!(trie.len(), 4);
        assert_eq!(skips.len(), 1);
        // The LCP of "abc" and "abd" is 2 bytes + some bits.
        // "abc" = 01100001 01100010 01100011
        // "abd" = 01100001 01100010 01100100
        // LCP in bits = 16 + leading_zeros(01100011 ^ 01100100) = 16 + leading_zeros(00000111) = 16 + 5 = 21
        assert_eq!(skips[0], 21);
    }

    #[test]
    fn test_trie_builder_three_keys() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"a");
        builder.push(b"b");
        builder.push(b"c");
        let (trie, skips, num_nodes) = builder.finish();
        // Three keys = two internal nodes
        assert_eq!(num_nodes, 2);
        assert_eq!(trie.len(), 6); // 1 (1()0) (1()0) 0 = but actually nested
        assert_eq!(skips.len(), 2);
    }

    #[test]
    fn test_trie_builder_many_keys() {
        let mut builder = HollowTrieBuilder::new();
        let keys: Vec<String> = (0..100).map(|i| format!("key_{:04}", i)).collect();
        for key in &keys {
            builder.push(key.as_bytes());
        }
        let (trie, skips, num_nodes) = builder.finish();
        // 100 keys = 99 internal nodes
        assert_eq!(num_nodes, 99);
        assert_eq!(trie.len(), 2 * 99 + 2);
        assert_eq!(skips.len(), 99);
    }

    #[cfg(feature = "rayon")]
    mod distributor_tests {
        use super::*;
        use dsi_progress_logger::no_logging;

        #[test]
        fn test_distributor_directly() {
            let keys: Vec<&str> = vec!["alpha", "beta", "delta", "gamma", "omega"];
            let log2_bs = 0;
            let dist = HtDist::try_new(&keys, log2_bs, no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    dist.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }

        #[test]
        fn test_ht_dist_mmphf_small() {
            let keys: Vec<&str> = vec!["alpha", "beta", "delta", "gamma", "omega"];
            let func =
                HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key:?} at position {i}");
            }
        }

        #[test]
        fn test_ht_dist_mmphf_many() {
            let keys: Vec<String> = (0..500).map(|i| format!("key_{:06}", i)).collect();
            let func =
                HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key:?} at position {i}");
            }
        }

        #[test]
        fn test_ht_dist_mmphf_two_keys() {
            let keys: Vec<&str> = vec!["aaa", "zzz"];
            let func =
                HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i);
            }
        }

        #[test]
        fn test_ht_dist_mmphf_single_key() {
            let keys: Vec<&str> = vec!["only"];
            let func =
                HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            assert_eq!(func.get("only"), 0);
        }

        #[test]
        fn test_ht_dist_mmphf_large() {
            let keys: Vec<String> = (0..5000).map(|i| format!("key_{:08}", i)).collect();
            let func =
                HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key:?} at position {i}");
            }
        }

        #[test]
        fn test_ht_dist_mmphf_prefix_keys() {
            let keys: Vec<&str> = vec![
                "0",
                "00",
                "000",
                "0000",
                "00000",
                "000000",
                "0000000",
                "00000000",
                "000000000",
                "0000000000",
                "1",
                "10",
                "100",
                "2",
                "20",
                "200",
            ];
            let func =
                HtDistMmphfStr::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key:?} at position {i}");
            }
        }
    }

    // ── HtDistMmphfInt tests ──────────────────────────────────────

    #[cfg(feature = "rayon")]
    mod int_tests {
        use super::*;
        use dsi_progress_logger::no_logging;

        #[test]
        fn test_int_small_u64() {
            let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key} at position {i}");
            }
        }

        #[test]
        fn test_int_many_u64() {
            let keys: Vec<u64> = (0..500).map(|i| i * 7 + 1000).collect();
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key} at position {i}");
            }
        }

        #[test]
        fn test_int_two_keys() {
            let keys: Vec<u64> = vec![0, u64::MAX];
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i);
            }
        }

        #[test]
        fn test_int_single_key() {
            let keys: Vec<u64> = vec![42];
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            assert_eq!(func.get(42), 0);
        }

        #[test]
        fn test_int_large_u64() {
            let keys: Vec<u64> = (0..5000).map(|i| i * 3).collect();
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key} at position {i}");
            }
        }

        #[test]
        fn test_int_u32() {
            let keys: Vec<u32> = (0..200).map(|i| i * 11 + 5).collect();
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key} at position {i}");
            }
        }

        #[test]
        fn test_int_signed_i64() {
            let keys: Vec<i64> = (-250..250).collect();
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key} at position {i}");
            }
        }

        #[test]
        fn test_int_consecutive_u64() {
            let keys: Vec<u64> = (0..100).collect();
            let func =
                HtDistMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![]).unwrap();
            for (i, &key) in keys.iter().enumerate() {
                assert_eq!(func.get(key), i, "Mismatch for key {key} at position {i}");
            }
        }
    }
}
