/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! Hollow trie distributor for monotone minimal perfect hashing.
//!
//! A hollow trie is a compacted (Patricia) trie whose edge labels have been
//! replaced with their lengths. Combined with behaviour functions (stored as
//! [`VFunc`](crate::func::VFunc)s), it distributes sorted keys into
//! equal-size buckets in O(log *u*) time per query, where *u* is the
//! universe size.
//!
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
//! [Theory and practice of monotone minimal perfect
//! hashing](https://doi.org/10.1145/2775054). *Journal of Experimental
//! Algorithmics* 20(3):1−26, 2016.

#[cfg(feature = "rayon")]
use {
    crate::bits::BitFieldVec,
    crate::func::VFunc,
    crate::utils::*,
    anyhow::Result,
    dsi_progress_logger::ProgressLog,
    log::info,
    mem_dbg::*,
    succinctly::trees::BalancedParens,
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
#[cfg(feature = "rayon")]
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

/// Read bit `i` from a word-packed bit vector (LSB-first within each word).
#[cfg(feature = "rayon")]
#[inline]
fn get_bit(words: &[u64], i: usize) -> bool {
    (words[i / 64] >> (i % 64)) & 1 != 0
}

/// Append a single bit to a word-packed bit vector.
#[cfg(feature = "rayon")]
#[inline]
fn push_bit(words: &mut Vec<u64>, len: &mut usize, bit: bool) {
    if *len % 64 == 0 {
        words.push(0);
    }
    if bit {
        let last = words.len() - 1;
        words[last] |= 1u64 << (*len % 64);
    }
    *len += 1;
}

/// Append `src_len` bits from `src` (word-packed, LSB-first) to `dst`.
#[cfg(feature = "rayon")]
fn push_bits(dst: &mut Vec<u64>, dst_len: &mut usize, src: &[u64], src_len: usize) {
    // TODO: optimize with word-level copy + shift
    for i in 0..src_len {
        push_bit(dst, dst_len, get_bit(src, i));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Trie construction (online, stack-based)
// ═══════════════════════════════════════════════════════════════════

/// LCP in bits between two byte slices (MSB-first, with virtual NUL).
/// LCP in bits between two byte slices (MSB-first) with virtual NUL
/// terminator (all-zero byte after the last real byte). Since actual
/// bytes are assumed nonzero, the NUL is always distinct from any
/// continuation byte, ensuring prefix-freeness while preserving
/// lexicographic order.
#[cfg(feature = "rayon")]
fn lcp_bits_nul(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    for i in 0..min_len {
        if a[i] != b[i] {
            return i * 8 + (a[i] ^ b[i]).leading_zeros() as usize;
        }
    }
    if a.len() == b.len() {
        // Identical keys (including virtual NUL). The virtual NUL is
        // 0x00 for both, so all bits match. Return len*8 + 8 as the
        // full extent (the NUL byte is fully shared).
        min_len * 8 + 8
    } else {
        // The shorter key has virtual NUL 0x00 at position min_len.
        // The longer key has byte b[min_len] (nonzero by assumption).
        // NUL XOR nonzero byte = the byte itself.
        let next_byte = if a.len() > b.len() { a[min_len] } else { b[min_len] };
        min_len * 8 + next_byte.leading_zeros() as usize
    }
}

#[cfg(feature = "rayon")]
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
    repr: Vec<u64>,
    /// Number of bits used in `repr`.
    repr_len: usize,
    /// Skip values for internal nodes in the left subtree (DFS order).
    repr_skips: Vec<usize>,
}

#[cfg(feature = "rayon")]
impl SpineNode {
    fn new(skip: usize) -> Self {
        Self {
            skip,
            repr: Vec::new(),
            repr_len: 0,
            repr_skips: Vec::new(),
        }
    }
}

#[cfg(feature = "rayon")]
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

#[cfg(feature = "rayon")]
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
    fn serialize_chain(
        nodes: &[SpineNode],
    ) -> (Vec<u64>, usize, Vec<usize>) {
        let mut repr = Vec::new();
        let mut repr_len = 0usize;
        let mut skips = Vec::new();

        for node in nodes {
            push_bit(&mut repr, &mut repr_len, true);  // (
            push_bits(&mut repr, &mut repr_len, &node.repr, node.repr_len);
            push_bit(&mut repr, &mut repr_len, false); // )
            skips.push(node.skip);
            skips.extend_from_slice(&node.repr_skips);
        }

        (repr, repr_len, skips)
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

        let lcp = lcp_bits_nul(&self.prev, key);

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
            let (repr, repr_len, repr_skips) = Self::serialize_chain(&adjusted);

            let mut new_node = SpineNode::new(prefix);
            new_node.repr = repr;
            new_node.repr_len = repr_len;
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
    /// - `trie_words`: word-packed balanced-parentheses bit vector
    /// - `trie_len`: number of bits in the trie
    /// - `skips`: skip values in DFS preorder
    /// - `num_nodes`: number of internal nodes
    pub fn finish(self) -> (Vec<u64>, usize, Vec<usize>, usize) {
        if self.count <= 1 {
            // Empty or single-element trie.
            let mut words = Vec::new();
            let mut len = 0;
            push_bit(&mut words, &mut len, true);
            push_bit(&mut words, &mut len, false);
            return (words, len, Vec::new(), 0);
        }

        // Serialize the remaining right spine.
        let (chain_repr, chain_repr_len, chain_skips) =
            Self::serialize_chain(&self.stack);

        // Wrap in fake root brackets: 1 [chain] 0
        let mut trie = Vec::new();
        let mut trie_len = 0;
        push_bit(&mut trie, &mut trie_len, true);
        push_bits(&mut trie, &mut trie_len, &chain_repr, chain_repr_len);
        push_bit(&mut trie, &mut trie_len, false);

        debug_assert_eq!(
            trie_len,
            2 * self.num_nodes + 2,
            "trie length mismatch: expected {}, got {}",
            2 * self.num_nodes + 2,
            trie_len
        );

        (trie, trie_len, chain_skips, self.num_nodes)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Behaviour key encoding
// ═══════════════════════════════════════════════════════════════════

/// Encodes a (node position, path fragment) pair as a byte vector for
/// use as a VFunc key. The encoding is:
///   - 8 bytes: node position as u64 LE
///   - 8 bytes: path length in bits as u64 LE
///   - ceil((bit_end - bit_start) / 8) bytes: path bits packed MSB-first
///
/// The encoding must be injective for correctness.
///
/// Returns the number of bytes written into `buf`.
#[cfg(feature = "rayon")]
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
    let total = 16 + packed_bytes;
    debug_assert!(buf.len() >= total);
    buf[0..8].copy_from_slice(&(node_pos as u64).to_le_bytes());
    buf[8..16].copy_from_slice(&(path_len as u64).to_le_bytes());

    // Pack path bits MSB-first into bytes. Since key_bytes is MSB-first,
    // we can use byte-level operations when bit_start is byte-aligned,
    // and shift operations otherwise.
    let start_byte = bit_start / 8;
    let start_bit_offset = bit_start % 8;

    if start_bit_offset == 0 {
        // Byte-aligned: copy whole bytes, mask the last one.
        let copy_bytes = packed_bytes.min(key_bytes.len().saturating_sub(start_byte));
        buf[16..16 + copy_bytes]
            .copy_from_slice(&key_bytes[start_byte..start_byte + copy_bytes]);
        // Zero any bytes beyond the key (virtual NUL).
        for i in copy_bytes..packed_bytes {
            buf[16 + i] = 0;
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
            buf[16 + b] = (hi << shift) | (lo >> (8 - shift));
        }
    }

    // Mask off trailing bits in the last byte.
    let trail = path_len % 8;
    if trail != 0 && packed_bytes > 0 {
        buf[16 + packed_bytes - 1] &= !((1u8 << (8 - trail)) - 1);
    }

    total
}

/// Convenience wrapper that allocates a `Vec` — used during construction
/// where keys need to be stored.
#[cfg(feature = "rayon")]
fn encode_behaviour_key(
    node_pos: usize,
    key_bytes: &[u8],
    bit_start: usize,
    bit_end: usize,
) -> Vec<u8> {
    let path_len = bit_end - bit_start;
    let packed_bytes = path_len.div_ceil(8);
    let mut buf = vec![0u8; 16 + packed_bytes];
    encode_behaviour_key_into(&mut buf, node_pos, key_bytes, bit_start, bit_end);
    buf
}

// ═══════════════════════════════════════════════════════════════════
// HollowTrieDistributor — the main structure
// ═══════════════════════════════════════════════════════════════════

/// A hollow trie distributor that assigns sorted byte-sequence keys
/// to bucket indices.
///
/// Built from sorted keys and a bucket size. Uses a hollow trie on
/// the bucket delimiters combined with behaviour functions stored as
/// [`VFunc`](crate::func::VFunc)s.
#[cfg(feature = "rayon")]
#[derive(Debug)]
pub struct HollowTrieDistributor {
    /// Balanced-parentheses support structure for the trie.
    bal_paren: BalancedParens,
    /// Skip values stored as a prefix-sum list over Elias-Fano.
    skips: crate::list::PrefixSumIntList,
    /// Number of internal nodes (= number of delimiters - 1).
    #[allow(dead_code)]
    num_nodes: usize,
    /// Number of delimiters.
    num_delimiters: usize,
    /// Detects false follows: maps (node, path) -> 0 (true follow) or
    /// 1 (false follow).
    #[cfg(feature = "rayon")]
    false_follows_detector: VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    /// External behaviour: maps (node, path) -> LEFT (0) or RIGHT (1).
    #[cfg(feature = "rayon")]
    external_behaviour: VFunc<[u8], BitFieldVec<Box<[usize]>>>,
}

#[cfg(feature = "rayon")]
impl MemSize for HollowTrieDistributor {
    fn mem_size_rec(
        &self,
        flags: SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        let mut size = core::mem::size_of::<Self>();
        // BalancedParens internal memory is opaque; estimate from words
        size += self.bal_paren.words().len() * 8;
        size += self.skips.mem_size_rec(flags, refs);
        #[cfg(feature = "rayon")]
        {
            size += self.false_follows_detector.mem_size_rec(flags, refs);
            size += self.external_behaviour.mem_size_rec(flags, refs);
        }
        size
    }
}

#[cfg(feature = "rayon")]
impl MemDbgImpl for HollowTrieDistributor {}

/// Exit on the left (closer to left delimiter).
#[cfg(feature = "rayon")]
const LEFT: usize = 0;
/// Exit on the right (closer to right delimiter).
#[cfg(feature = "rayon")]
const RIGHT: usize = 1;
/// Follow the trie edge (true follow).
#[cfg(feature = "rayon")]
const FOLLOW: usize = 2;

#[cfg(feature = "rayon")]
impl HollowTrieDistributor {
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

        let (trie_words, trie_len, raw_skips, num_nodes) = builder.finish();
        let bal_paren = BalancedParens::new(trie_words.clone(), trie_len);

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
        let num_buckets = n.div_ceil(bucket_size);
        let mut right_delimiter: Option<&[u8]> = None;
        let mut delimiter_lcp: Option<usize>;

        for b in 0..num_buckets {
            let bucket_start = b * bucket_size;
            let bucket_end = ((b + 1) * bucket_size).min(n);
            let real_bucket_size = bucket_end - bucket_start;

            // Update delimiters: right_delimiter is only set for full buckets
            let left_delimiter = right_delimiter;
            right_delimiter = if real_bucket_size == bucket_size {
                Some(keys[bucket_end - 1].as_ref())
            } else {
                None
            };
            delimiter_lcp = match (left_delimiter, right_delimiter) {
                (Some(l), Some(r)) => Some(lcp_bits_nul(l, r)),
                _ => None,
            };

            // Stack for resuming the trie walk across keys in the same
            // bucket (keys are sorted, so we can skip the common prefix).
            let mut stack_p: Vec<usize> = vec![1];
            let mut stack_r: Vec<usize> = vec![0];
            let mut stack_s: Vec<usize> = vec![0];
            let mut stack_index: Vec<usize> = vec![0];
            let mut depth: usize = 0;

            let mut last_node: Option<usize> = None;
            let mut last_path: Option<Vec<u8>> = None;
            let mut prev_key: Option<&[u8]> = None;

            for j in 0..real_bucket_size {
                let curr = keys[bucket_start + j].as_ref();
                let length = curr.len() * 8 + 8; // bit length including prefix-free terminator

                // Adjust stack using LCP with previous key in bucket
                if let Some(prev) = prev_key {
                    let prefix = lcp_bits_nul(prev, curr);
                    while depth > 0 && stack_s[depth] > prefix {
                        depth -= 1;
                    }
                }

                let mut p = stack_p[depth];
                let mut r = stack_r[depth];
                let mut s = stack_s[depth];
                let mut index = stack_index[depth];

                // Determine exit direction and max descent length
                let (exit_left, max_descent_length) = match (left_delimiter, right_delimiter) {
                    (None, Some(rd)) => {
                        // First bucket: no left delimiter
                        (true, lcp_bits_nul(curr, rd) + 1)
                    }
                    (Some(ld), None) => {
                        // Last (partial) bucket: no right delimiter
                        (false, lcp_bits_nul(curr, ld) + 1)
                    }
                    (Some(ld), Some(rd)) => {
                        let dlcp = delimiter_lcp.unwrap();
                        let el = get_key_bit(curr, dlcp);
                        if el {
                            // Closer to right delimiter
                            (true, lcp_bits_nul(curr, rd) + 1)
                        } else {
                            // Closer to left delimiter
                            (false, lcp_bits_nul(curr, ld) + 1)
                        }
                    }
                    (None, None) => {
                        // Single bucket edge case
                        (true, length + 1)
                    }
                };

                // Walk the trie
                let mut is_internal;
                let mut skip = 0usize;

                loop {
                    is_internal = get_bit(&trie_words, p);
                    if is_internal {
                        use value_traits::slices::SliceByValue;
                        {
                            use value_traits::slices::SliceByValue;
                            skip = skips.index_value(r);
                        }
                    }

                    // If this is an internal node, first-time visit, and
                    // within the descent range: record true follow.
                    if is_internal
                        && s + skip < max_descent_length
                        && !emitted[r]
                    {
                        emitted[r] = true;
                        let key = encode_behaviour_key(
                            p - 1,
                            curr,
                            s,
                            (s + skip).min(length),
                        );
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
                    if depth >= stack_p.len() {
                        stack_p.resize(depth + 1, 0);
                        stack_r.resize(depth + 1, 0);
                        stack_s.resize(depth + 1, 0);
                        stack_index.resize(depth + 1, 0);
                    }
                    stack_p[depth] = p;
                    stack_r[depth] = r;
                    stack_s[depth] = s;
                    stack_index[depth] = index;
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
                    ext_keys.push(path_key.clone());
                    ext_values.push(if exit_left { LEFT } else { RIGHT });

                    // If exiting at an internal node (false follow), also
                    // record it in the false-follows detector.
                    if is_internal {
                        last_path = Some(path_key.clone());
                        last_node = Some(p - 1);
                        ff_keys.push(path_key);
                        ff_values.push(1); // false follow
                    }
                }

                prev_key = Some(keys[bucket_start + j].as_ref());
            }
        }

        pl.info(format_args!(
            "Building false-follows detector ({} keys)...",
            ff_keys.len()
        ));

        let false_follows_detector =
            <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
                FromSlice::new(&ff_keys),
                FromCloneableIntoIterator::new(ff_values.iter().copied()),
                ff_keys.len(),
                pl,
            )?;

        pl.info(format_args!(
            "Building external behaviour ({} keys)...",
            ext_keys.len()
        ));

        let external_behaviour =
            <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
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

        let trie_words = self.bal_paren.words();
        let length = key.len() * 8 + 8; // including virtual NUL terminator
        let mut p: usize = 1;
        let mut index: usize = 0;
        let mut r: usize = 0;
        let mut s: usize = 0;
        let mut last_left_turn: usize = 0;
        let mut last_left_turn_index: usize = 0;
        // Buffer for behaviour key encoding. Reused across loop iterations
        // to avoid per-node allocation. Max size: 16 header + key path bytes.
        let buf_size = 16 + key.len() + 1;
        let mut key_buf_storage;
        let mut key_buf_vec;
        let mut key_buf: &mut [u8] = if buf_size <= 528 {
            key_buf_storage = [0u8; 528];
            &mut key_buf_storage[..buf_size]
        } else {
            key_buf_vec = vec![0u8; buf_size];
            &mut key_buf_vec
        };

        loop {
            let is_internal = get_bit(trie_words, p);
            let skip: usize = if is_internal {
                use value_traits::slices::SliceByValue;
                self.skips.index_value(r)
            } else {
                0
            };

            let behaviour = if is_internal {
                let n = encode_behaviour_key_into(
                    &mut key_buf, p - 1, key, s, (s + skip).min(length),
                );
                if self.false_follows_detector.get(&key_buf[..n]) == 0 {
                    FOLLOW
                } else {
                    self.external_behaviour.get(&key_buf[..n])
                }
            } else {
                let n = encode_behaviour_key_into(
                    &mut key_buf, p - 1, key, s, length,
                );
                self.external_behaviour.get(&key_buf[..n])
            };

            if behaviour != FOLLOW || !is_internal || {
                s += skip;
                s >= length
            } {
                if behaviour == LEFT {
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

            // Turn left or right
            if get_key_bit(key, s) {
                // Right
                let q = self
                    .bal_paren
                    .find_close(p)
                    .expect("balanced parentheses broken")
                    + 1;
                index += (q - p) / 2;
                r += (q - p) / 2;
                p = q;
            } else {
                // Left
                last_left_turn = p;
                last_left_turn_index = index;
                p += 1;
                r += 1;
            }

            s += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// HtDistMmphf — Hollow Trie Distributor MMPHF
// ═══════════════════════════════════════════════════════════════════

/// A monotone minimal perfect hash function based on a
/// [`HollowTrieDistributor`] and per-bucket offsets stored in a
/// [`VFunc`].
///
/// Given *n* byte-sequence keys in sorted order, the structure maps each
/// key to its rank (0 to *n* − 1). Querying a key not in the original
/// set returns an arbitrary value.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::hollow_trie::HtDistMmphf;
/// let keys: Vec<&str> = vec!["alpha", "beta", "delta", "gamma"];
/// let func = HtDistMmphf::try_new(&keys, no_logging![])?;
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_bytes()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
#[derive(Debug)]
#[cfg(feature = "rayon")]
pub struct HtDistMmphf {
    /// The hollow trie distributor.
    distributor: HollowTrieDistributor,
    /// Per-key offset within the bucket.
    offset: VFunc<[u8], BitFieldVec<Box<[usize]>>>,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
    /// Number of keys.
    n: usize,
}

#[cfg(feature = "rayon")]
impl MemSize for HtDistMmphf {
    fn mem_size_rec(
        &self,
        flags: SizeFlags,
        refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        let mut size = core::mem::size_of::<Self>();
        size += self.distributor.mem_size_rec(flags, refs);
        size += self.offset.mem_size_rec(flags, refs);
        size
    }
}

#[cfg(feature = "rayon")]
impl MemDbgImpl for HtDistMmphf {}

#[cfg(feature = "rayon")]
impl HtDistMmphf {
    /// Builds a new hollow-trie-distributor-based monotone minimal
    /// perfect hash function from sorted byte-sequence keys.
    ///
    /// The keys must be in strictly increasing lexicographic order.
    pub fn try_new(
        keys: &[impl AsRef<[u8]>],
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_new_with_log2_bucket_size(keys, None, pl)
    }

    /// Builds with an explicit log2 bucket size. If `None`, it is
    /// computed automatically.
    pub fn try_new_with_log2_bucket_size(
        keys: &[impl AsRef<[u8]>],
        log2_bucket_size: Option<usize>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        let n = keys.len();
        if n == 0 {
            return Ok(Self {
                distributor: HollowTrieDistributor {
                    bal_paren: BalancedParens::new(vec![0b10], 2),
                    skips: crate::list::PrefixSumIntList::new(&Vec::<usize>::new()),
                    num_nodes: 0,
                    num_delimiters: 0,
                    false_follows_detector: VFunc::empty(),
                    external_behaviour: VFunc::empty(),
                },
                offset: VFunc::empty(),
                log2_bucket_size: 0,
                n: 0,
            });
        }

        // Compute average key length in bits (including virtual NUL)
        let total_bits: usize = keys.iter().map(|k| k.as_ref().len() * 8 + 8).sum();
        let avg_bits = total_bits as f64 / n as f64;

        // Compute log2 bucket size following the Java formula.
        // The constant C is the overhead of the VFunc (GOV3/fuse).
        // For VFunc with fuse graphs, C ≈ 1.125.
        let log2_bs = log2_bucket_size.unwrap_or_else(|| {
            if n <= 1 {
                return 0;
            }
            let c = 1.10_f64; // GOV3/VFunc overhead constant
            let val = (avg_bits.ln() + 2.0) * f64::ln(2.0) / c;
            let l = val.max(1.0).round() as usize;
            let l = if l.is_power_of_two() {
                l.ilog2() as usize
            } else {
                l.next_power_of_two().ilog2() as usize
            };
            // Ensure we have at least 2 buckets
            if n / (1usize << l) <= 1 { 0 } else { l }
        });
        let bucket_size = 1usize << log2_bs;

        pl.info(format_args!(
            "HtDistMmphf: {n} keys, bucket_size=2^{log2_bs}={bucket_size}, avg_key_bits={avg_bits:.0}"
        ));

        let distributor = HollowTrieDistributor::try_new(keys, log2_bs, pl)?;

        // Build offset VFunc
        let bucket_mask = bucket_size - 1;
        pl.info(format_args!("Building offset VFunc..."));

        // Convert keys to Vec<u8> so they implement Borrow<[u8]>
        let byte_keys: Vec<Vec<u8>> = keys.iter().map(|k| k.as_ref().to_vec()).collect();
        let offset = <VFunc<[u8], BitFieldVec<Box<[usize]>>>>::try_new(
            FromSlice::new(&byte_keys),
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
        let dist_bp = result.distributor.bal_paren.words().len() * 8 * 8;
        let dist_skips = result.distributor.skips.mem_size(flags) * 8;
        let dist_ff = result.distributor.false_follows_detector.mem_size(flags) * 8;
        let dist_ext = result.distributor.external_behaviour.mem_size(flags) * 8;
        let offset_bits = result.offset.mem_size(flags) * 8;
        info!(
            "HtDistMmphf: {:.2} bits/key ({total_bits} bits for {n} keys)",
            total_bits as f64 / n as f64
        );
        info!(
            "  Trie BP: {dist_bp} bits ({:.2}/key), skips: {dist_skips} bits ({:.2}/key)",
            dist_bp as f64 / n as f64,
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

    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary.
    #[inline]
    pub fn get(&self, key: &[u8]) -> usize {
        if self.n <= 1 {
            return 0;
        }
        let bucket = self.distributor.get(key);
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

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests {
    use super::*;

    #[test]
    fn test_trie_builder_empty() {
        let builder = HollowTrieBuilder::new();
        let (_words, len, skips, num_nodes) = builder.finish();
        assert_eq!(len, 2); // just ()
        assert_eq!(num_nodes, 0);
        assert!(skips.is_empty());
    }

    #[test]
    fn test_trie_builder_single() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"hello");
        let (_words, len, _skips, num_nodes) = builder.finish();
        assert_eq!(len, 2); // just ()
        assert_eq!(num_nodes, 0);
    }

    #[test]
    fn test_trie_builder_two_keys() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"abc");
        builder.push(b"abd");
        let (_words, len, skips, num_nodes) = builder.finish();
        // Two keys = one internal node = 1()0 + fake = 1 1 0 0 = 4 bits
        assert_eq!(num_nodes, 1);
        assert_eq!(len, 4);
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
        let (_words, len, skips, num_nodes) = builder.finish();
        // Three keys = two internal nodes
        assert_eq!(num_nodes, 2);
        assert_eq!(len, 6); // 1 (1()0) (1()0) 0 = but actually nested
        assert_eq!(skips.len(), 2);
    }

    #[test]
    fn test_trie_builder_many_keys() {
        let mut builder = HollowTrieBuilder::new();
        let keys: Vec<String> = (0..100).map(|i| format!("key_{:04}", i)).collect();
        for key in &keys {
            builder.push(key.as_bytes());
        }
        let (_words, len, skips, num_nodes) = builder.finish();
        // 100 keys = 99 internal nodes
        assert_eq!(num_nodes, 99);
        assert_eq!(len, 2 * 99 + 2);
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
            let dist = HollowTrieDistributor::try_new(&keys, log2_bs, no_logging![]).unwrap();
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
            let func = HtDistMmphf::try_new(&keys, no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    func.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }

        #[test]
        fn test_ht_dist_mmphf_many() {
            let keys: Vec<String> =
                (0..500).map(|i| format!("key_{:06}", i)).collect();
            let func =
                HtDistMmphf::try_new(&keys, no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    func.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }

        #[test]
        fn test_ht_dist_mmphf_two_keys() {
            let keys: Vec<&str> = vec!["aaa", "zzz"];
            let func = HtDistMmphf::try_new(&keys, no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(func.get(key.as_bytes()), i);
            }
        }

        #[test]
        fn test_ht_dist_mmphf_fixed_bucket_size() {
            let keys: Vec<String> =
                (0..200).map(|i| format!("key_{:06}", i)).collect();
            let func = HtDistMmphf::try_new_with_log2_bucket_size(
                &keys,
                Some(4),
                no_logging![],
            )
            .unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    func.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }

        #[test]
        fn test_ht_dist_mmphf_single_key() {
            let keys: Vec<&str> = vec!["only"];
            let func = HtDistMmphf::try_new(&keys, no_logging![]).unwrap();
            assert_eq!(func.get(b"only"), 0);
        }

        #[test]
        fn test_ht_dist_mmphf_large() {
            let keys: Vec<String> =
                (0..5000).map(|i| format!("key_{:08}", i)).collect();
            let func = HtDistMmphf::try_new(&keys, no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    func.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }

        #[test]
        fn test_ht_dist_mmphf_bucket_size_1() {
            // With bucket_size=1 (log2=0), every key is its own bucket.
            let keys: Vec<String> =
                (0..50).map(|i| format!("x_{:04}", i)).collect();
            let func = HtDistMmphf::try_new_with_log2_bucket_size(
                &keys,
                Some(0),
                no_logging![],
            )
            .unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    func.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }

        #[test]
        fn test_ht_dist_mmphf_prefix_keys() {
            let keys: Vec<&str> = vec![
                "0", "00", "000", "0000", "00000", "000000", "0000000",
                "00000000", "000000000", "0000000000", "1", "10", "100",
                "2", "20", "200",
            ];
            let func = HtDistMmphf::try_new(&keys, no_logging![]).unwrap();
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    func.get(key.as_bytes()),
                    i,
                    "Mismatch for key {key:?} at position {i}"
                );
            }
        }
    }
}
