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

use mem_dbg::*;
use succinctly::trees::BalancedParens;

// ═══════════════════════════════════════════════════════════════════
// Bit manipulation helpers
// ═══════════════════════════════════════════════════════════════════

/// Read bit `i` from a byte slice (MSB-first within each byte).
/// Returns `false` for positions beyond the key length (virtual NUL).
#[inline]
fn get_key_bit(key: &[u8], i: usize) -> bool {
    let byte_idx = i / 8;
    let bit_idx = 7 - (i % 8); // MSB first
    if byte_idx < key.len() {
        (key[byte_idx] >> bit_idx) & 1 != 0
    } else {
        false // virtual NUL
    }
}

/// Read bit `i` from a word-packed bit vector (LSB-first within each word).
#[inline]
fn get_bit(words: &[u64], i: usize) -> bool {
    (words[i / 64] >> (i % 64)) & 1 != 0
}

/// Append a single bit to a word-packed bit vector.
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
fn lcp_bits_nul(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    for i in 0..min_len {
        if a[i] != b[i] {
            return i * 8 + (a[i] ^ b[i]).leading_zeros() as usize;
        }
    }
    if a.len() == b.len() {
        a.len() * 8 + 8 // identical + virtual NUL
    } else {
        let longer = if a.len() > b.len() { a } else { b };
        min_len * 8 + longer[min_len].leading_zeros() as usize
    }
}

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

impl SpineNode {
    fn leaf() -> Self {
        Self {
            skip: 0,
            repr: Vec::new(),
            repr_len: 0,
            repr_skips: Vec::new(),
        }
    }

    fn new(skip: usize) -> Self {
        Self {
            skip,
            repr: Vec::new(),
            repr_len: 0,
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
            // Serialize the popped chain into a new node's left subtree.
            let (repr, repr_len, repr_skips) = Self::serialize_chain(&popped);

            // Adjust the topmost popped node's skip: the new internal
            // node takes `prefix` bits, plus 1 branching bit.
            // The popped chain's root had a skip that included these bits.
            let mut new_node = SpineNode::new(prefix);
            new_node.repr = repr;
            new_node.repr_len = repr_len;
            new_node.repr_skips = repr_skips;

            // If the top of the remaining stack exists, adjust its
            // right child's skip.
            if let Some(top) = self.stack.last_mut() {
                // nothing to adjust on stack — the popped nodes' skips
                // were already correct relative to their parent.
            }

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
// HollowTrieDistributor — the main structure
// ═══════════════════════════════════════════════════════════════════

/// A hollow trie distributor that assigns sorted byte-sequence keys
/// to bucket indices.
///
/// Built from sorted keys and a bucket size. Uses a hollow trie on
/// the bucket delimiters combined with behaviour functions stored as
/// [`VFunc`](crate::func::VFunc)s.
#[derive(Debug)]
pub struct HollowTrieDistributor {
    /// Balanced-parentheses bit vector of the trie.
    trie_words: Vec<u64>,
    /// Length in bits.
    trie_len: usize,
    /// Balanced-parentheses support.
    bal_paren: BalancedParens,
    /// Skip values indexed by DFS rank.
    skips: Vec<usize>,
    /// Number of internal nodes (= number of delimiters - 1).
    num_nodes: usize,
    /// Number of keys.
    n: usize,
    /// Log2 of bucket size.
    log2_bucket_size: usize,
}

impl HollowTrieDistributor {
    /// Returns the bucket index for the given key.
    ///
    /// The key is navigated through the hollow trie using the balanced
    /// parentheses structure and skip values. At each internal node,
    /// the behaviour functions determine whether to follow the trie
    /// edge or exit left/right.
    pub fn get(&self, _key: &[u8]) -> usize {
        // TODO: implement the full query with behaviour functions
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_builder_empty() {
        let builder = HollowTrieBuilder::new();
        let (words, len, skips, num_nodes) = builder.finish();
        assert_eq!(len, 2); // just ()
        assert_eq!(num_nodes, 0);
        assert!(skips.is_empty());
    }

    #[test]
    fn test_trie_builder_single() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"hello");
        let (words, len, skips, num_nodes) = builder.finish();
        assert_eq!(len, 2); // just ()
        assert_eq!(num_nodes, 0);
    }

    #[test]
    fn test_trie_builder_two_keys() {
        let mut builder = HollowTrieBuilder::new();
        builder.push(b"abc");
        builder.push(b"abd");
        let (words, len, skips, num_nodes) = builder.finish();
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
        let (words, len, skips, num_nodes) = builder.finish();
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
        let (words, len, skips, num_nodes) = builder.finish();
        // 100 keys = 99 internal nodes
        assert_eq!(num_nodes, 99);
        assert_eq!(len, 2 * 99 + 2);
        assert_eq!(skips.len(), 99);
    }
}
