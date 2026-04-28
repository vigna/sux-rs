/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Shared primitives for 3-uniform hypergraph peeling.
//!
//! This module holds the data structures used by [`VBuilder`] and [`CompVFunc`]
//! to peel graphs.

use std::ops::BitXorAssign;
use std::slice::Iter;

// ── XorGraph ────────────────────────────────────────────────────────

/// A graph represented compactly.
///
/// Each (*k*-hyper)edge is a set of *k* vertices (by construction fuse graphs
/// do not have degenerate edges), but we represent it internally as a vector.
/// We call *side* the position of a vertex in the edge.
///
/// For each vertex we store information about the edges incident to the vertex
/// and the sides of the vertex in such edges. While technically not necessary
/// to perform peeling, the knowledge of the sides speeds up the peeling visit
/// by reducing the number of tests that are necessary to update the degrees
/// once the edge is peeled (see the `peel_by_*` methods). For the same reason
/// it also speeds up assignment.
///
/// Depending on the peeling method (by signature or by index), the graph will
/// store edge indices or signature/value pairs (the generic parameter `X`).
///
/// Edge information is packed together using Djamal's XOR trick (see
/// [“Cache-Oblivious Peeling of Random Hypergraphs”]): since during the
/// peeling visit we need to know the content of the list only when a single
/// edge index is present, we can XOR together all the edge information.
///
/// We use a single byte to store the degree (six upper bits) and the XOR of the
/// sides (lower two bits). The degree can be stored with a small number of bits
/// because the graph is random, so the maximum degree is *O*(log log *n*).
/// In any case, the Boolean field `overflow` will become `true` in case of
/// overflow.
///
/// When we peel an edge, we just [zero the degree], leaving the edge
/// information and the side in place for further processing later.
///
/// [zero the degree]: Self::zero
///
/// ["Cache-Oblivious Peeling of Random Hypergraphs"]: https://doi.org/10.1109/DCC.2014.48
pub(crate) struct XorGraph<X: Copy + Default + BitXorAssign> {
    edges: Box<[X]>,
    degrees_sides: Box<[u8]>,
    pub(crate) overflow: bool,
}

impl<X: BitXorAssign + Default + Copy> XorGraph<X> {
    pub(crate) fn new(n: usize) -> XorGraph<X> {
        XorGraph {
            edges: vec![X::default(); n].into(),
            degrees_sides: vec![0; n].into(),
            overflow: false,
        }
    }

    #[inline(always)]
    pub(crate) fn add(&mut self, v: usize, x: X, side: usize) {
        debug_assert!(side < 3);
        let (degree_size, overflow) = self.degrees_sides[v].overflowing_add(4);
        self.degrees_sides[v] = degree_size;
        self.overflow |= overflow;
        self.degrees_sides[v] ^= side as u8;
        self.edges[v] ^= x;
    }

    #[inline(always)]
    pub(crate) fn remove(&mut self, v: usize, x: X, side: usize) {
        debug_assert!(side < 3);
        self.degrees_sides[v] -= 4;
        self.degrees_sides[v] ^= side as u8;
        self.edges[v] ^= x;
    }

    #[inline(always)]
    pub(crate) fn zero(&mut self, v: usize) {
        self.degrees_sides[v] &= 0b11;
    }

    #[inline(always)]
    pub(crate) fn edge_and_side(&self, v: usize) -> (X, usize) {
        debug_assert!(self.degree(v) < 2);
        (self.edges[v] as _, (self.degrees_sides[v] & 0b11) as _)
    }

    #[inline(always)]
    pub(crate) fn degree(&self, v: usize) -> u8 {
        self.degrees_sides[v] >> 2
    }

    pub(crate) fn degrees(
        &self,
    ) -> std::iter::Map<std::iter::Copied<std::slice::Iter<'_, u8>>, fn(u8) -> u8> {
        self.degrees_sides.iter().copied().map(|d| d >> 2)
    }
}

// ── FastStack ───────────────────────────────────────────────────────

/// A preallocated stack implementation that avoids the expensive (even if
/// rarely taken) branch of the [`Vec`] implementation in which memory is
/// reallocated. Note that using [`Vec::with_capacity`] is not enough, because
/// for the CPU the branch is still there.
///
/// [`Vec`]: std::vec::Vec
/// [`Vec::with_capacity`]: std::vec::Vec::with_capacity
pub(crate) struct FastStack<X: Copy + Default> {
    stack: Vec<X>,
    top: usize,
}

impl<X: Copy + Default> FastStack<X> {
    pub(crate) fn new(n: usize) -> FastStack<X> {
        FastStack {
            stack: vec![X::default(); n],
            top: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, x: X) {
        debug_assert!(self.top < self.stack.len());
        self.stack[self.top] = x;
        self.top += 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.top
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, X> {
        self.stack[..self.top].iter()
    }
}

// ── DoubleStack ─────────────────────────────────────────────────────

/// Two stacks in the same vector.
///
/// This struct implements a pair of stacks sharing the same memory. The lower
/// stack grows from the beginning of the vector, the upper stack grows from the
/// end of the vector. Since we use the lower stack for the visit and the upper
/// stack for peeled edges (possibly represented by the vertex from which they
/// have been peeled), the sum of the lengths of the two stacks cannot exceed
/// the length of the vector.
#[derive(Debug)]
pub(crate) struct DoubleStack<V> {
    stack: Vec<V>,
    lower: usize,
    upper: usize,
}

impl<V: Default + Copy> DoubleStack<V> {
    pub(crate) fn new(n: usize) -> DoubleStack<V> {
        DoubleStack {
            stack: vec![V::default(); n],
            lower: 0,
            upper: n,
        }
    }
}

impl<V: Copy> DoubleStack<V> {
    #[inline(always)]
    pub(crate) fn push_lower(&mut self, v: V) {
        debug_assert!(self.lower < self.upper);
        self.stack[self.lower] = v;
        self.lower += 1;
    }

    #[inline(always)]
    pub(crate) fn push_upper(&mut self, v: V) {
        debug_assert!(self.lower < self.upper);
        self.upper -= 1;
        self.stack[self.upper] = v;
    }

    #[inline(always)]
    pub(crate) fn pop_lower(&mut self) -> Option<V> {
        if self.lower == 0 {
            None
        } else {
            self.lower -= 1;
            Some(self.stack[self.lower])
        }
    }

    pub(crate) fn upper_len(&self) -> usize {
        self.stack.len() - self.upper
    }

    pub(crate) fn iter_upper(&self) -> Iter<'_, V> {
        self.stack[self.upper..].iter()
    }
}

// ── remove_edge! macro ─────────────────────────────────────────────

/// Removes a peeled edge from the [`XorGraph`], pushing onto `$stack`
/// (via `$push`) any of the edge's *other* two vertices that just
/// dropped to degree 2 (and will become degree 1 after the removal).
///
/// `$xor_graph`, `$stack` are identifiers; `$e` is the 3-vertex array
/// (`[usize; 3]`); `$side` is the pivot side (0/1/2); `$edge` is the
/// payload to XOR out of the other two vertices; `$conv` is a closure
/// `usize -> StackElem` that converts a vertex index into whatever the
/// stack stores.
macro_rules! remove_edge {
    ($xor_graph: ident, $e: ident, $side: ident, $edge: ident, $stack: ident, $push:ident, $conv: expr) => {
        match $side {
            0 => {
                if $xor_graph.degree($e[1] as usize) == 2 {
                    $stack.$push($conv($e[1]));
                }
                $xor_graph.remove($e[1] as usize, $edge, 1);
                if $xor_graph.degree($e[2] as usize) == 2 {
                    $stack.$push($conv($e[2]));
                }
                $xor_graph.remove($e[2] as usize, $edge, 2);
            }
            1 => {
                if $xor_graph.degree($e[0] as usize) == 2 {
                    $stack.$push($conv($e[0]));
                }
                $xor_graph.remove($e[0] as usize, $edge, 0);
                if $xor_graph.degree($e[2] as usize) == 2 {
                    $stack.$push($conv($e[2]));
                }
                $xor_graph.remove($e[2] as usize, $edge, 2);
            }
            2 => {
                if $xor_graph.degree($e[0] as usize) == 2 {
                    $stack.$push($conv($e[0]));
                }
                $xor_graph.remove($e[0] as usize, $edge, 0);
                if $xor_graph.degree($e[1] as usize) == 2 {
                    $stack.$push($conv($e[1]));
                }
                $xor_graph.remove($e[1] as usize, $edge, 1);
            }
            // SAFETY: side is always 0, 1, or 2 (encoded as a 2-bit
            // field in the degrees_sides array).
            _ => unsafe { ::std::hint::unreachable_unchecked() },
        }
    };
}

pub(crate) use remove_edge;
