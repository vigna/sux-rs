/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Generic peeling primitives for 3-uniform F₂ hypergraph linear systems.
//!
//! This module factors out the data structures and peel-loop bodies that
//! [`VBuilder`](crate::func::VBuilder) and
//! [`CompVFunc`](crate::func::CompVFunc) share. It exposes:
//!
//! * [`XorGraph`] / [`DoubleStack`] / [`FastStack`] — the graph and stack
//!   primitives used by all peelers (cache-oblivious peeling à la
//!   ["Cache-Oblivious Peeling of Random Hypergraphs"][cop]).
//! * [`peel_by_index`] — the index-based peeler (X = small id), supports
//!   partial peeling for an LGE residual fallback.
//! * [`peel_by_data_high_mem`] / [`peel_by_data_low_mem`] — data-based
//!   peelers where the [`XorGraph`] payload `X` carries enough
//!   information to reconstruct the edge at assign time. Both drop any
//!   data the caller's `gen_edges` closure owns as soon as the graph is
//!   built.
//!
//! All three peelers are **value-agnostic**: they don't take an `rhs`
//! array, they don't touch the output data, and they don't know what an
//! "edge value" is. The caller passes a `gen_edges` closure that fills
//! the `XorGraph` (typically by iterating its own data and calling
//! [`XorGraph::add`]) plus a `verts_of` closure that recovers the 3
//! vertices of an edge from its `X` payload (used by the inner
//! [`remove_edge!`] expansion). On success the peeler returns the peel
//! stacks; assignment is the caller's responsibility, which keeps the
//! peeler's signature value-free.
//!
//! This separation is what lets [`VFunc`](crate::func::VFunc) (one edge
//! per key) and [`CompVFunc`] (`L` ≥ 1 edges per key, one per codeword
//! bit) share the *same* peeler core, with their differences confined
//! to the closures.
//!
//! [cop]: https://doi.org/10.1109/DCC.2014.48

use std::ops::BitXorAssign;
use std::slice::Iter;
use std::sync::atomic::{AtomicBool, Ordering};

// ── Primitives ──────────────────────────────────────────────────────

/// A vertex/edge XOR-cancellation graph for 3-uniform hypergraphs,
/// adapted from ["Cache-Oblivious Peeling of Random Hypergraphs"][cop].
///
/// The XorGraph stores, for each vertex `v`, a `degree | side` byte and
/// a payload `X`. Every edge incident on `v` contributes its own `X`
/// payload (XOR-accumulated into `edges[v]`) and increments the vertex
/// degree by one. When `degree(v)` drops to 1, `edges[v]` equals the
/// payload of the single remaining incident edge — that's how the peel
/// loop recovers an edge's identity from a leaf vertex.
///
/// The payload type `X` is generic so the same primitive can store
/// either compact integer ids (for [`peel_by_index`], where the caller
/// keeps the shard alive and looks the edge up at assign time) or
/// self-contained edge data (for [`peel_by_data_high_mem`] /
/// [`peel_by_data_low_mem`], where the shard is dropped after graph
/// construction).
///
/// [cop]: https://doi.org/10.1109/DCC.2014.48
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

/// A preallocated stack that avoids the rare-but-expensive Vec-grow
/// branch.
///
/// Even `Vec::with_capacity` keeps the grow branch in the generated
/// code; this struct compiles to a flat `stack[top] = x; top += 1`
/// without the branch. Used for the peel record where the maximum size
/// is known upfront.
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

/// Two stacks sharing a single backing buffer. The lower stack grows
/// from index 0 forward; the upper stack grows from `len-1` backward.
///
/// Used by the index peeler: the lower half is the visit stack
/// (vertices to be processed), the upper half is the peeled-edge
/// record. The two halves cannot overlap, so the total of both lengths
/// is bounded by the buffer size.
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
                if $xor_graph.degree($e[1]) == 2 {
                    $stack.$push($conv($e[1]));
                }
                $xor_graph.remove($e[1], $edge, 1);
                if $xor_graph.degree($e[2]) == 2 {
                    $stack.$push($conv($e[2]));
                }
                $xor_graph.remove($e[2], $edge, 2);
            }
            1 => {
                if $xor_graph.degree($e[0]) == 2 {
                    $stack.$push($conv($e[0]));
                }
                $xor_graph.remove($e[0], $edge, 0);
                if $xor_graph.degree($e[2]) == 2 {
                    $stack.$push($conv($e[2]));
                }
                $xor_graph.remove($e[2], $edge, 2);
            }
            2 => {
                if $xor_graph.degree($e[0]) == 2 {
                    $stack.$push($conv($e[0]));
                }
                $xor_graph.remove($e[0], $edge, 0);
                if $xor_graph.degree($e[1]) == 2 {
                    $stack.$push($conv($e[1]));
                }
                $xor_graph.remove($e[1], $edge, 1);
            }
            // SAFETY: side is always 0, 1, or 2 (encoded as a 2-bit
            // field in the degrees_sides array).
            _ => unsafe { ::std::hint::unreachable_unchecked() },
        }
    };
}

// `remove_edge!` is only used inside this module (by the generic
// peelers below); no need to re-export it.

// ── Generic peelers ─────────────────────────────────────────────────

/// Output of [`peel_by_index`].
///
/// In both variants, the peel stack lives in the upper half of
/// `double_stack`; entries are in **newest-first** order (because
/// [`DoubleStack::push_upper`] writes to decreasing indices).
/// `sides_stack[i]` is the pivot side of the *i*-th push, in
/// **oldest-first** order — to iterate both in reverse peel order,
/// reverse only `sides_stack`. The helper [`iter_reverse_peel`]
/// on each variant performs that reversal for you.
///
/// [`iter_reverse_peel`]: IndexPeelOutput::iter_reverse_peel
pub(crate) enum IndexPeelOutput<I: Copy + Default> {
    /// Every edge was peeled. The caller can drive reverse-peel
    /// assignment directly from `double_stack` / `sides_stack`.
    Complete {
        double_stack: DoubleStack<I>,
        sides_stack: Vec<u8>,
    },
    /// Peeling left a non-empty core. The caller can fall back to
    /// lazy Gaussian elimination on the unpeeled remainder, then
    /// drive the reverse peel from the same stacks.
    Partial {
        double_stack: DoubleStack<I>,
        sides_stack: Vec<u8>,
    },
}

impl<I: Copy + Default> IndexPeelOutput<I> {
    /// Iterates `(edge_id, side)` pairs in **reverse peel order**
    /// (newest peeled first) — the order in which a reverse-peel
    /// assignment loop should consume them. Works for both
    /// [`Complete`](IndexPeelOutput::Complete) and
    /// [`Partial`](IndexPeelOutput::Partial) (the partial variant
    /// also exposes its peeled portion this way).
    pub(crate) fn iter_reverse_peel(&self) -> impl Iterator<Item = (I, u8)> + '_ {
        let (double_stack, sides_stack) = match self {
            IndexPeelOutput::Complete {
                double_stack,
                sides_stack,
            }
            | IndexPeelOutput::Partial {
                double_stack,
                sides_stack,
            } => (double_stack, sides_stack),
        };
        double_stack
            .iter_upper()
            .copied()
            .zip(sides_stack.iter().copied().rev())
    }
}

/// Output of [`peel_by_data_high_mem`].
pub(crate) struct DataHighMemPeelOutput<X: Copy + Default> {
    pub(crate) payloads: FastStack<X>,
    pub(crate) sides: FastStack<u8>,
}

impl<X: Copy + Default> DataHighMemPeelOutput<X> {
    /// Iterates `(payload, side)` pairs in **reverse peel order**
    /// (newest peeled first), the order in which the caller's
    /// reverse-peel assignment loop should consume them.
    ///
    /// Hides the `iter().rev().zip(sides.iter().copied().rev())`
    /// idiom so call sites don't have to remember it.
    pub(crate) fn iter_reverse_peel(&self) -> impl Iterator<Item = (X, u8)> + '_ {
        self.payloads
            .iter()
            .copied()
            .rev()
            .zip(self.sides.iter().copied().rev())
    }
}

/// Output of [`peel_by_data_low_mem`].
///
/// The peel stack stores **pivot vertex indices** (not payloads) in
/// the upper half of `visit_stack`. The caller looks each one up in
/// `xor_graph` at assign time via [`XorGraph::edge_and_side`] — the
/// payload survives `XorGraph::zero` (which only clears the degree).
pub(crate) struct DataLowMemPeelOutput<X: Copy + Default + BitXorAssign> {
    pub(crate) visit_stack: DoubleStack<u32>,
    pub(crate) xor_graph: XorGraph<X>,
}

impl<X: Copy + Default + BitXorAssign> DataLowMemPeelOutput<X> {
    /// Iterates `(payload, side)` pairs in **reverse peel order**
    /// (newest peeled first) by re-querying the [`XorGraph`] for
    /// each pivot vertex stored in the upper half of `visit_stack`.
    pub(crate) fn iter_reverse_peel(&self) -> impl Iterator<Item = (X, u8)> + '_ {
        self.visit_stack.iter_upper().map(move |&v| {
            let (payload, side) = self.xor_graph.edge_and_side(v as usize);
            (payload, side as u8)
        })
    }
}

/// Index-based peeler.
///
/// `gen_edges` is consumed (`FnOnce`) so any data the caller captures
/// — e.g., a borrowed shard reference — can be released as soon as the
/// graph is built. The generic `I` type is the [`XorGraph`] payload
/// type (typically a vertex/edge id like `u32` or `usize`).
///
/// The peeler stores both vertex indices (in the lower half of the
/// double stack) and edge id payloads (in the upper half). Conversion
/// between `usize` vertex indices and `I` is done via the `from_usize`
/// and `to_usize` closures the caller supplies — this keeps the
/// peeler free of `num_primitive` trait bounds and lets the call site
/// use whatever conversion is most natural for its `I` type.
///
/// The peeler does not perform value assignment. On success
/// (`complete = true`) the caller iterates `output.double_stack.iter_upper()`
/// (in newest-first order) zipped with `output.sides_stack.iter().rev()`
/// to drive its own assignment loop. On partial peeling (`complete =
/// false`) the caller can fall back to lazy Gaussian elimination on the
/// unpeeled core (using `output.double_stack.iter_upper()` to identify
/// peeled edges).
///
/// Returns `Err(())` only if the abort flag (`failed`) is set during
/// peeling — used by the parallel solve loop in
/// [`VBuilder::par_solve`](crate::func::VBuilder) to short-circuit
/// concurrent work after one shard has failed.
pub(crate) fn peel_by_index<I, FromUsize, ToUsize, GenEdges, VertsOf>(
    num_vertices: usize,
    num_edges: usize,
    failed: &AtomicBool,
    from_usize: FromUsize,
    to_usize: ToUsize,
    gen_edges: GenEdges,
    verts_of: VertsOf,
) -> Result<IndexPeelOutput<I>, ()>
where
    I: Copy + Default + BitXorAssign,
    FromUsize: Fn(usize) -> I + Copy,
    ToUsize: Fn(I) -> usize + Copy,
    GenEdges: FnOnce(&mut XorGraph<I>),
    VertsOf: Fn(I) -> [usize; 3],
{
    let mut xor_graph = XorGraph::<I>::new(num_vertices);
    gen_edges(&mut xor_graph);

    assert!(!xor_graph.overflow, "XorGraph degree overflow during peeling");

    if failed.load(Ordering::Relaxed) {
        return Err(());
    }

    let mut double_stack = DoubleStack::<I>::new(num_vertices);
    let mut sides_stack: Vec<u8> = Vec::with_capacity(num_edges);

    for (v, deg) in xor_graph.degrees().enumerate() {
        if deg == 1 {
            double_stack.push_lower(from_usize(v));
        }
    }

    while let Some(v) = double_stack.pop_lower() {
        let v: usize = to_usize(v);
        if xor_graph.degree(v) == 0 {
            continue;
        }
        debug_assert!(xor_graph.degree(v) == 1);
        let (edge_payload, side) = xor_graph.edge_and_side(v);
        xor_graph.zero(v);
        double_stack.push_upper(edge_payload);
        sides_stack.push(side as u8);
        let e = verts_of(edge_payload);
        remove_edge!(
            xor_graph,
            e,
            side,
            edge_payload,
            double_stack,
            push_lower,
            from_usize
        );
    }

    if double_stack.upper_len() == num_edges {
        Ok(IndexPeelOutput::Complete {
            double_stack,
            sides_stack,
        })
    } else {
        Ok(IndexPeelOutput::Partial {
            double_stack,
            sides_stack,
        })
    }
}

/// Data-based peeler with a [`FastStack`] of payloads + sides
/// (high memory, fast assign).
///
/// `gen_edges` should fill the [`XorGraph`] with `X` payloads that
/// carry enough information for the caller to perform assignment
/// without consulting any external data structure. As `gen_edges` is
/// `FnOnce`, the caller can move the original data into the closure
/// and let it drop at closure end — the peeler then proceeds with the
/// `XorGraph` alone.
///
/// On peeling failure (some edges remain unpeeled), returns `None`.
/// There is no LGE fallback for this strategy: the original data is
/// gone, so the residual system can't be reconstructed. The caller
/// should treat `None` as an `UnsolvableShard` and retry with a new
/// seed.
pub(crate) fn peel_by_data_high_mem<X, GenEdges, VertsOf>(
    num_vertices: usize,
    num_edges: usize,
    failed: &AtomicBool,
    gen_edges: GenEdges,
    verts_of: VertsOf,
) -> Result<Option<DataHighMemPeelOutput<X>>, ()>
where
    X: Copy + Default + BitXorAssign,
    GenEdges: FnOnce(&mut XorGraph<X>),
    VertsOf: Fn(X) -> [usize; 3],
{
    let mut xor_graph = XorGraph::<X>::new(num_vertices);
    gen_edges(&mut xor_graph);

    assert!(!xor_graph.overflow, "XorGraph degree overflow during peeling");

    if failed.load(Ordering::Relaxed) {
        return Err(());
    }

    let mut payloads = FastStack::<X>::new(num_edges);
    let mut sides = FastStack::<u8>::new(num_edges);
    // Experimentally this stack never grows beyond a little more than
    // num_vertices / 4.
    let mut visit_stack = Vec::<u32>::with_capacity(num_vertices / 3);

    for (v, deg) in xor_graph.degrees().enumerate() {
        if deg == 1 {
            visit_stack.push(v as u32);
        }
    }

    while let Some(v) = visit_stack.pop() {
        let v: usize = v as usize;
        if xor_graph.degree(v) == 0 {
            continue;
        }
        let (payload, side) = xor_graph.edge_and_side(v);
        xor_graph.zero(v);
        payloads.push(payload);
        sides.push(side as u8);
        let e = verts_of(payload);
        remove_edge!(xor_graph, e, side, payload, visit_stack, push, |v| v as u32);
    }

    if payloads.len() != num_edges {
        return Ok(None);
    }

    Ok(Some(DataHighMemPeelOutput { payloads, sides }))
}

/// Data-based peeler with a [`DoubleStack`] of pivot vertices and
/// XorGraph re-lookup at assign time (low memory, slightly slower).
///
/// Same `gen_edges`/`verts_of` contract as [`peel_by_data_high_mem`],
/// but instead of pushing the `X` payload onto a `FastStack`, the
/// peeler pushes the *pivot vertex index* (as a `u32`) onto the upper
/// half of a [`DoubleStack`]. At assign time the caller calls
/// [`XorGraph::edge_and_side`] on each pivot vertex to recover the
/// payload (which survived `zero` because `zero` only clears the
/// degree, not the edge slot).
///
/// The `xor_graph` is moved into the output so the caller can perform
/// these lookups during assignment.
///
/// **Note:** the visit stack stores vertex indices as `u32`, so this
/// peeler can address at most 2³² − 1 vertices per shard. All current
/// [`ShardEdge`] implementations sharded with `Vertex = u32` are
/// already capped at that limit; the unsharded `Vertex = usize`
/// implementations should pick a different peeler if they ever need
/// more than 4 G vertices in one shard.
///
/// On peeling failure, returns `None` (no LGE fallback — see
/// [`peel_by_data_high_mem`] for the rationale).
pub(crate) fn peel_by_data_low_mem<X, GenEdges, VertsOf>(
    num_vertices: usize,
    num_edges: usize,
    failed: &AtomicBool,
    gen_edges: GenEdges,
    verts_of: VertsOf,
) -> Result<Option<DataLowMemPeelOutput<X>>, ()>
where
    X: Copy + Default + BitXorAssign,
    GenEdges: FnOnce(&mut XorGraph<X>),
    VertsOf: Fn(X) -> [usize; 3],
{
    assert!(
        num_vertices <= u32::MAX as usize,
        "peel_by_data_low_mem: num_vertices ({}) exceeds 2³² − 1; use peel_by_data_high_mem instead",
        num_vertices
    );

    let mut xor_graph = XorGraph::<X>::new(num_vertices);
    gen_edges(&mut xor_graph);

    assert!(!xor_graph.overflow, "XorGraph degree overflow during peeling");

    if failed.load(Ordering::Relaxed) {
        return Err(());
    }

    let mut visit_stack = DoubleStack::<u32>::new(num_vertices);

    for (v, deg) in xor_graph.degrees().enumerate() {
        if deg == 1 {
            visit_stack.push_lower(v as u32);
        }
    }

    while let Some(v) = visit_stack.pop_lower() {
        let v: usize = v as usize;
        if xor_graph.degree(v) == 0 {
            continue;
        }
        let (payload, side) = xor_graph.edge_and_side(v);
        xor_graph.zero(v);
        visit_stack.push_upper(v as u32);
        let e = verts_of(payload);
        remove_edge!(xor_graph, e, side, payload, visit_stack, push_lower, |v| v as u32);
    }

    if visit_stack.upper_len() != num_edges {
        return Ok(None);
    }

    Ok(Some(DataLowMemPeelOutput {
        visit_stack,
        xor_graph,
    }))
}
