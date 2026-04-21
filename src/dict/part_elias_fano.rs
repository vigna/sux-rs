/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::bits::BitVec;
use crate::dict::elias_fano::{EfSeqDict, EliasFanoBuilder};
use crate::rank_sel::{RankSmall, SelectSmall};
use crate::traits::indexed_dict::Types;
use crate::traits::iter::{BidiIterator, SwappedIter};
use crate::traits::{
    BitVecOpsMut, IndexedSeq, Pred, PredUnchecked, RankUnchecked, SelectUnchecked, Succ,
    SuccUnchecked,
};
use mem_dbg::{MemDbg, MemSize};
use std::borrow::Borrow;

type DenseBV = SelectSmall<1, 11, RankSmall<64, 1, 11, BitVec<Box<[usize]>>>>;

#[allow(dead_code)]
#[derive(MemSize, MemDbg)]
enum Chunk {
    EliasFano {
        ef: EfSeqDict<usize>,
        len: usize,
    },
    Dense {
        bv: DenseBV,
        len: usize,
        universe: usize,
    },
}

impl Chunk {
    fn len(&self) -> usize {
        match self {
            Chunk::EliasFano { len, .. } => *len,
            Chunk::Dense { len, .. } => *len,
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> usize {
        match self {
            Chunk::EliasFano { ef, .. } => unsafe { ef.get_unchecked(index) },
            Chunk::Dense { bv, .. } => unsafe { bv.select_unchecked(index) },
        }
    }

    unsafe fn succ_unchecked(&self, value: usize) -> (usize, usize) {
        match self {
            Chunk::EliasFano { ef, .. } => unsafe { ef.succ_unchecked::<false>(value) },
            Chunk::Dense { bv, .. } => {
                let rank = unsafe { bv.rank_unchecked(value) };
                let pos = unsafe { bv.select_unchecked(rank) };
                if pos >= value {
                    (rank, pos)
                } else {
                    (rank + 1, unsafe { bv.select_unchecked(rank + 1) })
                }
            }
        }
    }

    unsafe fn succ_strict_unchecked(&self, value: usize) -> (usize, usize) {
        match self {
            Chunk::EliasFano { ef, .. } => unsafe { ef.succ_unchecked::<true>(value) },
            Chunk::Dense { bv, universe, .. } => {
                let rank = if value >= *universe {
                    (unsafe { bv.rank_unchecked(*universe) }) + 1
                } else {
                    unsafe { bv.rank_unchecked(value + 1) }
                };
                (rank, unsafe { bv.select_unchecked(rank) })
            }
        }
    }

    unsafe fn pred_unchecked(&self, value: usize) -> (usize, usize) {
        match self {
            Chunk::EliasFano { ef, .. } => unsafe { ef.pred_unchecked::<false>(value) },
            Chunk::Dense { bv, len, universe } => {
                let rank = if value >= *universe {
                    *len
                } else {
                    unsafe { bv.rank_unchecked(value + 1) }
                };
                (rank - 1, unsafe { bv.select_unchecked(rank - 1) })
            }
        }
    }

    unsafe fn pred_strict_unchecked(&self, value: usize) -> (usize, usize) {
        match self {
            Chunk::EliasFano { ef, .. } => unsafe { ef.pred_unchecked::<true>(value) },
            Chunk::Dense { bv, .. } => {
                let rank = unsafe { bv.rank_unchecked(value) };
                (rank - 1, unsafe { bv.select_unchecked(rank - 1) })
            }
        }
    }
}

#[allow(dead_code)]
#[derive(MemSize, MemDbg)]
pub struct PartEliasFano {
    n: usize,
    u: usize,
    endpoints: EfSeqDict<usize>,
    upper_bounds: EfSeqDict<usize>,
    bases: Vec<usize>,
    chunks: Vec<Chunk>,
}

// -- Iterator types --

pub struct PartEliasFanoIter<'a> {
    pef: &'a PartEliasFano,
    pos: usize,
}

impl Iterator for PartEliasFanoIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.pos >= self.pef.n {
            return None;
        }
        let val = unsafe { self.pef.get_unchecked(self.pos) };
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.pef.n - self.pos;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for PartEliasFanoIter<'_> {}

pub struct PartEliasFanoBidiIter<'a> {
    pef: &'a PartEliasFano,
    pos: usize,
}

impl Iterator for PartEliasFanoBidiIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.pos >= self.pef.n {
            return None;
        }
        let val = unsafe { self.pef.get_unchecked(self.pos) };
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.pef.n - self.pos;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for PartEliasFanoBidiIter<'_> {}

impl BidiIterator for PartEliasFanoBidiIter<'_> {
    type SwappedIter = SwappedIter<Self>;

    #[inline(always)]
    fn swap(self) -> SwappedIter<Self> {
        SwappedIter(self)
    }

    #[inline]
    fn prev(&mut self) -> Option<usize> {
        if self.pos == 0 {
            return None;
        }
        self.pos -= 1;
        Some(unsafe { self.pef.get_unchecked(self.pos) })
    }

    fn prev_size_hint(&self) -> (usize, Option<usize>) {
        (self.pos, Some(self.pos))
    }
}

impl PartEliasFano {
    pub fn num_partitions(&self) -> usize {
        self.chunks.len()
    }
}

// -- Trait implementations --

impl Types for PartEliasFano {
    type Output<'a> = usize;
    type Input = usize;
}

impl IndexedSeq for PartEliasFano {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        let (partition_idx, _) = unsafe { self.endpoints.succ_unchecked::<false>(index + 1) };
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_unchecked(partition_idx - 1) }
        };
        let local_index = index - partition_start;
        (unsafe { self.chunks[partition_idx].get_unchecked(local_index) }) + self.bases[partition_idx]
    }

    fn len(&self) -> usize {
        self.n
    }
}

impl SuccUnchecked for PartEliasFano {
    type Iter<'a> = PartEliasFanoIter<'a>;
    type BidiIter<'a> = PartEliasFanoBidiIter<'a>;

    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, usize) {
        let value = *value.borrow();
        let (partition_idx, _) = unsafe { self.upper_bounds.succ_unchecked::<false>(value) };
        let base = self.bases[partition_idx];
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_unchecked(partition_idx - 1) }
        };

        if value < base {
            let local_val = unsafe { self.chunks[partition_idx].get_unchecked(0) };
            return (partition_start, local_val + base);
        }

        let relative = value - base;
        let (local_idx, local_val) = if STRICT {
            unsafe { self.chunks[partition_idx].succ_strict_unchecked(relative) }
        } else {
            unsafe { self.chunks[partition_idx].succ_unchecked(relative) }
        };
        (partition_start + local_idx, local_val + base)
    }

    unsafe fn iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::Iter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (idx, PartEliasFanoIter { pef: self, pos: idx })
    }

    unsafe fn iter_bidi_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BidiIter<'_>) {
        let (idx, _) = unsafe { self.succ_unchecked::<STRICT>(value) };
        (
            idx,
            PartEliasFanoBidiIter { pef: self, pos: idx },
        )
    }
}

impl Succ for PartEliasFano {
    fn succ(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<false>(value) })
        }
    }

    fn succ_strict(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.succ_unchecked::<true>(value) })
        }
    }

    fn iter_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<false>(value) })
        }
    }

    fn iter_from_succ_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccUnchecked>::Iter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<true>(value) })
        }
    }

    fn iter_bidi_from_succ(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value > unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<false>(value) })
        }
    }

    fn iter_bidi_from_succ_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, <Self as SuccUnchecked>::BidiIter<'_>)> {
        let value = *value.borrow();
        if value >= unsafe { self.get_unchecked(self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_succ_unchecked::<true>(value) })
        }
    }
}

impl PredUnchecked for PartEliasFano {
    type BackIter<'a> = SwappedIter<PartEliasFanoBidiIter<'a>>;
    type BidiIter<'a> = PartEliasFanoBidiIter<'a>;

    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, usize) {
        let value = *value.borrow();
        let (partition_idx, _) = unsafe { self.upper_bounds.succ_unchecked::<false>(value) };
        let base = self.bases[partition_idx];
        let partition_start = if partition_idx == 0 {
            0
        } else {
            unsafe { self.endpoints.get_unchecked(partition_idx - 1) }
        };

        if value < base {
            // pred is in the previous partition (last element)
            let prev_part = partition_idx - 1;
            let prev_end = if prev_part == 0 {
                (unsafe { self.endpoints.get_unchecked(0) }) - 1
            } else {
                (unsafe { self.endpoints.get_unchecked(prev_part) }) - 1
            };
            let prev_base = self.bases[prev_part];
            let local_last = self.chunks[prev_part].len() - 1;
            let local_val = unsafe { self.chunks[prev_part].get_unchecked(local_last) };
            return (prev_end, local_val + prev_base);
        }

        let relative = value - base;
        let (local_idx, local_val) = if STRICT {
            unsafe { self.chunks[partition_idx].pred_strict_unchecked(relative) }
        } else {
            unsafe { self.chunks[partition_idx].pred_unchecked(relative) }
        };
        (partition_start + local_idx, local_val + base)
    }

    unsafe fn iter_back_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BackIter<'_>) {
        let (idx, _) = unsafe { self.pred_unchecked::<STRICT>(value) };
        (
            idx,
            SwappedIter(PartEliasFanoBidiIter {
                pef: self,
                pos: idx + 1,
            }),
        )
    }

    unsafe fn iter_bidi_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<usize>,
    ) -> (usize, Self::BidiIter<'_>) {
        let (idx, _) = unsafe { self.pred_unchecked::<STRICT>(value) };
        (
            idx,
            PartEliasFanoBidiIter {
                pef: self,
                pos: idx + 1,
            },
        )
    }
}

impl Pred for PartEliasFano {
    fn pred(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<false>(value) })
        }
    }

    fn pred_strict(&self, value: impl Borrow<usize>) -> Option<(usize, usize)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.pred_unchecked::<true>(value) })
        }
    }

    fn iter_back_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<false>(value) })
        }
    }

    fn iter_back_from_pred_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, Self::BackIter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_back_from_pred_unchecked::<true>(value) })
        }
    }

    fn iter_bidi_from_pred(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        let value = *value.borrow();
        if self.n == 0 || value < unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<false>(value) })
        }
    }

    fn iter_bidi_from_pred_strict(
        &self,
        value: impl Borrow<usize>,
    ) -> Option<(usize, Self::BidiIter<'_>)> {
        let value = *value.borrow();
        if value <= unsafe { self.get_unchecked(0) } {
            None
        } else {
            Some(unsafe { self.iter_bidi_from_pred_unchecked::<true>(value) })
        }
    }
}

// -- Builder and partitioning --

pub struct PartEliasFanoBuilder {
    n: usize,
    u: usize,
    eps1: f64,
    eps2: f64,
    fix_cost: usize,
    values: Vec<usize>,
}

impl PartEliasFanoBuilder {
    pub fn new(n: usize, u: usize) -> Self {
        Self {
            n,
            u,
            eps1: 0.03,
            eps2: 0.3,
            fix_cost: 448,
            values: Vec::with_capacity(n),
        }
    }

    pub fn eps1(mut self, eps1: f64) -> Self {
        self.eps1 = eps1;
        self
    }

    pub fn eps2(mut self, eps2: f64) -> Self {
        self.eps2 = eps2;
        self
    }

    pub fn fix_cost(mut self, fix_cost: usize) -> Self {
        self.fix_cost = fix_cost;
        self
    }

    pub fn push(&mut self, value: usize) {
        debug_assert!(
            self.values.is_empty() || value >= *self.values.last().unwrap(),
            "values must be monotone non-decreasing"
        );
        self.values.push(value);
    }

    pub fn build(self) -> PartEliasFano {
        assert_eq!(self.values.len(), self.n);
        if self.n == 0 {
            let efb = EliasFanoBuilder::new(0, 0usize);
            let empty_ef = efb.build_with_seq_and_dict();
            return PartEliasFano {
                n: 0,
                u: self.u,
                endpoints: {
                    let efb = EliasFanoBuilder::new(0, 0usize);
                    efb.build_with_seq_and_dict()
                },
                upper_bounds: empty_ef,
                bases: Vec::new(),
                chunks: Vec::new(),
            };
        }

        let partition_points = optimal_partition(
            &self.values,
            self.eps1,
            self.eps2,
            self.fix_cost,
        );

        let num_partitions = partition_points.len() - 1;
        let mut bases = Vec::with_capacity(num_partitions);
        let mut chunks = Vec::with_capacity(num_partitions);
        let mut cumulative_sizes = Vec::with_capacity(num_partitions);
        let mut upper_bound_values = Vec::with_capacity(num_partitions);

        let mut cumulative = 0usize;
        for p in 0..num_partitions {
            let start = partition_points[p];
            let end = partition_points[p + 1];
            let chunk_size = end - start;
            cumulative += chunk_size;
            cumulative_sizes.push(cumulative);

            let base = self.values[start];
            let upper = self.values[end - 1];
            bases.push(base);
            upper_bound_values.push(upper);

            let universe = upper - base;
            let chunk = build_chunk(&self.values[start..end], base, universe, chunk_size);
            chunks.push(chunk);
        }

        let mut ep_builder = EliasFanoBuilder::new(num_partitions, self.n);
        for &c in &cumulative_sizes {
            unsafe { ep_builder.push_unchecked(c) };
        }
        let endpoints = ep_builder.build_with_seq_and_dict();

        let mut ub_builder = EliasFanoBuilder::new(num_partitions, self.u);
        for &ub in &upper_bound_values {
            unsafe { ub_builder.push_unchecked(ub) };
        }
        let upper_bounds = ub_builder.build_with_seq_and_dict();

        PartEliasFano {
            n: self.n,
            u: self.u,
            endpoints,
            upper_bounds,
            bases,
            chunks,
        }
    }
}

/// Cost in bits of an Elias-Fano representation.
/// `universe` is the range size (max_value + 1), `n` is the element count.
fn ef_cost(universe: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if universe <= 1 {
        return n;
    }
    let l = if universe > n {
        (universe / n).ilog2() as usize
    } else {
        0
    };
    let high_bits = n + (universe >> l) + 2;
    let low_bits = n * l;
    high_bits + low_bits
}

/// Cost in bits of a dense bitvector representation.
/// `universe` is the range size (max_value + 1).
fn dense_cost(universe: usize) -> usize {
    universe
}

fn chunk_cost(universe: usize, n: usize) -> usize {
    ef_cost(universe, n).min(dense_cost(universe))
}

fn build_chunk(values: &[usize], base: usize, universe: usize, n: usize) -> Chunk {
    let use_dense = n <= universe + 1 && dense_cost(universe + 1) < ef_cost(universe + 1, n);

    if use_dense {
        let mut bv = BitVec::new(universe + 1);
        for &v in values {
            bv.set(v - base, true);
        }
        let bv: BitVec<Box<[usize]>> = bv.into();
        let rs = crate::rank_small![3; bv];
        let bv: DenseBV = SelectSmall::<1, 11, _>::new(rs);
        Chunk::Dense {
            bv,
            len: n,
            universe,
        }
    } else {
        let mut efb = EliasFanoBuilder::new(n, universe);
        for &v in values {
            unsafe { efb.push_unchecked(v - base) };
        }
        let ef = efb.build_with_seq_and_dict();
        Chunk::EliasFano { ef, len: n }
    }
}

struct CostWindow {
    start: usize,
    end: usize,
    min_p: usize,
    max_p: usize,
    cost_upper_bound: usize,
}

pub(super) fn optimal_partition(
    values: &[usize],
    eps1: f64,
    eps2: f64,
    fix_cost: usize,
) -> Vec<usize> {
    optimal_partition_with(values, eps1, eps2, |universe, n| {
        chunk_cost(universe, n) + fix_cost
    })
}

pub(super) fn optimal_partition_with(
    values: &[usize],
    eps1: f64,
    eps2: f64,
    cost_fun: impl Fn(usize, usize) -> usize,
) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return vec![0];
    }
    if n == 1 {
        return vec![0, 1];
    }

    let single_block_cost = cost_fun(values[n - 1] - values[0] + 1, n);

    let mut min_cost = vec![single_block_cost; n + 1];
    min_cost[0] = 0;
    let mut path = vec![0usize; n + 1];

    let cost_lb = cost_fun(1, 1);
    let mut windows: Vec<CostWindow> = Vec::new();
    let mut cost_bound = cost_lb;
    loop {
        windows.push(CostWindow {
            start: 0,
            end: 0,
            min_p: values[0],
            max_p: 0,
            cost_upper_bound: cost_bound,
        });
        if cost_bound >= single_block_cost {
            break;
        }
        if eps1 == 0.0 {
            break;
        }
        let next = ((cost_bound as f64) * (1.0 + eps2)) as usize;
        cost_bound = next.max(cost_bound + 1);
        if (cost_bound as f64) >= (cost_lb as f64) / eps1 {
            break;
        }
    }

    for i in 0..n {
        let mut last_end = i + 1;
        for w in windows.iter_mut() {
            assert_eq!(w.start, i);

            while w.end < last_end {
                w.max_p = values[w.end];
                w.end += 1;
            }

            loop {
                let universe = w.max_p.saturating_sub(w.min_p) + 1;
                let size = w.end - w.start;
                let window_cost = cost_fun(universe, size);

                if min_cost[i] + window_cost < min_cost[w.end] {
                    min_cost[w.end] = min_cost[i] + window_cost;
                    path[w.end] = i;
                }

                last_end = w.end;
                if w.end == n {
                    break;
                }
                if window_cost >= w.cost_upper_bound {
                    break;
                }
                w.max_p = values[w.end];
                w.end += 1;
            }

            w.min_p = values[w.start] + 1;
            w.start += 1;
        }
    }

    let mut result = Vec::new();
    let mut pos = n;
    while pos > 0 {
        result.push(pos);
        pos = path[pos];
    }
    result.push(0);
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let values: Vec<usize> = vec![0, 3, 7, 15, 20, 30, 50, 100, 200, 500];
        let n = values.len();
        let u = 500;

        let mut builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }
    }

    #[test]
    fn test_succ() {
        let values: Vec<usize> = vec![10, 20, 30, 40, 50, 100, 200, 300];
        let n = values.len();
        let u = 300;

        let mut builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        assert_eq!(pef.succ(0), Some((0, 10)));
        assert_eq!(pef.succ(10), Some((0, 10)));
        assert_eq!(pef.succ(11), Some((1, 20)));
        assert_eq!(pef.succ(25), Some((2, 30)));
        assert_eq!(pef.succ(300), Some((7, 300)));
        assert_eq!(pef.succ(301), None);
    }

    #[test]
    fn test_pred() {
        let values: Vec<usize> = vec![10, 20, 30, 40, 50, 100, 200, 300];
        let n = values.len();
        let u = 300;

        let mut builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        assert_eq!(pef.pred(300), Some((7, 300)));
        assert_eq!(pef.pred(299), Some((6, 200)));
        assert_eq!(pef.pred(50), Some((4, 50)));
        assert_eq!(pef.pred(10), Some((0, 10)));
        assert_eq!(pef.pred(9), None);
    }

    #[test]
    fn test_empty() {
        let builder = PartEliasFanoBuilder::new(0, 0);
        let pef = builder.build();
        assert_eq!(pef.len(), 0);
        assert_eq!(pef.succ(0), None);
        assert_eq!(pef.pred(0), None);
    }

    #[test]
    fn test_single() {
        let mut builder = PartEliasFanoBuilder::new(1, 42);
        builder.push(42);
        let pef = builder.build();
        assert_eq!(pef.get(0), 42);
        assert_eq!(pef.succ(42), Some((0, 42)));
        assert_eq!(pef.succ(43), None);
        assert_eq!(pef.pred(42), Some((0, 42)));
        assert_eq!(pef.pred(41), None);
    }

    #[test]
    fn test_large_sequence() {
        let values: Vec<usize> = (0..10000).map(|i| i * 3).collect();
        let n = values.len();
        let u = values[n - 1];

        let mut builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }

        assert_eq!(pef.succ(0), Some((0, 0)));
        assert_eq!(pef.succ(1), Some((1, 3)));
        assert_eq!(pef.succ(29997), Some((9999, 29997)));
        assert_eq!(pef.succ(29998), None);

        assert_eq!(pef.pred(29997), Some((9999, 29997)));
        assert_eq!(pef.pred(29996), Some((9998, 29994)));
        assert_eq!(pef.pred(0), Some((0, 0)));
    }

    #[test]
    fn test_iterator() {
        let values: Vec<usize> = vec![5, 10, 15, 20, 25, 100, 200, 300];
        let n = values.len();
        let u = 300;

        let mut builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        if let Some((idx, iter)) = pef.iter_from_succ(12) {
            assert_eq!(idx, 2);
            let collected: Vec<usize> = iter.collect();
            assert_eq!(collected, vec![15, 20, 25, 100, 200, 300]);
        } else {
            panic!("iter_from_succ should not be None");
        }
    }

    #[test]
    fn test_dense_chunks() {
        // ~50% density forces the partitioner to pick dense bitvector encoding
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let universe = 2000;
        let n = 1000; // 50% density
        let mut rng = SmallRng::seed_from_u64(42);
        let mut values: Vec<usize> = (0..universe)
            .collect::<Vec<_>>()
            .into_iter()
            .filter(|_| rng.random_bool(0.5))
            .take(n)
            .collect();
        values.sort_unstable();
        values.dedup();

        let n = values.len();
        let u = *values.last().unwrap();

        let mut builder = PartEliasFanoBuilder::new(n, u);
        for &v in &values {
            builder.push(v);
        }
        let pef = builder.build();

        // Verify at least one chunk is dense
        let has_dense = pef.chunks.iter().any(|c| matches!(c, Chunk::Dense { .. }));
        assert!(has_dense, "Expected at least one dense chunk at 50% density");

        // Test get
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i), expected, "get({i})");
        }

        // Test succ
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.succ(v), Some((i, v)), "succ({v})");
        }
        // Values just after each element
        for i in 0..values.len() - 1 {
            if values[i] + 1 < values[i + 1] {
                assert_eq!(
                    pef.succ(values[i] + 1),
                    Some((i + 1, values[i + 1])),
                    "succ({} + 1)",
                    values[i]
                );
            }
        }
        assert_eq!(pef.succ(u + 1), None);

        // Test pred
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.pred(v), Some((i, v)), "pred({v})");
        }
        // Values just before each element
        for i in 1..values.len() {
            if values[i] - 1 > values[i - 1] {
                assert_eq!(
                    pef.pred(values[i] - 1),
                    Some((i - 1, values[i - 1])),
                    "pred({} - 1)",
                    values[i]
                );
            }
        }
        assert_eq!(pef.pred(values[0].wrapping_sub(1)), None);
    }
}
