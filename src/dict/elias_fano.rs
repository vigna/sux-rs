/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Implementation of an [`IndexedDict`] using the Elias–Fano representation of monotone sequences.

There are two ways to build a base [`EliasFano`] structure: using
an [`EliasFanoBuilder`] or an [`EliasFanoConcurrentBuilder`].

Once the base structure has been built, it is possible to enrich it with
indices that will make operations faster, using the same mechanism with which
[you can add ranking and selection structures to bit vectors](`crate::rank_sel`),
that is, by calling [`ConvertTo::convert_to`] towards the desired type. For example,
```rust
use sux::prelude::*;
let mut efb = EliasFanoBuilder::new(2, 5);
efb.push(0);
efb.push(2);
let ef = efb.build();
// Add a selection structure for ones (accelerates get operations).
let ef: EliasFano<SelectFixed2> =
    ef.convert_to().unwrap();
// Add also a selection structure for zeros (accelerates precedessor and successor).
let ef: EliasFano<SelectZeroFixed2<SelectFixed2>> =
    ef.convert_to().unwrap();
```

*/

use crate::prelude::*;
use crate::traits::bit_field_slice::*;
use anyhow::{bail, Result};
use core::sync::atomic::Ordering;
use epserde::*;
use mem_dbg::*;

/// A sequential builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoBuilder::push`] to add new values.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct EliasFanoBuilder {
    u: usize,
    n: usize,
    l: usize,
    low_bits: BitFieldVec,
    high_bits: BitVec,
    last_value: usize,
    count: usize,
}

impl EliasFanoBuilder {
    /// Create a builder for an [`EliasFano`] containing
    /// `n` numbers smaller than or equal to `u`.
    pub fn new(n: usize, u: usize) -> Self {
        let l = if u >= n {
            (u as f64 / n as f64).log2().floor() as usize
        } else {
            0
        };

        Self {
            u,
            n,
            l,
            low_bits: BitFieldVec::new(l, n),
            high_bits: BitVec::new(n + (u >> l) + 1),
            last_value: 0,
            count: 0,
        }
    }

    /// Add a new value to the builder.
    ///
    /// # Panic
    /// May panic if the value is smaller than the last provided
    /// value, or if too many values are provided.
    pub fn push(&mut self, value: usize) -> Result<()> {
        if self.count == self.n {
            bail!("Too many values");
        }
        if value > self.u {
            bail!("Value too large: {} > {}", value, self.u);
        }
        if value < self.last_value {
            bail!(
                "The values provided are not monotone: {} < {}",
                value,
                self.last_value
            );
        }
        unsafe {
            self.push_unchecked(value);
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Values passed to this function must be smaller than or equal `u` and must be monotone.
    /// Moreover, the function should not be called more than `n` times.
    pub unsafe fn push_unchecked(&mut self, value: usize) {
        let low = value & ((1 << self.l) - 1);
        self.low_bits.set(self.count, low);

        let high = (value >> self.l) + self.count;
        self.high_bits.set(high, true);

        self.count += 1;
        self.last_value = value;
    }

    pub fn build(self) -> EliasFano {
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits,
            high_bits: self.high_bits.with_count(self.n),
        }
    }
}

/// A parallel builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoConcurrentBuilder::set`]
/// to set the values concurrently. However, this operation is inherently
/// unsafe as no check is performed on the provided data (e.g., duplicate
/// indices and lack of monotonicity are not detected).
#[derive(MemDbg, MemSize)]
pub struct EliasFanoConcurrentBuilder {
    u: usize,
    n: usize,
    l: usize,
    low_bits: AtomicBitFieldVec,
    high_bits: AtomicBitVec,
}

impl EliasFanoConcurrentBuilder {
    /// Create a builder for an [`EliasFano`] containing
    /// `n` numbers smaller than or equal to `u`.
    pub fn new(n: usize, u: usize) -> Self {
        let l = if u >= n {
            (u as f64 / n as f64).log2().floor() as usize
        } else {
            0
        };

        Self {
            u,
            n,
            l,
            low_bits: AtomicBitFieldVec::new(l, n),
            high_bits: AtomicBitVec::new(n + (u >> l) + 1),
        }
    }

    /// Concurrently set values.
    ///
    /// # Safety
    /// - All indices must be distinct.
    /// - All values must be smaller than or equal to `u`.
    /// - All indices must be smaller than `n`.
    /// - You must call this function exactly `n` times.
    pub unsafe fn set(&self, index: usize, value: usize, order: Ordering) {
        let low = value & ((1 << self.l) - 1);
        // Note that the concurrency guarantees of BitFieldVec
        // are sufficient for us.
        self.low_bits.set_atomic_unchecked(index, low, order);

        let high = (value >> self.l) + index;
        self.high_bits.set(high, true, order);
    }

    pub fn build(self) -> EliasFano {
        let bit_vec: BitVec = self.high_bits.into();
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits.into(),
            high_bits: bit_vec.with_count(self.n),
        }
    }
}

#[derive(Epserde, Debug, Clone, Hash, MemDbg, MemSize)]
pub struct EliasFano<H = CountBitVec, L = BitFieldVec> {
    /// An upper bound to the values.
    u: usize,
    /// The number of values.
    n: usize,
    /// The number of lower bits.
    l: usize,
    /// The lower-bits array.
    low_bits: L,
    /// the higher-bits array.
    high_bits: H,
}

impl<H, L> EliasFano<H, L> {
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimate the size of an instance.
    pub fn estimate_size(u: usize, n: usize) -> usize {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as usize)
    }

    pub fn transform<F, H2, L2>(self, func: F) -> EliasFano<H2, L2>
    where
        F: Fn(H, L) -> (H2, L2),
    {
        let (high_bits, low_bits) = func(self.high_bits, self.low_bits);
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits,
            high_bits,
        }
    }
}

impl<H, L> EliasFano<H, L> {
    /// # Safety
    /// No check is performed.
    #[inline(always)]
    pub unsafe fn from_raw_parts(u: usize, n: usize, l: usize, low_bits: L, high_bits: H) -> Self {
        Self {
            u,
            n,
            l,
            low_bits,
            high_bits,
        }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (usize, usize, usize, L, H) {
        (self.u, self.n, self.l, self.low_bits, self.high_bits)
    }
}

impl<H: AsRef<[usize]> + Select + SelectZero, L: BitFieldSlice<usize>> IndexedDict
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    type Output = usize;
    type Input = usize;

    #[inline]
    fn len(&self) -> usize {
        self.n
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        let high_bits = self.high_bits.select_unchecked(index) - index;
        let low_bits = self.low_bits.get_unchecked(index);
        (high_bits << self.l) | low_bits
    }

    fn contains(&self, value: &Self::Input) -> bool {
        if *value > self.u {
            return false;
        }
        let zeros_to_skip = *value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            self.high_bits.select_zero(zeros_to_skip - 1).unwrap() + 1
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_unchecked_iter_from(rank);
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a successor for sure.

        let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
            & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                if word_idx >= self.high_bits.as_ref().len() {
                    return false;
                }
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            // find the lowest bit set index in the word
            let bit_idx = window.trailing_zeros() as usize;
            // compute the global bit index
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            // compose the value
            let res = (high_bits << self.l) | unsafe { iter.next_unchecked() };

            if res == *value {
                return true;
            }
            if res > *value {
                return false;
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }

    fn index_of(&self, value: &Self::Input) -> Option<usize> {
        if *value > self.u {
            return None;
        }
        let zeros_to_skip = *value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            self.high_bits.select_zero(zeros_to_skip - 1).unwrap() + 1
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_unchecked_iter_from(rank);
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a successor for sure.

        let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
            & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                if word_idx >= self.high_bits.as_ref().len() {
                    return None;
                }
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            // find the lowest bit set index in the word
            let bit_idx = window.trailing_zeros() as usize;
            // compute the global bit index
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            // compose the value
            let res = (high_bits << self.l) | unsafe { iter.next_unchecked() };
            if res == *value {
                return Some(rank);
            }
            if res > *value {
                return None;
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }
}

impl<'a, H: AsRef<[usize]> + Select, L: BitFieldSlice<usize>> IntoIterator for &'a EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;
    type IntoIter = EliasFanoIterator<'a, H, L>;
    #[inline(always)]

    fn into_iter(self) -> Self::IntoIter {
        EliasFanoIterator::new(self)
    }
}

impl<H: AsRef<[usize]> + Select, L: BitFieldSlice<usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    pub fn into_iter_from(&self, from: usize) -> EliasFanoIterator<'_, H, L> {
        EliasFanoIterator::new_from(self, from)
    }
}

impl<H1, L1, H2, L2> ConvertTo<EliasFano<H1, L1>> for EliasFano<H2, L2>
where
    H2: ConvertTo<H1>,
    L2: ConvertTo<L1>,
{
    #[inline(always)]
    fn convert_to(self) -> Result<EliasFano<H1, L1>> {
        Ok(EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits.convert_to()?,
            high_bits: self.high_bits.convert_to()?,
        })
    }
}

/// An iterator streaming over the Elias--Fano representation.
#[derive(MemDbg, MemSize)]
pub struct EliasFanoIterator<'a, H: AsRef<[usize]>, L: BitFieldSlice<usize>>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    ef: &'a EliasFano<H, L>,
    /// The index of the next value it will be returned when `next` is called.
    index: usize,
    /// Index of the word loaded in the `word` field.
    word_idx: usize,
    /// Current window on the high bits.
    /// This is an usize because BitVec is implemented only for `Vec<usize>` and `&[usize]`.
    window: usize,
    low_bits: <&'a L as IntoUncheckedIterator>::IntoUncheckedIter,
}

impl<'a, H: Select + AsRef<[usize]>, L: BitFieldSlice<usize>> EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    pub fn new(ef: &'a EliasFano<H, L>) -> Self {
        let word = if ef.high_bits.as_ref().is_empty() {
            0
        } else {
            unsafe { *ef.high_bits.as_ref().get_unchecked(0) }
        };
        Self {
            ef,
            index: 0,
            word_idx: 0,
            window: word,
            low_bits: ef.low_bits.into_unchecked_iter(),
        }
    }

    pub fn new_from(ef: &'a EliasFano<H, L>, start_index: usize) -> Self {
        if start_index > ef.len() {
            panic!("Index out of bounds: {} > {}", start_index, ef.len());
        }
        let bit_pos = unsafe { ef.high_bits.select_unchecked(start_index) };
        let word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        let window = if ef.high_bits.as_ref().is_empty() {
            0
        } else {
            // get the word from the high bits
            let word = unsafe { *ef.high_bits.as_ref().get_unchecked(word_idx) };
            // clean off the bits that we don't care about
            word & (usize::MAX << bits_to_clean)
        };

        Self {
            ef,
            index: start_index,
            word_idx,
            window,
            low_bits: ef.low_bits.into_unchecked_iter_from(start_index),
        }
    }
}

impl<'a, H: AsRef<[usize]>, L: BitFieldSlice<usize>> Iterator for EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.ef.len() {
            return None;
        }
        // find the next word with zeros
        while self.window == 0 {
            self.word_idx += 1;
            debug_assert!(self.word_idx < self.ef.high_bits.as_ref().len());
            self.window = unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = self.window.trailing_zeros() as usize;
        // compute the global bit index
        let high_bits = (self.word_idx * usize::BITS as usize) + bit_idx - self.index;
        // clear the lowest bit set
        self.window &= self.window - 1;
        // compose the value
        let res = (high_bits << self.ef.l) | unsafe { self.low_bits.next_unchecked() };
        self.index += 1;
        Some(res)
    }
}

impl<'a, H: AsRef<[usize]>, L: BitFieldSlice<usize>> ExactSizeIterator
    for EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.ef.len() - self.index
    }
}

#[allow(clippy::collapsible_else_if)]
impl<H: SelectZero + Select + AsRef<[usize]>, L: BitFieldSlice<usize>> Succ for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: &Self::Input,
    ) -> (usize, Self::Output) {
        if STRICT {
            debug_assert!(*value < self.get(self.len() - 1));
        } else {
            debug_assert!(*value <= self.get(self.len() - 1));
        }
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            self.high_bits.select_zero(zeros_to_skip - 1).unwrap() + 1
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_unchecked_iter_from(rank);
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a successor for sure.

        let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
            & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                debug_assert!(word_idx < self.high_bits.as_ref().len());
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            // find the lowest bit set index in the word
            let bit_idx = window.trailing_zeros() as usize;
            // compute the global bit index
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            // compose the value
            let res = (high_bits << self.l) | unsafe { iter.next_unchecked() };

            if STRICT {
                if res > *value {
                    return (rank, res);
                }
            } else {
                if res >= *value {
                    return (rank, res);
                }
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }
}

#[allow(clippy::collapsible_else_if)]
impl<H: SelectZero + Select + AsRef<[usize]>, L: BitFieldSlice<usize>> Pred for EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: &Self::Input,
    ) -> (usize, Self::Output) {
        if STRICT {
            debug_assert!(*value > self.get(0));
        } else {
            debug_assert!(*value >= self.get(0));
        }

        let zeros_to_skip = value >> self.l;
        let mut bit_pos = self.high_bits.select_zero(zeros_to_skip).unwrap() - 1;

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_rev_unchecked_iter_from(rank + 1);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a predecessor for sure.

        loop {
            let lower_bits = unsafe { iter.next_unchecked() };
            let mut word_idx = bit_pos / (usize::BITS as usize);
            let bit_idx = bit_pos % (usize::BITS as usize);
            if self.high_bits.get(word_idx) & (1_usize << bit_idx) == 0 {
                let mut zeros = bit_idx;
                let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
                    & !(usize::MAX << bit_idx);
                while window == 0 {
                    word_idx -= 1;
                    window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
                    zeros += usize::BITS as usize;
                }
                return (
                    rank,
                    ((usize::BITS as usize) - 1 + bit_pos
                        - zeros
                        - window.leading_zeros() as usize
                        - rank)
                        << self.l
                        | lower_bits,
                );
            }

            if STRICT {
                if lower_bits < value & ((1 << self.l) - 1) {
                    return (rank, (bit_pos - rank) << self.l | lower_bits);
                }
            } else {
                if lower_bits <= value & ((1 << self.l) - 1) {
                    return (rank, (bit_pos - rank) << self.l | lower_bits);
                }
            }

            bit_pos -= 1;
            rank -= 1;
        }
    }
}
