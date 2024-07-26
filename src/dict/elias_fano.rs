/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! An implementation of the Elias–Fano representation of monotone sequences.
//!
//! Given a monotone sequence 0 ≤ *x*₀ ≤ *x*₁ ≤ ... ≤ *x*<sub>*n* – 1</sub> ≤
//! *u*, where *u* is a given upper bound, the Elias–Fano representation makes
//! it possible to store the sequence using at most 2 + lg(*u*/*n*) bits per
//! element, which is very close to the information-theoretical lower bound ≈ lg
//! *e* + lg(*u*/*n*) when *n* is much smaller than *u*. A typical example is a
//! list of pointer into records of a large file: instead of using, for each
//! pointer, a number of bit sufficient to express the length of the file, the
//! Elias–Fano representation makes it possible to use, for each pointer, a
//! number of bits roughly equal to the logarithm of the average length of a
//! record.
//!
//! The representation was introduced in Peter Elias in “[Efficient storage and
//! retrieval by content and address of static
//! files](https://dl.acm.org/doi/abs/10.1145/321812.321820)”, *J. Assoc.
//! Comput. Mach.*, 21(2):246–260, 1974, and also independently by Robert Fano
//! in “[On the number of bits required to implement an associative
//! memory](http://csg.csail.mit.edu/pubs/memos/Memo-61/Memo-61.pdf)”,
//! Memorandum 61, Computer Structures Group, Project MAC, MIT, Cambridge,
//! Mass., n.d., 1971.
//!
//! This implementation is based on algorithmic engineering ideas proposed by
//! Sebastiano Vigna in “[Quasi-succinct
//! indices](https://dl.acm.org/doi/10.1145/2433396.2433409)”, *Proceedings of
//! the 6th ACM International Conference on Web Search and Data Mining,
//! WSDM'13*, pages 83–92, ACM, 2013. The name “Elias–Fano” for this
//! representation was used for the first time by  Sebastiano Vigna in
//! “[Broadword Implementation of Rank/Select
//! Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
//! _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
//! 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
//! Springer, 2008.
//!
//! The elements of the sequence are recorded by storing separately the lower
//! *s* = ⌊lg(*u*/*n*)⌋ bits and the remaining upper bits. The lower bits are
//! stored contiguously, whereas the upper bits are stored in an array of *n* +
//! ⌊*u* / 2<sup>*s*</sup>⌋ bits by setting, for each 0 ≤ *i* < *n*, the bit of
//! index ⌊*x*<sub>*i*</sub> / 2<sup>*s*</sup>⌋ + *i*; the value can then be
//! recovered by selecting the *i*-th bit of the resulting bit array and
//! subtracting *i* (note that this will work because the upper bits are
//! nondecreasing).
//!
//!

use crate::prelude::*;
use crate::traits::bit_field_slice::*;
use core::sync::atomic::Ordering;
use epserde::*;
use mem_dbg::*;
use std::borrow::Borrow;

/// An [`IndexedDict`] that stores a monotone sequence of integers using the
/// Elias–Fano representation.
///
/// There are two ways to build a base [`EliasFano`] structure: using an
/// [`EliasFanoBuilder`] or an [`EliasFanoConcurrentBuilder`].
///
/// Once the base structure has been built, one can only iterate over the
/// sequence, but it is possible to enrich the structure with indices that will
/// make more operations available by calling [`EliasFano::map_high_bits`]. To
/// use the structure as an [`IndexedSeq`] you need to add a selection structure
/// for ones, whereas to use it as an [`IndexedDict`] with [`Succ`] and [`Pred`]
/// you need to add a selection structure for zeros. [`SelectAdaptConst`] and
/// [`SelectZeroAdaptConst`] are the structures of choice for this purpose.
///
/// # Examples
///
/// ```rust
/// # use sux::rank_sel::{SelectAdaptConst, SelectZeroAdaptConst};
/// # use sux::dict::{EliasFanoBuilder};
/// # use sux::traits::{Types,IndexedSeq,IndexedDict,Succ};
/// let mut efb = EliasFanoBuilder::new(4, 10);
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build();
/// // Add a selection structure for zeros (implements IndexedSeq)
/// let ef = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _>::new) };
///
/// assert_eq!(ef.get(0), 0);
/// assert_eq!(ef.get(1), 2);
///
/// // Add a selection structure for zeros (implements IndexedDict, Succ, Pred)
/// let ef = unsafe { ef.map_high_bits(SelectZeroAdaptConst::<_, _>::new) };
///
/// assert_eq!(ef.succ(6), Some((2, 8)));
/// assert_eq!(ef.succ(11), None);
/// ```

#[derive(Epserde, Debug, Clone, Hash, MemDbg, MemSize)]
pub struct EliasFano<H = BitVec<Box<[usize]>>, L = BitFieldVec<usize, Box<[usize]>>> {
    /// The number of values.
    n: usize,
    /// An upper bound to the values.
    u: usize,
    /// The number of lower bits.
    l: usize,
    /// The lower-bits array.
    low_bits: L,
    /// the higher-bits array.
    high_bits: H,
}

impl<H, L> EliasFano<H, L> {
    /// Estimate the size of an instance.
    pub fn estimate_size(u: usize, n: usize) -> usize {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as usize)
    }

    /// Returns the number elements in the sequence.
    ///
    /// This method is equivalent to [`IndexedSeq::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Replaces the high bits.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new high bits are identical to the old ones as a bit vector.
    pub unsafe fn map_high_bits<F, H2>(self, func: F) -> EliasFano<H2, L>
    where
        F: FnOnce(H) -> H2,
    {
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: self.low_bits,
            high_bits: func(self.high_bits),
        }
    }

    /// Replaces the low bits.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new low bits are identical to the old ones as vector.
    pub unsafe fn map_low_bits<F, L2>(self, func: F) -> EliasFano<H, L2>
    where
        F: FnOnce(L) -> L2,
    {
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: func(self.low_bits),
            high_bits: self.high_bits,
        }
    }
}

impl<H: AsRef<[usize]>, L: BitFieldSlice<usize>> Types for EliasFano<H, L> {
    type Output = usize;
    type Input = usize;
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: BitFieldSlice<usize>> IndexedSeq for EliasFano<H, L> {
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
}

impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: BitFieldSlice<usize>> IndexedDict
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize> {
        let value = *value.borrow();
        if value > self.u {
            return None;
        }
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip - 1) + 1 }
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
            if res == value {
                return Some(rank);
            }
            if res > value {
                return None;
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }
}

#[allow(clippy::collapsible_else_if)]
impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: BitFieldSlice<usize>> SuccUnchecked
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output) {
        let value = *value.borrow();
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            self.high_bits.select_zero_unchecked(zeros_to_skip - 1) + 1
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
                if res > value {
                    return (rank, res);
                }
            } else {
                if res >= value {
                    return (rank, res);
                }
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: BitFieldSlice<usize>> Succ
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
}

#[allow(clippy::collapsible_else_if)]
impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: BitFieldSlice<usize>> PredUnchecked
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output) {
        let value = *value.borrow();
        let zeros_to_skip = value >> self.l;
        let mut bit_pos = self.high_bits.select_zero_unchecked(zeros_to_skip) - 1;

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

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: BitFieldSlice<usize>> Pred
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
}

impl<'a, H: AsRef<[usize]>, L: BitFieldSlice<usize>> IntoIterator for &'a EliasFano<H, L>
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

impl<H: AsRef<[usize]>, L: BitFieldSlice<usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    pub fn iter(&self) -> EliasFanoIterator<'_, H, L> {
        EliasFanoIterator::new(self)
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: BitFieldSlice<usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    pub fn iter_from(&self, from: usize) -> EliasFanoIterator<'_, H, L> {
        EliasFanoIterator::new_from(self, from)
    }
}

/// An iterator for [`EliasFano`].
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

impl<'a, H: AsRef<[usize]>, L: BitFieldSlice<usize>> EliasFanoIterator<'a, H, L>
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
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: BitFieldSlice<usize>> EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
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

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
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

/// A sequential builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoBuilder::push`] to add
/// new values, and then call [`EliasFanoBuilder::build`] to create the
/// [`EliasFano`] instance.
///
/// # Examples
///
/// ```rust
/// # use sux::dict::EliasFanoBuilder;
/// let mut efb = EliasFanoBuilder::new(4, 10);
///
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build();
/// let mut iter = ef.iter();
/// assert_eq!(iter.next(), Some(0));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(8));
/// assert_eq!(iter.next(), Some(10));
/// assert_eq!(iter.next(), None);
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct EliasFanoBuilder {
    n: usize,
    u: usize,
    l: usize,
    low_bits: BitFieldVec,
    high_bits: BitVec,
    last_value: usize,
    count: usize,
}

impl EliasFanoBuilder {
    /// Creates a builder for an [`EliasFano`] containing
    /// `n` numbers smaller than or equal to `u`.
    pub fn new(n: usize, u: usize) -> Self {
        let l = if u >= n {
            (u as f64 / n as f64).log2().floor() as usize
        } else {
            0
        };

        Self {
            n,
            u,
            l,
            low_bits: BitFieldVec::new(l, n),
            high_bits: BitVec::new(n + (u >> l) + 1),
            last_value: 0,
            count: 0,
        }
    }

    /// Adds a new value to the builder.
    ///
    /// # Panic
    /// May panic if the value is smaller than the last provided
    /// value, or if too many values are provided.
    pub fn push(&mut self, value: usize) {
        if self.count == self.n {
            panic!("Too many values");
        }
        if value > self.u {
            panic!("Value too large: {} > {}", value, self.u);
        }
        if value < self.last_value {
            panic!(
                "The values provided are not monotone: {} < {}",
                value, self.last_value
            );
        }
        unsafe {
            self.push_unchecked(value);
        }
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
        let high_bits: BitVec<Box<[usize]>> = self.high_bits.into();
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: self.low_bits.into(),
            // SAFETY: n is the number of ones in the high_bits.
            high_bits,
        }
    }
}

/// A concurrent builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoConcurrentBuilder::set`]
/// to set the values concurrently. However, this operation is inherently
/// unsafe as no check is performed on the provided data (e.g., duplicate
/// indices and lack of monotonicity are not detected).
///
/// # Examples
///
/// ```rust
/// # use sux::dict::EliasFanoConcurrentBuilder;
/// let mut efcb = EliasFanoConcurrentBuilder::new(4, 10);
/// std::thread::scope(|s| {
///     s.spawn(|| { unsafe { efcb.set(0, 0); } });
///     s.spawn(|| { unsafe { efcb.set(1, 2); } });
///     s.spawn(|| { unsafe { efcb.set(2, 8); } });
///     s.spawn(|| { unsafe { efcb.set(3, 10); } });
/// });
///
/// let ef = efcb.build();
/// let mut iter = ef.iter();
/// assert_eq!(iter.next(), Some(0));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(8));
/// assert_eq!(iter.next(), Some(10));
/// assert_eq!(iter.next(), None);
/// ```

#[derive(MemDbg, MemSize)]
pub struct EliasFanoConcurrentBuilder {
    n: usize,
    u: usize,
    l: usize,
    low_bits: AtomicBitFieldVec,
    high_bits: AtomicBitVec,
}

impl EliasFanoConcurrentBuilder {
    /// Creates a concurrent builder for a sequence containing `n` nonnegative
    /// numbers smaller than or equal to `u`.
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

    /// Sets a value concurrently.
    ///
    /// # Safety
    /// - All indices must be distinct.
    /// - All values must be smaller than or equal to `u`.
    /// - All indices must be smaller than `n`.
    /// - You must call this function exactly `n` times.
    pub unsafe fn set(&self, index: usize, value: usize) {
        let low = value & ((1 << self.l) - 1);
        // Note that the concurrency guarantees of BitFieldVec
        // are sufficient for us.
        self.low_bits
            .set_atomic_unchecked(index, low, Ordering::Relaxed);

        let high = (value >> self.l) + index;
        self.high_bits.set(high, true, Ordering::Relaxed);
    }

    pub fn build(self) -> EliasFano {
        let high_bits: BitVec<Vec<usize>> = self.high_bits.into();
        let high_bits: BitVec<Box<[usize]>> = high_bits.into();
        let low_bits: BitFieldVec<usize, Vec<usize>> = self.low_bits.into();
        let low_bits: BitFieldVec<usize, Box<[usize]>> = low_bits.into();
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits,
            high_bits,
        }
    }
}
