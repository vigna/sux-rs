/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Implementation of the Elias--Fano representation of monotone sequences.

There are two ways to build an [`EliasFano`] structure: using
an [`EliasFanoBuilder`] or an [`EliasFanoAtomicBuilder`].

The main trait implemented by [`EliasFano`] is [`IndexedDict`], which
makes it possible to access its values with [`IndexedDict::get`].

 */
use crate::prelude::*;
use anyhow::{bail, Result};
use core::sync::atomic::{AtomicUsize, Ordering};
use epserde::*;

/// The default combination of parameters return by the builders
pub type DefaultEliasFano = EliasFano<CountBitVec, CompactArray>;

/// A sequential builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoBuilder::push`] to add new values.
pub struct EliasFanoBuilder {
    u: usize,
    n: usize,
    l: usize,
    low_bits: CompactArray<Vec<usize>>,
    high_bits: BitVec<Vec<usize>>,
    last_value: usize,
    count: usize,
}

impl EliasFanoBuilder {
    /// Create a builder for an [`EliasFano`] containing
    /// `n` numbers smaller than `u`.
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
            low_bits: CompactArray::new(l, n),
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
        if value >= self.u {
            bail!("Value too large: {} >= {}", value, self.u);
        }
        if value < self.last_value {
            bail!("The values given to elias-fano are not monotone");
        }
        unsafe {
            self.push_unchecked(value);
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Values passed to this function must be smaller than `u` and must be monotone.
    /// Moreover, the function should not be called more than `n` times.
    pub unsafe fn push_unchecked(&mut self, value: usize) {
        let low = value & ((1 << self.l) - 1);
        self.low_bits.set(self.count, low);

        let high = (value >> self.l) + self.count;
        self.high_bits.set(high, true);

        self.count += 1;
        self.last_value = value;
    }

    pub fn build(self) -> DefaultEliasFano {
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
/// After creating an instance, you can use [`EliasFanoAtomicBuilder::set`]
/// to set the values concurrently. However, this operation is inherently
/// unsafe as no check is performed on the provided data (e.g., duplicate
/// indices and lack of monotonicity are not detected).
pub struct EliasFanoAtomicBuilder {
    u: usize,
    n: usize,
    l: usize,
    low_bits: CompactArray<Vec<AtomicUsize>>,
    high_bits: BitVec<Vec<AtomicUsize>>,
}

impl EliasFanoAtomicBuilder {
    /// Create a builder for an [`EliasFano`] containing
    /// `n` numbers smaller than `u`.
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
            low_bits: CompactArray::new_atomic(l, n),
            high_bits: BitVec::new_atomic(n + (u >> l) + 1),
        }
    }

    /// Concurrently set values.
    ///
    /// # Safety
    /// - All indices must be distinct.
    /// - All values must be smaller than `u`.
    /// - All indices must be smaller than `n`.
    /// - You must call this function exactly `n` times.
    pub unsafe fn set(&self, index: usize, value: usize, order: Ordering) {
        let low = value & ((1 << self.l) - 1);
        // Note that the concurrency guarantees of CompactArray
        // are sufficient for us.
        self.low_bits.set_unchecked(index, low, order);

        let high = (value >> self.l) + index;
        self.high_bits.set(high, true, order);
    }

    pub fn build(self) -> DefaultEliasFano {
        let bit_vec: BitVec<Vec<usize>> = self.high_bits.into();
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits.into(),
            high_bits: bit_vec.with_count(self.n),
        }
    }
}

#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EliasFano<H, L> {
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

/**
Implementation of the Elias--Fano representation of monotone sequences.

There are two ways to build an [`EliasFano`] structure: using
an [`EliasFanoBuilder`] or an [`EliasFanoAtomicBuilder`].

Once the structure has been built, it is possible to enrich it with
indices that will make operations faster. This is done by calling
[ConvertTo::convert_to] towards the desired type. For example,
```rust
use sux::prelude::*;
let mut efb = EliasFanoBuilder::new(2, 5);
efb.push(0);
efb.push(1);
let ef = efb.build();
// Add an index on the ones (accelerates get operations).
let efo: EliasFano<QuantumIndex<CountBitVec>, CompactArray> =
    ef.convert_to().unwrap();
// Add also an index on the zeros  (accelerates precedessor and successor).
let efoz: EliasFano<QuantumZeroIndex<QuantumIndex<CountBitVec>>, CompactArray> =
    efo.convert_to().unwrap();
```

The main trait implemented is [`IndexedDict`], which
makes it possible to access values with [`IndexedDict::get`].
 */
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

impl<H: Select + AsRef<[usize]>, L: VSlice> IndexedDict for EliasFano<H, L> {
    type Value = usize;

    type Iterator<'a> = EliasFanoIterator<'a, H, L>
    where
        Self: 'a;

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

    #[inline(always)]
    fn iter(&self) -> Self::Iterator<'_> {
        EliasFanoIterator::new(self)
    }

    #[inline(always)]
    fn iter_from(&self, start_index: usize) -> Self::Iterator<'_> {
        EliasFanoIterator::new_from(self, start_index)
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

pub struct EliasFanoIterator<'a, H: Select + AsRef<[usize]>, L: VSlice> {
    ef: &'a EliasFano<H, L>,
    /// the index of the next value it will be returned when `next` is called
    index: usize,
    /// Index of the word loaded in the `word` field
    word_idx: usize,
    //// Current word we use to compute the next high bit by finding the lowest bit set
    /// This is an usize because BitVec is implemented only for Vec<usize> and &[usize]
    window: usize,
}

impl<'a, H: Select + AsRef<[usize]>, L: VSlice> EliasFanoIterator<'a, H, L> {
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
        }
    }

    pub fn new_from(ef: &'a EliasFano<H, L>, start_index: usize) -> Self {
        if start_index > ef.len() {
            panic!("Index out of bounds: {} > {}", start_index, ef.len());
        }
        let bit_pos = unsafe { ef.high_bits.select_unchecked(start_index) };
        let word_idx = bit_pos / (core::mem::size_of::<usize>() * 8);
        let bits_to_clean = bit_pos % (core::mem::size_of::<usize>() * 8);

        let window = if ef.high_bits.as_ref().is_empty() {
            0
        } else {
            // get the word from the high bits
            let word = unsafe { *ef.high_bits.as_ref().get_unchecked(word_idx) };
            // clean off the bits that we don't care about
            word & !((1 << bits_to_clean) - 1)
        };

        Self {
            ef,
            index: start_index,
            word_idx,
            window,
        }
    }
}

impl<'a, H: Select + AsRef<[usize]>, L: VSlice> Iterator for EliasFanoIterator<'a, H, L> {
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
        let high_bits = (self.word_idx * core::mem::size_of::<usize>() * 8) + bit_idx - self.index;
        // clear the lowest bit set
        self.window &= self.window - 1;
        // compose the value
        let res = (high_bits << self.ef.l) | unsafe { self.ef.low_bits.get_unchecked(self.index) };
        self.index += 1;
        Some(res)
    }
}

impl<'a, H: Select + AsRef<[usize]>, L: VSlice> ExactSizeIterator for EliasFanoIterator<'a, H, L> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.ef.len() - self.index
    }
}
