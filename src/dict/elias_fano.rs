/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::{
    bits::bit_vec::BitVec, bits::bit_vec::CountBitVec, bits::compact_array::CompactArray, traits::*,
};
use anyhow::{bail, Result};
use core::sync::atomic::{AtomicUsize, Ordering};
use epserde::*;

/// The default combination of parameters for Elias-Fano which is returned
/// by the builders
pub type DefaultEliasFano = EliasFano<CountBitVec<Vec<usize>>, CompactArray<Vec<usize>>>;

/// A sequential builder for elias-fano
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
    pub fn new(u: usize, n: usize) -> Self {
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

    pub fn mem_upperbound(u: usize, n: usize) -> usize {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as usize)
    }

    pub fn push(&mut self, value: usize) -> Result<()> {
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
    pub unsafe fn push_unchecked(&mut self, value: usize) {
        let low = value & ((1 << self.l) - 1);
        // TODO
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
            high_bits: self.high_bits.with_count(self.n as _),
        }
    }
}

/// A concurrent builder for elias-fano
pub struct EliasFanoAtomicBuilder {
    u: usize,
    n: usize,
    l: usize,
    low_bits: CompactArray<Vec<AtomicUsize>>,
    high_bits: BitVec<Vec<AtomicUsize>>,
}

impl EliasFanoAtomicBuilder {
    pub fn new(u: usize, n: usize) -> Self {
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

    pub fn mem_upperbound(u: u64, n: u64) -> u64 {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as u64)
    }

    /// Concurrently set values
    ///
    /// # Safety
    /// The values and indices have to be right and the values should be monotone
    pub unsafe fn set(&self, index: usize, value: usize, order: Ordering) {
        let low = value & ((1 << self.l) - 1);
        // TODO
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
    /// upperbound of the values
    u: usize,
    /// number of values
    n: usize,
    /// the size of the lower bits
    l: usize,
    /// A structure that stores the `l` lowest bits of the values
    low_bits: L,
    /// The bitmap containing the gaps between high bits as unary codes
    high_bits: H,
}

impl<H, L> EliasFano<H, L> {
    pub fn monad<F, H2, L2>(self, func: F) -> EliasFano<H2, L2>
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
    /// TODO: this function is never used
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

impl<H, L> BitLength for EliasFano<H, L> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.u
    }
}

impl<H, L> BitCount for EliasFano<H, L> {
    #[inline(always)]
    fn count(&self) -> usize {
        self.n
    }
}

impl<H: Select, L: VSlice> Select for EliasFano<H, L> {
    #[inline]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        self.get_unchecked(rank)
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

impl<H: Select, L: VSlice> IndexedDict for EliasFano<H, L> {
    type Value = usize;
    #[inline]
    fn len(&self) -> usize {
        self.count()
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        let high_bits = self.high_bits.select_unchecked(index) - index;
        let low_bits = self.low_bits.get_unchecked(index);
        (high_bits << self.l) | low_bits
    }
}
