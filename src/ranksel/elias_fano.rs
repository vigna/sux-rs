use crate::{bitmap::BitMap, compact_array::CompactArray, prelude::CountingBitmap, traits::*};
use anyhow::{bail, Result};
use core::sync::atomic::{AtomicU64, Ordering};
use epserde::*;

/// The default combination of parameters for Elias-Fano which is returned
/// by the builders
pub type DefaultEliasFano = EliasFano<CountingBitmap<Vec<u64>, usize>, CompactArray<Vec<u64>>>;

/// A sequential builder for elias-fano
pub struct EliasFanoBuilder {
    u: u64,
    n: u64,
    l: u64,
    low_bits: CompactArray<Vec<u64>>,
    high_bits: BitMap<Vec<u64>>,
    last_value: u64,
    count: u64,
}

impl EliasFanoBuilder {
    pub fn new(u: u64, n: u64) -> Self {
        let l = if u >= n {
            (u as f64 / n as f64).log2().floor() as u64
        } else {
            0
        };

        Self {
            u,
            n,
            l,
            low_bits: CompactArray::new(l as usize, n as usize),
            high_bits: BitMap::new(n as usize + (u as usize >> l) + 1),
            last_value: 0,
            count: 0,
        }
    }

    pub fn mem_upperbound(u: u64, n: u64) -> u64 {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as u64)
    }

    pub fn push(&mut self, value: u64) -> Result<()> {
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
    pub unsafe fn push_unchecked(&mut self, value: u64) {
        let low = value & ((1 << self.l) - 1);
        self.low_bits.set(self.count as usize, low);

        let high = (value >> self.l) + self.count;
        self.high_bits.set(high as usize, 1);

        self.count += 1;
        self.last_value = value;
    }

    pub fn build(self) -> DefaultEliasFano {
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits,
            high_bits: self.high_bits.with_count(self.n as _).into(),
        }
    }
}

/// A concurrent builder for elias-fano
pub struct EliasFanoAtomicBuilder {
    u: u64,
    n: u64,
    l: u64,
    low_bits: CompactArray<Vec<AtomicU64>>,
    high_bits: BitMap<Vec<AtomicU64>>,
}

impl EliasFanoAtomicBuilder {
    pub fn new(u: u64, n: u64) -> Self {
        let l = if u >= n {
            (u as f64 / n as f64).log2().floor() as u64
        } else {
            0
        };

        Self {
            u,
            n,
            l,
            low_bits: CompactArray::new_atomic(l as usize, n as usize),
            high_bits: BitMap::new_atomic(n as usize + (u as usize >> l) + 1),
        }
    }

    pub fn mem_upperbound(u: u64, n: u64) -> u64 {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as u64)
    }

    /// Concurrently set values
    ///
    /// # Safety
    /// The values and indices have to be right and the values should be monotone
    pub unsafe fn set(&self, index: usize, value: u64, order: Ordering) {
        let low = value & ((1 << self.l) - 1);
        self.low_bits.set_atomic_unchecked(index, low, order);

        let high = (value >> self.l) + index as u64;
        self.high_bits.set_atomic_unchecked(high as usize, 1, order);
    }

    pub fn build(self) -> DefaultEliasFano {
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits.into(),
            high_bits: self.high_bits.with_count(self.n as _).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EliasFano<H, L> {
    /// upperbound of the values
    u: u64,
    /// number of values
    n: u64,
    /// the size of the lower bits
    l: u64,
    /// A structure that stores the `l` lowest bits of the values
    low_bits: L,
    /// The bitmap containing the gaps between high bits as unary codes
    high_bits: H,
}

impl<H, L> EliasFano<H, L> {
    /// # Safety
    /// TODO: this function is never used
    #[inline(always)]
    pub unsafe fn from_raw_parts(u: u64, n: u64, l: u64, low_bits: L, high_bits: H) -> Self {
        Self {
            u,
            n,
            l,
            low_bits,
            high_bits,
        }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (u64, u64, u64, L, H) {
        (self.u, self.n, self.l, self.low_bits, self.high_bits)
    }
}

impl<H, L> BitLength for EliasFano<H, L> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.u as usize
    }
}

impl<H, L> BitCount for EliasFano<H, L> {
    #[inline(always)]
    fn count(&self) -> usize {
        self.n as usize
    }
}

impl<H: Select, L: VSlice> Select for EliasFano<H, L> {
    #[inline]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let high_bits = self.high_bits.select_unchecked(rank) - rank;
        let low_bits = self.low_bits.get_unchecked(rank);
        (high_bits << self.l) | low_bits as usize
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
    type Value = u64;
    #[inline]
    fn len(&self) -> usize {
        self.count()
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        self.select_unchecked(index) as u64
    }
}

impl<H: MemSize, L: MemSize> MemSize for EliasFano<H, L> {
    fn mem_size(&self) -> usize {
        self.u.mem_size()
            + self.n.mem_size()
            + self.l.mem_size()
            + self.high_bits.mem_size()
            + self.low_bits.mem_size()
    }
    fn mem_used(&self) -> usize {
        self.u.mem_used()
            + self.n.mem_used()
            + self.l.mem_used()
            + self.high_bits.mem_used()
            + self.low_bits.mem_used()
    }
}
