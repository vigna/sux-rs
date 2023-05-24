use crate::{bitmap::BitMap, compact_array::CompactArray, traits::*};
use anyhow::{bail, Result};
use std::io::{Seek, Write};

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

    pub fn push(&mut self, value: u64) -> Result<()> {
        if value < self.last_value {
            bail!("The values given to elias-fano are not monotone");
        }
        unsafe {
            self.push_unchecked(value);
        }
        Ok(())
    }

    pub unsafe fn push_unchecked(&mut self, value: u64) {
        let low = value & ((1 << self.l) - 1);
        self.low_bits.set(self.count as usize, low).unwrap();

        let high = (value >> self.l) + self.count;
        self.high_bits.set(high as usize, 1).unwrap();

        self.count += 1;
        self.last_value = value;
    }

    pub fn build(self) -> EliasFano<BitMap<Vec<u64>>, CompactArray<Vec<u64>>> {
        EliasFano {
            u: self.u,
            n: self.n,
            l: self.l,
            low_bits: self.low_bits,
            high_bits: self.high_bits,
        }
    }
}

pub struct EliasFano<H, L> {
    /// upperbound of the values
    u: u64,
    /// number of values
    n: u64,
    /// the size of the lower bits
    l: u64,
    /// A structure that stores the `l` lowest bits of the values
    low_bits: L,

    high_bits: H,
}

impl<H, L> BitLength for EliasFano<H, L> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.u as usize
    }

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

impl<H: Select, L: VSlice> VSlice for EliasFano<H, L> {
    #[inline]
    fn bit_width(&self) -> usize {
        self.u.next_power_of_two().ilog2() as _
    }

    #[inline]
    fn len(&self) -> usize {
        self.count()
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        self.select_unchecked(index) as u64
    }
}

impl<H: core::fmt::Debug, L: core::fmt::Debug> core::fmt::Debug for EliasFano<H, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EliasFano")
            .field("n", &self.n)
            .field("u", &self.u)
            .field("l", &self.l)
            .field("low_bits", &self.low_bits)
            .field("high_bits", &self.high_bits)
            .finish()
    }
}

impl<H: Clone, L: Clone> Clone for EliasFano<H, L> {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: self.low_bits.clone(),
            high_bits: self.high_bits.clone(),
        }
    }
}

impl<H: Serialize, L: Serialize> Serialize for EliasFano<H, L> {
    fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
        let mut bytes = 0;
        bytes += self.u.serialize(backend)?;
        bytes += self.n.serialize(backend)?;
        bytes += self.l.serialize(backend)?;
        bytes += self.low_bits.serialize(backend)?;
        bytes += self.high_bits.serialize(backend)?;
        Ok(bytes)
    }
}

impl<'a, H: Deserialize<'a>, L: Deserialize<'a>> Deserialize<'a> for EliasFano<H, L> {
    fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
        let (u, backend) = u64::deserialize(&backend)?;
        let (n, backend) = u64::deserialize(&backend)?;
        let (l, backend) = u64::deserialize(&backend)?;
        let (low_bits, backend) = L::deserialize(&backend)?;
        let (high_bits, backend) = H::deserialize(&backend)?;

        Ok((
            Self {
                u,
                n,
                l,
                high_bits,
                low_bits,
            },
            backend,
        ))
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
