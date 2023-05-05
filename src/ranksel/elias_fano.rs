use anyhow::{Result, bail};
use crate::{traits::*, compact_array::CompactArray, bitmap::BitMap};

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
        unsafe{self.push_unchecked(value);}
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