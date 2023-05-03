use crate::traits::*;

pub struct EliasFano<H, L> {
    high_bits: H,
    low_bits: L,
    /// upperbound of the values 
    u: u64,
    /// number of values
    n: u64,
    /// the size of the lower bits
    l: u64,
}

impl<H, L> BitLength for EliasFano<H, L> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.u
    }

    #[inline(always)]
    fn count(&self) -> usize {
        self.n
    }
}

impl<H: Select, L: VSlice> Select for EliasFano<H, L> {
    #[inline]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let high_bits = self.high_bits.select_unchecked(rank) - rank;
        let low_bits = self.low_bits.get_unchecked(rank);
        (high_bits << self.l) | low_bits
    }
}
