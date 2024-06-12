use sux::bits::{BitVec, CountBitVec};
use sux::rank_sel::{Rank9, RankSmall};
use sux::rank_sel::{Select9, SimpleSelect};
use sux::traits::{BitLength, NumBits, Select, SelectHinted, SelectUnchecked};

use super::Build;

macro_rules! impl_simple {
    ($name:ident, $subinv: literal) => {
        pub struct $name<B> {
            inner: SimpleSelect<B>,
        }

        impl Build<BitVec> for $name<CountBitVec<BitVec>> {
            fn new(bits: BitVec) -> Self {
                let bits: CountBitVec = bits.into();
                Self {
                    inner: SimpleSelect::new(bits, $subinv),
                }
            }
        }
        impl<B: BitLength + SelectHinted + AsRef<[usize]>> BitLength for $name<B> {
            fn len(&self) -> usize {
                self.inner.len()
            }
        }
        impl NumBits for $name<CountBitVec> {
            fn num_ones(&self) -> usize {
                self.inner.num_ones()
            }
        }
        impl SelectUnchecked for $name<CountBitVec> {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                self.inner.select_unchecked(rank)
            }
        }
        impl Select for $name<CountBitVec> {
            fn select(&self, rank: usize) -> Option<usize> {
                self.inner.select(rank)
            }
        }
    };
}

impl_simple!(SimpleSelect0, 0);
impl_simple!(SimpleSelect1, 1);
impl_simple!(SimpleSelect2, 2);
impl_simple!(SimpleSelect3, 3);

impl Build<BitVec> for Select9 {
    fn new(bits: BitVec) -> Self {
        Select9::new(Rank9::new(bits))
    }
}

impl Build<BitVec> for Rank9 {
    fn new(bits: BitVec) -> Self {
        Rank9::new(bits)
    }
}

impl Build<BitVec> for RankSmall<2, 9> {
    fn new(bits: BitVec) -> Self {
        RankSmall::<2, 9>::new(bits)
    }
}

impl Build<BitVec> for RankSmall<1, 9> {
    fn new(bits: BitVec) -> Self {
        RankSmall::<1, 9>::new(bits)
    }
}

impl Build<BitVec> for RankSmall<1, 10> {
    fn new(bits: BitVec) -> Self {
        RankSmall::<1, 10>::new(bits)
    }
}

impl Build<BitVec> for RankSmall<1, 11> {
    fn new(bits: BitVec) -> Self {
        RankSmall::<1, 11>::new(bits)
    }
}

impl Build<BitVec> for RankSmall<3, 13> {
    fn new(bits: BitVec) -> Self {
        RankSmall::<3, 13>::new(bits)
    }
}
