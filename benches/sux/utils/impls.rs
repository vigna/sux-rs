use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};
use sux::bits::BitVec;
use sux::rank_sel::{Rank9, RankSmall};
use sux::rank_sel::{Select9, SelectAdapt, SelectAdaptConst, SelectSmall};
use sux::traits::{AddNumBits, BitLength, NumBits, Select, SelectHinted, SelectUnchecked};

use super::Build;

macro_rules! impl_select_adapt {
    ($name:ident, $subinv: literal) => {
        #[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
        pub struct $name<B> {
            inner: SelectAdapt<B>,
        }

        impl Build<BitVec> for $name<AddNumBits<BitVec>> {
            fn new(bits: BitVec) -> Self {
                let bits: AddNumBits<_> = bits.into();
                Self {
                    inner: SelectAdapt::new(bits, $subinv),
                }
            }
        }
        impl<B: BitLength + SelectHinted + AsRef<[usize]>> BitLength for $name<B> {
            fn len(&self) -> usize {
                self.inner.len()
            }
        }
        impl NumBits for $name<AddNumBits<BitVec>> {
            fn num_ones(&self) -> usize {
                self.inner.num_ones()
            }
        }
        impl SelectUnchecked for $name<AddNumBits<BitVec>> {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                self.inner.select_unchecked(rank)
            }
        }
        impl Select for $name<AddNumBits<BitVec>> {
            fn select(&self, rank: usize) -> Option<usize> {
                self.inner.select(rank)
            }
        }
    };
}

impl_select_adapt!(SelectAdapt0, 0);
impl_select_adapt!(SelectAdapt1, 1);
impl_select_adapt!(SelectAdapt2, 2);
impl_select_adapt!(SelectAdapt3, 3);

// Default value
const LOG2_ONES_PER_INVENTORY: usize = 12;

macro_rules! impl_select_adapt_const {
    ($name:ident, $subinv: literal) => {
        #[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
        pub struct $name<B> {
            inner: SelectAdaptConst<B, Box<[usize]>, LOG2_ONES_PER_INVENTORY, $subinv>,
        }

        impl Build<BitVec> for $name<AddNumBits<BitVec>> {
            fn new(bits: BitVec) -> Self {
                let bits: AddNumBits<_> = bits.into();
                Self {
                    inner: SelectAdaptConst::<_, _, LOG2_ONES_PER_INVENTORY, $subinv>::new(bits),
                }
            }
        }
        impl<B: BitLength + SelectHinted + AsRef<[usize]>> BitLength for $name<B> {
            fn len(&self) -> usize {
                self.inner.len()
            }
        }
        impl NumBits for $name<AddNumBits<BitVec>> {
            fn num_ones(&self) -> usize {
                self.inner.num_ones()
            }
        }
        impl SelectUnchecked for $name<AddNumBits<BitVec>> {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                self.inner.select_unchecked(rank)
            }
        }
        impl Select for $name<AddNumBits<BitVec>> {
            fn select(&self, rank: usize) -> Option<usize> {
                self.inner.select(rank)
            }
        }
    };
}

impl_select_adapt_const!(SelectAdaptConst0, 0);
impl_select_adapt_const!(SelectAdaptConst1, 1);
impl_select_adapt_const!(SelectAdaptConst2, 2);
impl_select_adapt_const!(SelectAdaptConst3, 3);

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

impl Build<BitVec> for SelectSmall<2, 9, RankSmall<2, 9>> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<2, 9, _>::new(RankSmall::<2, 9>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<1, 9, RankSmall<1, 9>> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<1, 9, _>::new(RankSmall::<1, 9>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<1, 10, RankSmall<1, 10>> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<1, 10, _>::new(RankSmall::<1, 10>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<1, 11, RankSmall<1, 11>> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<1, 11, _>::new(RankSmall::<1, 11>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<3, 13, RankSmall<3, 13>> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<3, 13, _>::new(RankSmall::<3, 13>::new(bits))
    }
}
