use epserde::Epserde;
use mem_dbg::{MemDbg, MemSize};
use sux::bits::BitVec;
use sux::rank_sel::{Rank10, Rank11, Rank9, RankSmall, SelectSmall};
use sux::rank_sel::{Select9, SelectAdapt, SimpleSelect, SimpleSelectConst};
use sux::traits::{BitCount, BitLength, Select, SelectHinted};

use super::Build;

macro_rules! impl_simple {
    ($name:ident, $subinv: literal) => {
        #[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
        pub struct $name<B> {
            inner: SimpleSelect<B>,
        }

        impl Build<BitVec> for $name<BitVec> {
            fn new(bits: BitVec) -> Self {
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
        impl<B: BitCount + SelectHinted + AsRef<[usize]>> BitCount for $name<B> {
            fn count_ones(&self) -> usize {
                self.inner.count_ones()
            }
        }
        impl Select for $name<BitVec> {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                self.inner.select_unchecked(rank)
            }

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

macro_rules! impl_adapt {
    ($name:ident, $subinv: literal) => {
        #[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
        pub struct $name<B> {
            inner: SelectAdapt<B>,
        }

        impl Build<BitVec> for $name<BitVec> {
            fn new(bits: BitVec) -> Self {
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
        impl<B: BitCount + SelectHinted + AsRef<[usize]>> BitCount for $name<B> {
            fn count_ones(&self) -> usize {
                self.inner.count_ones()
            }
        }
        impl Select for $name<BitVec> {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                self.inner.select_unchecked(rank)
            }

            fn select(&self, rank: usize) -> Option<usize> {
                self.inner.select(rank)
            }
        }
    };
}

impl_adapt!(SelectAdapt0, 0);
impl_adapt!(SelectAdapt1, 1);
impl_adapt!(SelectAdapt2, 2);
impl_adapt!(SelectAdapt3, 3);

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
impl<const LOG2_LOWER_BLOCK_SIZE: usize> Build<BitVec> for Rank10<LOG2_LOWER_BLOCK_SIZE> {
    fn new(bits: BitVec) -> Self {
        Rank10::new(bits)
    }
}
impl Build<BitVec> for Rank11 {
    fn new(bits: BitVec) -> Self {
        Rank11::new(bits)
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

impl Build<BitVec> for SelectSmall<2, 9> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<2, 9>::new(RankSmall::<2, 9>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<1, 9> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<1, 9>::new(RankSmall::<1, 9>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<1, 10> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<1, 10>::new(RankSmall::<1, 10>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<1, 11> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<1, 11>::new(RankSmall::<1, 11>::new(bits))
    }
}

impl Build<BitVec> for SelectSmall<3, 13> {
    fn new(bits: BitVec) -> Self {
        SelectSmall::<3, 13>::new(RankSmall::<3, 13>::new(bits))
    }
}

macro_rules! impl_simple_const {
    ([$($inv_size:literal),+], $subinv_size:tt) => {
        $(
            impl_simple_const!($inv_size, $subinv_size);
        )+
    };
    ($inv_size:literal, [$($subinv_size:literal),+]) => {
        $(
            impl_simple_const!($inv_size, $subinv_size);
        )+
    };
    ($log_inv_size:literal, $log_subinv_size:literal) => {
        impl Build<BitVec> for SimpleSelectConst<BitVec, Vec<usize>, $log_inv_size, $log_subinv_size> {
            fn new(bits: BitVec) -> Self {
                SimpleSelectConst::<BitVec, Vec<usize>,$log_inv_size, $log_subinv_size>::new(bits)
            }
        }
    };
}

impl_simple_const!([8, 9, 10, 11, 12, 13], [1, 2, 3, 4, 5]);
