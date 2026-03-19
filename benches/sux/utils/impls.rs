use mem_dbg::{MemDbg, MemSize};
use sux::bits::BitVec;
use sux::rank_sel::{Rank9, RankSmall};
use sux::rank_sel::{Select9, SelectAdapt, SelectAdaptConst, SelectSmall};
use sux::traits::{AddNumBits, BitLength, NumBits, Select, SelectUnchecked};

use super::Build;

const LOG2_ONES_PER_INVENTORY: usize = 12;

/// Forward [`BitLength`], [`NumBits`], [`SelectUnchecked`], and [`Select`]
/// to field `.0` of a newtype wrapper.
macro_rules! forward_select {
    ($name:ident) => {
        impl BitLength for $name {
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        impl NumBits for $name {
            fn num_ones(&self) -> usize {
                self.0.num_ones()
            }
        }
        impl SelectUnchecked for $name {
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                unsafe { self.0.select_unchecked(rank) }
            }
        }
        impl Select for $name {
            fn select(&self, rank: usize) -> Option<usize> {
                self.0.select(rank)
            }
        }
    };
}

macro_rules! select_adapt_wrapper {
    ($name:ident, $subinv:literal) => {
        #[derive(MemDbg, MemSize)]
        pub struct $name(SelectAdapt<AddNumBits<BitVec>>);
        impl Build<BitVec> for $name {
            fn new(bits: BitVec) -> Self {
                Self(SelectAdapt::new(bits.into(), $subinv))
            }
        }
        forward_select!($name);
    };
}

macro_rules! select_adapt_const_wrapper {
    ($name:ident, $subinv:literal) => {
        #[derive(MemDbg, MemSize)]
        pub struct $name(
            SelectAdaptConst<AddNumBits<BitVec>, Box<[usize]>, LOG2_ONES_PER_INVENTORY, $subinv>,
        );
        impl Build<BitVec> for $name {
            fn new(bits: BitVec) -> Self {
                Self(SelectAdaptConst::<_, _, LOG2_ONES_PER_INVENTORY, $subinv>::new(bits.into()))
            }
        }
        forward_select!($name);
    };
}

select_adapt_wrapper!(SelectAdapt0, 0);
select_adapt_wrapper!(SelectAdapt1, 1);
select_adapt_wrapper!(SelectAdapt2, 2);
select_adapt_wrapper!(SelectAdapt3, 3);

select_adapt_const_wrapper!(SelectAdaptConst0, 0);
select_adapt_const_wrapper!(SelectAdaptConst1, 1);
select_adapt_const_wrapper!(SelectAdaptConst2, 2);
select_adapt_const_wrapper!(SelectAdaptConst3, 3);

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

macro_rules! impl_build_rank_small {
    ($($a:literal, $b:literal);+ $(;)?) => {
        $(
            impl Build<BitVec> for RankSmall<$a, $b> {
                fn new(bits: BitVec) -> Self {
                    RankSmall::<$a, $b>::new(bits)
                }
            }
            impl Build<BitVec> for SelectSmall<$a, $b, RankSmall<$a, $b>> {
                fn new(bits: BitVec) -> Self {
                    SelectSmall::<$a, $b, _>::new(RankSmall::<$a, $b>::new(bits))
                }
            }
        )+
    };
}

impl_build_rank_small!(
    2, 9;
    1, 9;
    1, 10;
    1, 11;
    3, 13;
);
