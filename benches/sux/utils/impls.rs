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

impl Build<BitVec<Vec<u64>>> for Select9<Rank9<BitVec<Vec<u64>>>> {
    fn new(bits: BitVec<Vec<u64>>) -> Self {
        Select9::new(Rank9::new(bits))
    }
}

impl Build<BitVec<Vec<u64>>> for Rank9<BitVec<Vec<u64>>> {
    fn new(bits: BitVec<Vec<u64>>) -> Self {
        Rank9::new(bits)
    }
}

// These RankSmall variants require 64-bit words; on 32-bit platforms
// usize is 32 bits and would fail the compile-time word-size assert.
#[cfg(target_pointer_width = "64")]
macro_rules! impl_build_rank_small {
    ($($a:literal, $b:literal);+ $(;)?) => {
        $(
            impl Build<BitVec> for RankSmall<64, $a, $b> {
                fn new(bits: BitVec) -> Self {
                    RankSmall::<64, $a, $b>::new(bits)
                }
            }
            impl Build<BitVec> for SelectSmall<$a, $b, RankSmall<64, $a, $b>> {
                fn new(bits: BitVec) -> Self {
                    SelectSmall::<$a, $b, _>::new(RankSmall::<64, $a, $b>::new(bits))
                }
            }
        )+
    };
}

#[cfg(target_pointer_width = "64")]
impl_build_rank_small!(
    2, 9;
    1, 9;
    1, 10;
    1, 11;
    3, 13;
);

// 32-bit word variants (RankSmall<32,1,7> and RankSmall<32,1,8>): always available.
impl Build<BitVec<Vec<u32>>> for RankSmall<32, 1, 7, BitVec<Vec<u32>>> {
    fn new(bits: BitVec<Vec<u32>>) -> Self {
        <Self>::new(bits)
    }
}

impl Build<BitVec<Vec<u32>>> for RankSmall<32, 1, 8, BitVec<Vec<u32>>> {
    fn new(bits: BitVec<Vec<u32>>) -> Self {
        <Self>::new(bits)
    }
}

impl Build<BitVec<Vec<u32>>> for SelectSmall<1, 7, RankSmall<32, 1, 7, BitVec<Vec<u32>>>> {
    fn new(bits: BitVec<Vec<u32>>) -> Self {
        <Self>::new(<RankSmall<32, 1, 7, BitVec<Vec<u32>>>>::new(bits))
    }
}

impl Build<BitVec<Vec<u32>>> for SelectSmall<1, 8, RankSmall<32, 1, 8, BitVec<Vec<u32>>>> {
    fn new(bits: BitVec<Vec<u32>>) -> Self {
        <Self>::new(<RankSmall<32, 1, 8, BitVec<Vec<u32>>>>::new(bits))
    }
}
