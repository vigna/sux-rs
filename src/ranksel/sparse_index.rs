use anyhow::Result;
use crate::traits::*;

pub struct SparseIndex<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize = 6> {
    bits: B,
    ones: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> SparseIndex<B, O, QUANTUM_LOG2>{
    fn build_ones(&mut self) {
        todo!();
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> Select for SparseIndex<B, O, QUANTUM_LOG2> {
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let idx = rank >> QUANTUM_LOG2;
        let pos = self.ones.get_unchecked(idx);
        let rank_at_pos = idx << QUANTUM_LOG2;

        self.bits.select_unchecked_hinted(
            rank,
            pos as usize,
            rank_at_pos,
        )
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZero, O: VSlice, const QUANTUM_LOG2: usize> SelectZero for SparseIndex<B, O, QUANTUM_LOG2> {
    #[inline(always)]
    fn select_zero(&self, rank: usize) -> Option<usize> {
        self.bits.select_zero(rank)
    }
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.bits.select_zero_unchecked(rank)
    }
}

/*
/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZeroHinted for SparseIndex<B, O, QUANTUM_LOG2> {
    #[inline(always)]
    unsafe fn select_zero_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        self.bits.select_zero_unchecked_hinted(rank, pos, rank_at_pos)
    }
} */

/// Forward the lengths
impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> BitLength for SparseIndex<B, O, QUANTUM_LOG2> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

impl<B: SelectHinted, const QUANTUM_LOG2: usize> ConvertTo<B> for SparseIndex<B, Vec<u64>, QUANTUM_LOG2> {
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

impl<B: SelectHinted, const QUANTUM_LOG2: usize> ConvertTo<SparseIndex<B, Vec<u64>, QUANTUM_LOG2>> for B {
    #[inline(always)]
    fn convert_to(self) -> Result<SparseIndex<B, Vec<u64>, QUANTUM_LOG2>> {
        let mut res = SparseIndex {
            ones: Vec::with_capacity(self.count()),
            bits: self,
            _marker: core::marker::PhantomData::default(),
        };
        res.build_ones();
        Ok(res)
    }
}