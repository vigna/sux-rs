use crate::traits::*;

pub struct SparseIndex<B: SelectHinted, O: VSlice> {
    bits: B,
    ones: O,
    quantum_log2: usize,
}

/// Forward the lengths
impl<B: SelectHinted, O: VSlice> BitLength for SparseIndex<B, O> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectHinted, O: VSlice> Select for SparseIndex<B, O> {
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let idx = rank >> quantum_log2;
        let pos = self.ones.get_unchecked(idx);
        let rank_at_pos = idx << quantum_log2;

        self.bits.select_unchecked_hinted(
            rank,
            pos,
            rank_at_pos,
        )
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZero> SelectZero for SparseIndex<B, O> {
    #[inline(always)]
    fn select_zero(&self, rank: usize) -> Option<usize> {
        self.bits.select_zero(rank)
    }
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.bits.select_zero_unchecked(rank)
    }
}
