use crate::traits::*;
use crate::utils::select_in_word;
use anyhow::Result;

pub struct SparseIndex<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize = 6> {
    bits: B,
    ones: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectHinted + AsRef<[u64]>, O: VSliceMut, const QUANTUM_LOG2: usize>
    SparseIndex<B, O, QUANTUM_LOG2>
{
    fn build_ones(&mut self) -> Result<()> {
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ones_index = 0;

        for (i, word) in self.bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = select_in_word(word, (next_quantum - number_of_ones) as usize);
                let index = (i * 64) as u64 + in_word_index as u64;
                self.ones.set(ones_index, index).unwrap();
                next_quantum += 1 << QUANTUM_LOG2;
                ones_index += 1;
            }

            number_of_ones += ones_in_word;
        }
        Ok(())
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> Select
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let pos = self.ones.get_unchecked(index);
        let rank_at_pos = index << QUANTUM_LOG2;

        self.bits
            .select_unchecked_hinted(rank, pos as usize, rank_at_pos)
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZero, O: VSlice, const QUANTUM_LOG2: usize> SelectZero
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn select_zero(&self, rank: usize) -> Option<usize> {
        self.bits.select_zero(rank)
    }
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.bits.select_zero_unchecked(rank)
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZeroHinted
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        self.bits
            .select_zero_unchecked_hinted(rank, pos, rank_at_pos)
    }
}

/// Forward the lengths
impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> BitLength
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

impl<B: SelectHinted, const QUANTUM_LOG2: usize> ConvertTo<B>
    for SparseIndex<B, Vec<u64>, QUANTUM_LOG2>
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

impl<B: SelectHinted + AsRef<[u64]>, const QUANTUM_LOG2: usize>
    ConvertTo<SparseIndex<B, Vec<u64>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SparseIndex<B, Vec<u64>, QUANTUM_LOG2>> {
        let mut res = SparseIndex {
            ones: vec![0; (self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2],
            bits: self,
            _marker: core::marker::PhantomData::default(),
        };
        res.build_ones()?;
        Ok(res)
    }
}

impl<B, O, const QUANTUM_LOG2: usize> AsRef<[u64]> for SparseIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[u64]> + SelectHinted,
    O: VSlice,
{
    fn as_ref(&self) -> &[u64] {
        self.bits.as_ref()
    }
}

impl<B, D, O, const QUANTUM_LOG2: usize> ConvertTo<SparseIndex<B, O, QUANTUM_LOG2>>
    for SparseIndex<D, O, QUANTUM_LOG2>
where
    B: SelectHinted + AsRef<[u64]>,
    D: SelectHinted + AsRef<[u64]> + ConvertTo<B>,
    O: VSlice,
{
    #[inline(always)]
    fn convert_to(self) -> Result<SparseIndex<B, O, QUANTUM_LOG2>> {
        Ok(SparseIndex {
            ones: self.ones,
            bits: self.bits.convert_to()?,
            _marker: core::marker::PhantomData::default(),
        })
    }
}

impl<B, O, const QUANTUM_LOG2: usize> core::fmt::Debug for SparseIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[u64]> + SelectHinted + core::fmt::Debug,
    O: VSlice + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseIndex")
            .field("bits", &self.bits)
            .field("ones", &self.ones)
            .finish()
    }
}

impl<B, O, const QUANTUM_LOG2: usize> Clone for SparseIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[u64]> + SelectHinted + Clone,
    O: VSlice + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bits: self.bits.clone(),
            ones: self.ones.clone(),
            _marker: core::marker::PhantomData::default(),
        }
    }
}
