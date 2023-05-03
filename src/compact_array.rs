
pub struct CompactArray<B> {
    data: B,
    bit_width: usize,
    mask: u64,
    len: usize,
}


// TODO!: add invariant that Self::bit_width() <= B::bit_width()
impl<B: VSlice> VSlice for CompactArray<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        let pos = index * self.bit_width;
        let o1 = pos & self.mask;
        let o2 = self.data.bit_width() - o1;
    
        let mask = (1 << value_size) - 1;
        let base = (pos >> self.data.bit_width()) as usize;
        let lower = (self.data.get_unchecked(base) >> o1) & mask;
        let higher = self.data.get_unchecked(base + 1) >> o2;
        (higher | lower) & mask
    }
}