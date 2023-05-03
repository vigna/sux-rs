use crate::traits::*;

pub struct BitMap<B> {
    data: B,
    len: usize,
    number_of_ones: usize,
}

impl<B> BitLength for BitMap<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.number_of_ones
    }
}
