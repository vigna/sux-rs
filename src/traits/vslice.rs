

pub trait VSlice {
    fn len(&self) -> usize;

    unsafe fn get_unchecked(&self, index: usize) -> u64;

    fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len() {
            None
        } else {
            Some(unsafe{self.get_unchecked(index)})
        }
    }
}

pub trait VSliceMut: VSlice {
    unsafe fn set_unchecked(&self, index: usize, value: u64);
    fn set(&self, index: usize, value: u64) -> Option<()>;
}