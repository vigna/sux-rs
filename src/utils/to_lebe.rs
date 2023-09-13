use crate::traits::*;
use core::sync::atomic::Ordering;

/// On-the-fly conversion to little-endian of a VSlice
pub struct ToLe<B>(B);
/// On-the-fly conversion to big-endian of a VSlice
pub struct ToBe<B>(B);

impl<B: VSliceCore> VSliceCore for ToLe<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        self.0.bit_width()
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<B: VSliceCore> VSliceCore for ToBe<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        self.0.bit_width()
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<B: VSlice> VSlice for ToLe<B> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        self.0.get_unchecked(index).to_le()
    }
}
impl<B: VSlice> VSlice for ToBe<B> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        self.0.get_unchecked(index).to_be()
    }
}

impl<B: VSliceMut> VSliceMut for ToLe<B> {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        self.0.set_unchecked(index, value.to_le());
    }
}
impl<B: VSliceMut> VSliceMut for ToBe<B> {
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        self.0.set_unchecked(index, value.to_be());
    }
}

impl<B: VSliceAtomic> VSliceAtomic for ToLe<B> {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize {
        self.0.get_atomic_unchecked(index, order).to_le()
    }
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering) {
        self.0.set_atomic_unchecked(index, value.to_le(), order);
    }
}
impl<B: VSliceAtomic> VSliceAtomic for ToBe<B> {
    #[inline(always)]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> usize {
        self.0.get_atomic_unchecked(index, order).to_be()
    }
    unsafe fn set_atomic_unchecked(&self, index: usize, value: usize, order: Ordering) {
        self.0.set_atomic_unchecked(index, value.to_be(), order);
    }
}

impl<B: VSliceMutAtomicCmpExchange> VSliceMutAtomicCmpExchange for ToLe<B> {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        self.0
            .compare_exchange_unchecked(index, current.to_le(), new.to_le(), success, failure)
    }
}
impl<B: VSliceMutAtomicCmpExchange> VSliceMutAtomicCmpExchange for ToBe<B> {
    #[inline(always)]
    unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        self.0
            .compare_exchange_unchecked(index, current.to_be(), new.to_be(), success, failure)
    }
}
