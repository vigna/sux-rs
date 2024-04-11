//! Submodule implementing core operations for `BitFieldVec`.

use super::{
    bit_field_slice::{panic_if_value, Word},
    BitFieldVec,
};
use crate::traits::bit_field_slice::BitFieldSliceApply;
use core::ops::*;

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> AddAssign<E> for BitFieldVec<W, B>
where
    W: Add<E, Output = W>,
{
    #[inline]
    fn add_assign(&mut self, rhs: E) {
        self.apply_inplace(|x| x + rhs);
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> Add<E> for BitFieldVec<W, B>
where
    W: Add<E, Output = W>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: E) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> SubAssign<E> for BitFieldVec<W, B>
where
    W: Sub<E, Output = W>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: E) {
        self.apply_inplace(|x| x - rhs);
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> Sub<E> for BitFieldVec<W, B>
where
    W: Sub<E, Output = W>,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: E) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> MulAssign<E> for BitFieldVec<W, B>
where
    W: Mul<E, Output = W>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: E) {
        self.apply_inplace(|x| x * rhs);
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> Mul<E> for BitFieldVec<W, B>
where
    W: Mul<E, Output = W>,
{
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: E) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> DivAssign<E> for BitFieldVec<W, B>
where
    W: Div<E, Output = W>,
{
    #[inline]
    fn div_assign(&mut self, rhs: E) {
        // We can safely apply the division operation to the entire slice
        // since it cannot yield an invalid value, except for when `rhs` is zero,
        // but that is already checked when executing the operation.
        unsafe { self.apply_inplace_unchecked(|x| x / rhs) };
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> Div<E> for BitFieldVec<W, B>
where
    W: Div<E, Output = W>,
{
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: E) -> Self::Output {
        self.div_assign(rhs);
        self
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> RemAssign<E> for BitFieldVec<W, B>
where
    W: Rem<E, Output = W>,
{
    #[inline]
    fn rem_assign(&mut self, rhs: E) {
        // We can safely apply the remainder operation to the entire slice
        // since it cannot yield an invalid value, except for when `rhs` is zero,
        // but that is already checked when executing the operation.
        unsafe { self.apply_inplace_unchecked(|x| x % rhs) };
    }
}

impl<E: Copy, W: Word, B: AsRef<[W]> + AsMut<[W]>> Rem<E> for BitFieldVec<W, B>
where
    W: Rem<E, Output = W>,
{
    type Output = Self;

    #[inline]
    fn rem(mut self, rhs: E) -> Self::Output {
        self.rem_assign(rhs);
        self
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]>> Not for BitFieldVec<W, B>
where
    W: Not<Output = W>,
{
    type Output = Self;

    #[inline]
    fn not(mut self) -> Self::Output {
        // We can safely apply the not operation to the entire slice
        // since it cannot yield an invalid value.
        unsafe { self.apply_inplace_unchecked(|x| !x) };
        self
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> BitAndAssign<W> for BitFieldVec<W, B>
where
    W: BitAnd<Output = W>,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: W) {
        // We can safely apply the bitwise and operation to the entire slice
        // since it cannot yield an invalid value.
        unsafe { self.apply_inplace_unchecked(|x| x & rhs) };
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> BitAnd<W> for BitFieldVec<W, B>
where
    W: BitAnd<Output = W>,
{
    type Output = Self;

    #[inline]
    fn bitand(mut self, rhs: W) -> Self::Output {
        self.bitand_assign(rhs);
        self
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> BitOrAssign<W> for BitFieldVec<W, B>
where
    W: BitOr<Output = W>,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: W) {
        // We can safely apply the bitwise or operation to the entire slice
        // since it cannot yield an invalid value if the original value is valid.
        panic_if_value!(rhs, self.mask, self.bit_width);
        unsafe { self.apply_inplace_unchecked(|x| x | rhs) };
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> BitOr<W> for BitFieldVec<W, B>
where
    W: BitOr<Output = W>,
{
    type Output = Self;

    #[inline]
    fn bitor(mut self, rhs: W) -> Self::Output {
        self.bitor_assign(rhs);
        self
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> BitXorAssign<W> for BitFieldVec<W, B>
where
    W: BitXor<Output = W>,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: W) {
        // We can safely apply the bitwise xor operation to the entire slice
        // since it cannot yield an invalid value if the original value is valid.
        panic_if_value!(rhs, self.mask, self.bit_width);
        unsafe { self.apply_inplace_unchecked(|x| x ^ rhs) };
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> BitXor<W> for BitFieldVec<W, B>
where
    W: BitXor<Output = W>,
{
    type Output = Self;

    #[inline]
    fn bitxor(mut self, rhs: W) -> Self::Output {
        self.bitxor_assign(rhs);
        self
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> ShlAssign<usize> for BitFieldVec<W, B>
where
    W: Shl<usize, Output = W>,
{
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        self.apply_inplace(|x| x << rhs);
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> Shl<usize> for BitFieldVec<W, B>
where
    W: Shl<usize, Output = W>,
{
    type Output = Self;

    #[inline]
    fn shl(mut self, rhs: usize) -> Self::Output {
        self.shl_assign(rhs);
        self
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> ShrAssign<usize> for BitFieldVec<W, B>
where
    W: Shr<usize, Output = W>,
{
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        // We can safely apply the shift right operation to the entire slice
        // since it cannot yield an invalid value if the original value is valid.
        unsafe { self.apply_inplace_unchecked(|x| x >> rhs) };
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]> + Copy> Shr<usize> for BitFieldVec<W, B>
where
    W: Shr<usize, Output = W>,
{
    type Output = Self;

    #[inline]
    fn shr(mut self, rhs: usize) -> Self::Output {
        self.shr_assign(rhs);
        self
    }
}
