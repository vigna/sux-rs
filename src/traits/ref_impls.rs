/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Implementations of basic traits for references to types that implement such traits.

See <https://www.judy.co.uk/blog/rust-traits-and-references/>

*/

use crate::traits::*;

macro_rules! impl_for_refs {
    ($ty:ty) => {
        impl<T> BitLength for $ty
        where
            T: BitLength,
        {
            #[inline(always)]
            fn len(&self) -> usize {
                (**self).len()
            }
        }

        impl<T> BitCount for $ty
        where
            T: BitCount,
        {
            #[inline(always)]
            fn count(&self) -> usize {
                (**self).count()
            }
        }

        impl<T> Rank for $ty
        where
            T: Rank,
        {
            #[inline(always)]
            fn rank(&self, pos: usize) -> usize {
                (**self).rank(pos)
            }
            #[inline(always)]
            unsafe fn rank_unchecked(&self, pos: usize) -> usize {
                (**self).rank_unchecked(pos)
            }
        }

        impl<T> RankZero for $ty
        where
            T: RankZero,
        {
            #[inline(always)]
            fn rank_zero(&self, pos: usize) -> usize {
                (**self).rank_zero(pos)
            }
            #[inline(always)]
            unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
                (**self).rank_zero_unchecked(pos)
            }
        }

        impl<T> Select for $ty
        where
            T: Select,
        {
            #[inline(always)]
            fn select(&self, rank: usize) -> Option<usize> {
                (**self).select(rank)
            }
            #[inline(always)]
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                (**self).select_unchecked(rank)
            }
        }

        impl<T> SelectZero for $ty
        where
            T: SelectZero,
        {
            #[inline(always)]
            fn select_zero(&self, rank: usize) -> Option<usize> {
                (**self).select_zero(rank)
            }
            #[inline(always)]
            unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
                (**self).select_zero_unchecked(rank)
            }
        }

        impl<T> SelectHinted for $ty
        where
            T: SelectHinted,
        {
            #[inline(always)]
            fn select_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> Option<usize> {
                (**self).select_hinted(rank, pos, rank_at_pos)
            }
            #[inline(always)]
            unsafe fn select_hinted_unchecked(
                &self,
                rank: usize,
                pos: usize,
                rank_at_pos: usize,
            ) -> usize {
                (**self).select_hinted_unchecked(rank, pos, rank_at_pos)
            }
        }

        impl<T> SelectZeroHinted for $ty
        where
            T: SelectZeroHinted,
        {
            #[inline(always)]
            unsafe fn select_zero_hinted_unchecked(
                &self,
                rank: usize,
                pos: usize,
                rank_at_pos: usize,
            ) -> usize {
                (**self).select_zero_hinted_unchecked(rank, pos, rank_at_pos)
            }
            #[inline(always)]
            fn select_zero_hinted(
                &self,
                rank: usize,
                pos: usize,
                rank_at_pos: usize,
            ) -> Option<usize> {
                (**self).select_zero_hinted(rank, pos, rank_at_pos)
            }
        }
    };
}

impl_for_refs!(&T);
impl_for_refs!(&mut T);
impl_for_refs!(Box<T>);
