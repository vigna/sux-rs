#![feature(prelude_import)]
#![deny(unconditional_recursion)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
pub mod ranksel {
    pub mod elias_fano {
        use crate::{bitmap::BitMap, compact_array::CompactArray, traits::*};
        use anyhow::{bail, Result};
        use core::fmt::Debug;
        use core::sync::atomic::{AtomicU64, Ordering};
        use std::io::{Seek, Write};
        pub struct EliasFanoBuilder {
            u: u64,
            n: u64,
            l: u64,
            low_bits: CompactArray<Vec<u64>>,
            high_bits: BitMap<Vec<u64>>,
            last_value: u64,
            count: u64,
        }
        impl EliasFanoBuilder {
            pub fn new(u: u64, n: u64) -> Self {
                let l = if u >= n {
                    (u as f64 / n as f64).log2().floor() as u64
                } else {
                    0
                };
                Self {
                    u,
                    n,
                    l,
                    low_bits: CompactArray::new(l as usize, n as usize),
                    high_bits: BitMap::new(n as usize + (u as usize >> l) + 1, false),
                    last_value: 0,
                    count: 0,
                }
            }
            pub fn mem_upperbound(u: u64, n: u64) -> u64 {
                2 * n + (n * (u as f64 / n as f64).log2().ceil() as u64)
            }
            pub fn push(&mut self, value: u64) -> Result<()> {
                if value < self.last_value {
                    return ::anyhow::__private::Err({
                        let error = ::anyhow::__private::format_err(format_args!(
                            "The values given to elias-fano are not monotone"
                        ));
                        error
                    });
                }
                unsafe {
                    self.push_unchecked(value);
                }
                Ok(())
            }
            /// # Safety
            ///
            /// Values passed to this function must be smaller than `u` and must be monotone.
            pub unsafe fn push_unchecked(&mut self, value: u64) {
                let low = value & ((1 << self.l) - 1);
                self.low_bits.set(self.count as usize, low);
                let high = (value >> self.l) + self.count;
                self.high_bits.set(high as usize, 1);
                self.count += 1;
                self.last_value = value;
            }
            pub fn build(self) -> EliasFano<BitMap<Vec<u64>>, CompactArray<Vec<u64>>> {
                EliasFano {
                    u: self.u,
                    n: self.n,
                    l: self.l,
                    low_bits: self.low_bits,
                    high_bits: self.high_bits,
                }
            }
        }
        use rkyv::{Archive, Deserialize as RDeserialize, Serialize as RSerialize};
        #[archive_attr(derive(Debug))]
        pub struct EliasFano<H: Archive, L: Archive>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            /// upperbound of the values
            u: u64,
            /// number of values
            n: u64,
            /// the size of the lower bits
            l: u64,
            /// A structure that stores the `l` lowest bits of the values
            low_bits: L,
            high_bits: H,
        }
        #[automatically_derived]
        impl<H: ::core::fmt::Debug + Archive, L: ::core::fmt::Debug + Archive> ::core::fmt::Debug
            for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field5_finish(
                    f,
                    "EliasFano",
                    "u",
                    &self.u,
                    "n",
                    &self.n,
                    "l",
                    &self.l,
                    "low_bits",
                    &self.low_bits,
                    "high_bits",
                    &&self.high_bits,
                )
            }
        }
        #[automatically_derived]
        impl<H: ::core::clone::Clone + Archive, L: ::core::clone::Clone + Archive>
            ::core::clone::Clone for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            #[inline]
            fn clone(&self) -> EliasFano<H, L> {
                EliasFano {
                    u: ::core::clone::Clone::clone(&self.u),
                    n: ::core::clone::Clone::clone(&self.n),
                    l: ::core::clone::Clone::clone(&self.l),
                    low_bits: ::core::clone::Clone::clone(&self.low_bits),
                    high_bits: ::core::clone::Clone::clone(&self.high_bits),
                }
            }
        }
        #[automatically_derived]
        impl<H: Archive, L: Archive> ::core::marker::StructuralPartialEq for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
        }
        #[automatically_derived]
        impl<H: ::core::cmp::PartialEq + Archive, L: ::core::cmp::PartialEq + Archive>
            ::core::cmp::PartialEq for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            #[inline]
            fn eq(&self, other: &EliasFano<H, L>) -> bool {
                self.u == other.u
                    && self.n == other.n
                    && self.l == other.l
                    && self.low_bits == other.low_bits
                    && self.high_bits == other.high_bits
            }
        }
        #[automatically_derived]
        impl<H: Archive, L: Archive> ::core::marker::StructuralEq for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
        }
        #[automatically_derived]
        impl<H: ::core::cmp::Eq + Archive, L: ::core::cmp::Eq + Archive> ::core::cmp::Eq for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            #[inline]
            #[doc(hidden)]
            #[no_coverage]
            fn assert_receiver_is_total_eq(&self) -> () {
                let _: ::core::cmp::AssertParamIsEq<u64>;
                let _: ::core::cmp::AssertParamIsEq<L>;
                let _: ::core::cmp::AssertParamIsEq<H>;
            }
        }
        #[automatically_derived]
        impl<H: ::core::hash::Hash + Archive, L: ::core::hash::Hash + Archive> ::core::hash::Hash
            for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            #[inline]
            fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
                ::core::hash::Hash::hash(&self.u, state);
                ::core::hash::Hash::hash(&self.n, state);
                ::core::hash::Hash::hash(&self.l, state);
                ::core::hash::Hash::hash(&self.low_bits, state);
                ::core::hash::Hash::hash(&self.high_bits, state)
            }
        }
        #[automatically_derived]
        ///An archived [`EliasFano`]
        #[repr()]
        pub struct ArchivedEliasFano<H: Archive, L: Archive>
        where
            H::Archived: Debug,
            L::Archived: Debug,
            u64: ::rkyv::Archive,
            u64: ::rkyv::Archive,
            u64: ::rkyv::Archive,
            L: ::rkyv::Archive,
            H: ::rkyv::Archive,
        {
            ///The archived counterpart of [`EliasFano::u`]
            u: ::rkyv::Archived<u64>,
            ///The archived counterpart of [`EliasFano::n`]
            n: ::rkyv::Archived<u64>,
            ///The archived counterpart of [`EliasFano::l`]
            l: ::rkyv::Archived<u64>,
            ///The archived counterpart of [`EliasFano::low_bits`]
            low_bits: ::rkyv::Archived<L>,
            ///The archived counterpart of [`EliasFano::high_bits`]
            high_bits: ::rkyv::Archived<H>,
        }
        #[automatically_derived]
        impl<H: ::core::fmt::Debug + Archive, L: ::core::fmt::Debug + Archive> ::core::fmt::Debug
            for ArchivedEliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
            u64: ::rkyv::Archive,
            u64: ::rkyv::Archive,
            u64: ::rkyv::Archive,
            L: ::rkyv::Archive,
            H: ::rkyv::Archive,
        {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field5_finish(
                    f,
                    "ArchivedEliasFano",
                    "u",
                    &self.u,
                    "n",
                    &self.n,
                    "l",
                    &self.l,
                    "low_bits",
                    &self.low_bits,
                    "high_bits",
                    &&self.high_bits,
                )
            }
        }
        #[automatically_derived]
        ///The resolver for an archived [`EliasFano`]
        pub struct EliasFanoResolver<H: Archive, L: Archive>
        where
            H::Archived: Debug,
            L::Archived: Debug,
            u64: ::rkyv::Archive,
            u64: ::rkyv::Archive,
            u64: ::rkyv::Archive,
            L: ::rkyv::Archive,
            H: ::rkyv::Archive,
        {
            u: ::rkyv::Resolver<u64>,
            n: ::rkyv::Resolver<u64>,
            l: ::rkyv::Resolver<u64>,
            low_bits: ::rkyv::Resolver<L>,
            high_bits: ::rkyv::Resolver<H>,
        }
        #[automatically_derived]
        const _: () = {
            use ::core::marker::PhantomData;
            use ::rkyv::{out_field, Archive, Archived};
            impl<H: Archive, L: Archive> Archive for EliasFano<H, L>
            where
                H::Archived: Debug,
                L::Archived: Debug,
                u64: ::rkyv::Archive,
                u64: ::rkyv::Archive,
                u64: ::rkyv::Archive,
                L: ::rkyv::Archive,
                H: ::rkyv::Archive,
            {
                type Archived = ArchivedEliasFano<H, L>;
                type Resolver = EliasFanoResolver<H, L>;
                #[allow(clippy::unit_arg)]
                #[inline]
                unsafe fn resolve(
                    &self,
                    pos: usize,
                    resolver: Self::Resolver,
                    out: *mut Self::Archived,
                ) {
                    let (fp, fo) = {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let fo = &raw mut (*out).u;
                            (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                        }
                    };
                    ::rkyv::Archive::resolve((&self.u), pos + fp, resolver.u, fo);
                    let (fp, fo) = {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let fo = &raw mut (*out).n;
                            (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                        }
                    };
                    ::rkyv::Archive::resolve((&self.n), pos + fp, resolver.n, fo);
                    let (fp, fo) = {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let fo = &raw mut (*out).l;
                            (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                        }
                    };
                    ::rkyv::Archive::resolve((&self.l), pos + fp, resolver.l, fo);
                    let (fp, fo) = {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let fo = &raw mut (*out).low_bits;
                            (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                        }
                    };
                    ::rkyv::Archive::resolve((&self.low_bits), pos + fp, resolver.low_bits, fo);
                    let (fp, fo) = {
                        #[allow(unused_unsafe)]
                        unsafe {
                            let fo = &raw mut (*out).high_bits;
                            (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                        }
                    };
                    ::rkyv::Archive::resolve((&self.high_bits), pos + fp, resolver.high_bits, fo);
                }
            }
        };
        #[automatically_derived]
        const _: () = {
            use ::rkyv::{Archive, Archived, Deserialize, Fallible};
            impl<__D: Fallible + ?Sized, H: Archive, L: Archive> Deserialize<EliasFano<H, L>, __D>
                for Archived<EliasFano<H, L>>
            where
                H::Archived: Debug,
                L::Archived: Debug,
                u64: Archive,
                Archived<u64>: Deserialize<u64, __D>,
                u64: Archive,
                Archived<u64>: Deserialize<u64, __D>,
                u64: Archive,
                Archived<u64>: Deserialize<u64, __D>,
                L: Archive,
                Archived<L>: Deserialize<L, __D>,
                H: Archive,
                Archived<H>: Deserialize<H, __D>,
            {
                #[inline]
                fn deserialize(
                    &self,
                    deserializer: &mut __D,
                ) -> ::core::result::Result<EliasFano<H, L>, __D::Error> {
                    Ok(EliasFano {
                        u: Deserialize::<u64, __D>::deserialize(&self.u, deserializer)?,
                        n: Deserialize::<u64, __D>::deserialize(&self.n, deserializer)?,
                        l: Deserialize::<u64, __D>::deserialize(&self.l, deserializer)?,
                        low_bits: Deserialize::<L, __D>::deserialize(&self.low_bits, deserializer)?,
                        high_bits: Deserialize::<H, __D>::deserialize(
                            &self.high_bits,
                            deserializer,
                        )?,
                    })
                }
            }
        };
        #[automatically_derived]
        const _: () = {
            use ::rkyv::{Archive, Fallible, Serialize};
            impl<__S: Fallible + ?Sized, H: Archive, L: Archive> Serialize<__S> for EliasFano<H, L>
            where
                H::Archived: Debug,
                L::Archived: Debug,
                u64: Serialize<__S>,
                u64: Serialize<__S>,
                u64: Serialize<__S>,
                L: Serialize<__S>,
                H: Serialize<__S>,
            {
                #[inline]
                fn serialize(
                    &self,
                    serializer: &mut __S,
                ) -> ::core::result::Result<Self::Resolver, __S::Error> {
                    Ok(EliasFanoResolver {
                        u: Serialize::<__S>::serialize(&self.u, serializer)?,
                        n: Serialize::<__S>::serialize(&self.n, serializer)?,
                        l: Serialize::<__S>::serialize(&self.l, serializer)?,
                        low_bits: Serialize::<__S>::serialize(&self.low_bits, serializer)?,
                        high_bits: Serialize::<__S>::serialize(&self.high_bits, serializer)?,
                    })
                }
            }
        };
        impl<H: Archive, L: Archive> EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            /// # Safety
            /// TODO: this function is never used
            #[inline(always)]
            pub unsafe fn from_raw_parts(
                u: u64,
                n: u64,
                l: u64,
                low_bits: L,
                high_bits: H,
            ) -> Self {
                Self {
                    u,
                    n,
                    l,
                    low_bits,
                    high_bits,
                }
            }
            #[inline(always)]
            pub fn into_raw_parts(self) -> (u64, u64, u64, L, H) {
                (self.u, self.n, self.l, self.low_bits, self.high_bits)
            }
        }
        impl<H: Archive, L: Archive> BitLength for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            #[inline(always)]
            fn len(&self) -> usize {
                self.u as usize
            }
            #[inline(always)]
            fn count(&self) -> usize {
                self.n as usize
            }
        }
        impl<H: Select + Archive, L: VSlice + Archive> Select for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            #[inline]
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                let high_bits = self.high_bits.select_unchecked(rank) - rank;
                let low_bits = self.low_bits.get_unchecked(rank);
                (high_bits << self.l) | low_bits as usize
            }
        }
        impl<H1: Archive, L1: Archive, H2: Archive, L2: Archive> ConvertTo<EliasFano<H1, L1>>
            for EliasFano<H2, L2>
        where
            H2: ConvertTo<H1>,
            L2: ConvertTo<L1>,
            H1::Archived: Debug,
            L1::Archived: Debug,
            H2::Archived: Debug,
            L2::Archived: Debug,
        {
            #[inline(always)]
            fn convert_to(self) -> Result<EliasFano<H1, L1>> {
                Ok(EliasFano {
                    u: self.u,
                    n: self.n,
                    l: self.l,
                    low_bits: self.low_bits.convert_to()?,
                    high_bits: self.high_bits.convert_to()?,
                })
            }
        }
        impl<H: Select + Archive, L: VSlice + Archive> IndexedDict for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            type Value = u64;
            #[inline]
            fn len(&self) -> usize {
                self.count()
            }
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> u64 {
                self.select_unchecked(index) as u64
            }
        }
        impl<H: MemSize + Archive, L: MemSize + Archive> MemSize for EliasFano<H, L>
        where
            H::Archived: Debug,
            L::Archived: Debug,
        {
            fn mem_size(&self) -> usize {
                self.u.mem_size()
                    + self.n.mem_size()
                    + self.l.mem_size()
                    + self.high_bits.mem_size()
                    + self.low_bits.mem_size()
            }
            fn mem_used(&self) -> usize {
                self.u.mem_used()
                    + self.n.mem_used()
                    + self.l.mem_used()
                    + self.high_bits.mem_used()
                    + self.low_bits.mem_used()
            }
        }
    }
    pub mod sparse_index {
        use crate::traits::*;
        use crate::utils::select_in_word;
        use anyhow::Result;
        use std::io::{Seek, Write};
        pub struct SparseIndex<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize = 6> {
            bits: B,
            ones: O,
            _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
        }
        #[automatically_derived]
        impl<
                B: ::core::fmt::Debug + SelectHinted,
                O: ::core::fmt::Debug + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::fmt::Debug for SparseIndex<B, O, QUANTUM_LOG2>
        {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field3_finish(
                    f,
                    "SparseIndex",
                    "bits",
                    &self.bits,
                    "ones",
                    &self.ones,
                    "_marker",
                    &&self._marker,
                )
            }
        }
        #[automatically_derived]
        impl<
                B: ::core::clone::Clone + SelectHinted,
                O: ::core::clone::Clone + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::clone::Clone for SparseIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            fn clone(&self) -> SparseIndex<B, O, QUANTUM_LOG2> {
                SparseIndex {
                    bits: ::core::clone::Clone::clone(&self.bits),
                    ones: ::core::clone::Clone::clone(&self.ones),
                    _marker: ::core::clone::Clone::clone(&self._marker),
                }
            }
        }
        #[automatically_derived]
        impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize>
            ::core::marker::StructuralPartialEq for SparseIndex<B, O, QUANTUM_LOG2>
        {
        }
        #[automatically_derived]
        impl<
                B: ::core::cmp::PartialEq + SelectHinted,
                O: ::core::cmp::PartialEq + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::cmp::PartialEq for SparseIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            fn eq(&self, other: &SparseIndex<B, O, QUANTUM_LOG2>) -> bool {
                self.bits == other.bits && self.ones == other.ones && self._marker == other._marker
            }
        }
        #[automatically_derived]
        impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> ::core::marker::StructuralEq
            for SparseIndex<B, O, QUANTUM_LOG2>
        {
        }
        #[automatically_derived]
        impl<
                B: ::core::cmp::Eq + SelectHinted,
                O: ::core::cmp::Eq + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::cmp::Eq for SparseIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            #[doc(hidden)]
            #[no_coverage]
            fn assert_receiver_is_total_eq(&self) -> () {
                let _: ::core::cmp::AssertParamIsEq<B>;
                let _: ::core::cmp::AssertParamIsEq<O>;
                let _: ::core::cmp::AssertParamIsEq<core::marker::PhantomData<[(); QUANTUM_LOG2]>>;
            }
        }
        #[automatically_derived]
        impl<
                B: ::core::hash::Hash + SelectHinted,
                O: ::core::hash::Hash + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::hash::Hash for SparseIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
                ::core::hash::Hash::hash(&self.bits, state);
                ::core::hash::Hash::hash(&self.ones, state);
                ::core::hash::Hash::hash(&self._marker, state)
            }
        }
        impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> SparseIndex<B, O, QUANTUM_LOG2> {
            /// # Safety
            /// TODO: this function is never used
            #[inline(always)]
            pub unsafe fn from_raw_parts(bits: B, ones: O) -> Self {
                Self {
                    bits,
                    ones,
                    _marker: core::marker::PhantomData,
                }
            }
            #[inline(always)]
            pub fn into_raw_parts(self) -> (B, O) {
                (self.bits, self.ones)
            }
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
                    while number_of_ones + ones_in_word > next_quantum {
                        let in_word_index =
                            select_in_word(word, (next_quantum - number_of_ones) as usize);
                        let index = (i * 64) as u64 + in_word_index as u64;
                        self.ones.set(ones_index, index);
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
        impl<B: SelectHinted + SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize>
            SelectZeroHinted for SparseIndex<B, O, QUANTUM_LOG2>
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
        /// Allow the use of multiple indices, this might not be the best way to do it
        /// but it works
        impl<B: SelectHinted + SelectZero, O: VSlice, const QUANTUM_LOG2: usize> SelectHinted
            for SparseIndex<B, O, QUANTUM_LOG2>
        {
            #[inline(always)]
            unsafe fn select_unchecked_hinted(
                &self,
                rank: usize,
                pos: usize,
                rank_at_pos: usize,
            ) -> usize {
                let index = rank >> QUANTUM_LOG2;
                let this_pos = self.ones.get_unchecked(index) as usize;
                let this_rank_at_pos = index << QUANTUM_LOG2;
                if rank_at_pos > this_rank_at_pos {
                    self.bits.select_unchecked_hinted(rank, pos, rank_at_pos)
                } else {
                    self.bits
                        .select_unchecked_hinted(rank, this_pos, this_rank_at_pos)
                }
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
                    ones: ::alloc::vec::from_elem(
                        0,
                        (self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2,
                    ),
                    bits: self,
                    _marker: core::marker::PhantomData,
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
        impl<B: SelectHinted + Serialize, O: VSlice + Serialize, const QUANTUM_LOG2: usize>
            Serialize for SparseIndex<B, O, QUANTUM_LOG2>
        {
            fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
                let mut bytes = 0;
                bytes += self.bits.serialize(backend)?;
                bytes += self.ones.serialize(backend)?;
                Ok(bytes)
            }
        }
        impl<
                'a,
                B: SelectHinted + Deserialize<'a>,
                O: VSlice + Deserialize<'a>,
                const QUANTUM_LOG2: usize,
            > Deserialize<'a> for SparseIndex<B, O, QUANTUM_LOG2>
        {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (bits, backend) = B::deserialize(backend)?;
                let (ones, backend) = O::deserialize(backend)?;
                Ok((
                    Self {
                        bits,
                        ones,
                        _marker: Default::default(),
                    },
                    backend,
                ))
            }
        }
        impl<B: SelectHinted + MemSize, O: VSlice + MemSize, const QUANTUM_LOG2: usize> MemSize
            for SparseIndex<B, O, QUANTUM_LOG2>
        {
            fn mem_size(&self) -> usize {
                self.bits.mem_size() + self.ones.mem_size()
            }
            fn mem_used(&self) -> usize {
                self.bits.mem_used() + self.ones.mem_used()
            }
        }
    }
    pub mod sparse_zero_index {
        use crate::traits::*;
        use crate::utils::select_in_word;
        use anyhow::Result;
        use std::io::{Seek, Write};
        pub struct SparseZeroIndex<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize = 6> {
            bits: B,
            zeros: O,
            _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
        }
        #[automatically_derived]
        impl<
                B: ::core::fmt::Debug + SelectZeroHinted,
                O: ::core::fmt::Debug + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::fmt::Debug for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field3_finish(
                    f,
                    "SparseZeroIndex",
                    "bits",
                    &self.bits,
                    "zeros",
                    &self.zeros,
                    "_marker",
                    &&self._marker,
                )
            }
        }
        #[automatically_derived]
        impl<
                B: ::core::clone::Clone + SelectZeroHinted,
                O: ::core::clone::Clone + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::clone::Clone for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            fn clone(&self) -> SparseZeroIndex<B, O, QUANTUM_LOG2> {
                SparseZeroIndex {
                    bits: ::core::clone::Clone::clone(&self.bits),
                    zeros: ::core::clone::Clone::clone(&self.zeros),
                    _marker: ::core::clone::Clone::clone(&self._marker),
                }
            }
        }
        #[automatically_derived]
        impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize>
            ::core::marker::StructuralPartialEq for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
        }
        #[automatically_derived]
        impl<
                B: ::core::cmp::PartialEq + SelectZeroHinted,
                O: ::core::cmp::PartialEq + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::cmp::PartialEq for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            fn eq(&self, other: &SparseZeroIndex<B, O, QUANTUM_LOG2>) -> bool {
                self.bits == other.bits
                    && self.zeros == other.zeros
                    && self._marker == other._marker
            }
        }
        #[automatically_derived]
        impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> ::core::marker::StructuralEq
            for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
        }
        #[automatically_derived]
        impl<
                B: ::core::cmp::Eq + SelectZeroHinted,
                O: ::core::cmp::Eq + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::cmp::Eq for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            #[doc(hidden)]
            #[no_coverage]
            fn assert_receiver_is_total_eq(&self) -> () {
                let _: ::core::cmp::AssertParamIsEq<B>;
                let _: ::core::cmp::AssertParamIsEq<O>;
                let _: ::core::cmp::AssertParamIsEq<core::marker::PhantomData<[(); QUANTUM_LOG2]>>;
            }
        }
        #[automatically_derived]
        impl<
                B: ::core::hash::Hash + SelectZeroHinted,
                O: ::core::hash::Hash + VSlice,
                const QUANTUM_LOG2: usize,
            > ::core::hash::Hash for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline]
            fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
                ::core::hash::Hash::hash(&self.bits, state);
                ::core::hash::Hash::hash(&self.zeros, state);
                ::core::hash::Hash::hash(&self._marker, state)
            }
        }
        impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize>
            SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            /// # Safety
            /// TODO: this function is never used
            #[inline(always)]
            pub unsafe fn from_raw_parts(bits: B, zeros: O) -> Self {
                Self {
                    bits,
                    zeros,
                    _marker: core::marker::PhantomData,
                }
            }
            #[inline(always)]
            pub fn into_raw_parts(self) -> (B, O) {
                (self.bits, self.zeros)
            }
        }
        impl<B: SelectZeroHinted + AsRef<[u64]>, O: VSliceMut, const QUANTUM_LOG2: usize>
            SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            fn build_zeros(&mut self) -> Result<()> {
                let mut number_of_ones = 0;
                let mut next_quantum = 0;
                let mut ones_index = 0;
                for (i, mut word) in self.bits.as_ref().iter().copied().enumerate() {
                    word = !word;
                    let ones_in_word = word.count_ones() as u64;
                    while number_of_ones + ones_in_word > next_quantum {
                        let in_word_index =
                            select_in_word(word, (next_quantum - number_of_ones) as usize);
                        let index = (i * 64) as u64 + in_word_index as u64;
                        if index >= self.len() as _ {
                            return Ok(());
                        }
                        self.zeros.set(ones_index, index);
                        next_quantum += 1 << QUANTUM_LOG2;
                        ones_index += 1;
                    }
                    number_of_ones += ones_in_word;
                }
                Ok(())
            }
        }
        /// Provide the hint to the underlying structure
        impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZero
            for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline(always)]
            unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
                let index = rank >> QUANTUM_LOG2;
                let pos = self.zeros.get_unchecked(index);
                let rank_at_pos = index << QUANTUM_LOG2;
                self.bits
                    .select_zero_unchecked_hinted(rank, pos as usize, rank_at_pos)
            }
        }
        /// If the underlying implementation has select zero, forward the methods
        impl<B: SelectZeroHinted + Select, O: VSlice, const QUANTUM_LOG2: usize> Select
            for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline(always)]
            fn select(&self, rank: usize) -> Option<usize> {
                self.bits.select(rank)
            }
            #[inline(always)]
            unsafe fn select_unchecked(&self, rank: usize) -> usize {
                self.bits.select_unchecked(rank)
            }
        }
        /// If the underlying implementation has select zero, forward the methods
        impl<B: SelectZeroHinted + SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectHinted
            for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline(always)]
            unsafe fn select_unchecked_hinted(
                &self,
                rank: usize,
                pos: usize,
                rank_at_pos: usize,
            ) -> usize {
                self.bits.select_unchecked_hinted(rank, pos, rank_at_pos)
            }
        }
        /// Allow the use of multiple indices, this might not be the best way to do it
        /// but it works
        impl<B: SelectZeroHinted + SelectHinted, O: VSlice, const QUANTUM_LOG2: usize>
            SelectZeroHinted for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            #[inline(always)]
            unsafe fn select_zero_unchecked_hinted(
                &self,
                rank: usize,
                pos: usize,
                rank_at_pos: usize,
            ) -> usize {
                let index = rank >> QUANTUM_LOG2;
                let this_pos = self.zeros.get_unchecked(index) as usize;
                let this_rank_at_pos = index << QUANTUM_LOG2;
                if rank_at_pos > this_rank_at_pos {
                    self.bits
                        .select_zero_unchecked_hinted(rank, pos, rank_at_pos)
                } else {
                    self.bits
                        .select_zero_unchecked_hinted(rank, this_pos, this_rank_at_pos)
                }
            }
        }
        /// Forward the lengths
        impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> BitLength
            for SparseZeroIndex<B, O, QUANTUM_LOG2>
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
        impl<B: SelectZeroHinted, const QUANTUM_LOG2: usize> ConvertTo<B>
            for SparseZeroIndex<B, Vec<u64>, QUANTUM_LOG2>
        {
            #[inline(always)]
            fn convert_to(self) -> Result<B> {
                Ok(self.bits)
            }
        }
        impl<B: SelectZeroHinted + AsRef<[u64]>, const QUANTUM_LOG2: usize>
            ConvertTo<SparseZeroIndex<B, Vec<u64>, QUANTUM_LOG2>> for B
        {
            #[inline(always)]
            fn convert_to(self) -> Result<SparseZeroIndex<B, Vec<u64>, QUANTUM_LOG2>> {
                let mut res = SparseZeroIndex {
                    zeros: ::alloc::vec::from_elem(
                        0,
                        (self.len() - self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2,
                    ),
                    bits: self,
                    _marker: core::marker::PhantomData,
                };
                res.build_zeros()?;
                Ok(res)
            }
        }
        impl<B, O, const QUANTUM_LOG2: usize> AsRef<[u64]> for SparseZeroIndex<B, O, QUANTUM_LOG2>
        where
            B: AsRef<[u64]> + SelectZeroHinted,
            O: VSlice,
        {
            fn as_ref(&self) -> &[u64] {
                self.bits.as_ref()
            }
        }
        impl<B: SelectZeroHinted + Serialize, O: VSlice + Serialize, const QUANTUM_LOG2: usize>
            Serialize for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
                let mut bytes = 0;
                bytes += self.bits.serialize(backend)?;
                bytes += self.zeros.serialize(backend)?;
                Ok(bytes)
            }
        }
        impl<
                'a,
                B: SelectZeroHinted + Deserialize<'a>,
                O: VSlice + Deserialize<'a>,
                const QUANTUM_LOG2: usize,
            > Deserialize<'a> for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (bits, backend) = B::deserialize(backend)?;
                let (zeros, backend) = O::deserialize(backend)?;
                Ok((
                    Self {
                        bits,
                        zeros,
                        _marker: Default::default(),
                    },
                    backend,
                ))
            }
        }
        impl<B: SelectZeroHinted + MemSize, O: VSlice + MemSize, const QUANTUM_LOG2: usize> MemSize
            for SparseZeroIndex<B, O, QUANTUM_LOG2>
        {
            fn mem_size(&self) -> usize {
                self.bits.mem_size() + self.zeros.mem_size()
            }
            fn mem_used(&self) -> usize {
                self.bits.mem_used() + self.zeros.mem_used()
            }
        }
    }
    pub mod prelude {
        pub use super::elias_fano::*;
        pub use super::sparse_index::*;
        pub use super::sparse_zero_index::*;
    }
}
pub mod traits {
    //! # Traits
    //! This modules contains basic traits related to succinct data structures.
    //! The train `Length` provides information about the length of the
    //! underlying bit vector, independently of its implementation.
    //!
    //! Traits are collected into a module so you can do `use sux::traits::*;`
    //! for ease of use.
    pub mod serdes {
        use crate::utils::*;
        use anyhow::Result;
        use std::{
            io::{Read, Seek, Write},
            mem::MaybeUninit,
            ops::Deref,
            path::Path,
            ptr::addr_of_mut,
        };
        use bitflags::bitflags;
        pub struct Flags(<Flags as ::bitflags::__private::PublicFlags>::Internal);
        #[automatically_derived]
        impl ::core::fmt::Debug for Flags {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Flags", &&self.0)
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for Flags {
            #[inline]
            fn clone(&self) -> Flags {
                let _: ::core::clone::AssertParamIsClone<
                    <Flags as ::bitflags::__private::PublicFlags>::Internal,
                >;
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for Flags {}
        #[automatically_derived]
        impl ::core::marker::StructuralPartialEq for Flags {}
        #[automatically_derived]
        impl ::core::cmp::PartialEq for Flags {
            #[inline]
            fn eq(&self, other: &Flags) -> bool {
                self.0 == other.0
            }
        }
        #[automatically_derived]
        impl ::core::marker::StructuralEq for Flags {}
        #[automatically_derived]
        impl ::core::cmp::Eq for Flags {
            #[inline]
            #[doc(hidden)]
            #[no_coverage]
            fn assert_receiver_is_total_eq(&self) -> () {
                let _: ::core::cmp::AssertParamIsEq<
                    <Flags as ::bitflags::__private::PublicFlags>::Internal,
                >;
            }
        }
        #[automatically_derived]
        impl ::core::cmp::PartialOrd for Flags {
            #[inline]
            fn partial_cmp(&self, other: &Flags) -> ::core::option::Option<::core::cmp::Ordering> {
                ::core::cmp::PartialOrd::partial_cmp(&self.0, &other.0)
            }
        }
        #[automatically_derived]
        impl ::core::cmp::Ord for Flags {
            #[inline]
            fn cmp(&self, other: &Flags) -> ::core::cmp::Ordering {
                ::core::cmp::Ord::cmp(&self.0, &other.0)
            }
        }
        #[automatically_derived]
        impl ::core::hash::Hash for Flags {
            #[inline]
            fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
                ::core::hash::Hash::hash(&self.0, state)
            }
        }
        impl Flags {
            #[allow(deprecated, non_upper_case_globals)]
            pub const MMAP: Self = Self::from_bits_retain(1 << 0);
            #[allow(deprecated, non_upper_case_globals)]
            pub const TRANSPARENT_HUGE_PAGES: Self = Self::from_bits_retain(1 << 1);
        }
        impl ::bitflags::Flags for Flags {
            const FLAGS: &'static [::bitflags::Flag<Flags>] = &[
                {
                    #[allow(deprecated, non_upper_case_globals)]
                    ::bitflags::Flag::new("MMAP", Flags::MMAP)
                },
                {
                    #[allow(deprecated, non_upper_case_globals)]
                    ::bitflags::Flag::new("TRANSPARENT_HUGE_PAGES", Flags::TRANSPARENT_HUGE_PAGES)
                },
            ];
            type Bits = u32;
            fn bits(&self) -> u32 {
                Flags::bits(self)
            }
            fn from_bits_retain(bits: u32) -> Flags {
                Flags::from_bits_retain(bits)
            }
        }
        #[allow(
            dead_code,
            deprecated,
            unused_doc_comments,
            unused_attributes,
            unused_mut,
            unused_imports,
            non_upper_case_globals,
            clippy::assign_op_pattern
        )]
        const _: () = { # [repr (transparent)] pub struct InternalBitFlags (u32) ; # [automatically_derived] impl :: core :: clone :: Clone for InternalBitFlags { # [inline] fn clone (& self) -> InternalBitFlags { let _ : :: core :: clone :: AssertParamIsClone < u32 > ; * self } } # [automatically_derived] impl :: core :: marker :: Copy for InternalBitFlags { } # [automatically_derived] impl :: core :: marker :: StructuralPartialEq for InternalBitFlags { } # [automatically_derived] impl :: core :: cmp :: PartialEq for InternalBitFlags { # [inline] fn eq (& self , other : & InternalBitFlags) -> bool { self . 0 == other . 0 } } # [automatically_derived] impl :: core :: marker :: StructuralEq for InternalBitFlags { } # [automatically_derived] impl :: core :: cmp :: Eq for InternalBitFlags { # [inline] # [doc (hidden)] # [no_coverage] fn assert_receiver_is_total_eq (& self) -> () { let _ : :: core :: cmp :: AssertParamIsEq < u32 > ; } } # [automatically_derived] impl :: core :: cmp :: PartialOrd for InternalBitFlags { # [inline] fn partial_cmp (& self , other : & InternalBitFlags) -> :: core :: option :: Option < :: core :: cmp :: Ordering > { :: core :: cmp :: PartialOrd :: partial_cmp (& self . 0 , & other . 0) } } # [automatically_derived] impl :: core :: cmp :: Ord for InternalBitFlags { # [inline] fn cmp (& self , other : & InternalBitFlags) -> :: core :: cmp :: Ordering { :: core :: cmp :: Ord :: cmp (& self . 0 , & other . 0) } } # [automatically_derived] impl :: core :: hash :: Hash for InternalBitFlags { # [inline] fn hash < __H : :: core :: hash :: Hasher > (& self , state : & mut __H) -> () { :: core :: hash :: Hash :: hash (& self . 0 , state) } } impl :: bitflags :: __private :: PublicFlags for Flags { type Primitive = u32 ; type Internal = InternalBitFlags ; } impl :: bitflags :: __private :: core :: default :: Default for InternalBitFlags { # [inline] fn default () -> Self { InternalBitFlags :: empty () } } impl :: bitflags :: __private :: core :: fmt :: Debug for InternalBitFlags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter < '_ >) -> :: bitflags :: __private :: core :: fmt :: Result { if self . is_empty () { f . write_fmt (format_args ! ("{0:#x}" , < u32 as :: bitflags :: Bits >:: EMPTY)) } else { :: bitflags :: __private :: core :: fmt :: Display :: fmt (self , f) } } } impl :: bitflags :: __private :: core :: fmt :: Display for InternalBitFlags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter < '_ >) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: parser :: to_writer (& Flags (* self) , f) } } impl :: bitflags :: __private :: core :: str :: FromStr for InternalBitFlags { type Err = :: bitflags :: parser :: ParseError ; fn from_str (s : & str) -> :: bitflags :: __private :: core :: result :: Result < Self , Self :: Err > { :: bitflags :: parser :: from_str :: < Flags > (s) . map (| flags | flags . 0) } } impl :: bitflags :: __private :: core :: convert :: AsRef < u32 > for InternalBitFlags { fn as_ref (& self) -> & u32 { & self . 0 } } impl :: bitflags :: __private :: core :: convert :: From < u32 > for InternalBitFlags { fn from (bits : u32) -> Self { Self :: from_bits_retain (bits) } } # [allow (dead_code , deprecated , unused_attributes)] impl InternalBitFlags { # [doc = " Returns an empty set of flags."] # [inline] pub const fn empty () -> Self { { Self (< u32 as :: bitflags :: Bits > :: EMPTY) } } # [doc = " Returns the set containing all flags."] # [inline] pub const fn all () -> Self { { Self :: from_bits_truncate (< u32 as :: bitflags :: Bits > :: ALL) } } # [doc = " Returns the raw value of the flags currently stored."] # [inline] pub const fn bits (& self) -> u32 { let f = self ; { f . 0 } } # [doc = " Convert from underlying bit representation, unless that"] # [doc = " representation contains bits that do not correspond to a flag."] # [inline] pub const fn from_bits (bits : u32) -> :: bitflags :: __private :: core :: option :: Option < Self > { let bits = bits ; { let truncated = Self :: from_bits_truncate (bits) . 0 ; if truncated == bits { :: bitflags :: __private :: core :: option :: Option :: Some (Self (bits)) } else { :: bitflags :: __private :: core :: option :: Option :: None } } } # [doc = " Convert from underlying bit representation, dropping any bits"] # [doc = " that do not correspond to flags."] # [inline] pub const fn from_bits_truncate (bits : u32) -> Self { let bits = bits ; { if bits == < u32 as :: bitflags :: Bits > :: EMPTY { return Self (bits) } let mut truncated = < u32 as :: bitflags :: Bits > :: EMPTY ; { if bits & Flags :: MMAP . bits () == Flags :: MMAP . bits () { truncated = truncated | Flags :: MMAP . bits () } } ; { if bits & Flags :: TRANSPARENT_HUGE_PAGES . bits () == Flags :: TRANSPARENT_HUGE_PAGES . bits () { truncated = truncated | Flags :: TRANSPARENT_HUGE_PAGES . bits () } } ; Self (truncated) } } # [doc = " Convert from underlying bit representation, preserving all"] # [doc = " bits (even those not corresponding to a defined flag)."] # [inline] pub const fn from_bits_retain (bits : u32) -> Self { let bits = bits ; { Self (bits) } } # [doc = " Get the value for a flag from its stringified name."] # [doc = ""] # [doc = " Names are _case-sensitive_, so must correspond exactly to"] # [doc = " the identifier given to the flag."] # [inline] pub fn from_name (name : & str) -> :: bitflags :: __private :: core :: option :: Option < Self > { let name = name ; { { if name == "MMAP" { return :: bitflags :: __private :: core :: option :: Option :: Some (Self (Flags :: MMAP . bits ())) ; } } ; { if name == "TRANSPARENT_HUGE_PAGES" { return :: bitflags :: __private :: core :: option :: Option :: Some (Self (Flags :: TRANSPARENT_HUGE_PAGES . bits ())) ; } } ; let _ = name ; :: bitflags :: __private :: core :: option :: Option :: None } } # [doc = " Returns `true` if no flags are currently stored."] # [inline] pub const fn is_empty (& self) -> bool { let f = self ; { f . bits () == < u32 as :: bitflags :: Bits > :: EMPTY } } # [doc = " Returns `true` if all flags are currently set."] # [inline] pub const fn is_all (& self) -> bool { let f = self ; { Self :: all () . bits () | f . bits () == f . bits () } } # [doc = " Returns `true` if there are flags common to both `self` and `other`."] # [inline] pub const fn intersects (& self , other : Self) -> bool { let f = self ; let other = other ; { f . bits () & other . bits () != < u32 as :: bitflags :: Bits > :: EMPTY } } # [doc = " Returns `true` if all of the flags in `other` are contained within `self`."] # [inline] pub const fn contains (& self , other : Self) -> bool { let f = self ; let other = other ; { f . bits () & other . bits () == other . bits () } } # [doc = " Inserts the specified flags in-place."] # [doc = ""] # [doc = " This method is equivalent to `union`."] # [inline] pub fn insert (& mut self , other : Self) { let f = self ; let other = other ; { * f = Self :: from_bits_retain (f . bits () | other . bits ()) ; } } # [doc = " Removes the specified flags in-place."] # [doc = ""] # [doc = " This method is equivalent to `difference`."] # [inline] pub fn remove (& mut self , other : Self) { let f = self ; let other = other ; { * f = Self :: from_bits_retain (f . bits () & ! other . bits ()) ; } } # [doc = " Toggles the specified flags in-place."] # [doc = ""] # [doc = " This method is equivalent to `symmetric_difference`."] # [inline] pub fn toggle (& mut self , other : Self) { let f = self ; let other = other ; { * f = Self :: from_bits_retain (f . bits () ^ other . bits ()) ; } } # [doc = " Inserts or removes the specified flags depending on the passed value."] # [inline] pub fn set (& mut self , other : Self , value : bool) { let f = self ; let other = other ; let value = value ; { if value { f . insert (other) ; } else { f . remove (other) ; } } } # [doc = " Returns the intersection between the flags in `self` and"] # [doc = " `other`."] # [doc = ""] # [doc = " Calculating `self` bitwise and (`&`) other, including"] # [doc = " any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn intersection (self , other : Self) -> Self { let f = self ; let other = other ; { Self :: from_bits_retain (f . bits () & other . bits ()) } } # [doc = " Returns the union of between the flags in `self` and `other`."] # [doc = ""] # [doc = " Calculates `self` bitwise or (`|`) `other`, including"] # [doc = " any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn union (self , other : Self) -> Self { let f = self ; let other = other ; { Self :: from_bits_retain (f . bits () | other . bits ()) } } # [doc = " Returns the difference between the flags in `self` and `other`."] # [doc = ""] # [doc = " Calculates `self` bitwise and (`&!`) the bitwise negation of `other`,"] # [doc = " including any bits that don't correspond to a defined flag."] # [doc = ""] # [doc = " This method is _not_ equivalent to `a & !b` when there are bits set that"] # [doc = " don't correspond to a defined flag. The `!` operator will unset any"] # [doc = " bits that don't correspond to a flag, so they'll always be unset by `a &! b`,"] # [doc = " but respected by `a.difference(b)`."] # [inline] # [must_use] pub const fn difference (self , other : Self) -> Self { let f = self ; let other = other ; { Self :: from_bits_retain (f . bits () & ! other . bits ()) } } # [doc = " Returns the symmetric difference between the flags"] # [doc = " in `self` and `other`."] # [doc = ""] # [doc = " Calculates `self` bitwise exclusive or (`^`) `other`,"] # [doc = " including any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn symmetric_difference (self , other : Self) -> Self { let f = self ; let other = other ; { Self :: from_bits_retain (f . bits () ^ other . bits ()) } } # [doc = " Returns the complement of this set of flags."] # [doc = ""] # [doc = " Calculates the bitwise negation (`!`) of `self`,"] # [doc = " **unsetting** any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn complement (self) -> Self { let f = self ; { Self :: from_bits_truncate (! f . bits ()) } } } impl :: bitflags :: __private :: core :: fmt :: Binary for InternalBitFlags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: Binary :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: fmt :: Octal for InternalBitFlags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: Octal :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: fmt :: LowerHex for InternalBitFlags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: LowerHex :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: fmt :: UpperHex for InternalBitFlags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: UpperHex :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: ops :: BitOr for InternalBitFlags { type Output = Self ; # [doc = " Returns the union of the two sets of flags."] # [inline] fn bitor (self , other : InternalBitFlags) -> Self { self . union (other) } } impl :: bitflags :: __private :: core :: ops :: BitOrAssign for InternalBitFlags { # [doc = " Adds the set of flags."] # [inline] fn bitor_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . union (other) ; } } impl :: bitflags :: __private :: core :: ops :: BitXor for InternalBitFlags { type Output = Self ; # [doc = " Returns the left flags, but with all the right flags toggled."] # [inline] fn bitxor (self , other : Self) -> Self { self . symmetric_difference (other) } } impl :: bitflags :: __private :: core :: ops :: BitXorAssign for InternalBitFlags { # [doc = " Toggles the set of flags."] # [inline] fn bitxor_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . symmetric_difference (other) ; } } impl :: bitflags :: __private :: core :: ops :: BitAnd for InternalBitFlags { type Output = Self ; # [doc = " Returns the intersection between the two sets of flags."] # [inline] fn bitand (self , other : Self) -> Self { self . intersection (other) } } impl :: bitflags :: __private :: core :: ops :: BitAndAssign for InternalBitFlags { # [doc = " Disables all flags disabled in the set."] # [inline] fn bitand_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . intersection (other) ; } } impl :: bitflags :: __private :: core :: ops :: Sub for InternalBitFlags { type Output = Self ; # [doc = " Returns the set difference of the two sets of flags."] # [inline] fn sub (self , other : Self) -> Self { self . difference (other) } } impl :: bitflags :: __private :: core :: ops :: SubAssign for InternalBitFlags { # [doc = " Disables all flags enabled in the set."] # [inline] fn sub_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . difference (other) ; } } impl :: bitflags :: __private :: core :: ops :: Not for InternalBitFlags { type Output = Self ; # [doc = " Returns the complement of this set of flags."] # [inline] fn not (self) -> Self { self . complement () } } impl :: bitflags :: __private :: core :: iter :: Extend < InternalBitFlags > for InternalBitFlags { fn extend < T : :: bitflags :: __private :: core :: iter :: IntoIterator < Item = Self > > (& mut self , iterator : T) { for item in iterator { self . insert (item) } } } impl :: bitflags :: __private :: core :: iter :: FromIterator < InternalBitFlags > for InternalBitFlags { fn from_iter < T : :: bitflags :: __private :: core :: iter :: IntoIterator < Item = Self > > (iterator : T) -> Self { use :: bitflags :: __private :: core :: iter :: Extend ; let mut result = Self :: empty () ; result . extend (iterator) ; result } } impl InternalBitFlags { # [doc = " Iterate over enabled flag values."] # [inline] pub const fn iter (& self) -> :: bitflags :: iter :: Iter < Flags > { :: bitflags :: iter :: Iter :: __private_const_new (< Flags as :: bitflags :: Flags > :: FLAGS , Flags :: from_bits_retain (self . bits ()) , Flags :: from_bits_retain (self . bits ())) } # [doc = " Iterate over enabled flag values with their stringified names."] # [inline] pub const fn iter_names (& self) -> :: bitflags :: iter :: IterNames < Flags > { :: bitflags :: iter :: IterNames :: __private_const_new (< Flags as :: bitflags :: Flags > :: FLAGS , Flags :: from_bits_retain (self . bits ()) , Flags :: from_bits_retain (self . bits ())) } } impl :: bitflags :: __private :: core :: iter :: IntoIterator for InternalBitFlags { type Item = Flags ; type IntoIter = :: bitflags :: iter :: Iter < Flags > ; fn into_iter (self) -> Self :: IntoIter { self . iter () } } impl InternalBitFlags { # [doc = " Returns a mutable reference to the raw value of the flags currently stored."] # [inline] pub fn bits_mut (& mut self) -> & mut u32 { & mut self . 0 } } # [allow (dead_code , deprecated , unused_attributes)] impl Flags { # [doc = " Returns an empty set of flags."] # [inline] pub const fn empty () -> Self { { Self (InternalBitFlags :: empty ()) } } # [doc = " Returns the set containing all flags."] # [inline] pub const fn all () -> Self { { Self (InternalBitFlags :: all ()) } } # [doc = " Returns the raw value of the flags currently stored."] # [inline] pub const fn bits (& self) -> u32 { let f = self ; { f . 0 . bits () } } # [doc = " Convert from underlying bit representation, unless that"] # [doc = " representation contains bits that do not correspond to a flag."] # [inline] pub const fn from_bits (bits : u32) -> :: bitflags :: __private :: core :: option :: Option < Self > { let bits = bits ; { match InternalBitFlags :: from_bits (bits) { :: bitflags :: __private :: core :: option :: Option :: Some (bits) => :: bitflags :: __private :: core :: option :: Option :: Some (Self (bits)) , :: bitflags :: __private :: core :: option :: Option :: None => :: bitflags :: __private :: core :: option :: Option :: None , } } } # [doc = " Convert from underlying bit representation, dropping any bits"] # [doc = " that do not correspond to flags."] # [inline] pub const fn from_bits_truncate (bits : u32) -> Self { let bits = bits ; { Self (InternalBitFlags :: from_bits_truncate (bits)) } } # [doc = " Convert from underlying bit representation, preserving all"] # [doc = " bits (even those not corresponding to a defined flag)."] # [inline] pub const fn from_bits_retain (bits : u32) -> Self { let bits = bits ; { Self (InternalBitFlags :: from_bits_retain (bits)) } } # [doc = " Get the value for a flag from its stringified name."] # [doc = ""] # [doc = " Names are _case-sensitive_, so must correspond exactly to"] # [doc = " the identifier given to the flag."] # [inline] pub fn from_name (name : & str) -> :: bitflags :: __private :: core :: option :: Option < Self > { let name = name ; { match InternalBitFlags :: from_name (name) { :: bitflags :: __private :: core :: option :: Option :: Some (bits) => :: bitflags :: __private :: core :: option :: Option :: Some (Self (bits)) , :: bitflags :: __private :: core :: option :: Option :: None => :: bitflags :: __private :: core :: option :: Option :: None , } } } # [doc = " Returns `true` if no flags are currently stored."] # [inline] pub const fn is_empty (& self) -> bool { let f = self ; { f . 0 . is_empty () } } # [doc = " Returns `true` if all flags are currently set."] # [inline] pub const fn is_all (& self) -> bool { let f = self ; { f . 0 . is_all () } } # [doc = " Returns `true` if there are flags common to both `self` and `other`."] # [inline] pub const fn intersects (& self , other : Self) -> bool { let f = self ; let other = other ; { f . 0 . intersects (other . 0) } } # [doc = " Returns `true` if all of the flags in `other` are contained within `self`."] # [inline] pub const fn contains (& self , other : Self) -> bool { let f = self ; let other = other ; { f . 0 . contains (other . 0) } } # [doc = " Inserts the specified flags in-place."] # [doc = ""] # [doc = " This method is equivalent to `union`."] # [inline] pub fn insert (& mut self , other : Self) { let f = self ; let other = other ; { f . 0 . insert (other . 0) } } # [doc = " Removes the specified flags in-place."] # [doc = ""] # [doc = " This method is equivalent to `difference`."] # [inline] pub fn remove (& mut self , other : Self) { let f = self ; let other = other ; { f . 0 . remove (other . 0) } } # [doc = " Toggles the specified flags in-place."] # [doc = ""] # [doc = " This method is equivalent to `symmetric_difference`."] # [inline] pub fn toggle (& mut self , other : Self) { let f = self ; let other = other ; { f . 0 . toggle (other . 0) } } # [doc = " Inserts or removes the specified flags depending on the passed value."] # [inline] pub fn set (& mut self , other : Self , value : bool) { let f = self ; let other = other ; let value = value ; { f . 0 . set (other . 0 , value) } } # [doc = " Returns the intersection between the flags in `self` and"] # [doc = " `other`."] # [doc = ""] # [doc = " Calculating `self` bitwise and (`&`) other, including"] # [doc = " any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn intersection (self , other : Self) -> Self { let f = self ; let other = other ; { Self (f . 0 . intersection (other . 0)) } } # [doc = " Returns the union of between the flags in `self` and `other`."] # [doc = ""] # [doc = " Calculates `self` bitwise or (`|`) `other`, including"] # [doc = " any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn union (self , other : Self) -> Self { let f = self ; let other = other ; { Self (f . 0 . union (other . 0)) } } # [doc = " Returns the difference between the flags in `self` and `other`."] # [doc = ""] # [doc = " Calculates `self` bitwise and (`&!`) the bitwise negation of `other`,"] # [doc = " including any bits that don't correspond to a defined flag."] # [doc = ""] # [doc = " This method is _not_ equivalent to `a & !b` when there are bits set that"] # [doc = " don't correspond to a defined flag. The `!` operator will unset any"] # [doc = " bits that don't correspond to a flag, so they'll always be unset by `a &! b`,"] # [doc = " but respected by `a.difference(b)`."] # [inline] # [must_use] pub const fn difference (self , other : Self) -> Self { let f = self ; let other = other ; { Self (f . 0 . difference (other . 0)) } } # [doc = " Returns the symmetric difference between the flags"] # [doc = " in `self` and `other`."] # [doc = ""] # [doc = " Calculates `self` bitwise exclusive or (`^`) `other`,"] # [doc = " including any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn symmetric_difference (self , other : Self) -> Self { let f = self ; let other = other ; { Self (f . 0 . symmetric_difference (other . 0)) } } # [doc = " Returns the complement of this set of flags."] # [doc = ""] # [doc = " Calculates the bitwise negation (`!`) of `self`,"] # [doc = " **unsetting** any bits that don't correspond to a defined flag."] # [inline] # [must_use] pub const fn complement (self) -> Self { let f = self ; { Self (f . 0 . complement ()) } } } impl :: bitflags :: __private :: core :: fmt :: Binary for Flags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: Binary :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: fmt :: Octal for Flags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: Octal :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: fmt :: LowerHex for Flags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: LowerHex :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: fmt :: UpperHex for Flags { fn fmt (& self , f : & mut :: bitflags :: __private :: core :: fmt :: Formatter) -> :: bitflags :: __private :: core :: fmt :: Result { :: bitflags :: __private :: core :: fmt :: UpperHex :: fmt (& self . 0 , f) } } impl :: bitflags :: __private :: core :: ops :: BitOr for Flags { type Output = Self ; # [doc = " Returns the union of the two sets of flags."] # [inline] fn bitor (self , other : Flags) -> Self { self . union (other) } } impl :: bitflags :: __private :: core :: ops :: BitOrAssign for Flags { # [doc = " Adds the set of flags."] # [inline] fn bitor_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . union (other) ; } } impl :: bitflags :: __private :: core :: ops :: BitXor for Flags { type Output = Self ; # [doc = " Returns the left flags, but with all the right flags toggled."] # [inline] fn bitxor (self , other : Self) -> Self { self . symmetric_difference (other) } } impl :: bitflags :: __private :: core :: ops :: BitXorAssign for Flags { # [doc = " Toggles the set of flags."] # [inline] fn bitxor_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . symmetric_difference (other) ; } } impl :: bitflags :: __private :: core :: ops :: BitAnd for Flags { type Output = Self ; # [doc = " Returns the intersection between the two sets of flags."] # [inline] fn bitand (self , other : Self) -> Self { self . intersection (other) } } impl :: bitflags :: __private :: core :: ops :: BitAndAssign for Flags { # [doc = " Disables all flags disabled in the set."] # [inline] fn bitand_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . intersection (other) ; } } impl :: bitflags :: __private :: core :: ops :: Sub for Flags { type Output = Self ; # [doc = " Returns the set difference of the two sets of flags."] # [inline] fn sub (self , other : Self) -> Self { self . difference (other) } } impl :: bitflags :: __private :: core :: ops :: SubAssign for Flags { # [doc = " Disables all flags enabled in the set."] # [inline] fn sub_assign (& mut self , other : Self) { * self = Self :: from_bits_retain (self . bits ()) . difference (other) ; } } impl :: bitflags :: __private :: core :: ops :: Not for Flags { type Output = Self ; # [doc = " Returns the complement of this set of flags."] # [inline] fn not (self) -> Self { self . complement () } } impl :: bitflags :: __private :: core :: iter :: Extend < Flags > for Flags { fn extend < T : :: bitflags :: __private :: core :: iter :: IntoIterator < Item = Self > > (& mut self , iterator : T) { for item in iterator { self . insert (item) } } } impl :: bitflags :: __private :: core :: iter :: FromIterator < Flags > for Flags { fn from_iter < T : :: bitflags :: __private :: core :: iter :: IntoIterator < Item = Self > > (iterator : T) -> Self { use :: bitflags :: __private :: core :: iter :: Extend ; let mut result = Self :: empty () ; result . extend (iterator) ; result } } impl Flags { # [doc = " Iterate over enabled flag values."] # [inline] pub const fn iter (& self) -> :: bitflags :: iter :: Iter < Flags > { :: bitflags :: iter :: Iter :: __private_const_new (< Flags as :: bitflags :: Flags > :: FLAGS , Flags :: from_bits_retain (self . bits ()) , Flags :: from_bits_retain (self . bits ())) } # [doc = " Iterate over enabled flag values with their stringified names."] # [inline] pub const fn iter_names (& self) -> :: bitflags :: iter :: IterNames < Flags > { :: bitflags :: iter :: IterNames :: __private_const_new (< Flags as :: bitflags :: Flags > :: FLAGS , Flags :: from_bits_retain (self . bits ()) , Flags :: from_bits_retain (self . bits ())) } } impl :: bitflags :: __private :: core :: iter :: IntoIterator for Flags { type Item = Flags ; type IntoIter = :: bitflags :: iter :: Iter < Flags > ; fn into_iter (self) -> Self :: IntoIter { self . iter () } } };
        impl Flags {
            pub fn mmap_flags(&self) -> mmap_rs::MmapFlags {
                match self.contains(Flags::TRANSPARENT_HUGE_PAGES) {
                    true => {
                        mmap_rs::MmapFlags::TRANSPARENT_HUGE_PAGES
                            | mmap_rs::MmapFlags::COPY_ON_WRITE
                    }
                    false => mmap_rs::MmapFlags::empty(),
                }
            }
        }
        /// Possible backends of a [`MemCase`]. The `None` variant is used when the data structure is
        /// created in memory; the `Memory` variant is used when the data structure is deserialized
        /// from a file loaded into an allocated memory region; the `Mmap` variant is used when
        /// the data structure is deserialized from a memory-mapped region.
        pub enum MemBackend {
            /// No backend. The data structure is a standard Rust data structure.
            None,
            /// The backend is an allocated in a memory region aligned to 64 bits.
            Memory(Vec<u64>),
            /// The backend is a memory-mapped region.
            Mmap(mmap_rs::Mmap),
        }
        /// A wrapper keeping together an immutable structure and the memory
        /// it was deserialized from. It is specifically designed for
        /// the case of memory-mapped regions, where the mapping must
        /// be kept alive for the whole lifetime of the data structure.
        /// [`MemCase`] instances can not be cloned, but references
        /// to such instances can be shared freely.
        ///
        /// [`MemCase`] can also be used with data structures deserialized from
        /// memory, although in that case it is not strictly necessary;
        /// nonetheless, reading a single block of memory with [`Read::read_exact`] can be
        /// very fast, and using [`load`] to create a [`MemCase`]
        /// is a way to prevent cloning of the immutable
        /// structure.
        ///
        /// [`MemCase`] implements [`Deref`] and [`AsRef`] to the
        /// wrapped type, so it can be used almost transparently and
        /// with no performance cost. However,
        /// if you need to use a memory-mapped structure as a field in
        /// a struct and you want to avoid `dyn`, you will have
        /// to use [`MemCase`] as the type of the field.
        /// [`MemCase`] implements [`From`] for the
        /// wrapped type, using the no-op [`None`](`MemBackend#variant.None`) variant
        /// of [`MemBackend`], so a data structure can be [encased](encase_mem)
        /// almost transparently.
        pub struct MemCase<S>(pub S, MemBackend);
        unsafe impl<S: Send> Send for MemCase<S> {}
        unsafe impl<S: Sync> Sync for MemCase<S> {}
        impl<S> Deref for MemCase<S> {
            type Target = S;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl<S> AsRef<S> for MemCase<S> {
            #[inline(always)]
            fn as_ref(&self) -> &S {
                &self.0
            }
        }
        /// Encases a data structure in a [`MemCase`] with no backend.
        pub fn encase_mem<S>(s: S) -> MemCase<S> {
            MemCase(s, MemBackend::None)
        }
        impl<S: Send + Sync> From<S> for MemCase<S> {
            fn from(s: S) -> Self {
                encase_mem(s)
            }
        }
        /// Mamory map a file and deserialize a data structure from it,
        /// returning a [`MemCase`] containing the data structure and the
        /// memory mapping.
        #[allow(clippy::uninit_vec)]
        pub fn map<'a, P: AsRef<Path>, S: Deserialize<'a>>(
            path: P,
            flags: &Flags,
        ) -> Result<MemCase<S>> {
            let file_len = path.as_ref().metadata()?.len();
            let file = std::fs::File::open(path)?;
            Ok({
                let mut uninit: MaybeUninit<MemCase<S>> = MaybeUninit::uninit();
                let ptr = uninit.as_mut_ptr();
                let mmap = unsafe {
                    mmap_rs::MmapOptions::new(file_len as _)?
                        .with_flags(flags.mmap_flags())
                        .with_file(file, 0)
                        .map()?
                };
                unsafe {
                    (&raw mut (*ptr).1).write(MemBackend::Mmap(mmap));
                }
                if let MemBackend::Mmap(mmap) = unsafe { &(*ptr).1 } {
                    let (s, _) = S::deserialize(mmap)?;
                    unsafe {
                        (&raw mut (*ptr).0).write(s);
                    }
                    unsafe { uninit.assume_init() }
                } else {
                    ::core::panicking::panic("internal error: entered unreachable code")
                }
            })
        }
        /// Load a file into memory and deserialize a data structure from it,
        /// returning a [`MemCase`] containing the data structure and the
        /// memory. Excess bytes are zeroed out.
        #[allow(clippy::uninit_vec)]
        pub fn load<'a, P: AsRef<Path>, S: Deserialize<'a>>(
            path: P,
            flags: &Flags,
        ) -> Result<MemCase<S>> {
            let file_len = path.as_ref().metadata()?.len() as usize;
            let mut file = std::fs::File::open(path)?;
            let capacity = (file_len + 7) / 8;
            if flags.contains(Flags::MMAP) {
                let mut mmap = mmap_rs::MmapOptions::new(capacity * 8)?
                    .with_flags(flags.mmap_flags())
                    .map_mut()?;
                Ok({
                    let mut uninit: MaybeUninit<MemCase<S>> = MaybeUninit::uninit();
                    let ptr = uninit.as_mut_ptr();
                    file.read_exact(&mut mmap[..file_len])?;
                    mmap[file_len..].fill(0);
                    unsafe {
                        if let Ok(mmap_ro) = mmap.make_read_only() {
                            (&raw mut (*ptr).1).write(MemBackend::Mmap(mmap_ro));
                        } else {
                            {
                                ::core::panicking::panic_fmt(format_args!(
                                    "internal error: entered unreachable code: {0}",
                                    format_args!("make_read_only() failed")
                                ));
                            }
                        }
                    }
                    if let MemBackend::Mmap(mmap) = unsafe { &mut (*ptr).1 } {
                        let (s, _) = S::deserialize(mmap)?;
                        unsafe {
                            (&raw mut (*ptr).0).write(s);
                        }
                        unsafe { uninit.assume_init() }
                    } else {
                        ::core::panicking::panic("internal error: entered unreachable code")
                    }
                })
            } else {
                let mut mem = Vec::<u64>::with_capacity(capacity);
                unsafe {
                    mem.set_len(capacity);
                }
                Ok({
                    let mut uninit: MaybeUninit<MemCase<S>> = MaybeUninit::uninit();
                    let ptr = uninit.as_mut_ptr();
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut::<u64, u8>(mem.as_mut_slice());
                    file.read_exact(&mut bytes[..file_len])?;
                    bytes[file_len..].fill(0);
                    unsafe {
                        (&raw mut (*ptr).1).write(MemBackend::Memory(mem));
                    }
                    if let MemBackend::Memory(mem) = unsafe { &mut (*ptr).1 } {
                        let (s, _) = S::deserialize(bytemuck::cast_slice::<u64, u8>(mem))?;
                        unsafe {
                            (&raw mut (*ptr).0).write(s);
                        }
                        unsafe { uninit.assume_init() }
                    } else {
                        ::core::panicking::panic("internal error: entered unreachable code")
                    }
                })
            }
        }
        pub trait Serialize {
            fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize>;
        }
        pub trait Deserialize<'a>: Sized {
            /// a function that return a deserialzied values that might contains
            /// references to the backend
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])>;
        }
        impl Serialize for usize {
            #[inline(always)]
            fn serialize<F: Write>(&self, backend: &mut F) -> Result<usize> {
                Ok(backend.write(&self.to_ne_bytes())?)
            }
        }
        impl<'a> Deserialize<'a> for usize {
            #[inline(always)]
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                Ok((
                    <usize>::from_ne_bytes(
                        backend[..core::mem::size_of::<usize>()].try_into().unwrap(),
                    ),
                    &backend[core::mem::size_of::<usize>()..],
                ))
            }
        }
        impl<'a> Deserialize<'a> for &'a [usize] {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (len, backend) = usize::deserialize(backend)?;
                let bytes = len * core::mem::size_of::<usize>();
                let (_pre, data, after) = unsafe { backend[..bytes].align_to() };
                if !after.is_empty() {
                    ::core::panicking::panic("assertion failed: after.is_empty()")
                };
                Ok((data, &backend[bytes..]))
            }
        }
        impl Serialize for u8 {
            #[inline(always)]
            fn serialize<F: Write>(&self, backend: &mut F) -> Result<usize> {
                Ok(backend.write(&self.to_ne_bytes())?)
            }
        }
        impl<'a> Deserialize<'a> for u8 {
            #[inline(always)]
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                Ok((
                    <u8>::from_ne_bytes(backend[..core::mem::size_of::<u8>()].try_into().unwrap()),
                    &backend[core::mem::size_of::<u8>()..],
                ))
            }
        }
        impl<'a> Deserialize<'a> for &'a [u8] {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (len, backend) = usize::deserialize(backend)?;
                let bytes = len * core::mem::size_of::<u8>();
                let (_pre, data, after) = unsafe { backend[..bytes].align_to() };
                if !after.is_empty() {
                    ::core::panicking::panic("assertion failed: after.is_empty()")
                };
                Ok((data, &backend[bytes..]))
            }
        }
        impl Serialize for u16 {
            #[inline(always)]
            fn serialize<F: Write>(&self, backend: &mut F) -> Result<usize> {
                Ok(backend.write(&self.to_ne_bytes())?)
            }
        }
        impl<'a> Deserialize<'a> for u16 {
            #[inline(always)]
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                Ok((
                    <u16>::from_ne_bytes(
                        backend[..core::mem::size_of::<u16>()].try_into().unwrap(),
                    ),
                    &backend[core::mem::size_of::<u16>()..],
                ))
            }
        }
        impl<'a> Deserialize<'a> for &'a [u16] {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (len, backend) = usize::deserialize(backend)?;
                let bytes = len * core::mem::size_of::<u16>();
                let (_pre, data, after) = unsafe { backend[..bytes].align_to() };
                if !after.is_empty() {
                    ::core::panicking::panic("assertion failed: after.is_empty()")
                };
                Ok((data, &backend[bytes..]))
            }
        }
        impl Serialize for u32 {
            #[inline(always)]
            fn serialize<F: Write>(&self, backend: &mut F) -> Result<usize> {
                Ok(backend.write(&self.to_ne_bytes())?)
            }
        }
        impl<'a> Deserialize<'a> for u32 {
            #[inline(always)]
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                Ok((
                    <u32>::from_ne_bytes(
                        backend[..core::mem::size_of::<u32>()].try_into().unwrap(),
                    ),
                    &backend[core::mem::size_of::<u32>()..],
                ))
            }
        }
        impl<'a> Deserialize<'a> for &'a [u32] {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (len, backend) = usize::deserialize(backend)?;
                let bytes = len * core::mem::size_of::<u32>();
                let (_pre, data, after) = unsafe { backend[..bytes].align_to() };
                if !after.is_empty() {
                    ::core::panicking::panic("assertion failed: after.is_empty()")
                };
                Ok((data, &backend[bytes..]))
            }
        }
        impl Serialize for u64 {
            #[inline(always)]
            fn serialize<F: Write>(&self, backend: &mut F) -> Result<usize> {
                Ok(backend.write(&self.to_ne_bytes())?)
            }
        }
        impl<'a> Deserialize<'a> for u64 {
            #[inline(always)]
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                Ok((
                    <u64>::from_ne_bytes(
                        backend[..core::mem::size_of::<u64>()].try_into().unwrap(),
                    ),
                    &backend[core::mem::size_of::<u64>()..],
                ))
            }
        }
        impl<'a> Deserialize<'a> for &'a [u64] {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (len, backend) = usize::deserialize(backend)?;
                let bytes = len * core::mem::size_of::<u64>();
                let (_pre, data, after) = unsafe { backend[..bytes].align_to() };
                if !after.is_empty() {
                    ::core::panicking::panic("assertion failed: after.is_empty()")
                };
                Ok((data, &backend[bytes..]))
            }
        }
        impl<T: Serialize> Serialize for Vec<T> {
            fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
                let len = self.len();
                let mut bytes = 0;
                bytes += backend.write(&len.to_ne_bytes())?;
                let file_pos = backend.stream_position()? as usize;
                for _ in 0..pad_align_to(file_pos, core::mem::size_of::<T>()) {
                    bytes += backend.write(&[0])?;
                }
                for item in self {
                    bytes += item.serialize(backend)?;
                }
                Ok(bytes)
            }
        }
    }
    pub use serdes::*;
    mod memory {
        /// Like `core::mem::size_of()` but also for complex objects
        pub trait MemSize {
            /// Memory Owned, i.e. how much data is copied on Clone
            fn mem_size(&self) -> usize;
            /// Memory Owned + Borrowed, i.e. also slices sizes
            fn mem_used(&self) -> usize;
        }
        impl MemSize for u8 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for u16 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for u32 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for u64 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for u128 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for usize {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for i8 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for i16 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for i32 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for i64 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for i128 {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl MemSize for isize {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>()
            }
        }
        impl<'a, T: MemSize> MemSize for &'a [T] {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                self.mem_size() + self.iter().map(|x| x.mem_used()).sum::<usize>()
            }
        }
        impl<T: MemSize> MemSize for Vec<T> {
            #[inline(always)]
            fn mem_size(&self) -> usize {
                core::mem::size_of::<Self>() + self.iter().map(|x| x.mem_size()).sum::<usize>()
            }
            #[inline(always)]
            fn mem_used(&self) -> usize {
                core::mem::size_of::<Self>() + self.iter().map(|x| x.mem_used()).sum::<usize>()
            }
        }
    }
    pub use memory::*;
    mod vslice {
        //! # VSlice
        //!
        //! This module defines the `VSlice` and `VSliceMut` traits, which are accessed
        //! with a logic similar to slices, but when indexed with `get` return a value.
        //! Implementing the slice trait would be more natural, but it would be very complicated
        //! because there is no easy way to return a reference to a bit segment
        //! (see, e.g., [BitSlice](https://docs.rs/bitvec/latest/bitvec/slice/struct.BitSlice.html)).
        //!
        //! Each `VSlice` has an associated [`VSlice::bit_width`]. All stored values must fit
        //! within this bit width.
        //!
        //! Implementations must return always zero on a [`VSlice::get`] when the bit
        //! width is zero. The behavior of a [`VSliceMut::set`] in the same context is not defined.
        use core::sync::atomic::{AtomicU64, Ordering};
        /// Trait for common bits between [`VSlice`] and [`VSliceAtomic`]
        pub trait VSliceCore {
            /// Return the width of the slice. All elements stored in the slice must
            /// fit within this bit width.
            fn bit_width(&self) -> usize;
            /// Return the length of the slice.
            fn len(&self) -> usize;
            /// Return if the slice has length zero
            fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }
        pub trait VSlice: VSliceCore {
            /// Return the value at the specified index.
            ///
            /// # Safety
            /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
            unsafe fn get_unchecked(&self, index: usize) -> u64;
            /// Return the value at the specified index.
            ///
            /// # Panics
            /// May panic if the index is not in in [0..[len](`VSlice::len`))
            fn get(&self, index: usize) -> u64 {
                if index >= self.len() {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Index out of bounds: {0} >= {1}",
                            index,
                            self.len()
                        ));
                    };
                } else {
                    unsafe { self.get_unchecked(index) }
                }
            }
        }
        pub trait VSliceMut: VSlice {
            /// Set the element of the slice at the specified index.
            /// No bounds checking is performed.
            ///
            /// # Safety
            /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
            unsafe fn set_unchecked(&mut self, index: usize, value: u64);
            /// Set the element of the slice at the specified index.
            ///
            ///
            /// May panic if the index is not in in [0..[len](`VSlice::len`))
            /// or the value does not fit in [`VSlice::bit_width`] bits.
            fn set(&mut self, index: usize, value: u64) {
                if index >= self.len() {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Index out of bounds {0} on a vector of len {1}",
                            index,
                            self.len()
                        ));
                    }
                }
                let bw = self.bit_width();
                let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
                if value & mask != value {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Value {0} does not fit in {1} bits",
                            value, bw
                        ));
                    }
                }
                unsafe {
                    self.set_unchecked(index, value);
                }
            }
        }
        pub trait VSliceAtomic: VSliceCore {
            /// Return the value at the specified index.
            ///
            /// # Safety
            /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64;
            /// Return the value at the specified index.
            ///
            /// # Panics
            /// May panic if the index is not in in [0..[len](`VSlice::len`))
            fn get_atomic(&self, index: usize, order: Ordering) -> u64 {
                if index >= self.len() {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Index out of bounds: {0} >= {1}",
                            index,
                            self.len()
                        ));
                    };
                } else {
                    unsafe { self.get_atomic_unchecked(index, order) }
                }
            }
            /// Return if the slice has length zero
            fn is_empty(&self) -> bool {
                self.len() == 0
            }
            /// Set the element of the slice at the specified index.
            /// No bounds checking is performed.
            ///
            /// # Safety
            /// `index` must be in [0..[len](`VSlice::len`)). No bounds checking is performed.
            unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering);
            /// Set the element of the slice at the specified index.
            ///
            ///
            /// May panic if the index is not in in [0..[len](`VSlice::len`))
            /// or the value does not fit in [`VSlice::bit_width`] bits.
            fn set_atomic(&self, index: usize, value: u64, order: Ordering) {
                if index >= self.len() {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Index out of bounds {0} on a vector of len {1}",
                            index,
                            self.len()
                        ));
                    }
                }
                let bw = self.bit_width();
                let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
                if value & mask != value {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Value {0} does not fit in {1} bits",
                            value, bw
                        ));
                    }
                }
                unsafe {
                    self.set_atomic_unchecked(index, value, order);
                }
            }
        }
        pub trait VSliceMutAtomicCmpExchange: VSliceAtomic {
            #[inline(always)]
            /// Compare and exchange the value at the specified index.
            /// If the current value is equal to `current`, set it to `new` and return
            /// `Ok(current)`. Otherwise, return `Err(current)`.
            fn compare_exchange(
                &self,
                index: usize,
                current: u64,
                new: u64,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u64, u64> {
                if index >= self.len() {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Index out of bounds {0} on a vector of len {1}",
                            index,
                            self.len()
                        ));
                    }
                }
                let bw = self.bit_width();
                let mask = u64::MAX.wrapping_shr(64 - bw as u32) & !((bw as i64 - 1) >> 63) as u64;
                if current & mask != current {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Value {0} does not fit in {1} bits",
                            current, bw
                        ));
                    }
                }
                if new & mask != new {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Value {0} does not fit in {1} bits",
                            new, bw
                        ));
                    }
                }
                unsafe { self.compare_exchange_unchecked(index, current, new, success, failure) }
            }
            /// Compare and exchange the value at the specified index.
            /// If the current value is equal to `current`, set it to `new` and return
            /// `Ok(current)`. Otherwise, return `Err(current)`.
            ///
            /// # Safety
            /// The caller must ensure that `index` is in [0..[len](`VSlice::len`)) and that
            /// `current` and `new` fit in [`VSlice::bit_width`] bits.
            unsafe fn compare_exchange_unchecked(
                &self,
                index: usize,
                current: u64,
                new: u64,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u64, u64>;
        }
        impl<'a> VSliceCore for &'a [u64] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[u64]>::len(self)
            }
        }
        impl<'a> VSlice for &'a [u64] {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                *<[u64]>::get_unchecked(self, index)
            }
        }
        impl<'a> VSliceCore for &'a [AtomicU64] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[AtomicU64]>::len(self)
            }
        }
        impl<'a> VSliceAtomic for &'a [AtomicU64] {
            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index).load(order)
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index).store(value, order);
            }
        }
        impl<'a> VSliceCore for &'a mut [u64] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[u64]>::len(self)
            }
        }
        impl<'a> VSlice for &'a mut [u64] {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                *<[u64]>::get_unchecked(self, index)
            }
        }
        impl<'a> VSliceMut for &'a mut [u64] {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                *<[u64]>::get_unchecked_mut(self, index) = value;
            }
        }
        impl<'a> VSliceCore for &'a mut [AtomicU64] {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[AtomicU64]>::len(self)
            }
        }
        impl<'a> VSliceAtomic for &'a mut [AtomicU64] {
            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index).load(order)
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index).store(value, order);
            }
        }
        impl<'a> VSliceMutAtomicCmpExchange for &'a [AtomicU64] {
            #[inline(always)]
            unsafe fn compare_exchange_unchecked(
                &self,
                index: usize,
                current: u64,
                new: u64,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u64, u64> {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index)
                    .compare_exchange(current, new, success, failure)
            }
        }
        impl<'a> VSliceMutAtomicCmpExchange for &'a mut [AtomicU64] {
            #[inline(always)]
            unsafe fn compare_exchange_unchecked(
                &self,
                index: usize,
                current: u64,
                new: u64,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u64, u64> {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index)
                    .compare_exchange(current, new, success, failure)
            }
        }
        impl VSliceCore for Vec<u64> {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[u64]>::len(self)
            }
        }
        impl VSlice for Vec<u64> {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                *<[u64]>::get_unchecked(self, index)
            }
        }
        impl VSliceMut for Vec<u64> {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                *<[u64]>::get_unchecked_mut(self, index) = value;
            }
        }
        impl VSliceCore for Vec<AtomicU64> {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                <[AtomicU64]>::len(self)
            }
        }
        impl VSliceAtomic for Vec<AtomicU64> {
            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index).load(order)
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index).store(value, order);
            }
        }
        impl VSliceMutAtomicCmpExchange for Vec<AtomicU64> {
            #[inline(always)]
            unsafe fn compare_exchange_unchecked(
                &self,
                index: usize,
                current: u64,
                new: u64,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u64, u64> {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                <[AtomicU64]>::get_unchecked(self, index)
                    .compare_exchange(current, new, success, failure)
            }
        }
        impl VSliceCore for mmap_rs::Mmap {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.as_ref().len() / 8
            }
        }
        impl VSlice for mmap_rs::Mmap {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                let ptr = (self.as_ptr() as *const u64).add(index);
                std::ptr::read(ptr)
            }
        }
        impl VSliceCore for mmap_rs::MmapMut {
            #[inline(always)]
            fn bit_width(&self) -> usize {
                64
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.as_ref().len() / 8
            }
        }
        impl VSlice for mmap_rs::MmapMut {
            #[inline(always)]
            unsafe fn get_unchecked(&self, index: usize) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                let ptr = (self.as_ptr() as *const u64).add(index);
                std::ptr::read(ptr)
            }
        }
        impl VSliceAtomic for mmap_rs::MmapMut {
            #[inline(always)]
            unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                let ptr = (self.as_ptr() as *const AtomicU64).add(index);
                (*ptr).load(order)
            }
            #[inline(always)]
            unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                let ptr = (self.as_ptr() as *const AtomicU64).add(index);
                (*ptr).store(value, order)
            }
        }
        impl VSliceMut for mmap_rs::MmapMut {
            #[inline(always)]
            unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                let ptr = (self.as_ptr() as *mut u64).add(index);
                std::ptr::write(ptr, value);
            }
        }
        impl VSliceMutAtomicCmpExchange for mmap_rs::MmapMut {
            #[inline(always)]
            unsafe fn compare_exchange_unchecked(
                &self,
                index: usize,
                current: u64,
                new: u64,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u64, u64> {
                if true {
                    if !(index < self.len()) {
                        {
                            ::core::panicking::panic_fmt(format_args!(
                                "{0} {1}",
                                index,
                                self.len()
                            ));
                        }
                    };
                };
                let ptr = (self.as_ptr() as *const AtomicU64).add(index);
                (*ptr).compare_exchange(current, new, success, failure)
            }
        }
    }
    pub use vslice::*;
    mod indexed_dict {
        /// A dictionary of monotonically increasing values that can be indexed by a `usize`.
        pub trait IndexedDict {
            /// The type of the values stored in the dictionary.
            type Value;
            /// Return the value at the specified index.
            ///
            /// # Panics
            /// May panic if the index is not in in [0..[len](`IndexedDict::len`)).
            fn get(&self, index: usize) -> Self::Value {
                if index >= self.len() {
                    {
                        ::core::panicking::panic_fmt(format_args!(
                            "Index out of bounds: {0} >= {1}",
                            index,
                            self.len()
                        ));
                    }
                } else {
                    unsafe { self.get_unchecked(index) }
                }
            }
            /// Return the value at the specified index.
            ///
            /// # Safety
            ///
            /// `index` must be in [0..[len](`IndexedDict::len`)). No bounds checking is performed.
            unsafe fn get_unchecked(&self, index: usize) -> Self::Value;
            /// Return the length (number of items) of the dictionary.
            fn len(&self) -> usize;
            /// Return true of [`len`](`IndexedDict::len`) is zero.
            fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }
        pub trait Successor: IndexedDict {
            /// Return the index of the successor and the successor
            /// of the given value, or `None` if there is no successor.
            /// The successor is the first value in the dictionary
            /// that is greater than or equal to the given value.
            fn successor(&self, value: Self::Value) -> Option<(usize, Self::Value)>;
        }
        pub trait Predecessor: IndexedDict {
            /// Return the index of the predecessor and the predecessor
            /// of the given value, or `None` if there is no predecessor.
            /// The predecessor is the last value in the dictionary
            /// that is less than the given value.
            fn predecessor(&self, value: Self::Value) -> Option<Self::Value>;
        }
    }
    pub use indexed_dict::*;
    use anyhow::Result;
    /// Like Into but we need to avoid the orphan rule and error [E0210](https://github.com/rust-lang/rust/blob/master/compiler/rustc_error_codes/src/error_codes/E0210.md)
    ///
    /// Reference: https://rust-lang.github.io/chalk/book/clauses/coherence.html
    pub trait ConvertTo<B> {
        fn convert_to(self) -> Result<B>;
    }
    impl ConvertTo<Vec<u64>> for Vec<u64> {
        #[inline(always)]
        fn convert_to(self) -> Result<Self> {
            Ok(self)
        }
    }
    /// A trait specifying abstractly the length of the bit vector underlying
    /// a succint data structure.
    pub trait BitLength {
        /// Return the length in bits of the underlying bit vector.
        fn len(&self) -> usize;
        /// Return the number of ones in the underlying bit vector.
        fn count(&self) -> usize;
        /// Return if there are any ones
        fn is_empty(&self) -> bool {
            self.count() == 0
        }
    }
    /// Rank over a bit vector.
    pub trait Rank: BitLength {
        /// Return the number of ones preceding the specified position.
        ///
        /// # Arguments
        /// * `pos` : `usize` - The position to query.
        fn rank(&self, pos: usize) -> usize {
            unsafe { self.rank_unchecked(pos.min(self.len())) }
        }
        /// Return the number of ones preceding the specified position.
        ///
        /// # Arguments
        /// * `pos` : `usize` - The position to query; see Safety below for valid values.
        ///
        /// # Safety
        ///
        /// `pos` must be between 0 (included) and the [length of the underlying bit
        /// vector](`Length::len`) (included).
        unsafe fn rank_unchecked(&self, pos: usize) -> usize;
    }
    /// Rank zeros over a bit vector.
    pub trait RankZero: Rank + BitLength {
        /// Return the number of zeros preceding the specified position.
        ///
        /// # Arguments
        /// * `pos` : `usize` - The position to query.
        fn rank_zero(&self, pos: usize) -> usize {
            pos - self.rank(pos)
        }
        /// Return the number of zeros preceding the specified position.
        ///
        /// # Arguments
        /// * `pos` : `usize` - The position to query; see Safety below for valid values.
        ///
        /// # Safety
        ///
        /// `pos` must be between 0 and the [length of the underlying bit
        /// vector](`Length::len`) (included).
        unsafe fn rank_zero_unchecked(&self, pos: usize) -> usize {
            pos - self.rank_unchecked(pos)
        }
    }
    /// Select over a bit vector.
    pub trait Select: BitLength {
        /// Return the position of the one of given rank.
        ///
        /// # Arguments
        /// * `rank` : `usize` - The rank to query. If there is no
        /// one of given rank, this function return `None`.
        fn select(&self, rank: usize) -> Option<usize> {
            if rank >= self.count() {
                None
            } else {
                Some(unsafe { self.select_unchecked(rank) })
            }
        }
        /// Return the position of the one of given rank.
        ///
        /// # Arguments
        /// * `rank` : `usize` - The rank to query; see Seafety below for valid values.
        ///
        /// # Safety
        ///
        /// `rank` must be between zero (included) and the number of ones in the
        /// underlying bit vector (excluded).
        unsafe fn select_unchecked(&self, rank: usize) -> usize;
    }
    /// Select zeros over a bit vector.
    pub trait SelectZero: BitLength {
        /// Return the position of the zero of given rank.
        ///
        /// # Arguments
        /// * `rank` : `usize` - The rank to query. If there is no
        /// zero of given rank, this function return `None`.
        fn select_zero(&self, rank: usize) -> Option<usize> {
            if rank >= self.len() - self.count() {
                None
            } else {
                Some(unsafe { self.select_zero_unchecked(rank) })
            }
        }
        /// Return the position of the zero of given rank.
        ///
        /// # Arguments
        /// * `rank` : `usize` - The rank to query; see Safety below for valid values
        ///
        /// # Safety
        ///
        /// `rank` must be between zero (included) and the number of zeroes in the
        /// underlying bit vector (excluded).
        unsafe fn select_zero_unchecked(&self, rank: usize) -> usize;
    }
    pub trait SelectHinted: Select + BitLength {
        /// # Safety
        /// `rank` must be between zero (included) and the number of ones in the
        /// underlying bit vector (excluded). `pos` must be between 0 (included) and
        /// the [length of the underlying bit vector](`Length::len`) (included),
        /// and must be the position of a one in the underlying bit vector.
        /// `rank_at_pos` must be the number of ones in the underlying bit vector
        /// before `pos`.
        unsafe fn select_unchecked_hinted(
            &self,
            rank: usize,
            pos: usize,
            rank_at_pos: usize,
        ) -> usize;
    }
    pub trait SelectZeroHinted: SelectZero + BitLength {
        /// # Safety
        /// `rank` must be between zero (included) and the number of zeros in the
        /// underlying bit vector (excluded). `pos` must be between 0 (included) and
        /// the [length of the underlying bit vector](`Length::len`) (included),
        /// and must be the position of a zero in the underlying bit vector.
        /// `rank_at_pos` must be the number of zeros in the underlying bit vector
        /// before `pos`.
        unsafe fn select_zero_unchecked_hinted(
            &self,
            rank: usize,
            pos: usize,
            rank_at_pos: usize,
        ) -> usize;
    }
}
pub mod prelude {
    pub use crate::bitmap::*;
    pub use crate::compact_array::*;
    pub use crate::ranksel::prelude::*;
    pub use crate::rear_coded_array::*;
    pub use crate::traits::*;
}
mod bitmap {
    use crate::traits::*;
    use crate::utils::select_in_word;
    use anyhow::Result;
    use core::fmt::Debug;
    use rkyv::{Archive, Deserialize as RDeserialize, Serialize as RSerialize};
    use std::io::{Seek, Write};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    #[archive_attr(derive(Debug))]
    pub struct BitMap<B: Archive>
    where
        B::Archived: Debug,
    {
        data: B,
        len: usize,
        number_of_ones: AtomicUsize,
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Archive> ::core::fmt::Debug for BitMap<B>
    where
        B::Archived: Debug,
    {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "BitMap",
                "data",
                &self.data,
                "len",
                &self.len,
                "number_of_ones",
                &&self.number_of_ones,
            )
        }
    }
    #[automatically_derived]
    ///An archived [`BitMap`]
    #[repr()]
    pub struct ArchivedBitMap<B: Archive>
    where
        B::Archived: Debug,
        B: ::rkyv::Archive,
        usize: ::rkyv::Archive,
        AtomicUsize: ::rkyv::Archive,
    {
        ///The archived counterpart of [`BitMap::data`]
        data: ::rkyv::Archived<B>,
        ///The archived counterpart of [`BitMap::len`]
        len: ::rkyv::Archived<usize>,
        ///The archived counterpart of [`BitMap::number_of_ones`]
        number_of_ones: ::rkyv::Archived<AtomicUsize>,
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Archive> ::core::fmt::Debug for ArchivedBitMap<B>
    where
        B::Archived: Debug,
        B: ::rkyv::Archive,
        usize: ::rkyv::Archive,
        AtomicUsize: ::rkyv::Archive,
    {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "ArchivedBitMap",
                "data",
                &self.data,
                "len",
                &self.len,
                "number_of_ones",
                &&self.number_of_ones,
            )
        }
    }
    #[automatically_derived]
    ///The resolver for an archived [`BitMap`]
    pub struct BitMapResolver<B: Archive>
    where
        B::Archived: Debug,
        B: ::rkyv::Archive,
        usize: ::rkyv::Archive,
        AtomicUsize: ::rkyv::Archive,
    {
        data: ::rkyv::Resolver<B>,
        len: ::rkyv::Resolver<usize>,
        number_of_ones: ::rkyv::Resolver<AtomicUsize>,
    }
    #[automatically_derived]
    const _: () = {
        use ::core::marker::PhantomData;
        use ::rkyv::{out_field, Archive, Archived};
        impl<B: Archive> Archive for BitMap<B>
        where
            B::Archived: Debug,
            B: ::rkyv::Archive,
            usize: ::rkyv::Archive,
            AtomicUsize: ::rkyv::Archive,
        {
            type Archived = ArchivedBitMap<B>;
            type Resolver = BitMapResolver<B>;
            #[allow(clippy::unit_arg)]
            #[inline]
            unsafe fn resolve(
                &self,
                pos: usize,
                resolver: Self::Resolver,
                out: *mut Self::Archived,
            ) {
                let (fp, fo) = {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let fo = &raw mut (*out).data;
                        (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                    }
                };
                ::rkyv::Archive::resolve((&self.data), pos + fp, resolver.data, fo);
                let (fp, fo) = {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let fo = &raw mut (*out).len;
                        (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                    }
                };
                ::rkyv::Archive::resolve((&self.len), pos + fp, resolver.len, fo);
                let (fp, fo) = {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let fo = &raw mut (*out).number_of_ones;
                        (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                    }
                };
                ::rkyv::Archive::resolve(
                    (&self.number_of_ones),
                    pos + fp,
                    resolver.number_of_ones,
                    fo,
                );
            }
        }
    };
    #[automatically_derived]
    const _: () = {
        use ::rkyv::{Archive, Archived, Deserialize, Fallible};
        impl<__D: Fallible + ?Sized, B: Archive> Deserialize<BitMap<B>, __D> for Archived<BitMap<B>>
        where
            B::Archived: Debug,
            B: Archive,
            Archived<B>: Deserialize<B, __D>,
            usize: Archive,
            Archived<usize>: Deserialize<usize, __D>,
            AtomicUsize: Archive,
            Archived<AtomicUsize>: Deserialize<AtomicUsize, __D>,
        {
            #[inline]
            fn deserialize(
                &self,
                deserializer: &mut __D,
            ) -> ::core::result::Result<BitMap<B>, __D::Error> {
                Ok(BitMap {
                    data: Deserialize::<B, __D>::deserialize(&self.data, deserializer)?,
                    len: Deserialize::<usize, __D>::deserialize(&self.len, deserializer)?,
                    number_of_ones: Deserialize::<AtomicUsize, __D>::deserialize(
                        &self.number_of_ones,
                        deserializer,
                    )?,
                })
            }
        }
    };
    #[automatically_derived]
    const _: () = {
        use ::rkyv::{Archive, Fallible, Serialize};
        impl<__S: Fallible + ?Sized, B: Archive> Serialize<__S> for BitMap<B>
        where
            B::Archived: Debug,
            B: Serialize<__S>,
            usize: Serialize<__S>,
            AtomicUsize: Serialize<__S>,
        {
            #[inline]
            fn serialize(
                &self,
                serializer: &mut __S,
            ) -> ::core::result::Result<Self::Resolver, __S::Error> {
                Ok(BitMapResolver {
                    data: Serialize::<__S>::serialize(&self.data, serializer)?,
                    len: Serialize::<__S>::serialize(&self.len, serializer)?,
                    number_of_ones: Serialize::<__S>::serialize(&self.number_of_ones, serializer)?,
                })
            }
        }
    };
    impl BitMap<Vec<u64>> {
        pub fn new(len: usize, fill: bool) -> Self {
            let n_of_words = (len + 63) / 64;
            let word = if fill { u64::MAX } else { 0 };
            Self {
                data: ::alloc::vec::from_elem(word, n_of_words),
                len,
                number_of_ones: AtomicUsize::new(0),
            }
        }
    }
    impl BitMap<Vec<AtomicU64>> {
        pub fn new_atomic(len: usize, fill: bool) -> Self {
            let n_of_words = (len + 63) / 64;
            let word = if fill { u64::MAX } else { 0 };
            Self {
                data: (0..n_of_words).map(|_| AtomicU64::new(word)).collect(),
                len,
                number_of_ones: AtomicUsize::new(0),
            }
        }
    }
    impl<B: Archive> BitMap<B>
    where
        B::Archived: Debug,
    {
        /// # Safety
        /// TODO: this function is never used
        #[inline(always)]
        pub unsafe fn from_raw_parts(data: B, len: usize, number_of_ones: usize) -> Self {
            Self {
                data,
                len,
                number_of_ones: AtomicUsize::new(number_of_ones),
            }
        }
        #[inline(always)]
        pub fn into_raw_parts(self) -> (B, usize, usize) {
            (
                self.data,
                self.len,
                self.number_of_ones.load(Ordering::SeqCst),
            )
        }
    }
    impl<B: Archive> BitLength for BitMap<B>
    where
        B::Archived: Debug,
    {
        #[inline(always)]
        fn len(&self) -> usize {
            self.len
        }
        #[inline(always)]
        fn count(&self) -> usize {
            self.number_of_ones.load(Ordering::SeqCst)
        }
    }
    impl<B: VSliceCore + Archive> VSliceCore for BitMap<B>
    where
        B::Archived: Debug,
    {
        #[inline(always)]
        fn bit_width(&self) -> usize {
            if true {
                if !(1 <= self.data.bit_width()) {
                    ::core::panicking::panic("assertion failed: 1 <= self.data.bit_width()")
                };
            };
            1
        }
        #[inline(always)]
        fn len(&self) -> usize {
            self.len
        }
    }
    impl<B: VSlice + Archive> VSlice for BitMap<B>
    where
        B::Archived: Debug,
    {
        unsafe fn get_unchecked(&self, index: usize) -> u64 {
            let word_index = index / self.data.bit_width();
            let word = self.data.get_unchecked(word_index);
            (word >> (index % self.data.bit_width())) & 1
        }
    }
    impl<B: VSliceMutAtomicCmpExchange + Archive> VSliceAtomic for BitMap<B>
    where
        B::Archived: Debug,
    {
        unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
            let word_index = index / self.data.bit_width();
            let word = self.data.get_atomic_unchecked(word_index, order);
            (word >> (index % self.data.bit_width())) & 1
        }
        unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
            let word_index = index / self.data.bit_width();
            let bit_index = index % self.data.bit_width();
            let mut word = self.data.get_atomic_unchecked(word_index, order);
            let mut new_word;
            loop {
                new_word = word & !(1 << bit_index);
                new_word |= value << bit_index;
                match self
                    .data
                    .compare_exchange_unchecked(word_index, word, new_word, order, order)
                {
                    Ok(_) => break,
                    Err(w) => word = w,
                }
            }
            let inc = (new_word > word) as isize - (new_word < word) as isize;
            self.number_of_ones
                .fetch_add(inc as usize, Ordering::Relaxed);
        }
    }
    impl<B: VSliceMut + Archive> VSliceMut for BitMap<B>
    where
        B::Archived: Debug,
    {
        unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
            let word_index = index / self.data.bit_width();
            let bit_index = index % self.data.bit_width();
            let word = self.data.get_unchecked(word_index);
            let mut new_word = word & !(1 << bit_index);
            new_word |= value << bit_index;
            self.data.set_unchecked(word_index, new_word);
            self.number_of_ones
                .fetch_add((new_word > word) as usize, Ordering::Relaxed);
            self.number_of_ones
                .fetch_sub((new_word < word) as usize, Ordering::Relaxed);
        }
    }
    impl<B: VSliceMutAtomicCmpExchange + Archive> VSliceMutAtomicCmpExchange for BitMap<B>
    where
        B::Archived: Debug,
    {
        #[inline(always)]
        unsafe fn compare_exchange_unchecked(
            &self,
            index: usize,
            current: u64,
            new: u64,
            success: Ordering,
            failure: Ordering,
        ) -> Result<u64, u64> {
            let word_index = index / self.data.bit_width();
            let bit_index = index % self.data.bit_width();
            let word = self
                .data
                .get_atomic_unchecked(word_index, Ordering::Acquire);
            let clean_word = word & !(1 << bit_index);
            let cur_word = clean_word | (current << bit_index);
            let new_word = clean_word | (new << bit_index);
            let res = self
                .data
                .compare_exchange_unchecked(word_index, cur_word, new_word, success, failure);
            if res.is_ok() {
                let inc = (new > current) as isize - (new < current) as isize;
                self.number_of_ones
                    .fetch_add(inc as usize, Ordering::Relaxed);
            }
            res
        }
    }
    impl<B: VSlice + Archive> Select for BitMap<B>
    where
        B::Archived: Debug,
    {
        #[inline(always)]
        unsafe fn select_unchecked(&self, rank: usize) -> usize {
            self.select_unchecked_hinted(rank, 0, 0)
        }
    }
    impl<B: VSlice + Archive> SelectHinted for BitMap<B>
    where
        B::Archived: Debug,
    {
        unsafe fn select_unchecked_hinted(
            &self,
            rank: usize,
            pos: usize,
            rank_at_pos: usize,
        ) -> usize {
            let mut word_index = pos / self.data.bit_width();
            let bit_index = pos % self.data.bit_width();
            let mut residual = rank - rank_at_pos;
            let mut word = (self.data.get_unchecked(word_index) >> bit_index) << bit_index;
            loop {
                let bit_count = word.count_ones() as usize;
                if residual < bit_count {
                    break;
                }
                word_index += 1;
                word = self.data.get_unchecked(word_index);
                residual -= bit_count;
            }
            word_index * self.data.bit_width() + select_in_word(word, residual)
        }
    }
    impl<B: VSlice + Archive> SelectZero for BitMap<B>
    where
        B::Archived: Debug,
    {
        #[inline(always)]
        unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
            self.select_zero_unchecked_hinted(rank, 0, 0)
        }
    }
    impl<B: VSlice + Archive> SelectZeroHinted for BitMap<B>
    where
        B::Archived: Debug,
    {
        unsafe fn select_zero_unchecked_hinted(
            &self,
            rank: usize,
            pos: usize,
            rank_at_pos: usize,
        ) -> usize {
            let mut word_index = pos / self.data.bit_width();
            let bit_index = pos % self.data.bit_width();
            let mut residual = rank - rank_at_pos;
            let mut word = (!self.data.get_unchecked(word_index) >> bit_index) << bit_index;
            loop {
                let bit_count = word.count_ones() as usize;
                if residual < bit_count {
                    break;
                }
                word_index += 1;
                word = !self.data.get_unchecked(word_index);
                residual -= bit_count;
            }
            word_index * self.data.bit_width() + select_in_word(word, residual)
        }
    }
    impl<B: AsRef<[u64]> + Archive, D: AsRef<[u64]> + Archive> ConvertTo<BitMap<D>> for BitMap<B>
    where
        B: ConvertTo<D>,
        B::Archived: Debug,
        D::Archived: Debug,
    {
        fn convert_to(self) -> Result<BitMap<D>> {
            Ok(BitMap {
                len: self.len,
                number_of_ones: self.number_of_ones,
                data: self.data.convert_to()?,
            })
        }
    }
    impl<B: AsRef<[u64]> + Archive> AsRef<[u64]> for BitMap<B>
    where
        B::Archived: Debug,
    {
        fn as_ref(&self) -> &[u64] {
            self.data.as_ref()
        }
    }
    impl<B: AsRef<[AtomicU64]> + Archive> AsRef<[AtomicU64]> for BitMap<B>
    where
        B::Archived: Debug,
    {
        fn as_ref(&self) -> &[AtomicU64] {
            self.data.as_ref()
        }
    }
    impl<B: MemSize + Archive> MemSize for BitMap<B>
    where
        B::Archived: Debug,
    {
        fn mem_size(&self) -> usize {
            self.len.mem_size()
                + self.number_of_ones.load(Ordering::Relaxed).mem_size()
                + self.data.mem_size()
        }
        fn mem_used(&self) -> usize {
            self.len.mem_used()
                + self.number_of_ones.load(Ordering::Relaxed).mem_used()
                + self.data.mem_used()
        }
    }
}
mod compact_array {
    use crate::traits::*;
    use anyhow::Result;
    use core::fmt::Debug;
    use std::{
        io::{Seek, Write},
        sync::atomic::{compiler_fence, fence, AtomicU64, Ordering},
    };
    use rkyv::{Archive, Deserialize as RDeserialize, Serialize as RSerialize};
    #[archive_attr(derive(Debug))]
    pub struct CompactArray<B: Archive + Debug>
    where
        B::Archived: Debug,
    {
        data: B,
        bit_width: usize,
        len: usize,
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Archive + Debug> ::core::fmt::Debug for CompactArray<B>
    where
        B::Archived: Debug,
    {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "CompactArray",
                "data",
                &self.data,
                "bit_width",
                &self.bit_width,
                "len",
                &&self.len,
            )
        }
    }
    #[automatically_derived]
    impl<B: ::core::clone::Clone + Archive + Debug> ::core::clone::Clone for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        fn clone(&self) -> CompactArray<B> {
            CompactArray {
                data: ::core::clone::Clone::clone(&self.data),
                bit_width: ::core::clone::Clone::clone(&self.bit_width),
                len: ::core::clone::Clone::clone(&self.len),
            }
        }
    }
    #[automatically_derived]
    impl<B: Archive + Debug> ::core::marker::StructuralPartialEq for CompactArray<B> where
        B::Archived: Debug
    {
    }
    #[automatically_derived]
    impl<B: ::core::cmp::PartialEq + Archive + Debug> ::core::cmp::PartialEq for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        fn eq(&self, other: &CompactArray<B>) -> bool {
            self.data == other.data && self.bit_width == other.bit_width && self.len == other.len
        }
    }
    #[automatically_derived]
    impl<B: Archive + Debug> ::core::marker::StructuralEq for CompactArray<B> where B::Archived: Debug {}
    #[automatically_derived]
    impl<B: ::core::cmp::Eq + Archive + Debug> ::core::cmp::Eq for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        #[doc(hidden)]
        #[no_coverage]
        fn assert_receiver_is_total_eq(&self) -> () {
            let _: ::core::cmp::AssertParamIsEq<B>;
            let _: ::core::cmp::AssertParamIsEq<usize>;
        }
    }
    #[automatically_derived]
    impl<B: ::core::hash::Hash + Archive + Debug> ::core::hash::Hash for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
            ::core::hash::Hash::hash(&self.data, state);
            ::core::hash::Hash::hash(&self.bit_width, state);
            ::core::hash::Hash::hash(&self.len, state)
        }
    }
    #[automatically_derived]
    ///An archived [`CompactArray`]
    #[repr()]
    pub struct ArchivedCompactArray<B: Archive + Debug>
    where
        B::Archived: Debug,
        B: ::rkyv::Archive,
        usize: ::rkyv::Archive,
        usize: ::rkyv::Archive,
    {
        ///The archived counterpart of [`CompactArray::data`]
        data: ::rkyv::Archived<B>,
        ///The archived counterpart of [`CompactArray::bit_width`]
        bit_width: ::rkyv::Archived<usize>,
        ///The archived counterpart of [`CompactArray::len`]
        len: ::rkyv::Archived<usize>,
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Archive + Debug> ::core::fmt::Debug for ArchivedCompactArray<B>
    where
        B::Archived: Debug,
        B: ::rkyv::Archive,
        usize: ::rkyv::Archive,
        usize: ::rkyv::Archive,
    {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "ArchivedCompactArray",
                "data",
                &self.data,
                "bit_width",
                &self.bit_width,
                "len",
                &&self.len,
            )
        }
    }
    #[automatically_derived]
    ///The resolver for an archived [`CompactArray`]
    pub struct CompactArrayResolver<B: Archive + Debug>
    where
        B::Archived: Debug,
        B: ::rkyv::Archive,
        usize: ::rkyv::Archive,
        usize: ::rkyv::Archive,
    {
        data: ::rkyv::Resolver<B>,
        bit_width: ::rkyv::Resolver<usize>,
        len: ::rkyv::Resolver<usize>,
    }
    #[automatically_derived]
    const _: () = {
        use ::core::marker::PhantomData;
        use ::rkyv::{out_field, Archive, Archived};
        impl<B: Archive + Debug> Archive for CompactArray<B>
        where
            B::Archived: Debug,
            B: ::rkyv::Archive,
            usize: ::rkyv::Archive,
            usize: ::rkyv::Archive,
        {
            type Archived = ArchivedCompactArray<B>;
            type Resolver = CompactArrayResolver<B>;
            #[allow(clippy::unit_arg)]
            #[inline]
            unsafe fn resolve(
                &self,
                pos: usize,
                resolver: Self::Resolver,
                out: *mut Self::Archived,
            ) {
                let (fp, fo) = {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let fo = &raw mut (*out).data;
                        (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                    }
                };
                ::rkyv::Archive::resolve((&self.data), pos + fp, resolver.data, fo);
                let (fp, fo) = {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let fo = &raw mut (*out).bit_width;
                        (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                    }
                };
                ::rkyv::Archive::resolve((&self.bit_width), pos + fp, resolver.bit_width, fo);
                let (fp, fo) = {
                    #[allow(unused_unsafe)]
                    unsafe {
                        let fo = &raw mut (*out).len;
                        (fo.cast::<u8>().offset_from(out.cast::<u8>()) as usize, fo)
                    }
                };
                ::rkyv::Archive::resolve((&self.len), pos + fp, resolver.len, fo);
            }
        }
    };
    #[automatically_derived]
    const _: () = {
        use ::rkyv::{Archive, Archived, Deserialize, Fallible};
        impl<__D: Fallible + ?Sized, B: Archive + Debug> Deserialize<CompactArray<B>, __D>
            for Archived<CompactArray<B>>
        where
            B::Archived: Debug,
            B: Archive,
            Archived<B>: Deserialize<B, __D>,
            usize: Archive,
            Archived<usize>: Deserialize<usize, __D>,
            usize: Archive,
            Archived<usize>: Deserialize<usize, __D>,
        {
            #[inline]
            fn deserialize(
                &self,
                deserializer: &mut __D,
            ) -> ::core::result::Result<CompactArray<B>, __D::Error> {
                Ok(CompactArray {
                    data: Deserialize::<B, __D>::deserialize(&self.data, deserializer)?,
                    bit_width: Deserialize::<usize, __D>::deserialize(
                        &self.bit_width,
                        deserializer,
                    )?,
                    len: Deserialize::<usize, __D>::deserialize(&self.len, deserializer)?,
                })
            }
        }
    };
    #[automatically_derived]
    const _: () = {
        use ::rkyv::{Archive, Fallible, Serialize};
        impl<__S: Fallible + ?Sized, B: Archive + Debug> Serialize<__S> for CompactArray<B>
        where
            B::Archived: Debug,
            B: Serialize<__S>,
            usize: Serialize<__S>,
            usize: Serialize<__S>,
        {
            #[inline]
            fn serialize(
                &self,
                serializer: &mut __S,
            ) -> ::core::result::Result<Self::Resolver, __S::Error> {
                Ok(CompactArrayResolver {
                    data: Serialize::<__S>::serialize(&self.data, serializer)?,
                    bit_width: Serialize::<__S>::serialize(&self.bit_width, serializer)?,
                    len: Serialize::<__S>::serialize(&self.len, serializer)?,
                })
            }
        }
    };
    impl CompactArray<Vec<u64>> {
        pub fn new(bit_width: usize, len: usize) -> Self {
            #[cfg(not(any(feature = "testless_read", feature = "testless_write")))]
            let n_of_words = (len * bit_width + 63) / 64;
            Self {
                data: ::alloc::vec::from_elem(0, n_of_words),
                bit_width,
                len,
            }
        }
    }
    impl CompactArray<Vec<AtomicU64>> {
        pub fn new_atomic(bit_width: usize, len: usize) -> Self {
            #[cfg(not(any(feature = "testless_read", feature = "testless_write")))]
            let n_of_words = (len * bit_width + 63) / 64;
            Self {
                data: (0..n_of_words).map(|_| AtomicU64::new(0)).collect(),
                bit_width,
                len,
            }
        }
    }
    impl<B: Archive + Debug> CompactArray<B>
    where
        B::Archived: Debug,
    {
        /// # Safety
        /// TODO: this function is never used.
        #[inline(always)]
        pub unsafe fn from_raw_parts(data: B, bit_width: usize, len: usize) -> Self {
            Self {
                data,
                bit_width,
                len,
            }
        }
        #[inline(always)]
        pub fn into_raw_parts(self) -> (B, usize, usize) {
            (self.data, self.bit_width, self.len)
        }
    }
    impl<B: VSliceCore + Archive + Debug> VSliceCore for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline(always)]
        fn bit_width(&self) -> usize {
            if true {
                if !(self.bit_width <= self.data.bit_width()) {
                    ::core::panicking::panic(
                        "assertion failed: self.bit_width <= self.data.bit_width()",
                    )
                };
            };
            self.bit_width
        }
        #[inline(always)]
        fn len(&self) -> usize {
            self.len
        }
    }
    impl<B: VSlice + Archive + Debug> VSlice for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        unsafe fn get_unchecked(&self, index: usize) -> u64 {
            if true {
                if !(self.bit_width != 64) {
                    ::core::panicking::panic("assertion failed: self.bit_width != 64")
                };
            };
            #[cfg(not(feature = "testless_read"))]
            if self.bit_width == 0 {
                return 0;
            }
            let pos = index * self.bit_width;
            let word_index = pos / 64;
            let bit_index = pos % 64;
            #[cfg(not(feature = "testless_read"))]
            {
                let l = 64 - self.bit_width;
                if bit_index <= l {
                    self.data.get_unchecked(word_index) << (l - bit_index) >> l
                } else {
                    self.data.get_unchecked(word_index) >> bit_index
                        | self.data.get_unchecked(word_index + 1) << (64 + l - bit_index) >> l
                }
            }
        }
    }
    impl<B: VSliceMutAtomicCmpExchange + Archive + Debug> VSliceAtomic for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
            if true {
                if !(self.bit_width != 64) {
                    ::core::panicking::panic("assertion failed: self.bit_width != 64")
                };
            };
            if self.bit_width == 0 {
                return 0;
            }
            let pos = index * self.bit_width;
            let word_index = pos / 64;
            let bit_index = pos % 64;
            let l = 64 - self.bit_width;
            if bit_index <= l {
                self.data.get_atomic_unchecked(word_index, order) << (l - bit_index) >> l
            } else {
                self.data.get_atomic_unchecked(word_index, order) >> bit_index
                    | self.data.get_atomic_unchecked(word_index + 1, order) << (64 + l - bit_index)
                        >> l
            }
        }
        #[inline]
        unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
            if true {
                if !(self.bit_width != 64) {
                    ::core::panicking::panic("assertion failed: self.bit_width != 64")
                };
            };
            if self.bit_width == 0 {
                return;
            }
            let pos = index * self.bit_width;
            let word_index = pos / 64;
            let bit_index = pos % 64;
            let mask: u64 = (1_u64 << self.bit_width) - 1;
            let end_word_index = (pos + self.bit_width - 1) / 64;
            if word_index == end_word_index {
                let mut current = self.data.get_atomic_unchecked(word_index, order);
                loop {
                    let mut new = current;
                    new &= !(mask << bit_index);
                    new |= value << bit_index;
                    match self
                        .data
                        .compare_exchange(word_index, current, new, order, order)
                    {
                        Ok(_) => break,
                        Err(e) => current = e,
                    }
                }
            } else {
                let mut word = self.data.get_atomic_unchecked(word_index, order);
                fence(Ordering::Acquire);
                loop {
                    let mut new = word;
                    new &= (1 << bit_index) - 1;
                    new |= value << bit_index;
                    match self
                        .data
                        .compare_exchange(word_index, word, new, order, order)
                    {
                        Ok(_) => break,
                        Err(e) => word = e,
                    }
                }
                fence(Ordering::Release);
                compiler_fence(Ordering::SeqCst);
                let mut word = self.data.get_atomic_unchecked(end_word_index, order);
                fence(Ordering::Acquire);
                loop {
                    let mut new = word;
                    new &= !(mask >> (64 - bit_index));
                    new |= value >> (64 - bit_index);
                    match self
                        .data
                        .compare_exchange(end_word_index, word, new, order, order)
                    {
                        Ok(_) => break,
                        Err(e) => word = e,
                    }
                }
                fence(Ordering::Release);
            }
        }
    }
    impl<B: VSliceMut + Archive + Debug> VSliceMut for CompactArray<B>
    where
        B::Archived: Debug,
    {
        #[inline]
        unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
            if true {
                if !(self.bit_width != 64) {
                    ::core::panicking::panic("assertion failed: self.bit_width != 64")
                };
            };
            #[cfg(not(feature = "testless_write"))]
            if self.bit_width == 0 {
                return;
            }
            let pos = index * self.bit_width;
            let word_index = pos / 64;
            let bit_index = pos % 64;
            let mask: u64 = (1_u64 << self.bit_width) - 1;
            #[cfg(not(feature = "testless_write"))]
            {
                let end_word_index = (pos + self.bit_width - 1) / 64;
                if word_index == end_word_index {
                    let mut word = self.data.get_unchecked(word_index);
                    word &= !(mask << bit_index);
                    word |= value << bit_index;
                    self.data.set_unchecked(word_index, word);
                } else {
                    let mut word = self.data.get_unchecked(word_index);
                    word &= (1 << bit_index) - 1;
                    word |= value << bit_index;
                    self.data.set_unchecked(word_index, word);
                    let mut word = self.data.get_unchecked(end_word_index);
                    word &= !(mask >> (64 - bit_index));
                    word |= value >> (64 - bit_index);
                    self.data.set_unchecked(end_word_index, word);
                }
            }
        }
    }
    impl<B: Archive + Debug, D: Archive + Debug> ConvertTo<CompactArray<D>> for CompactArray<B>
    where
        B: ConvertTo<D> + VSlice,
        D: VSlice,
        B::Archived: Debug,
        D::Archived: Debug,
    {
        fn convert_to(self) -> Result<CompactArray<D>> {
            Ok(CompactArray {
                len: self.len,
                bit_width: self.bit_width,
                data: self.data.convert_to()?,
            })
        }
    }
    impl<B: VSlice + Archive + Debug> ConvertTo<Vec<u64>> for CompactArray<B>
    where
        B::Archived: Debug,
    {
        fn convert_to(self) -> Result<Vec<u64>> {
            Ok((0..self.len())
                .map(|i| unsafe { self.get_unchecked(i) })
                .collect::<Vec<_>>())
        }
    }
    impl<B: VSlice + MemSize + Archive + Debug> MemSize for CompactArray<B>
    where
        B::Archived: Debug,
    {
        fn mem_size(&self) -> usize {
            self.len.mem_size() + self.bit_width.mem_size() + self.data.mem_size()
        }
        fn mem_used(&self) -> usize {
            self.len.mem_used() + self.bit_width.mem_used() + self.data.mem_used()
        }
    }
}
mod rear_coded_array {
    use crate::traits::IndexedDict;
    use num_traits::AsPrimitive;
    pub struct Stats {
        /// Maximum block size in bytes
        pub max_block_bytes: usize,
        /// The total sum of the block size in bytes
        pub sum_block_bytes: usize,
        /// Maximum shared prefix in bytes
        pub max_lcp: usize,
        /// The total sum of the shared prefix in bytes
        pub sum_lcp: usize,
        /// maximum string length in bytes
        pub max_str_len: usize,
        /// the total sum of the strings length in bytes
        pub sum_str_len: usize,
        /// The bytes wasted writing without compression the first string in block
        pub redoundancy: isize,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Stats {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            let names: &'static _ = &[
                "max_block_bytes",
                "sum_block_bytes",
                "max_lcp",
                "sum_lcp",
                "max_str_len",
                "sum_str_len",
                "redoundancy",
            ];
            let values: &[&dyn ::core::fmt::Debug] = &[
                &self.max_block_bytes,
                &self.sum_block_bytes,
                &self.max_lcp,
                &self.sum_lcp,
                &self.max_str_len,
                &self.sum_str_len,
                &&self.redoundancy,
            ];
            ::core::fmt::Formatter::debug_struct_fields_finish(f, "Stats", names, values)
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for Stats {
        #[inline]
        fn clone(&self) -> Stats {
            Stats {
                max_block_bytes: ::core::clone::Clone::clone(&self.max_block_bytes),
                sum_block_bytes: ::core::clone::Clone::clone(&self.sum_block_bytes),
                max_lcp: ::core::clone::Clone::clone(&self.max_lcp),
                sum_lcp: ::core::clone::Clone::clone(&self.sum_lcp),
                max_str_len: ::core::clone::Clone::clone(&self.max_str_len),
                sum_str_len: ::core::clone::Clone::clone(&self.sum_str_len),
                redoundancy: ::core::clone::Clone::clone(&self.redoundancy),
            }
        }
    }
    #[automatically_derived]
    impl ::core::default::Default for Stats {
        #[inline]
        fn default() -> Stats {
            Stats {
                max_block_bytes: ::core::default::Default::default(),
                sum_block_bytes: ::core::default::Default::default(),
                max_lcp: ::core::default::Default::default(),
                sum_lcp: ::core::default::Default::default(),
                max_str_len: ::core::default::Default::default(),
                sum_str_len: ::core::default::Default::default(),
                redoundancy: ::core::default::Default::default(),
            }
        }
    }
    pub struct RearCodedArray<Ptr: AsPrimitive<usize> = usize>
    where
        usize: AsPrimitive<Ptr>,
    {
        /// The encoded strings \0 terminated
        data: Vec<u8>,
        /// The pointer to in which byte the k-th string start
        pointers: Vec<Ptr>,
        /// The number of strings in a block, this regulates the compression vs
        /// decompression speed tradeoff
        k: usize,
        /// Statistics of the encoded data
        pub stats: Stats,
        /// Number of encoded strings
        len: usize,
        /// Cache of the last encoded string for incremental encoding
        last_str: Vec<u8>,
    }
    #[automatically_derived]
    impl<Ptr: ::core::fmt::Debug + AsPrimitive<usize>> ::core::fmt::Debug for RearCodedArray<Ptr>
    where
        usize: AsPrimitive<Ptr>,
    {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            let names: &'static _ = &["data", "pointers", "k", "stats", "len", "last_str"];
            let values: &[&dyn ::core::fmt::Debug] = &[
                &self.data,
                &self.pointers,
                &self.k,
                &self.stats,
                &self.len,
                &&self.last_str,
            ];
            ::core::fmt::Formatter::debug_struct_fields_finish(f, "RearCodedArray", names, values)
        }
    }
    /// Copy a string until the first \0 from `data` to `result` and return the
    /// remaining data
    #[inline(always)]
    fn strcpy<'a>(mut data: &'a [u8], result: &mut Vec<u8>) -> &'a [u8] {
        loop {
            let c = data[0];
            data = &data[1..];
            if c == 0 {
                break;
            }
            result.push(c);
        }
        data
    }
    #[inline(always)]
    /// strcmp but string is a rust string and data is a \0 terminated string
    fn strcmp(string: &[u8], data: &[u8]) -> core::cmp::Ordering {
        for (i, c) in string.iter().enumerate() {
            match data[i].cmp(c) {
                core::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        data[string.len()].cmp(&0)
    }
    #[inline(always)]
    /// strcmp but both string are rust strings
    fn strcmp_rust(string: &[u8], other: &[u8]) -> core::cmp::Ordering {
        for (i, c) in string.iter().enumerate() {
            match other.get(i).unwrap_or(&0).cmp(c) {
                core::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        other.len().cmp(&string.len())
    }
    impl<Ptr: AsPrimitive<usize>> RearCodedArray<Ptr>
    where
        usize: AsPrimitive<Ptr>,
    {
        const COMPUTE_REDUNDANCY: bool = true;
        pub fn new(k: usize) -> Self {
            Self {
                data: Vec::with_capacity(1 << 20),
                last_str: Vec::with_capacity(1024),
                pointers: Vec::new(),
                len: 0,
                k,
                stats: Default::default(),
            }
        }
        pub fn shrink_to_fit(&mut self) {
            self.data.shrink_to_fit();
            self.pointers.shrink_to_fit();
            self.last_str.shrink_to_fit();
        }
        #[inline]
        pub fn push<S: AsRef<str>>(&mut self, string: S) {
            let string = string.as_ref();
            self.stats.max_str_len = self.stats.max_str_len.max(string.len());
            self.stats.sum_str_len += string.len();
            let to_encode = if self.len % self.k == 0 {
                let last_ptr = self.pointers.last().copied().unwrap_or(0.as_());
                let block_bytes = self.data.len() - last_ptr.as_();
                self.stats.max_block_bytes = self.stats.max_block_bytes.max(block_bytes);
                self.stats.sum_block_bytes += block_bytes;
                self.pointers.push(self.data.len().as_());
                if Self::COMPUTE_REDUNDANCY {
                    let lcp = longest_common_prefix(&self.last_str, string.as_bytes());
                    let rear_length = self.last_str.len() - lcp;
                    self.stats.redoundancy += lcp as isize;
                    self.stats.redoundancy -= encode_int_len(rear_length) as isize;
                }
                string.as_bytes()
            } else {
                let lcp = longest_common_prefix(&self.last_str, string.as_bytes());
                self.stats.max_lcp = self.stats.max_lcp.max(lcp);
                self.stats.sum_lcp += lcp;
                let rear_length = self.last_str.len() - lcp;
                encode_int(rear_length, &mut self.data);
                &string.as_bytes()[lcp..]
            };
            self.data.extend_from_slice(to_encode);
            self.data.push(0);
            self.last_str.clear();
            self.last_str.extend_from_slice(string.as_bytes());
            self.len += 1;
        }
        #[inline]
        pub fn extend<S: AsRef<str>, I: Iterator<Item = S>>(&mut self, iter: I) {
            for string in iter {
                self.push(string);
            }
        }
        /// Write the index-th string to `result` as bytes
        #[inline(always)]
        pub fn get_inplace(&self, index: usize, result: &mut Vec<u8>) {
            result.clear();
            let block = index / self.k;
            let offset = index % self.k;
            let start = self.pointers[block];
            let data = &self.data[start.as_()..];
            let mut data = strcpy(data, result);
            for _ in 0..offset {
                let (len, tmp) = decode_int(data);
                result.resize(result.len() - len, 0);
                let tmp = strcpy(tmp, result);
                data = tmp;
            }
        }
        /// Return whether the string is contained in the array.
        /// This can be used only if the strings inserted were sorted.
        pub fn contains(&self, string: &str) -> bool {
            let string = string.as_bytes();
            let block_idx = self
                .pointers
                .binary_search_by(|block_ptr| strcmp(string, &self.data[block_ptr.as_()..]));
            if block_idx.is_ok() {
                return true;
            }
            let mut block_idx = block_idx.unwrap_err();
            if block_idx == 0 || block_idx > self.pointers.len() {
                return false;
            }
            block_idx -= 1;
            let mut result = Vec::with_capacity(self.stats.max_str_len);
            let start = self.pointers[block_idx];
            let data = &self.data[start.as_()..];
            let mut data = strcpy(data, &mut result);
            let in_block = (self.k - 1).min(self.len - block_idx * self.k - 1);
            for _ in 0..in_block {
                let (len, tmp) = decode_int(data);
                let lcp = result.len() - len;
                result.resize(lcp, 0);
                let tmp = strcpy(tmp, &mut result);
                data = tmp;
                match strcmp_rust(string, &result) {
                    core::cmp::Ordering::Less => {}
                    core::cmp::Ordering::Equal => return true,
                    core::cmp::Ordering::Greater => return false,
                }
            }
            false
        }
        /// Return a sequential iterator over the strings
        pub fn iter(&self) -> RCAIter<'_, Ptr> {
            RCAIter {
                rca: self,
                index: 0,
                data: &self.data,
                buffer: Vec::with_capacity(self.stats.max_str_len),
            }
        }
        pub fn iter_from(&self, index: usize) -> RCAIter<'_, Ptr> {
            let block = index / self.k;
            let offset = index % self.k;
            let start = self.pointers[block];
            let mut res = RCAIter {
                rca: self,
                index,
                data: &self.data[start.as_()..],
                buffer: Vec::with_capacity(self.stats.max_str_len),
            };
            for _ in 0..offset {
                res.next();
            }
            res
        }
        pub fn print_stats(&self) {
            {
                ::std::io::_print(format_args!(
                    "max_block_bytes: {0}\n",
                    self.stats.max_block_bytes
                ));
            };
            {
                ::std::io::_print(format_args!(
                    "avg_block_bytes: {0:.3}\n",
                    self.stats.sum_block_bytes as f64 / self.len() as f64
                ));
            };
            {
                ::std::io::_print(format_args!("max_lcp: {0}\n", self.stats.max_lcp));
            };
            {
                ::std::io::_print(format_args!(
                    "avg_lcp: {0:.3}\n",
                    self.stats.sum_lcp as f64 / self.len() as f64
                ));
            };
            {
                ::std::io::_print(format_args!("max_str_len: {0}\n", self.stats.max_str_len));
            };
            {
                ::std::io::_print(format_args!(
                    "avg_str_len: {0:.3}\n",
                    self.stats.sum_str_len as f64 / self.len() as f64
                ));
            };
            let ptr_size: usize = self.pointers.len() * core::mem::size_of::<Ptr>();
            {
                ::std::io::_print(format_args!("data_bytes:  {0:>15}\n", self.data.len()));
            };
            {
                ::std::io::_print(format_args!("ptrs_bytes:  {0:>15}\n", ptr_size));
            };
            if Self::COMPUTE_REDUNDANCY {
                {
                    ::std::io::_print(format_args!(
                        "redundancy: {0:>15}\n",
                        self.stats.redoundancy
                    ));
                };
                let overhead = self.stats.redoundancy + ptr_size as isize;
                {
                    ::std::io::_print(format_args!(
                        "overhead_ratio: {0}\n",
                        overhead as f64 / (overhead + self.data.len() as isize) as f64
                    ));
                };
                {
                    ::std::io::_print(format_args!(
                        "no_overhead_compression_ratio: {0:.3}\n",
                        (self.data.len() as isize - self.stats.redoundancy) as f64
                            / self.stats.sum_str_len as f64
                    ));
                };
            }
            {
                ::std::io::_print(format_args!(
                    "compression_ratio: {0:.3}\n",
                    (ptr_size + self.data.len()) as f64 / self.stats.sum_str_len as f64
                ));
            };
        }
    }
    impl<Ptr: AsPrimitive<usize>> IndexedDict for RearCodedArray<Ptr>
    where
        usize: AsPrimitive<Ptr>,
    {
        type Value = String;
        unsafe fn get_unchecked(&self, index: usize) -> Self::Value {
            let mut result = Vec::with_capacity(self.stats.max_str_len);
            self.get_inplace(index, &mut result);
            String::from_utf8(result).unwrap()
        }
        #[inline(always)]
        fn len(&self) -> usize {
            self.len
        }
    }
    pub struct RCAIter<'a, Ptr: AsPrimitive<usize>>
    where
        usize: AsPrimitive<Ptr>,
    {
        rca: &'a RearCodedArray<Ptr>,
        buffer: Vec<u8>,
        data: &'a [u8],
        index: usize,
    }
    impl<'a, Ptr: AsPrimitive<usize>> RCAIter<'a, Ptr>
    where
        usize: AsPrimitive<Ptr>,
    {
        pub fn new(rca: &'a RearCodedArray<Ptr>) -> Self {
            Self {
                rca,
                buffer: Vec::with_capacity(rca.stats.max_str_len),
                data: &rca.data,
                index: 0,
            }
        }
    }
    impl<'a, Ptr: AsPrimitive<usize>> Iterator for RCAIter<'a, Ptr>
    where
        usize: AsPrimitive<Ptr>,
    {
        type Item = String;
        fn next(&mut self) -> Option<Self::Item> {
            if self.index >= self.rca.len() {
                return None;
            }
            if self.index % self.rca.k == 0 {
                self.buffer.clear();
                self.data = strcpy(self.data, &mut self.buffer);
            } else {
                let (len, tmp) = decode_int(self.data);
                self.buffer.resize(self.buffer.len() - len, 0);
                self.data = strcpy(tmp, &mut self.buffer);
            }
            self.index += 1;
            Some(String::from_utf8(self.buffer.clone()).unwrap())
        }
    }
    impl<'a, Ptr: AsPrimitive<usize>> ExactSizeIterator for RCAIter<'a, Ptr>
    where
        usize: AsPrimitive<Ptr>,
    {
        fn len(&self) -> usize {
            self.rca.len() - self.index
        }
    }
    #[inline(always)]
    fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
        let min_len = a.len().min(b.len());
        let mut i = 0;
        while i < min_len && a[i] == b[i] {
            i += 1;
        }
        i
    }
    /// Compute the length in bytes of value encoded as VByte
    #[inline(always)]
    fn encode_int_len(mut value: usize) -> usize {
        let mut len = 1;
        let mut max = 1 << 7;
        while value >= max {
            len += 1;
            value -= max;
            max <<= 7;
        }
        len
    }
    /// VByte encode an integer
    #[inline(always)]
    fn encode_int(mut value: usize, data: &mut Vec<u8>) {
        let mut len = 1_usize;
        let mut max = 1 << 7;
        while value >= max {
            value -= max;
            max <<= 7;
            len += 1;
        }
        let bits_in_first = 8 - len;
        let mut first = 1_u8 << bits_in_first;
        let mask = first.saturating_sub(1);
        first |= value as u8 & mask;
        data.push(first);
        value >>= bits_in_first;
        for _ in 0..len.saturating_sub(1) {
            data.push(value as u8);
            value >>= 8;
        }
    }
    #[inline(always)]
    fn decode_int(data: &[u8]) -> (usize, &[u8]) {
        let len = data[0].leading_zeros() as usize + 1;
        let mut base = 0;
        let mut res = (data[0] & (0xff >> len)) as usize;
        let mut shift = 8 - len;
        for value in &data[1..len] {
            base <<= 7;
            base += 1 << 7;
            res |= (*value as usize) << shift;
            shift += 8;
        }
        (res + base, &data[len..])
    }
}
pub mod utils {
    /// Compute the padding needed for alignement, i.e., the number so that
    /// `((value + pad_align_to(value, bits) & (bits - 1) == 0`.
    ///
    /// ```
    /// use sux::utils::pad_align_to;
    /// assert_eq!(7 + pad_align_to(7, 8), 8);
    /// assert_eq!(8 + pad_align_to(8, 8), 8);
    /// assert_eq!(9 + pad_align_to(9, 8), 16);
    /// ```
    pub fn pad_align_to(value: usize, bits: usize) -> usize {
        value.wrapping_neg() & (bits - 1)
    }
    #[inline(always)]
    /// Return the i-th
    /// ```
    /// use sux::utils::select_in_word;
    ///
    /// assert_eq!(select_in_word(0b1, 0), 0);
    /// assert_eq!(select_in_word(0b11, 1), 1);
    /// assert_eq!(select_in_word(0b101, 1), 2);
    /// assert_eq!(select_in_word(0x8000_0000_0000_0000, 0), 63);
    /// assert_eq!(select_in_word(0x8000_0000_8000_0000, 0), 31);
    /// assert_eq!(select_in_word(0x8000_0000_8000_0000, 1), 63);
    /// ```
    pub fn select_in_word(word: u64, rank: usize) -> usize {
        if true {
            if !(rank < word.count_ones() as _) {
                ::core::panicking::panic("assertion failed: rank < word.count_ones() as _")
            };
        };
        #[cfg(not(target_feature = "bmi2"))]
        {
            const ONES_STEP_4: u64 = 0x1111111111111111;
            const ONES_STEP_8: u64 = 0x0101010101010101;
            const LAMBDAS_STEP_8: u64 = 0x80 * ONES_STEP_8;
            let mut s = word;
            s = s - ((s & (0xA * ONES_STEP_4)) >> 1);
            s = (s & (0x3 * ONES_STEP_4)) + ((s >> 2) & (0x3 * ONES_STEP_4));
            s = (s + (s >> 4)) & (0xF * ONES_STEP_8);
            let byte_sums: u64 = s.wrapping_mul(ONES_STEP_8);
            let rank_step_8: u64 = rank as u64 * ONES_STEP_8;
            let geq_rank_step_8: u64 =
                ((rank_step_8 | LAMBDAS_STEP_8) - byte_sums) & LAMBDAS_STEP_8;
            let place = (geq_rank_step_8.count_ones() * 8) as usize;
            let byte_rank: u64 = rank as u64 - (((byte_sums << 8) >> place) & 0xFF_u64);
            let index = ((word >> place) & 0xFF) | (byte_rank << 8);
            place + SELECT_IN_BYTE[index as usize] as usize
        }
    }
    #[inline(always)]
    pub fn select_zero_in_word(word: u64, rank: usize) -> usize {
        select_in_word(!word, rank)
    }
    #[cfg(not(target_feature = "bmi2"))]
    const SELECT_IN_BYTE: [u8; 2048] = [
        8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
        1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
        2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
        1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
        3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
        1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
        2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0,
        1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 8, 8, 1, 8, 2, 2, 1, 8, 3, 3, 1, 3, 2,
        2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1,
        3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 6, 6, 1, 6, 2, 2, 1, 6, 3,
        3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1,
        5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 7, 7, 1, 7, 2,
        2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 7, 5, 5, 1,
        5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 7, 6,
        6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
        6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2,
        2, 1, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3,
        4, 3, 3, 2, 8, 8, 8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4,
        4, 3, 4, 3, 3, 2, 8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4, 4, 2,
        6, 4, 4, 3, 4, 3, 3, 2, 8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5, 5, 4, 5, 4,
        4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 7, 8, 7, 7, 2, 8, 7, 7, 3, 7, 3, 3, 2, 8, 7, 7, 4,
        7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2, 8, 7, 7, 5, 7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5,
        5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2,
        7, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3,
        3, 2, 6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8,
        8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 6,
        8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6, 6, 4, 6, 4, 4, 3, 8, 8, 8, 6, 8, 6,
        6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8,
        8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3, 8, 8,
        8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3,
        8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4, 6, 4,
        4, 3, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4,
        5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5,
        8, 8, 8, 5, 8, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8,
        8, 6, 8, 8, 8, 6, 8, 6, 6, 4, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6,
        8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8,
        8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5,
        8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7,
        7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6,
        7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7,
        7, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7,
        8, 7, 7, 6, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7,
        7, 6, 7, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 7,
    ];
}
