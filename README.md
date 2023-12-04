# `sux`

A pure Rust implementation of succinct and compressed data structures.

This crate is a work in progress: 
part of it is a port from [Sux](https://sux.di.unimi.it/) and from [the DSI Utilities](https://dsiutils.di.unimi.it/);
new data structures will be added over time. Presently,
we provide:

- the [`BitFieldSlice`](crate::traits::bit_field_slice::BitFieldSlice) trait---an
  alternative to [`Index`](core::ops::Index) returning values of fixed bit width;
- an implementation of [bit vectors](crate::bits::BitVec) and of [vectors of bit fields of fixed with](crate::bits::BitFieldVec);
- traits for building blocks and structures like [`Rank`](crate::traits::rank_sel::Rank) , 
  [`Select`](crate::traits::rank_sel::Select), and [`IndexedDict`](crate::traits::indexed_dict::IndexedDict);
- an implementation of the [Elias--Fano representation of monotone sequences](crate::dict::elias_fano::EliasFano);
- an implementation of lists of [strings compressed by rear-coded prefix omission](crate::dict::rear_coded_list::RearCodedList);
- an implementation of [static functions](crate::func::VFunc).

The focus is on efficiency (in particular, there are unchecked versions of all methods) and
on flexible composability (e.g., you can fine-tune your Elias–Fano instance by choosing different
types of internal indices, and whether to index zeros or ones).

## ε-serde support

All structures in this crate are designed to work with [ε-serde](https://crates.io/crates/epserde):
in particular, once you have created and serialized them, you can easily map them into memory
or load them in memory regions with specific `mmap()` attributes.

## [`MemDbg`](mem_dbg::MemDbg)/[`MemSize`](mem_dbg::MemSize) support

All structures in this crate support the [`MemDbg`](mem_dbg::MemDbg) and 
[`MemSize`](mem_dbg::MemSize) traits from the
[`mem_dbg` crate](https://crates.io/crates/mem_dbg), which provide convenient facilities
for inspecting memory usage and debugging memory-related issues. For example, this is
the output of `mem_dbg()` on a large `EliasFano` instance:
```text
  117_041_232 B 100.00% ⏺: sux::dict::elias_fano::EliasFano<sux::rank_sel::select_zero_fixed2::SelectZeroFixed2<sux::rank_sel::select_fixed2::SelectFixed2>>
           8 B   0.00% ├╴u: usize
           8 B   0.00% ├╴n: usize
           8 B   0.00% ├╴l: usize
  75_000_048 B  64.08% ├╴low_bits: sux::bits::bit_field_vec::BitFieldVec
  75_000_024 B  64.08% │ ├╴data: alloc::vec::Vec<usize>
           8 B   0.00% │ ├╴bit_width: usize
           8 B   0.00% │ ├╴mask: usize
           8 B   0.00% │ ╰╴len: usize
  42_041_160 B  35.92% ╰╴high_bits: sux::rank_sel::select_zero_fixed2::SelectZeroFixed2<sux::rank_sel::select_fixed2::SelectFixed2>
  35_937_608 B  30.71%   ├╴bits: sux::rank_sel::select_fixed2::SelectFixed2
  32_031_296 B  27.37%   │ ├╴bits: sux::bits::bit_vec::CountBitVec
  32_031_280 B  27.37%   │ │ ├╴data: alloc::vec::Vec<usize>
           8 B   0.00%   │ │ ├╴len: usize
           8 B   0.00%   │ │ ╰╴number_of_ones: usize
   3_906_312 B   3.34%   │ ╰╴inventory: alloc::vec::Vec<u64>
   6_103_552 B   5.21%   ╰╴inventory: alloc::vec::Vec<u64>
```

## Acknowledgments

This software has been partially supported by project SERICS (PE00000014) under the NRRP MUR program funded by the EU - NGEU,
and by project ANR COREGRAPHIE, grant ANR-20-CE23-0002 of the French Agence Nationale de la Recherche.
