# `sux`

[![downloads](https://img.shields.io/crates/d/sux)](https://crates.io/crates/sux)
[![dependents](https://img.shields.io/librariesio/dependents/cargo/sux)](https://crates.io/crates/sux/reverse_dependencies)
![GitHub CI](https://github.com/vigna/sux-rs/actions/workflows/rust.yml/badge.svg)
![license](https://img.shields.io/crates/l/sux)
[![](https://tokei.rs/b1/github/vigna/sux-rs?type=Rust,Python)](https://github.com/vigna/sux-rs)
[![Latest version](https://img.shields.io/crates/v/sux.svg)](https://crates.io/crates/sux)
[![Documentation](https://docs.rs/sux/badge.svg)](https://docs.rs/sux)

A pure Rust implementation of succinct and compressed data structures.

This crate is a work in progress: part of it is a port from
[Sux](https://sux.di.unimi.it/) and from [the DSI
Utilities](https://dsiutils.di.unimi.it/); new data structures will be added
over time. Presently, we provide:

- the [`BitFieldSlice`](crate::traits::bit_field_slice::BitFieldSlice) trait---an
  alternative to [`Index`](core::ops::Index) returning values of fixed bit width;
- an implementation of [bit vectors](crate::bits::BitVec) and of [vectors of bit fields of fixed with](crate::bits::BitFieldVec);
- traits for building blocks and structures like [`Rank`](crate::traits::rank_sel::Rank) ,
  [`Select`](crate::traits::rank_sel::Select), and [`IndexedDict`](crate::traits::indexed_dict::IndexedDict);
- an implementation of the [Elias--Fano representation of monotone sequences](crate::dict::elias_fano::EliasFano);
- an implementation of list of [strings compressed by rear-coded prefix omission](crate::dict::rear_coded_list::RearCodedList);
- an implementation of [static functions](crate::func::VFunc).

The focus is on efficiency (in particular, there are unchecked versions of all
methods) and on flexible composability (e.g., you can fine-tune your Elias–Fano
instance by choosing different types of internal indices, and whether to index
zeros or ones).

## ε-serde support

All structures in this crate are designed to work well with [ε-serde]: in
particular, once you have created and serialized them, you can easily map them
into memory or load them in memory regions with specific `mmap()` attributes.

## `MemDbg`/`MemSize` support

All structures in this crate support the [`MemDbg`] and [`MemSize`] traits from
the [`mem_dbg` crate], which provide convenient facilities for inspecting memory
usage and debugging memory-related issues. For example, this is the output of
`mem_dbg()` on a large [`EliasFano`] instance:

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

## Composability, functoriality, and performance

The design of this crate tries to satisfy the following principles:

- High performance: all implementations try to be as fast as possible (we
  minimize tests, cache misses, etc.).
- Composability: all structures are designed to be easily composed with each
  other: structures are built on top of other structures, which
  can be extracted with the usual `into_inner` idiom.
- Zero-cost abstraction: all structures forward conditionally all
  ranking/selection non-implemented methods on the underlying structures.
- Functoriality: whenever possible, there are mapping methods that replace an
  underlying structure with another one, provided it is compatible.

What this crate does not provide:

- High genericity: all bit vectors are based on the rather concrete trait combination
  `AsRef<[usize]>` + [`BitLength`].

For example, assuming we want to implement selection over a bit vector, we
could do as follows:

```rust
use sux::bit_vec;
use sux::rank_sel::SimpleSelect;
use sux::traits::SelectUnchecked;

let bv = bit_vec![0, 1, 0, 1, 1, 0, 1, 0];
let select = SimpleSelect::new(bv, 3);

assert_eq!(unsafe { select.select_unchecked(0) }, 1);
```

Note that we invoked [`select_unchecked`](SelectUnchecked::select_unchecked).
The [`select`](Select::select) method, indeed, requires the knowledge of the
number of ones in the bit vector to perform bound checks, and this number is not
available in constant time in a [`BitVec`]; we need a [`AddNumBits`], a thin
immutable wrapper around a bit vector that stores internally the number of ones
and thus implements the [`NumBits`] trait:

```rust
use sux::bit_vec;
use sux::bits::AddNumBits;
use sux::rank_sel::SimpleSelect;
use sux::traits::Select;

let bv: AddNumBits<_> = bit_vec![0, 1, 0, 1, 1, 0, 1, 0].into();
let select = SimpleSelect::new(bv, 3);

assert_eq!(select.select(0), Some(1));
```

Suppose instead we want to build our selection structure around a [`Rank9`]
structure: in this case, [`Rank9`] implements directly [`NumBits`], so we can
just use it:

```rust
use sux::{bit_vec, rank_small};
use sux::rank_sel::{Rank9, SimpleSelect};
use sux::traits::{Rank, Select};

let bv = bit_vec![0, 1, 0, 1, 1, 0, 1, 0];
let sel_rank9 = SimpleSelect::new(Rank9::new(bv), 3);

assert_eq!(sel_rank9.select(0), Some(1));
assert_eq!(sel_rank9.rank(4), 2);
assert!(!sel_rank9[0]);
assert!(sel_rank9[1]);

let sel_rank_small = sel_rank9.map(|x| rank_small![4; x.into_inner()]);
```

Note how [`SimpleSelect`] forwards not only [`Rank`] but also [`Index`], which
gives access to the bits of the underlying bit vector. The last line uses the
[`map`](Map::map) method to replace the underlying [`Rank9`] structure with
one that is slower but uses much less space; note how we use `into_inner()` to
get rid of the [`AddNumBits`] wrapper.

Some structures depend on the internals of others, and thus cannot be composed
freely: for example, a [`Select9`] must necessarily wrap a [`Rank9`]. In
general, in any case, we suggest embedding structure in the order rank, select,
and zero select, from inner to outer, because ranking structures usually
implement [`NumBits`].

## Acknowledgments

This software has been partially supported by project SERICS (PE00000014) under
the NRRP MUR program funded by the EU - NGEU, and by project ANR COREGRAPHIE,
grant ANR-20-CE23-0002 of the French Agence Nationale de la Recherche.

[`EliasFano`]: <https://docs.rs/sux/latest/sux/dict/elias_fano/struct.EliasFano.html>
[ε-serde]: <https://crates.io/crates/epserde>
[`MemDbg`]: <https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemDbg.html>
[`MemSize`]: <https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemSize.html>
[`mem_dbg` crate]: <https://crates.io/crates/mem_dbg>
