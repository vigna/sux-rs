# `sux-rs`

A pure Rust implementation of succinct and compressed data structures.

This crate is a work in progress: 
part of it  is a port from [Sux](https://sux.di.unimi.it/) and from [the DSI Utilities](https://dsiutils.di.unimi.it/);
new data structures will be added over time. Presently,
we provide:

- the [`BitFieldSlice`](crate::traits::bit_field_slice::BitFieldSlice) trait---an
  alternative to [`Index`](core::ops::Index) returning values of fixed bit width;
- an implementation of [bit vectors](crate::bits::BitVec) and of [vectors of bit fields of fixed with](crate::bits::BitFieldVec);
- traits for building blocks and structures like [`Rank`](crate::traits::rank_sel::Rank) , 
  [`Select`](crate::traits::rank_sel::Select), and [`IndexedDict`](crate::traits::indexed_dict::IndexedDict);
- an implementation of the [Elias--Fano representation of monotone sequences](crate::dict::elias_fano::EliasFano);
- an implementation of list of [strings compressed by rear-coded prefix omission](crate::dict::rear_coded_list::RearCodedList);
- an implementation of [static functions](crate::func::VFunc).

The focus is on efficiency (in particular, there are unchecked version of all methods) and
on flexible composability (e.g., you can fine tune your Elias–Fano instance choosing different
types of internal indices, and whether to index zeros or ones).

## ε-serde support

All structures in this crate are designed to work well with [ε-serde](https://crates.io/crates/epserde):
in particular, once you have created and serialized them, you can easily map them into memory
or load them in memory regions with specific `mmap()` attributes.
