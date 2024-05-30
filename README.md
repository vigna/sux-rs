# `sux-rs`

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
- an implementation of list of [strings compressed by rear-coded prefix omission](crate::dict::rear_coded_list::RearCodedList);
- an implementation of [static functions](crate::func::VFunc).

The focus is on efficiency (in particular, there are unchecked versions of all methods) and
on flexible composability (e.g., you can fine-tune your Elias–Fano instance by choosing different
types of internal indices, and whether to index zeros or ones).

## ε-serde support

All structures in this crate are designed to work well with [ε-serde]:
in particular, once you have created and serialized them, you can easily map them into memory
or load them in memory regions with specific `mmap()` attributes.

## [`MemDbg`]/[`MemSize`] support

All structures in this crate support the [`MemDbg`] and [`MemSize`] traits from the
[`mem_dbg`] crate, which provide convenient facilities
for inspecting memory usage and debugging memory-related issues.

## Acknowledgments

This software has been partially supported by project SERICS (PE00000014) under the NRRP MUR program funded by the EU - NGEU,
and by project ANR COREGRAPHIE, grant ANR-20-CE23-0002 of the French Agence Nationale de la Recherche.

[ε-serde]: <https://crates.io/crates/epserde>
[`mem_dbg`]: <https://crates.io/crates/mem_dbg>
[`MemDbg`]: <https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemDbg.html>
[`MemSize`]: <https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemSize.html>
