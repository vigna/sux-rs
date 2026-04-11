# `sux`

[![crates.io](https://img.shields.io/crates/v/sux.svg)](https://crates.io/crates/sux)
[![docs.rs](https://docs.rs/sux/badge.svg)](https://docs.rs/sux)
[![rustc](https://img.shields.io/badge/rustc-1.85+-red.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![CI](https://github.com/vigna/sux-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/vigna/sux-rs/actions)
![license](https://img.shields.io/crates/l/sux)
[![downloads](https://img.shields.io/crates/d/sux)](https://crates.io/crates/sux)
[![coveralls](https://coveralls.io/repos/github/vigna/sux-rs/badge.svg?branch=main)](https://coveralls.io/github/vigna/sux-rs?branch=main)

A pure Rust implementation of succinct and compressed data structures.

This crate started as part of the [Sux] project; it contains also code ported
from [the DSI Utilities] and new structures.

## Highlights

- Support for 32-bit and 64-bit architectures;
- [bit vectors and bit-field vectors];
- several structures for [rank and selection] with different tradeoffs;
- [indexed dictionaries], including an implementation of the popular [Elias–Fano
  representation of monotone sequences], and [lists of strings compressed by
  prefix omission].
- new state-of-the-art structures for [static functions] and [static filters],
  scaling to trillions of keys, and providing very fast queries;
- [monotone minimal perfect hash functions] based on longest common prefixes;
- support for [signed minimal perfect hash functions];
- [partial arrays], that is, “arrays with holes”, implemented using ranking or
  Elias–Fano;
- [lists] (e.g., [compressed lists of integers]);
- basic structures for [balanced parantheses].

The focus is on performance (e.g., there are unchecked versions of all methods
and support for [unaligned access]) and on flexible composability (e.g., you can
fine-tune your [`EliasFano`] instance by choosing different types of internal
indices, and whether to index zeros or ones). Whenever possible, there are
mapping methods that replace an underlying structure with another one, provided
it is compatible.

This crate does not provide high-level genericity on bit vectors: [operations on
bit vectors] are based on a word type `W`, on the [`BitLength`] trait, which
provides the bit length, and on the traits [`AsRef<W>`]/[`AsMut<W>`], which
provide concrete access to the underlying data. This approach makes it possible
to use any structure that implements these traits as a bit vector, and to
implement your own bit vector if you need specific features (e.g., support for
unaligned access). Additionally, we support [`Index<usize, Output = bool>`] for
bit vectors, so you can read them as you would read a slice of `bool`.

Rank/select structures are built on backends implementing the [`Backend`] trait,
providing via an associated type the word type, and on the traits above;
moreover, the strutures export the same traits by delegation, so you can use any
structure as a backend for (further) rank/select structures, making arbitrary
nesting of structures simple, typesafe and zero-cost. The same is true of
structures for balanced parentheses. All structures [`Deref`] to their backends,
too.

## Slice-by-value support

Wherever possible, we support the “slice by value” traits from the
[`value-traits`] crate, which make it possible to treat structures such as
bit-field vectors or succinct representations in a manner similar to slices.
This approach is also used to make [vectors of bit fields] and slices (where the
bit width of the fields is exactly that of the word type) interchangeable.

## Support for unaligned access

The [`TryIntoUnaligned`] trait provides a [`try_into_unaligned`] method that can
be used to convert recursively all components of a structure into components
using unaligned access to retrieve bit sequences. This approach makes unaligned
access typesafe.

## ε-serde support

All structures in this crate are designed to work well with [ε-serde]: in
particular, once you have created and serialized them, you can easily map them
into memory or load them in memory regions with specific `mmap()` attributes.
Support for ε-serde is provided by the feature `epserde`, and support for
memory mapping in ε-serde is provided by the `mmap` feature.

## serde support

All structures in this crate support serialization with [serde]. Support is
gated by the feature `serde`.

## `MemDbg`/`MemSize` support

All structures in this crate support the [`MemDbg`] and [`MemSize`] traits from
the [`mem_dbg`] crate, which provide convenient facilities for inspecting memory
usage and debugging memory-related issues. For example, this is the output of
`mem_dbg()` on an [`EliasFano`] instance:

```text
11.15 MB 100.00% ⏺: sux::dict::elias_fano::EliasFano<sux::rank_sel::select_zero_adapt_const::SelectZeroAdaptConst<sux::rank_sel::select_adapt_const::SelectAdaptConst<sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>>>>
7.500 MB  67.24% ├╴low_bits: sux::bits::bit_field_vec::BitFieldVec<usize, alloc::boxed::Box<[usize]>>
7.500 MB  67.24% │ ├╴bits: alloc::boxed::Box<[usize]>
    8  B   0.00% │ ├╴bit_width: usize
    8  B   0.00% │ ├╴mask: usize
    8  B   0.00% │ ╰╴len: usize
3.654 MB  32.76% ├╴high_bits: sux::rank_sel::select_zero_adapt_const::SelectZeroAdaptConst<sux::rank_sel::select_adapt_const::SelectAdaptConst<sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>>>
3.379 MB  30.29% │ ├╴bits: sux::rank_sel::select_adapt_const::SelectAdaptConst<sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>>
3.203 MB  28.72% │ │ ├╴bits: sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>
3.203 MB  28.72% │ │ │ ├╴bits: alloc::boxed::Box<[usize]>
    8  B   0.00% │ │ │ ╰╴len: usize
175.8 kB   1.58% │ │ ├╴inventory: alloc::boxed::Box<[usize]>
   16  B   0.00% │ │ ╰╴spill: alloc::boxed::Box<[usize]>
274.7 kB   2.46% │ ├╴inventory: alloc::boxed::Box<[usize]>
   16  B   0.00% │ ╰╴spill: alloc::boxed::Box<[usize]>
    8  B   0.00% ├╴n: usize
    8  B   0.00% ├╴u: usize
    8  B   0.00% ╰╴l: usize
```

## Binaries

A few binaries make it possible to build and serialize structures with ε-serde
(e.g., `rcl`, `vfunc`, and `vfilter`). Moreover, there are examples benchmarking
the structures (e.g., `bench_rear_coded_list`, `bench_vfunc`, and
`bench_vfilter`). You have to use the feature `cli` to build them.

## Features

The crate has the following features:

- `rayon`: enables support for parallel iterators using the `rayon` crate
  (default);
- `flate2`: enables support for gzip-compressed files in the [`lenders`] module
  (default);
- `zstd`: enables support for zstd-compressed files in the [`lenders`] module
  (default);
- `deko`: enables support for the [`deko`] crate, which provides dynamic
  detection of compressed files for the [`lenders`] module;
- `epserde`: enables support for [ε-serde];
- `serde`: enables support for [serde];
- `clap`: enables the `clap` crate for command-line argument parsing;
- `cli`: builds the binaries (implies `clap`, `epserde`, `deko`);
- `mmap`: enables support for memory mapping in ε-serde (implies `epserde`);
- `aarch64_prefetch`: enables prefetch support on aarch64 (requires nightly).

Note: The [`MemDbg`] and [`MemSize`] traits from the [`mem_dbg`] crate are
always available as `mem_dbg` is a required dependency.

## Benchmarks

A few benchmarks are available in the `benches` directory. The ones starting with
`bench_` can be just run as usual; for example,

```bash
cargo bench --bench bench_vfunc
```

The `sux` benchmark, which tests rank and select structures, is instead a CLI
command with options. Try

```bash
cargo bench --bench sux -- --help
```

to see the available tests. For example, with

```bash
cargo bench --bench sux -- Rank9 -d 0.5 -l 100000,1000000,10000000
```

you can test the [`Rank9`] structure with a density of 0.5 on a few bit sizes.
Afterwards, you can generate an SVG plot and CSV
data in the `plots` directory with

```bash
./python/plot_benches.py --benches-path ./target/criterion/ --plot-dir plots
```

You can then open the `plots/plot.svg` with a browser to see the results, or
inspect the directory `csv` for CSV data. Note that as you run benchmarks, the
results will cumulate in the `target/criterion` directory, so you can generate
plots for multiple runs.

By specifying multiple structures (using also substring matching), you can
compare the behavior of different structures. For example,

```bash
cargo bench --bench sux -- SelectSmall SelectAdapt0 -d 0.5 -l 100000,1000000,10000000
```

will test all variants of [`SelectSmall`] against a [`SelectAdapt`] with one (2⁰)
`u64` per subinventory. The plot will highlight the differences in performance:

```bash
./python/plot_benches.py --benches-path ./target/criterion/ --plot-dir plots
```

## Acknowledgments

This software has been partially supported by project SERICS (PE00000014) under
the NRRP MUR program funded by the EU - NGEU, and by project ANR COREGRAPHIE,
grant ANR-20-CE23-0002 of the French Agence Nationale de la Recherche. Views and
opinions expressed are however those of the authors only and do not necessarily
reflect those of the European Union or the Italian MUR. Neither the European
Union nor the Italian MUR can be held responsible for them.

[bit vectors and bit-field vectors]: https://docs.rs/sux/latest/sux/bits/index.html
[rank and selection]: https://docs.rs/sux/latest/sux/rank_sel/index.html
[indexed dictionaries]: https://docs.rs/sux/latest/sux/traits/indexed_dict/index.html
[`EliasFano`]: https://docs.rs/sux/latest/sux/dict/elias_fano/struct.EliasFano.html
[ε-serde]: https://crates.io/crates/epserde
[`MemDbg`]: https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemDbg.html
[`MemSize`]: https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemSize.html
[`mem_dbg`]: https://crates.io/crates/mem_dbg
[Elias–Fano representation of monotone sequences]: https://docs.rs/sux/latest/sux/dict/elias_fano/struct.EliasFano.html
[lists of strings compressed by prefix omission]: https://docs.rs/sux/latest/sux/dict/rear_coded_list/
[Sux]: https://sux.di.unimi.it/
[the DSI Utilities]: https://dsiutils.di.unimi.it/
[`BitLength`]: https://docs.rs/sux/latest/sux/traits/rank_sel/trait.BitLength.html
[`Rank9`]: https://docs.rs/sux/latest/sux/rank_sel/struct.Rank9.html
[`SelectSmall`]: https://docs.rs/sux/latest/sux/rank_sel/struct.SelectSmall.html
[`SelectAdapt`]: https://docs.rs/sux/latest/sux/rank_sel/struct.SelectAdapt.html
[static functions]: https://docs.rs/sux/latest/sux/func/vfunc/struct.VFunc.html
[monotone minimal perfect hash functions]: https://docs.rs/sux/latest/sux/func/lcp_mmphf/index.html
[static filters]: https://docs.rs/sux/latest/sux/dict/vfilter/struct.VFilter.html
[partial arrays]: https://docs.rs/sux/latest/sux/array/struct.PartialArray.html
[opertions on bit vectors]: https://docs.rs/sux/latest/sux/traits/bit_vec_ops/index.html
[unaligned access]: https://docs.rs/sux/latest/sux/traits/trait.TryIntoUnaligned.html
[`value-traits`]: https://crates.io/crates/value-traits
[serde]: https://crates.io/crates/serde/
[`deko`]: https://crates.io/crates/deko/
[`lenders`]: https://docs.rs/sux/latest/sux/utils/lenders/
[signed minimal perfect hash functions]: https://docs.rs/sux/latest/sux/func/signed/struct.SignedVFunc.html
[lists]: https://docs.rs/sux/latest/sux/list/index.html
[compressed lists of integers]: https://docs.rs/sux/latest/sux/list/comp_int_list/struct.CompIntList.html
[`AsRef<Backend:Word>`]: https://doc.rust-lang.org/core/convert/trait.AsRef.html
[`AsMut<Backend:Word>`]: https://doc.rust-lang.org/core/convert/trait.AsMut.html
[`AsRef<W>`]: https://doc.rust-lang.org/core/convert/trait.AsRef.html
[`AsMut<W>`]: https://doc.rust-lang.org/core/convert/trait.AsMut.html
[`Word`]: https://docs.rs/sux/latest/sux/traits/trait.Backend.html#associatedtype.Word
[`Backend`]: https://docs.rs/sux/latest/sux/traits/trait.Backend.html
[balanced parantheses]: https://docs.rs/sux/latest/sux/bal_paren/index.html
[`Index<usize, Output = bool>`]: https://doc.rust-lang.org/core/ops/index/trait.Index.html
[operations on bit vectors]: https://docs.rs/sux/latest/sux/traits/bit_vec_ops/index.html
[`TryIntoUnaligned`]: https://docs.rs/sux/latest/sux/traits/trait.TryIntoUnaligned.html
[`try_into_unaligned`]: https://docs.rs/sux/latest/sux/traits/trait.TryIntoUnaligned.html#tymethod.try_into_unaligned
[`Deref`]: https://doc.rust-lang.org/core/ops/deref/trait.Deref.html
