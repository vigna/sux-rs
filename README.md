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
- [indexed dictionaries][`dict`], including an implementation of the popular [Elias–Fano
  representation of monotone sequences], and [lists of strings compressed by
  prefix omission].
- new state-of-the-art structures for [static functions] and [static filters],
  scaling to trillions of keys, and providing very fast queries;
- [compressed static functions], which store values using a prefix-free code,
  and [two-step static functions], which handle frequent values separately:
  both use less space when the distribution of values is skewed;
- a [minimal perfect hash function] with very fast construction and queries,
  using less than two bits per key;
- [monotone minimal perfect hash functions] based on longest common prefixes,
  which are [the fastest available] (using less than log₂*n* space);
- support for [signed minimal perfect hash functions];
- [partial arrays], that is, “arrays with holes”, implemented using ranking or
  Elias–Fano;
- [lists] (e.g., [compressed lists of integers]);
- basic structures for [balanced parentheses].

The focus is on performance (e.g., query and access APIs commonly provide
unchecked variants, and there is support for [unaligned access]) and on flexible
composability (e.g., you can
fine-tune your [`EliasFano`] instance by choosing different types of internal
indices, and whether to index zeros or ones). Whenever possible, there are
mapping methods that replace an underlying structure with another one, provided
it is compatible.

You can run the [`vers_benchmarks`] and the [`cseq_benchmark`] benchmarks to see
the performance of different implementations on your hardware. In our tests the
structures in this crate are almost always as fast or faster than the ones in
other libraries.

This crate does not provide high-level genericity on bit vectors: [operations on
bit vectors] are based on a word type `W`, on the [`BitLength`] trait, which
provides the bit length, and on the traits `AsRef<[W]>`/`AsMut<[W]>`, which
provide concrete access to the underlying data. This approach makes it possible
to use any structure that implements these traits as a bit vector, and to
implement your own bit vector if you need specific features (e.g., support for
unaligned access). Additionally, we support [`Index<usize, Output = bool>`] for
bit vectors, so you can read them as you would read a slice of `bool`.

Rank/select structures are built on backends implementing the [`Backend`] trait,
providing via an associated type the word type, and on the traits above;
moreover, the structures export the same traits by delegation, so you can use any
structure as a backend for (further) rank/select structures, making arbitrary
nesting of structures simple, typesafe and zero-cost. The same is true of
structures for balanced parentheses. All structures [`Deref`] to their backends,
too.

## Choosing a structure

| You need… | Use |
|---|---|
| a bit vector, or a vector of fixed-width integers | [`BitVec`], [`BitFieldVec`] |
| rank and select on a bit vector | [`SelectAdapt`] over [`Rank9`] (see the [guide][rank and selection]) |
| a monotone sequence with access, successor, and predecessor | [`EliasFano`] |
| a static list of strings, with search | [`RearCodedList`] |
| a static list of integers | [`CompIntList`], [`PrefixSumIntList`] |
| an array with holes | [`PartialArray`] |
| a value for each key of a static set | [`VFunc`]; [`VFunc2`] or [`CompVFunc`] if the values are skewed |
| a bijection between a static set of keys and [0 . . *n*) | [`PHastR`] |
| the lexicographical rank of the keys of a static set | [`LcpMmphf`] |
| approximate membership in a static set | [`VFilter`], or a [`SignedFunc`] |
| balanced parentheses | [`JacobsonBalParen`] |

The [`prelude`] imports all structures and the most common traits. For
example, here is a bit vector with ranking and selection:

```rust
use sux::prelude::*;

// Rank9 requires 64-bit words, so we choose them explicitly
let bits = bit_vec![u64: 0, 1, 0, 1, 1, 0, 1, 0];
let rank_sel = SelectAdapt::new(Rank9::new(bits));

assert_eq!(rank_sel.rank(4), 2); // ones in positions [0 . . 4)
assert_eq!(rank_sel.select(2), Some(4)); // position of the third one
```

An Elias–Fano representation of a monotone sequence, with access and
successor queries (the traits of [indexed dictionaries] are not imported by
the prelude, as they would shadow some methods of slices):

```rust
use sux::prelude::{*, indexed_dict::*};

let mut efb = EliasFanoBuilder::new(4, 10_u64);
efb.extend([0, 2, 8, 10]);
let ef = efb.build_with_seq_and_dict();

assert_eq!(ef.get(2), 8);
assert_eq!(ef.succ(6), Some((2, 8))); // position and value of the successor
```

A minimal perfect hash function on a set of strings, built in parallel:

```rust
use dsi_progress_logger::no_logging;
use sux::prelude::*;

fn main() -> anyhow::Result<()> {
    let keys = ["apple", "banana", "cherry", "date"];
    let mphf = <PHastR<str>>::try_par_new(&keys, no_logging![])?;

    // The keys are mapped bijectively to [0 . . 4)
    let mut seen = [false; 4];
    for key in keys {
        seen[mphf.get(key)] = true;
    }
    assert!(seen.iter().all(|&s| s));
    Ok(())
}
```

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

Note that, both with ε-serde and with serde, the serialized form of some
structures depends on the endianness and on the pointer width of the platform
(for example, because of counters packed natively into words of type
`usize`): serialized data should be read on a platform with the same
endianness and pointer width.

## `MemDbg`/`MemSize` support

All structures in this crate support the [`MemDbg`] and [`MemSize`] traits from
the [`mem_dbg`] crate, which provide convenient facilities for inspecting memory
usage and debugging memory-related issues. For example, this is the output of
`mem_dbg()` on an [`EliasFano`] instance with ten million values:

```text
11.60 MB 100.00% ⏺: sux::dict::elias_fano::EliasFano<usize, sux::rank_sel::select_zero_adapt_const::SelectZeroAdaptConst<sux::rank_sel::select_adapt_const::SelectAdaptConst<sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>, alloc::boxed::Box<[usize]>, 11, 3>, alloc::boxed::Box<[usize]>, 11, 3>>
    8  B   0.00% ├╴n: usize
    8  B   0.00% ├╴u: usize
    8  B   0.00% ├╴l: usize
    8  B   0.00% ├╴first_val: usize
    8  B   0.00% ├╴last_val: usize
7.500 MB  64.63% ├╴low_bits: sux::bits::bit_field_vec::BitFieldVec<alloc::boxed::Box<[usize]>>
7.500 MB  64.63% │ ├╴bits: alloc::boxed::Box<[usize]>
    8  B   0.00% │ ├╴bit_width: usize
    8  B   0.00% │ ├╴mask: usize
    8  B   0.00% │ ╰╴len: usize
4.104 MB  35.37% ╰╴high_bits: sux::rank_sel::select_zero_adapt_const::SelectZeroAdaptConst<sux::rank_sel::select_adapt_const::SelectAdaptConst<sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>, alloc::boxed::Box<[usize]>, 11, 3>, alloc::boxed::Box<[usize]>, 11, 3>
3.555 MB  30.63%   ├╴bits: sux::rank_sel::select_adapt_const::SelectAdaptConst<sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>, alloc::boxed::Box<[usize]>, 11, 3>
3.203 MB  27.60%   │ ├╴bits: sux::bits::bit_vec::BitVec<alloc::boxed::Box<[usize]>>
3.203 MB  27.60%   │ │ ├╴bits: alloc::boxed::Box<[usize]>
    8  B   0.00%   │ │ ╰╴len: usize
351.6 kB   3.03%   │ ├╴inventory: alloc::boxed::Box<[usize]>
   16  B   0.00%   │ ╰╴spill: alloc::boxed::Box<[usize]>
549.4 kB   4.73%   ├╴inventory: alloc::boxed::Box<[usize]>
   16  B   0.00%   ╰╴spill: alloc::boxed::Box<[usize]>
```

## Binaries

The crate provides binaries that build structures and serialize them with
[ε-serde]; enable the `cli` feature to build them, or install them with
`cargo install sux --features cli`:

| Binary       | Builds                                                     | Notes                                                     |
| ------------ | ---------------------------------------------------------- | --------------------------------------------------------- |
| `rcl`        | a [`RearCodedListStr`] of the lines of a file              | `RearCodedListStr<false>` with `--unsorted`; `-` is stdin |
| `mrcl`       | a [`MappedRearCodedListStr`] from a stored rear-coded list | `MappedRearCodedListStr<false>` with `--unsorted`         |
| `vfunc`      | a [`VFunc`] mapping each key to its rank                   | signed with `--hash-type`                                 |
| `vfunc2`     | a [`VFunc2`]                                               | values from a file or generated                           |
| `comp_vfunc` | a [`CompVFunc`]                                            | values from a file or generated                           |
| `vfilter`    | a [`VFilter`]                                              | hashes of `--bits` bits                                   |
| `lcp_mmphf`  | an [`LcpMmphf`] (an [`Lcp2Mmphf`] with `-2`)               | keys must be sorted; signed with `--hash-type`            |
| `phastr`     | a [`PHastR`]                                               | signed with `--hash-type`                                 |
| `mem_usage`  | rank/select structures, printing their memory layout       | requires only the `clap` feature                          |

Use `--help` to see the options of each binary. Input files are read as they
are, unless a compression format is specified with `--decompress` (`gzip`,
`zstd`, or `deko`, which detects the format using the [`deko`] crate).

Signing a function with `--hash-type` yields a [`SignedFunc`]. The type of
the stored structure depends on the options (e.g., the type of the keys, the
signature, or `--unaligned`): binaries building functions and filters log the
full type of the structure they store. For example, after building a function
with `vfunc -f keys.txt func`, you can load it with

```ignore
use epserde::prelude::*;
use sux::bits::BitFieldVec;
use sux::func::VFunc;

let func = unsafe { VFunc::<str, BitFieldVec<Box<[usize]>>>::load_full("func") }?;
assert_eq!(func.get("first key"), 0);
```

or map it into memory with `mmap` (requires the `mmap` feature).

[`RearCodedListStr`]: https://docs.rs/sux/latest/sux/dict/rear_coded_list/type.RearCodedListStr.html
[`MappedRearCodedListStr`]: https://docs.rs/sux/latest/sux/dict/mapped_rear_coded_list/type.MappedRearCodedListStr.html
[`VFunc`]: https://docs.rs/sux/latest/sux/func/struct.VFunc.html
[`VFunc2`]: https://docs.rs/sux/latest/sux/func/struct.VFunc2.html
[`CompVFunc`]: https://docs.rs/sux/latest/sux/func/struct.CompVFunc.html
[`VFilter`]: https://docs.rs/sux/latest/sux/dict/vfilter/struct.VFilter.html
[`LcpMmphf`]: https://docs.rs/sux/latest/sux/func/lcp_mmphf/struct.LcpMmphf.html
[`Lcp2Mmphf`]: https://docs.rs/sux/latest/sux/func/lcp2_mmphf/struct.Lcp2Mmphf.html
[`PHastR`]: https://docs.rs/sux/latest/sux/func/struct.PHastR.html
[`SignedFunc`]: https://docs.rs/sux/latest/sux/func/signed/struct.SignedFunc.html

## Features

The crate has the following features:

- `rayon`: enables parallel construction using the `rayon` crate; it is
  required to build static functions, static filters, and monotone minimal
  perfect hash functions (default);
- `flate2`: enables support for gzip-compressed files in the [`lenders`] module
  (default);
- `zstd`: enables support for zstd-compressed files in the [`lenders`] module
  (default);
- `deko`: enables support for the [`deko`] crate, which detects dynamically
  the compression format of files read by the lenders of the [`lenders`]
  module;
- `epserde`: enables support for [ε-serde];
- `serde`: enables support for [serde];
- `clap`: enables the `clap` crate for command-line argument parsing;
- `cli`: builds the binaries (implies `clap`, `epserde`, `rayon`, `flate2`,
  `zstd`, and `deko`);
- `mmap`: enables support for memory mapping in ε-serde (implies `epserde`);
- `fuzz`: enables fuzzing support using the `arbitrary` crate;
- `mwhc`: compiles the MWHC shard/edge logic (mainly for benchmarking);
- `slow_tests`: enables slow tests (use with `--release`);
- `aarch64_prefetch`: enables prefetch support on aarch64 (requires nightly);
- `iter_advance_by`: enables support for the `advance_by` method on iterators
  (requires nightly).

Note: The [`MemDbg`] and [`MemSize`] traits from the [`mem_dbg`] crate are
always available as `mem_dbg` is a required dependency.

## Benchmarks

The examples starting with `bench_` measure the query speed of structures
built by the binaries: build a structure with a binary, and then pass it to the
example with the same type options (e.g., `--unaligned`, `--shard-edge`, or
`--bits`) and the same file of keys, from which queries are sampled. For
example,

```bash
cargo run --release --features cli --bin vfunc -- -f keys.txt func
cargo run --release --features cli --example bench_vfunc -- -f keys.txt 10000000 func
```

performs ten million queries on a [`VFunc`]. The examples `bench_vfunc`,
`bench_vfilter`, `bench_comp_vfunc`, and `bench_lcp_mmphf` work in this way;
`bench_phastr` instead builds and queries a [`PHastR`] on 64-bit keys by itself
(e.g., `cargo run --release --example bench_phastr -- 10000000`).

A few benchmarks are available in the `benches` directory. The ones starting with
`bench_` can be just run as usual; for example,

```bash
cargo bench --bench bench_vfunc
```

The `rank_sel` benchmark, which tests rank and select structures, is instead a CLI
command with options. Try

```bash
cargo bench --bench rank_sel -- --help
```

to see the available tests. For example, with

```bash
cargo bench --bench rank_sel -- Rank9 -d 0.5 -l 100000,1000000,10000000
```

you can test the [`Rank9`] structure with a density of 0.5 on a few bit sizes.
Afterwards, you can generate an SVG plot and CSV
data in the `plots` directory with

```bash
./python/plot_rank_sel.py --benches-path ./target/criterion/ --plot-dir plots
```

You can then open the `plots/plot.svg` with a browser to see the results, or
inspect the directory `csv` for CSV data. Note that as you run benchmarks, the
results will cumulate in the `target/criterion` directory, so you can generate
plots for multiple runs.

By specifying multiple structures (using also substring matching), you can
compare the behavior of different structures. For example,

```bash
cargo bench --bench rank_sel -- SelectSmall SelectAdapt0 -d 0.5 -l 100000,1000000,10000000
```

will test all variants of [`SelectSmall`] against a [`SelectAdapt`] with at most
one (2⁰) `usize` per subinventory (in `SelectAdaptN`, *N* is the base-2
logarithm of the maximum number of words per subinventory; the default
constructor of [`SelectAdapt`] corresponds to `SelectAdapt3`). The plot will highlight the differences in performance:

```bash
./python/plot_rank_sel.py --benches-path ./target/criterion/ --plot-dir plots
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
[`EliasFano`]: https://docs.rs/sux/latest/sux/dict/struct.EliasFano.html
[ε-serde]: https://crates.io/crates/epserde
[`MemDbg`]: https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemDbg.html
[`MemSize`]: https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemSize.html
[`mem_dbg`]: https://crates.io/crates/mem_dbg
[Elias–Fano representation of monotone sequences]: https://docs.rs/sux/latest/sux/dict/struct.EliasFano.html
[lists of strings compressed by prefix omission]: https://docs.rs/sux/latest/sux/dict/rear_coded_list/
[Sux]: https://sux.di.unimi.it/
[the DSI Utilities]: https://dsiutils.di.unimi.it/
[`BitLength`]: https://docs.rs/sux/latest/sux/traits/bit_vec_ops/trait.BitLength.html
[`Rank9`]: https://docs.rs/sux/latest/sux/rank_sel/struct.Rank9.html
[`SelectSmall`]: https://docs.rs/sux/latest/sux/rank_sel/struct.SelectSmall.html
[`SelectAdapt`]: https://docs.rs/sux/latest/sux/rank_sel/struct.SelectAdapt.html
[static functions]: https://docs.rs/sux/latest/sux/func/struct.VFunc.html
[compressed static functions]: https://docs.rs/sux/latest/sux/func/struct.CompVFunc.html
[two-step static functions]: https://docs.rs/sux/latest/sux/func/struct.VFunc2.html
[minimal perfect hash function]: https://docs.rs/sux/latest/sux/func/struct.PHastR.html
[monotone minimal perfect hash functions]: https://docs.rs/sux/latest/sux/func/lcp_mmphf/index.html
[static filters]: https://docs.rs/sux/latest/sux/dict/vfilter/struct.VFilter.html
[partial arrays]: https://docs.rs/sux/latest/sux/array/struct.PartialArray.html
[operations on bit vectors]: https://docs.rs/sux/latest/sux/traits/bit_vec_ops/index.html
[unaligned access]: https://docs.rs/sux/latest/sux/traits/trait.TryIntoUnaligned.html
[`value-traits`]: https://crates.io/crates/value-traits
[serde]: https://crates.io/crates/serde/
[`deko`]: https://crates.io/crates/deko/
[`lenders`]: https://docs.rs/sux/latest/sux/utils/lenders/
[signed minimal perfect hash functions]: https://docs.rs/sux/latest/sux/func/struct.SignedFunc.html
[lists]: https://docs.rs/sux/latest/sux/list/index.html
[compressed lists of integers]: https://docs.rs/sux/latest/sux/list/struct.CompIntList.html
[`Backend`]: https://docs.rs/sux/latest/sux/traits/trait.Backend.html
[balanced parentheses]: https://docs.rs/sux/latest/sux/bal_paren/index.html
[`Index<usize, Output = bool>`]: https://doc.rust-lang.org/core/ops/index/trait.Index.html
[`TryIntoUnaligned`]: https://docs.rs/sux/latest/sux/traits/trait.TryIntoUnaligned.html
[`try_into_unaligned`]: https://docs.rs/sux/latest/sux/traits/trait.TryIntoUnaligned.html#tymethod.try_into_unaligned
[`Deref`]: https://doc.rust-lang.org/core/ops/deref/trait.Deref.html
[vectors of bit fields]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/struct.BitFieldVec.html
[the fastest available]: https://github.com/vigna/MMPHF-Experiments
[`vers_benchmarks`]: https://github.com/Cydhra/vers_benchmarks
[`cseq_benchmark`]: https://github.com/beling/bsuccinct-rs/
[`BitVec`]: https://docs.rs/sux/latest/sux/bits/struct.BitVec.html
[`BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/struct.BitFieldVec.html
[`RearCodedList`]: https://docs.rs/sux/latest/sux/dict/rear_coded_list/struct.RearCodedList.html
[`CompIntList`]: https://docs.rs/sux/latest/sux/list/struct.CompIntList.html
[`PrefixSumIntList`]: https://docs.rs/sux/latest/sux/list/struct.PrefixSumIntList.html
[`PartialArray`]: https://docs.rs/sux/latest/sux/array/struct.PartialArray.html
[`VFunc`]: https://docs.rs/sux/latest/sux/func/struct.VFunc.html
[`VFunc2`]: https://docs.rs/sux/latest/sux/func/struct.VFunc2.html
[`CompVFunc`]: https://docs.rs/sux/latest/sux/func/struct.CompVFunc.html
[`PHastR`]: https://docs.rs/sux/latest/sux/func/struct.PHastR.html
[`LcpMmphf`]: https://docs.rs/sux/latest/sux/func/struct.LcpMmphf.html
[`VFilter`]: https://docs.rs/sux/latest/sux/dict/struct.VFilter.html
[`SignedFunc`]: https://docs.rs/sux/latest/sux/func/struct.SignedFunc.html
[`JacobsonBalParen`]: https://docs.rs/sux/latest/sux/bal_paren/struct.JacobsonBalParen.html
[`prelude`]: https://docs.rs/sux/latest/sux/prelude/index.html
[`dict`]: https://docs.rs/sux/latest/sux/dict/index.html
