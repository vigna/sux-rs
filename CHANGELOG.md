# Change Log

## [0.9.0]

### New

* New traits `BitVecOps`, `BitVecOpsMut`, as `AtomicBitVecOps` that
  provide bit-vector operations for types implementing `AsRef<[usize]>` and
  `BitLength`. Since all rank/select structures delegate `AsRef<[usize]>` and
  `BitLength`, by pulling in scope such traits you can use bit-vector
  operations on the underlying bit vector.

* `IndexedSeq` has now implementations for slices, vectors,
  and arrays. This should make `SliceSeq` unnecessary in most cases.

* `IntoIteratorFrom` has now implementations for (references of) slices,
  vectors, and boxed slices.

* `EliasFano` and `BitFieldVec` implement the relevant traits from
  the [`value-traits`](https://crates.io/crates/value-traits) crate.
  In particular, you can apply subslicing to the sequences undefined
  by those structures,

* Support for ε-serde depends now on the feature `epserde`. Support for memory
  mapping in ε-serde is provided by the `mmap` feature, which is on by default.

* serde support for all structures (with feature `serde`).

* New compressed, rewindable IO lenders based on the
  [`deko`](https://crates.io/crates/deko) crate. They are available with the
  `deko` feature.

* Many more implementations for `RewindableIoLender`.

* The structure `PartialArray` provides partial arrays (AKA “arrays with holes”)
  using ranking or Elias–Fano.

### Changed

* The associated output value for `Types` (and thus `IndexedSeq`, etc.) has now
  a lifetime to make it possible to return references (e.g., for sequences of
  strings). Bounds of the form `IndexedSeq<Input = I, Output = O>` will have to
  be rewritten as `for<'a> IndexedSeq<Input = I, Output<'a> = O>`.

* `BitFieldSlice` has no longer blanket implementations for `AsRef<[W]>`.
  Rather, we provide implementations for slices, vectors and arrays.
  Because of this change however, now importing both `IndexedSeq` and
  `BitFieldSlice` results in `len` and `get` becoming ambiguous. Thus, now the
  prelude imports the modules `indexed_dict` and `bit_field_slice`, but not the
  traits therein. You have to manually `use indexed_dict::*` or `use
  bit_field_slice::*` to use the traits.

* The unsafe `transmute_vec` and `transmute_boxed_slice` functions have
  been replaced by four specific, safe functions
  `transmute_(vec|boxed_slice)_(from|into)_atomic` that take care of the case of
  a transmute from a non-atomic type into an atomic type with strictly greater
  alignment requirements. In this case, we create a correctly aligned copy. We
  cannot do the same for references, so in that case what was previously a
  `From` implementation has been replaced by a `TryFrom` implementation.

* The iterator on atomic bit vectors no longer takes an exclusive reference.

* The inner structure supporting the high bits of Elias–Fano is now
  a boxed slice instead of a vector, saving a word. This might cause problems
  with structures serialized before ε-serde 0.10.

* The owned inner type of an `AtomicBitVec` is now a `Box<[AtomicUsize]>`
  instead of a `Vec<AtomicUsize>`, which might require some adaptation
  in user code.

* `AtomicHelper` has been removed.

* All rank/selection structures now `Deref` to their backend (only
  `Rank9` used to do so).

### Fixed

* Adding too few values to Elias-Fano structures was causing undefined
  behavior.

## [0.8.0] - 2025-06-21

### New

* `IntoIteratorFrom` implementation for `BitFieldVec`.

* `value-traits` implementations for `BitFieldVec` and `EliasFano`.

### Fixed

* Bug in `BitFieldVec::get_unaligned` only present with `W != u64` and `W
  != usize`.

### Improved

* All build methods of `VBuilder` now take a `RewindableIoLender`
  of `Borrow<T>` instead of `T`, which should make construction
  more flexible.

## [0.7.4] - 2025-04-15

### Changed

* The `cli` feature is now necessary to build all binaries. It is no
  longer necessary to run `cargo bench`.

### Fixed

* `cargo bench` now build its targets correctly.

## [0.7.3]

### New

* The `rayon` feature now gates all uses of rayon, including `VBuilder`.

* New `mmap` feature that is passed to ε-serde.

### Changed

* `BitFieldVec::get_unaligned` has now assertions for all undefined
  behaviors, which make it quite slow. Consider using
  `BitFieldVec::get_unaligned_unchecked`.

### Fixed

* Fuse-graph based constructions use now an expansion factor of 1.23 below 100
  keys. In a few cases a lower factor could lead to an infinite loop.

## [0.7.2] - 2025-03-25

### New

* Completed links to relevant papers.

## [0.7.1] - 2025-03-24

### New

* Better documentation for the `shard_edge` module, `VFunc` and `VFilter`.

## [0.7.0] - 2025-03-23

### New

* New `VFunc`/`VFilter` implementation of static functions and filters.

* New `copy` and `try_chunks_mut` methods for `BitFieldVec`.

* A `SigStore` can now be online (in-memory) or offline (on-disk).

## [0.6.0] - 2025-03-17

### New

* Updated dependencies.

* New explicit parallel methods enabled by the `rayon` feature for operations
  that were enabled with the `rayon` feature. Moreover, we use `with_min_len`
  to reduce overhead.

* New `FairChunks` struct that provides chunks of balanced weight using a
  `SuccUnchecked` structure.

## [0.5.0] - 2025-01-30

### New

* Implemented `Succ`, `Pred`, `UncheckedSucc`, `UncheckedPred` for `&T`.

* Added `#[delegatable_trait]` to the `indexed_dict` traits.

* New `FairChunks` structure providing chunks of balanced weight.

### Changed

* Trait `DivCeilUnchecked` is no longer needed. The difference in codegen
  between `a.div_ceil(b)` and `(a + b - 1) / b` on x86 compiling in release
  mode is given these two additional instructions for handling overflow:

```[asm]

cmpq $1, %rdx
sbbq $-1, %rax

```

  In our code all `div_ceil` are with power-of-two values, so there is no impact.

## [0.4.7] - 2024-10-07

### New

* There is a new `SliceSeq` adapter exhibiting a reference to a slice as
  an `IndexedSeq`.

* There is a new `IntoIteratorFrom` trait that is used to write trait
  bounds for iteration from a position, similarly to what happens with
  `IntoIterator`.

## [0.4.6] - 2024-10-07

### New

* Benchmarks are now available for rank/select structures.

## [0.4.5] - 2024-10-07

### New

* New SelectZeroSmall structure, freely composable with SelectSmall (over
  RankSmall).

* From implementation for EliasFano and Extend implementation for
  EliasFanoBuilder.

## [0.4.4] - 2024-09-16

### New

* We now provide access to the slice underlying a bit-field (atomic)
  vector.

## [0.4.3] - 2024-09-05

### Improved

* Better constructors for Elias–Fano.

## [0.4.2] - 2024-08-11

### Fixed

* Removed spurious dependencies.

### New

* More `vec!`-like `bit_field_vec![w => v; n]` macro syntax.

* Added size_hint to all exact-size iterators (see
  <https://doc.rust-lang.org/std/iter/trait.ExactSizeIterator.html>).

## [0.4.1] - 2024-07-24

### New

* New method BitFieldSliceMut.apply_in_place with optimized implementation in
  BitFieldVec.

## [0.4.0] - 2024-07-21

### New

* Major rewrite.

## [0.3.2] - 2024-03-21

### Improved

* Updated epserde and mem_dbg

## [0.3.1] - 2024-03-21

### Improved

* EliasFano has now a non-strict upper bound, allowing usize::MAX
  to be representable.

## [0.3.0] - 2024-03-18

### New

* Now the binaries are built only if the "cli" feature is enabled. The goal is
  to reduce the compile time when this crate is used as a library.

### Improved

* Migrated from stderr logger to envlogger
