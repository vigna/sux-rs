# Change Log

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
