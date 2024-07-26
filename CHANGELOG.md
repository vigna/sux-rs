# Change Log

## [0.4.2] - 2024-07-26

### New

* Added size_hint to all exact size iterators because if you do take of an exact
  size iterator that doesn't implement it it panics because of an assert in the
  default implementation of len of ExactSizeIterator.

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
