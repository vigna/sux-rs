# Change Log

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
