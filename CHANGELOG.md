# Change Log

## [0.15.3] - unreleased

### Improved

- `PHastR` construction is slightly faster: the scan finding the keys of the
  buckets without a seed prefetches the words of the bit vector of the buckets.

## [0.15.2] - 2026-10-14

### New

- `PHastRBuilder::sweep_chunks` sets the maximum number of chunks in which
  a level of a `PHastR` is swept in parallel, which by default is the number
  of threads of the current rayon pool: setting it makes the function built
  independent of the size of the pool. The `phastr` CLI utility has a
  corresponding `--sweep-chunks` option.

- `PHastRBuilder::temp_dir` and `VBuilder::temp_dir` set the directory of
  the temporary files of offline construction, which by default is the
  default temporary directory; the new function `sig_store::new_offline_in`
  creates an offline signature store in a given directory. The CLI
  utilities building structures offline have a corresponding `--temp-dir`
  option.

## [0.15.1] - 2026-10-14

### Improved

- `PHastR` queries with the default parameters extract the in-slice offset
  of the layout of a key with a load, rather than with a variable shift
  whose count depends on the seed (and thus on a cache miss), which reduced
  memory-level parallelism: on a billion keys, queries are about 15% faster
  on AMD Zen 5, and 10% faster on Intel Rocket Lake. The structure is
  unchanged.

- The documentation of `PHastR` cites the paper describing it.

## [0.15.0] - 2026-10-12

### New

- New `PHastR` minimal perfect hash function, based on bucket placement with
  rotated layouts, providing very fast construction and queries using less
  than two bits per key. It can be built from a slice (hashing keys in
  parallel) or from a lender, in memory or offline (keeping the hashes of
  the keys in temporary files).

- `SignedFunc` can wrap a `PHastR`, storing as verification hash a remix of
  the signature of each key; the new `phastr` CLI utility builds (possibly
  signed) `PHastR` functions.

- New `CompVFunc` compressed static function, using the algorithmic
  ideas of `VFunc` but representing data using the zero-th order entropy
  (i.e., values are represented by a Huffman code).

- New `vfunc2` and `comp_vfunc` CLI utilities, and a `--shard-edge` option
  of the utilities building functions and filters selecting the
  `ShardEdge` implementation.

- New `RearCodedListBuilder::try_push` method returning an error, instead of
  panicking, when the strings of a sorted list are out of order.

- New `VBuilder::retry_prob` setter (and `--retry-prob` CLI option) bounding
  the probability that a construction attempt must be retried because of
  duplicate hyperedges; the amount of sharding is limited accordingly. The
  default is 0.05, the previously hardwired value.

- New data-access parallel constructors starting from a slice, which are
  used by default by all CLI utilities; a switch reverts to the classical
  streaming constructors.

- New `Pred::rank` method, implemented by default using `Pred::pred`, but
  implemented much more efficiently for `EliasFano`.

- Several new trait implementations for `BitVec`, such as `SliceByValue`.

- All structures support `MemSize`.

- New `ParallelWithLen` extension trait setting both `with_min_len` and
  `with_max_len` in Rayon.

- Default `BitLength::is_empty` method.

- New `BuildError::MismatchedKeysAndValues` returned by `VFunc`, `VFunc2`,
  and `CompVFunc` construction when the number of values does not match
  the number of keys: slice-based constructors check the lengths exactly;
  lender-based constructors return an error if the values are exhausted
  before the keys (`CompVFunc`, which counts values beforehand, detects
  also the symmetric case). Previously, mismatches would cause a panic or
  be silently ignored.

- New `BitFieldVec::wrap` method to create a bit-field vector with a
  given backend.

- New opt-in `--check` flag for the `vfunc2` and `comp_vfunc` CLI utilities
  that verifies that the number of keys and values match.

- New `DensePartialArray` and `SparsePartialArray` type aliases (e.g., for
  ε-serde deserialization), and `PartialArrayBuilder::dense`/`sparse`
  constructors.

- Atomic and non-atomic bit vectors and bit-field vectors convert between
  any owned backends (`Vec`/`Box`) in both directions, and `AtomicBitVec`
  and `AtomicBitFieldVec` convert between `Vec` and `Box` backends.

- `bit_field_vec!` accepts an explicit word type (`W:` forms), and
  `bit_vec![expr; n]` accepts any expression.

- `SliceByValue`/`SliceByValueMut` are re-exported by
  `sux::traits::bit_field_slice`, and `value_traits` by `sux::traits`;
  `BitFieldVec::bit_width` and `BitFieldVec::len` are inherent for every
  backend, and `AtomicBitFieldVec::len` is inherent.

- The prelude includes `BitVecOps`, `BitVecOpsMut`, `AtomicBitVecOps`,
  `TryIntoUnaligned`, `Unaligned`, `Word`, `Backend`, and the lenders
  `FromSlice`, `FromCloneableIntoIterator`, and `LineLender`.

- New `seed()` accessors for `VFunc`, `VFunc2`, `VFilter`, `CompVFunc`, and
  the functions based on longest common prefixes.

- New `cli` helpers shared by the binaries: `store` (which logs the type of
  the stored structure), `KeyBuffer`, `ValueArgs`, `count_lines`, `bench`,
  and `gen_sorted_keys`.

### Fixed

- `BitVec::rank_hinted` no longer overflows in debug mode when the bound on
  the number of words to scan is `usize::MAX`.

- `FromIntoIterator` and `JacobsonBalParen` no longer fail to compile with
  an overflow evaluating `&_: IntoIterator` when the `objc2` crate is in the
  dependency graph (e.g., through `sysinfo` 0.39 on macOS), as their type
  parameters are now fixed before trait selection.

- Unsound reference-based `From` implementations between `BitVec`,
  `AtomicBitVec`, etc. have been removed.

- The safe predecessor methods of `EliasFano` (`pred`, `iter_from_pred`,
  `iter_back_from_pred`, `iter_bidi_from_pred`, and their strict variants)
  no longer cause undefined behavior when called with a value greater than
  the upper bound: they now return the last element, like `pred_strict`
  already did.

- The conversions between owned atomic and non-atomic backends no longer
  reuse the allocation when the alignment of the atomic type differs from
  that of its value type (e.g., `AtomicU64` vs. `u64` on 32-bit x86), as
  deallocating with a different alignment would be undefined behavior: they
  return an error containing the input instead. The check is resolved at
  compile time, so on targets with equal alignments the code is unchanged.

- `EliasFanoConcurrentBuilder::build` no longer panics on targets where the
  alignment of `Atomic<V>` differs from that of `V`, such as 32-bit x86 with
  `V` = `u64`: as the allocation of the low bits cannot be reused, it is now
  copied word by word.

- The temporary directory of an offline `SigStore` is now moved into the
  shard store and removed when the latter is dropped, after the bucket
  files have been closed; previously it was removed while the files were
  still open, which fails silently on Windows.

- Fixed bug in `Select9` affecting very sparse vectors.

- `BitVec::append` and `BitVec::append_value` now clear the last word before
  storing data in it.

- Fixed subtraction overflow when appending to a vector with excess capacity.

- Hardening pass over the safe query and access paths: they now use checked
  arithmetic and mask dirty padding bits, so structures built from unsanitized
  or deserialized input no longer panic or return spurious rank/select
  results. This covers the select builders (stale backing words), zero-width
  and full-word `BitFieldVec`, and `BitVec` length arithmetic.

- `VFunc2`, `CompVFunc`, and the LCP MMPHFs handle their edge cases losslessly
  and without panicking: wide (`u128`) and 32-bit values, absent-key queries,
  and LCP lengths (now stored as `u32` on all targets, and reported as errors
  when oversized).

- Builders and CLI utilities validate their input instead of panicking,
  dividing by zero, or looping forever: the parallel `VBuilder`, the
  `EliasFano` concurrent builder, and the Jacobson, `PartialArray`,
  `PrefixSumIntList`, `RearCodedListBuilder`, and `FairChunks` constructors,
  as well as several binaries.

- `Modulo2System::gaussian_elimination` accepts redundant identity rows and
  returns an error, instead of panicking, on an unsolvable system.

- Atomic bit-vector updates (`set_atomic`, `fill`, `flip`) use correct
  read-modify-write orderings.

- `VFunc2` honors all `VBuilder` options (offline mode, `check_dups`,
  `low_mem`, `retry_prob`) when building its internal functions; previously,
  only the number of threads and ε were forwarded.

- Structures based on a `ShardEdge` (including `LcpMmphf<str>`) can be
  deserialized with serde: the edge types did not implement `Deserialize`.

- A shard that cannot be solved after 100 attempts makes construction return
  `BuildError::UnsolvableShard` instead of panicking, and too large shards
  return the new `BuildError::MaxShardTooBig`.

- The construction of functions and filters based on `Fuse3NoShards` and
  `Fuse3Shards` no longer fails at some sizes below 100,000 keys: the
  rounding of the number of segments could leave the graph below the
  peelability threshold, so that almost all attempts failed (e.g., between
  11,464 and 11,521 keys up to 40% of the constructions returned
  `BuildError::UnsolvableShard`). The same happened to the default
  `LcpMmphf` around 184,000 keys, as its second function uses
  `Fuse3NoShards`. The number of segments is now increased, if necessary,
  to reach an empirical peelability threshold, and the ad hoc corrections
  to the expansion factor at 12,318–12,371 and 37,435–37,453 keys have been
  removed.

- The construction of `CompVFunc` no longer fails on distributions with
  rare values (e.g., 1% of escaped values at one million keys): as the edges
  of a key are correlated, the graph is now sized taking into account the
  distribution of codeword lengths, and each failed attempt makes the next
  graph sparser, so that construction eventually succeeds.

- Constructions using lazy Gaussian elimination (`FuseLge3Shards` and
  `FuseLge3FullSigs` below 800,000 keys) allow up to 1000 attempts: on some
  key sets of a few dozen keys more than 95% of the attempts fail because of
  duplicate edges, so the limit of 100 attempts was reached in about 1% of
  the constructions.

- `Mwhc3NoShards` (`mwhc` feature) generated correlated edges, as the third
  vertex was derived from the XOR of the words generating the first two, and
  with two keys it always generated duplicate edges, so construction failed.

- The construction of fuse-graph-based structures with huge key sets
  (e.g., `FuseLge3FullSigs` beyond a few dozen trillion keys) could create
  shards needing more than 2³² vertices, failing only after hashing all keys;
  the number of shards is now bounded from below, and construction fails
  immediately if no admissible number of shards exists.

- Structures with unsized keys, such as `VFunc<str>`, `CompVFunc<str>`,
  `VFilter<str>`, `LcpMmphfStr`, `Lcp2MmphfStr`, `RearCodedListStr`, and
  `MappedRearCodedListStr`, implement `Clone`.

- `MappedRearCodedListStr` can be deserialized with serde.

- The construction of `SelectSmall`, `SelectZeroSmall`, and `Select9` no
  longer overflows on 32-bit targets for bit vectors with almost `usize::MAX`
  ones (or zeros).

- `SelectAdapt` and `SelectZeroAdapt` no longer allocate huge, almost empty
  subinventories on very sparse vectors with a large inventory span.

- `bit_field_vec!` evaluates its length argument once.

- Structures with substructures (`VFunc2`, `LcpMmphf`, and `Lcp2Mmphf`) no
  longer leave stale log targets on the progress logger when the
  construction of a substructure fails: previously, the targets of retried
  attempts accumulated in log messages.

- The `rcl` CLI utility reports unsorted input as an error instead of
  panicking, and `RearCodedList` serialization functions writing to a file
  remove it on error instead of leaving a truncated file.

- `bit_vec!` no longer fails to compile on lists starting with expressions
  such as `*x`, `(…)`, or `unsafe {…}`, and `bit_field_vec!` evaluates each
  element once.

- Dense partial arrays deserialized with ε-serde (ε-copy or memory mapping)
  provide `len`, `get`, and `SliceByValue`, like sparse ones.

- The adaptive selection structures no longer trigger a debug assertion on
  bit vectors with dirty padding bits.

- Signing a `VFunc` built offline streams its shards from disk instead of
  loading all of them in memory.

### Improved

- Structured logging for structures with substructures.

- `BitVecOps::iter_ones` and `BitVecOps::iter_zeros` stop at the length of
  the bit vector rather than scanning the whole backend.

- Error messages of lenders opening files now include the path, and those of
  the Elias–Fano builders the offending values.

- Signing is parallel, and possibly from the store, when possible; in
  particular, `SignedFunc` constructors with `BitFieldVec` hashes sign in
  parallel (on platforms where the hash word type is less aligned than its
  atomic counterpart, such as `u64` on 32-bit x86, through a temporary
  atomic copy).

- `PHastR` in-memory construction builds the remapping of the later levels
  from bitmaps of used slots, as offline construction does, instead of
  keeping a vector of eight bytes per slot.

- `SelectSmall` and `SelectZeroSmall` constructors infer their const
  parameters from the wrapped structure, as in
  `SelectSmall::new(rank_small![bits])`.

- Documentation: guides for choosing a structure in the README and in the
  `func`, `rank_sel`, `bits`, `list`, and `array` modules, inline item lists
  in module pages, search aliases (e.g., `mphf`, `bloom`, `elias fano`),
  and examples showing their imports.

- Fewer allocations and less build-time work: buffer reuse in
  `RearCodedListStr::get_in_place`, capacity reservation in the `Extend`
  implementations, lazy `FilteredShardStore` iteration, and a restored early
  exit in the adaptive select builders.

- `JacobsonBalParen` pioneer construction is linear rather than quadratic on
  deeply nested parentheses, with 4.30x to 14.75x speedups at 4,096 to 16,384
  pairs on the benchmark host.

- Signed LCP and two-step LCP queries reuse the key signature for rank lookup
  and verification instead of hashing byte-sequence keys twice.

### Changed

- `ShardEdge::set_up_corr_graphs` has two new arguments: a statistic of the
  correlation of the edges of each key, and the number of previous failed
  construction attempts.

- `ShardEdge` is now an unsafe trait, as the builders rely on its
  implementations to generate indices within bounds and to provide a sort
  type with the same layout as `SigVal`; `FuseLge3NoShards` has been
  removed.

- The sequential constructors of `VFunc`, `VFunc2`, `CompVFunc`, `VFilter`,
  and of the corresponding signed structures no longer take the number of
  keys as a parameter.

- The `no_logging` feature has been removed.

- The `func::vfunc2` module is now private (`VFunc2` is still re-exported by
  `func`).

- The functions `transmute_(vec|boxed_slice)_(from|into)_atomic` return an
  error, rather than copying, when the alignment of the target type cannot be
  satisfied; since they consume their input, the error (the new generic
  `DifferentAlignmentError`) returns it, so that the caller can fall back to
  copying. Correspondingly, most conversion methods on bit(-field) vectors are
  now fallible (e.g., `TryFrom`). The new error types `InsufficientAlignmentError`
  and `AlignmentInfo` describe the alignment problem, and
  `CannotCastToAtomicError` has been removed.

- `CompIntList::into_inner` returns a pair `(B, D)` of backends.

- `BitVec::get_value_unaligned` and `BitVec::get_value_unaligned_unchecked`
  are now methods of `BitVecOps`.

- Iterators from successors and predecessors in indexed dictionaries have been
  extracted into the separate traits `SuccIter` and `PredIter`.

- `BitVec::to_owned` returns a `BitVec` with a `Vec` backend for every
  backend; previously, for borrowed backends it returned a borrowed copy.

- `BitVec::replace_value` and `BitVec::apply_in_place` panic if a value is
  not zero or one, as `BitVec::set_value` and width-one `BitFieldVec`s do;
  previously, any nonzero value set the bit.

- `Mwhc3NoShards` (`mwhc` feature) generates edges differently, so
  structures based on it serialized by previous versions must be rebuilt (see
  Fixed).

- The `mmap` feature implies `epserde`, and `mem_usage` requires only the
  `clap` feature.

- Dependencies: `lender` 0.8, `value-traits` 0.3, `epserde` 0.13.2,
  `serde_arrays` 0.2, and `tempfile` at least 3.11. With `epserde` 0.13.2
  the phantom key types of `VFunc`, `CompVFunc`, and `PHastR` are
  function-pointer markers, so their `Send` and `Sync` implementations are
  derived automatically, rather than being unsafe manual implementations.

- `IntoBackIteratorFrom::into_iter_back_from(from)` now excludes position
  `from`, yielding first the element of position `from` – 1, like
  `EliasFano::iter_back_from`, `IntoUncheckedBackIterator`,
  `BidiIterator::prev`, and `values[..from].iter().rev()` on slices.
  Previously, it included position `from`; code relying on the old behavior
  must pass `from + 1`.

- Sparse partial arrays no longer store the redundant end of their index, so
  their serialized layout has changed.

- CLI utilities and examples no longer detect compressed input files
  automatically: files are read as they are, unless a format is specified
  with `--decompress` (`gzip`, `zstd`, or `deko` for detection using the
  `deko` crate). The new `AnyLineLender` in `utils::lenders` (re-exported by
  `cli`) lends the lines of a file decompressed in a format chosen at
  runtime, and `cli::reservoir_sample` takes the format as an argument;
  `cli::read_concat_lines` and `cli::str_slice_from_offsets` are replaced by
  `cli::KeyBuffer`. The `cli` feature now enables the `flate2` and `zstd`
  features.

- CLI utilities follow a single rule for `-n` with `-f`: the first N keys of
  the file are used, and it is an error if the file has fewer keys (also in
  sequential mode, and for `lcp_mmphf`). `vfilter -u` with 8, 16, 32, or
  64-bit hashes warns instead of failing. `HashTypes` no longer implements
  `Display`, and `bench_phastr` has no short options.

- `SignedFunc` constructors with `BitFieldVec` hashes require the hash word
  type to implement `AtomicPrimitive` (so `u128` is no longer accepted).

- `BuildError::ValueTooLarge` has been removed. `SolveError`, the
  `VBuilder::shard_size_hint` setter, `VBuilder::try_build_func_with_store`,
  and the fields of `BlockInfo` are no longer public.

- `SignableFunc` provides the verification hash of a signature
  (`remixed_hash`) instead of the shard edge, so that functions without a
  shard edge can be signed.

- Field naming and logging have been reorganized. Serialized structures
  will be incompatible.

- `init_env_logger` is now in the `cli` module, gated by the `cli` feature
  (the crate-root path is kept as a hidden re-export), and the `env_logger`
  and `jiff` dependencies are now optional, enabled by that feature.

- Thanks to the new ε-serde, `VFilter` does not have `F` anymore
  as parameter. Moreover, other structures such as `VFunc2` will
  actually map into memory their bulk data.

- `BitVecValueOps::get_value`/`get_value_unchecked` are renamed
  to `get_bits`/`get_bits_unchecked`, to avoid clashing with the index-based
  `SliceByValue` methods of the same name.

- `UnalignedConversionError` is now a structured
  (`non_exhaustive`) enum instead of a wrapper around a `String`.

- `FairChunks::new_with` is now `unsafe`: the caller must uphold
  its documented contract (a nondecreasing cumulative weight function,
  `max_weight` equal to its last element, and `num_weights` one less than the
  length), which the constructor no longer checks.

- `VFilter` rejects boxed backends wider than 64 bits and caps
  the requested filter bit width at `min(64, W::BITS)`, matching its 64-bit
  verification hash.

- The `BitCount` trait has been removed; `count_ones` and
  `count_zeros` are now provided by `BitVecOps`.

- The `cli` feature now implies `rayon`.

- The license is now `Apache-2.0 OR MIT` (it was `Apache-2.0 OR
LGPL-2.1-or-later`).

- `ShardEdge::set_up_shards` takes an additional `retry_prob` parameter, and
  implementations must not select more than `LOG2_MAX_SHARDS` shard high
  bits.

## [0.14.0] - 2026-04-11

### New

- Everything. While the goal of this crate remains the same, this is a
  major rewrite, and the API is not compatible with the previous version.
  We might release bug fixes in the 0.13 branch, but no new features. Most
  structures which were requiring to specify the underlying word type and
  a backend now just require a backend, as backends have to implement the
  `Backend` trait which provides the necessary information. We suggest to
  use your favorite AI tool to handle the transition. The new APIs emerged
  from three years of experience with Rust and this library, and hopefully
  they will be stable.

## [0.13.1] - 2026-03-07

### Changed

- Fixed `num-primitive` version at 0.2.1.

## [0.13.0] - 2026-03-07

### Changed

- Removed dependency on `common_traits`, replaced by `num-primitive`
  and the new `atomic-primitive` crate. New `SelectInWord`, `AsU128` and
  `PrimitiveUnsignedExt` traits to compensate for the removed traits.

## [0.12.2] - 2026-02-18

### Changed

- Upgraded to `lender` 0.6.0.

## [0.12.1] - 2026-02-16

### Improved

- `BitVec::resize` now fills the new bits by word, and not by bit.

- Optimized methods for bidirectional iterators on `EliasFano`.

### Fixed

- Locked `dsi-progress-logger` version to 0.8.2 for Rust 1.85.

## [0.12.0] - 2026-02-15

### New

- New `BidiIterator` trait for bidirectional iterators. Returned by new methods
  in the `Succ`/`Pred` traits, and by types implementing
  `IntoBidiIterator`/`IntoBidiIteratorFrom`. The `EliasFano` implementation is
  slower than a forward-only or backward-only iterator, but much faster than
  calling select (and it does not need selection).

- New construction methods for iterators in indexed dictionaries. You can now
  obtain (bidirectional) iterators from a position, or from a
  successor/predecessor call.

- Iterators on `EliasFano` can be reversed with a cost lower than starting
  a new one (and without any selection structure).

### Changed

- Major trait and type renaming due to the previous unfortunate usage of
  "rev"/"reverse" in the sense of "back"/"backward". `DoubleEndedIterator::rev`
  has a very specific semantics whose meaning is entirely different from what we
  were trying to express. All traits, types and methods containing "rev"/"reverse"
  now contain "back" instead. This includes iterators on `BitFieldVec`.

- Moved to `mmap-rs` 0.7.0, `mem_dbg` 0.4.0 and `epserde` 0.12.0.

- `par_count_ones` was in `BitVecMutOps` instead of `BitVecOps`.

- Removed deprecated method `AtomicBitFieldVec::reset`.

- Types implementing `Iterator` end now in `Iter` rather than `Iterator`.

- `EliasFano` implements new backward and bidirectional iterators.

- For consistency, a number of types starting with `BitFieldVector`
  now start with `BitFieldVec`.

- Removed deprecated `bit_field_vec![width; length; value]` syntax.

## [0.11.1] - 2026-02-10

### Changed

- Moved to `rand` 0.10.0.

## [0.11.0] - 2026-02-10

### New

- Adapted to `Lender` 0.5.0 (i.e., we have covariance checks).

- New method `EliasFano::into_parts`.

- New features `clap`, `zstd` and `flate2` gate the corresponding
  dependencies.

### Fixed

- Fixed broken `DoubleEndedIterator` impl for bit vectors.

- Several minor bug fixes in bounds checking.

## [0.10.3] - 2025-12-09

### New

- New `SignedVFunc` and `BitSignedVFunc` structures that store signed
  index functions.

## [0.10.2] - 2025-11-28

### New

- The `vfunc`/`vfilter` CLI utilities now use the `deko` crate to
  transparently read compressed input files. The previous `zstd`
  switch has been removed.

### Fixed

- Fixed bug that made impossible to use the CLI utilities `vfunc`
  and `vfilter` with a file as input.

- `VBuilder` was not working properly without a provided expected
  number of keys.

## [0.10.1] - 2025-11-28

### New

- `PermRearCodedList` structure that combines a `RearCodedList` with a
  permutation to provide good compression even for non-sorted lists.

### Changed

- `RearCodedList`'s ratio parameter is now named `ratio` as in Java. Thus,
  instances previously serialized will not be compatible. The `rcl` CLI utility
  has now, correspondingly, a `--ratio` parameter instead of the previous `-k`.

## [0.10.0] - 2025-11-15

### Changed

- `RearCodedList` has been significantly extended and modified. It supports now
  both slices of bytes and strings, and it has a `SORTED` parameter to indicate
  whether the list is sorted. `IndexedDict` is now implemented only for the
  sorted version. Moreover, ε-serde serialized instances can be built directly
  on disk, with minimal core-memory usage. Users of the previous version should
  adapt their code to use `RearCodedListStr`.

- The crate now sets native CPU code generation in `.cargo/config.toml`.

- `RewindableIoLender` became `FallibleRewindableLender` and it is now based
  on the `FallibleLender` trait from the `lender` crate. Several new adapters
  are available (in particular, you can `enumerate`).

## [0.9.1] - 2025-10-16

- Updated to the last version of `Lender`.

## [0.9.0] - 2025-10-16

### New

- New traits `BitVecOps`, `BitVecOpsMut`, as `AtomicBitVecOps` that
  provide bit-vector operations for types implementing `AsRef<[usize]>` and
  `BitLength`. Since all rank/select structures delegate `AsRef<[usize]>` and
  `BitLength`, by pulling in scope such traits you can use bit-vector
  operations on the underlying bit vector.

- `IndexedSeq` has now implementations for slices, vectors,
  and arrays. This should make `SliceSeq` unnecessary in most cases.

- `IntoIteratorFrom` has now implementations for (references of) slices,
  vectors, and boxed slices.

- `EliasFano` implements `SliceByValue` from the
  [`value-traits`](https://crates.io/crates/value-traits) crate. In particular,
  you can apply subslicing.

- Support for ε-serde depends now on the feature `epserde`. Support for memory
  mapping in ε-serde is provided by the `mmap` feature.

- serde support for all structures (with feature `serde`).

- New compressed, rewindable IO lenders based on the
  [`deko`](https://crates.io/crates/deko) crate. They are available with the
  `deko` feature.

- Many more implementations for `FallibleRewindableLender`.

- The structure `PartialArray` provides partial arrays (AKA “arrays with holes”)
  using ranking or Elias–Fano.

### Changed

- The associated output value for `Types` (and thus `IndexedSeq`, etc.) has now
  a lifetime to make it possible to return references (e.g., for sequences of
  strings). Bounds of the form `IndexedSeq<Input = I, Output = O>` will have to
  be rewritten as `for<'a> IndexedSeq<Input = I, Output<'a> = O>`.

- `BitFieldSlice` and `BitFieldSliceMut` have been refactored to be based
  on `SliceByValue` and `SliceByValueMut` from the
  [`value-traits`](https://crates.io/crates/value-traits) crate. This avoids
  significant duplication of intent. Unfortunately this also means that the
  previous `get` method is now called `index_value`, and `get_unchecked` is
  now called `get_value_unchecked`. Similarly, `set` is now called
  `set_value`, and `set_unchecked` is now called `set_value_unchecked`.
  As a bonus, we have now subslicing for free.

- `BitFieldSlice` has no longer blanket implementations for `AsRef<[W]>`.
  Rather, we provide implementations for slices, vectors and arrays.
  Because of this change however, now importing both `IndexedSeq` and
  `BitFieldSlice` results in `len` and `get` becoming ambiguous. Thus, now the
  prelude imports the modules `indexed_dict` and `bit_field_slice`, but not the
  traits therein. You have to manually `use indexed_dict::*` or `use
bit_field_slice::*` to use the traits.

- Most previous occurrences of `BitFieldSlice` as a trait bound (e.g., for the
  lower bits of Elias–Fano) have been replaced by `SliceByValue`, which is
  a weaker trait but provide all necessary functionality.

- The unsafe `transmute_vec` and `transmute_boxed_slice` functions have
  been replaced by four specific, safe functions
  `transmute_(vec|boxed_slice)_(from|into)_atomic` that take care of the case of
  a transmute from a non-atomic type into an atomic type with strictly greater
  alignment requirements. In this case, we create a correctly aligned copy. We
  cannot do the same for references, so in that case what was previously a
  `From` implementation has been replaced by a `TryFrom` implementation.

- The iterator on atomic bit vectors no longer takes an exclusive reference.

- The inner structure supporting the high bits of Elias–Fano is now
  a boxed slice instead of a vector, saving a word. This might cause problems
  with structures serialized before ε-serde 0.10.

- The owned inner type of an `AtomicBitVec` is now a `Box<[AtomicUsize]>`
  instead of a `Vec<AtomicUsize>`, which might require some adaptation
  in user code.

- `AtomicHelper` has been removed.

- All rank/selection structures now `Deref` to their backend (only
  `Rank9` used to do so).

### Fixed

- Adding too few values to Elias-Fano structures was causing undefined
  behavior.

## [0.8.0] - 2025-06-21

### New

- `IntoIteratorFrom` implementation for `BitFieldVec`.

- `value-traits` implementations for `BitFieldVec` and `EliasFano`.

### Fixed

- Bug in `BitFieldVec::get_unaligned` only present with `W != u64` and `W
!= usize`.

### Improved

- All build methods of `VBuilder` now take a `FallibleRewindableLender`
  of `Borrow<T>` instead of `T`, which should make construction
  more flexible.

## [0.7.4] - 2025-04-15

### Changed

- The `cli` feature is now necessary to build all binaries. It is no
  longer necessary to run `cargo bench`.

### Fixed

- `cargo bench` now builds its targets correctly.

## [0.7.3]

### New

- The `rayon` feature now gates all uses of rayon, including `VBuilder`.

- New `mmap` feature that is passed to ε-serde.

### Changed

- `BitFieldVec::get_unaligned` has now assertions for all undefined
  behaviors, which make it quite slow. Consider using
  `BitFieldVec::get_unaligned_unchecked`.

### Fixed

- Fuse-graph based constructions use now an expansion factor of 1.23 below 100
  keys. In a few cases a lower factor could lead to an infinite loop.

## [0.7.2] - 2025-03-25

### New

- Completed links to relevant papers.

## [0.7.1] - 2025-03-24

### New

- Better documentation for the `shard_edge` module, `VFunc` and `VFilter`.

## [0.7.0] - 2025-03-23

### New

- New `VFunc`/`VFilter` implementation of static functions and filters.

- New `copy` and `try_chunks_mut` methods for `BitFieldVec`.

- A `SigStore` can now be online (in-memory) or offline (on-disk).

## [0.6.0] - 2025-03-17

### New

- Updated dependencies.

- New explicit parallel methods enabled by the `rayon` feature for operations
  that were enabled with the `rayon` feature. Moreover, we use `with_min_len`
  to reduce overhead.

- New `FairChunks` struct that provides chunks of balanced weight using a
  `SuccUnchecked` structure.

## [0.5.0] - 2025-01-30

### New

- Implemented `Succ`, `Pred`, `UncheckedSucc`, `UncheckedPred` for `&T`.

- Added `#[delegatable_trait]` to the `indexed_dict` traits.

- New `FairChunks` structure providing chunks of balanced weight.

### Changed

- Trait `DivCeilUnchecked` is no longer needed. The difference in codegen
  between `a.div_ceil(b)` and `(a + b - 1) / b` on x86 compiling in release
  mode is given these two additional instructions for handling overflow:

```[asm]

cmpq $1, %rdx
sbbq $-1, %rax

```

In our code all `div_ceil` are with power-of-two values, so there is no impact.

## [0.4.7] - 2024-10-07

### New

- There is a new `SliceSeq` adapter exhibiting a reference to a slice as
  an `IndexedSeq`.

- There is a new `IntoIteratorFrom` trait that is used to write trait
  bounds for iteration from a position, similarly to what happens with
  `IntoIterator`.

## [0.4.6] - 2024-10-07

### New

- Benchmarks are now available for rank/select structures.

## [0.4.5] - 2024-10-07

### New

- New SelectZeroSmall structure, freely composable with SelectSmall (over
  RankSmall).

- From implementation for EliasFano and Extend implementation for
  EliasFanoBuilder.

## [0.4.4] - 2024-09-16

### New

- We now provide access to the slice underlying a bit-field (atomic)
  vector.

## [0.4.3] - 2024-09-05

### Improved

- Better constructors for Elias–Fano.

## [0.4.2] - 2024-08-11

### Fixed

- Removed spurious dependencies.

### New

- More `vec!`-like `bit_field_vec![w => v; n]` macro syntax.

- Added size_hint to all exact-size iterators (see
  <https://doc.rust-lang.org/std/iter/trait.ExactSizeIterator.html>).

## [0.4.1] - 2024-07-24

### New

- New method BitFieldSliceMut.apply_in_place with optimized implementation in
  BitFieldVec.

## [0.4.0] - 2024-07-21

### New

- Major rewrite.

## [0.3.2] - 2024-03-21

### Improved

- Updated epserde and mem_dbg

## [0.3.1] - 2024-03-21

### Improved

- EliasFano has now a non-strict upper bound, allowing usize::MAX
  to be representable.

## [0.3.0] - 2024-03-18

### New

- Now the binaries are built only if the "cli" feature is enabled. The goal is
  to reduce the compile time when this crate is used as a library.

### Improved

- Migrated from stderr logger to envlogger
