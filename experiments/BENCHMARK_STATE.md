# Lcp2Mmphf One-Pass vs Two-Pass Benchmark State

## Goal
Compare construction performance of Lcp2MmphfInt between:
- **One-pass** (current HEAD, d1289ec): single pass over keys via `try_populate_and_build_with_fn` closure, LCP computed on-the-fly
- **Two-pass** (commit 38937f9): first pass computes LCP, second pass hashes keys into SigStore with packed `(lcp << log2_bs) | offset` values

Test with online and offline store, for 10M / 100M / 1B keys.

## Key Differences Between Approaches
- **Two-pass (38937f9)**: `new_with_builder` does first loop over keys computing lcp_bit_lengths + bucket_first_keys, then rewinds, calls `_try_build_func(..., keep_store=true)` packing LCP+offset into SigStore values. Then builds offsets/lcp_lengths/lcp2bucket from the retained store.
- **One-pass (HEAD)**: `try_new_with_builder` uses `try_populate_and_build_with_fn` with a stateful closure that computes LCP on-the-fly. SigStore values are just `idx`. Build phase uses separate `lcp_bit_lengths` Vec for lookups.

## What's Done
1. **Data generation**:
   - `experiments/gen_sorted_u64.rs` compiled to `experiments/gen_sorted_u64`
   - `experiments/sorted_10M.bin` (77MB) - DONE
   - `experiments/sorted_100M.bin` (763MB) - DONE
   - `experiments/sorted_1B.bin` (7.6GB) - WAS GENERATING IN BACKGROUND (task bbstkhl2b), CHECK IF COMPLETE

2. **Benchmark example** (partially written):
   - `examples/bench_lcp2_construction.rs` - written for HEAD/one-pass but needs adjustment:
     - Uses memmap2 which isn't a dependency. Switch to loading Vec<u64> from file instead, or use unsafe mmap via libc directly.
     - Alternatively, just read the whole file into a Vec<u64> — 62GB RAM available so even 8GB fits fine.

3. **Text files available**: trec.terms (35M lines, 420MB), uk-2007-05.urls (62M lines, 6.6GB)

## What's Left
1. Fix bench_lcp2_construction.rs: use std::fs::read + transmute, or add memmap2 dep
2. Build and run on HEAD for all 3 sizes, online + offline (6 runs)
3. Create git worktree at 38937f9, adapt benchmark for old API (`new_with_builder` with different args)
4. Build and run on worktree for all 3 sizes, online + offline (6 runs)
5. Collect and compare results

## System Info
- 62GB RAM, 115GB free disk, 20 cores, NVMe SSD
- IMPORTANT: Never run benchmarks in parallel (per user feedback)

## Key API Differences for Worktree
At 38937f9:
- Method is `new_with_builder` (not `try_new_with_builder`)
- `VBuilder` has `_try_build_func(..., keep_store=true)` and `try_build_func_from_store`
- `VFunc2` has `try_build_from_store` (not `try_build_from_store_with_freq`)
- Struct field: `short: Option<VFunc<...>>` (not always present)

## Git Commits Reference
- HEAD: d1289ec (one-pass, current)
- Two-pass: 38937f9 (VFunc2/Lcp2Mmphf)
- Between them: ~30 commits with substantial refactoring
