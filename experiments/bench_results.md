# Construction Benchmark Results

Machine: 12th Gen Intel i7-12700KF, 67 GB RAM, 8 P-cores
Rust: 1.94.0, release mode (default target, SSE2)

## Phase breakdown (VFunc<str>, 106M URLs, 8 threads)

| Phase | Per-shard time | % of total |
|---|---|---|
| Computing signatures (xxh3) | 4.0s (sequential) | 60% |
| Sorting shards (radix sort) | ~300ms | 5% |
| Generating graph (XorGraph) | ~430ms | 7% |
| Peeling (low-mem) | ~600ms | 9% |
| Assigning values | ~700ms | 11% |

## Configuration variants

| Variant | 100M u64 | 106M URLs | 35M terms |
|---|---|---|---|
| **Default (128-bit, 8 thr)** | **4.89s** | **6.53s** | **3.86s** |
| 64-bit sigs, no shards | 22.4s | 24.7s | 6.29s |
| Offline mode | 4.73s | 6.79s | 3.93s |
| 4 threads | 6.27s | 7.70s | 3.75s |
| Force high-mem peeler | 4.82s | 6.45s | 3.78s |
| Parallel hash (try_new_parallel) | 4.57s | 4.99s | 3.46s |

## Parallel hashing phase breakdown (106M URLs)

- Parallel hash (8 threads): 347ms
- Sequential SigStore push: 2.0s
- Sort + peel + assign: ~2.6s

## High-mem vs low-mem peeler

| Keys | Threads | high/low ratio |
|---|---|---|
| 100K | 1-8 | ~1.00 |
| 1M | 1-8 | 1.00-1.26 (noisy) |
| 10M | 1-8 | 0.92-0.93 (high-mem 7-8% faster) |
| 100M | 1-2 | 0.91-0.92 (high-mem 8-9% faster) |
| 100M | 4-8 | 0.98-0.99 (roughly tied) |

## AVX2 vs SSE2

No meaningful difference at 106M keys (memory-bandwidth bound).
