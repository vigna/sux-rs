# Results on Apple M1 Max (October 2026)

Machine: Apple M1 Max (8 performance + 2 efficiency cores), 64 GiB RAM,
macOS 26, Rust 1.98.0. Keys: distinct 64-bit integers; hash: XXH3-64 with seed
for both implementations. Reference: `ph` at commit `7d18454` of
`vigna/bsuccinct-rs` (branch `sux`).

These files were produced by the harness while it lived outside the
repository, so they are not in the format of `run.sh`, and there are some
differences in setup:

- the harness was built **without** `-C target-cpu=native` (on Apple silicon
  the default target CPU is already `apple-m1`, so this should be
  immaterial; on x86 it is not);
- for `S=8` the reference used `BitsFast(8)` (bit-packed) seed storage instead
  of `Bits8` (bytes); spot checks gave the same query times within noise. The
  current `cmp` uses `Bits8`;
- construction times are single runs; query times are the best of three
  averages over 10⁷ queries.

| File | Contents |
|---|---|
| `final_1e7.txt`/`.csv` | Table 1, reference rows (PHast-R rows superseded) |
| `final_1e7_r.txt` | Table 1, PHast-R rows (final code) |
| `final_1e8.txt`/`.csv` | Table 2, single-threaded construction (PHast-R rows superseded) |
| `final_1e8_r.txt` | Table 2, single-threaded, PHast-R rows (final code) |
| `final_1e8_mt.txt`/`.csv` | Table 2, 10 threads, reference rows (PHast-R rows superseded) |
| `final_1e8_mt_r.txt` | Table 2, 10 threads, PHast-R rows (final code, after the parallel-gap fix) |
| `evict_sweep.txt` | Repair sweep (patterns × candidates × depth), lab implementation, S=8 |
| `evict_bigS.txt` | Same for S=10/12 (S=12 rows are mistuned: λ too small, L too short) |
| `wt_8_d1.txt`, `wt_10_d1.txt` | Coordinate-descent tuning of the priority weights (now the sux defaults) |
