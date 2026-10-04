# Rehashing at levels after the first (64-bit collision tolerance)

Base: d2a9cb35 (`next_level` remix of (h, o)); V1/V2/V3: keys hashed again at
each level (V3, the adopted one: offsets from o ⊕ h', o of the first level
passed to the slow path). All runs single thread, pinned to core 2, GxHash.

- `query_ab.sh`/`query_ab.csv` (V1), `query_ab_v2.csv` (V2),
  `query_ab_v3.sh`/`query_ab_v3.csv` (V3): u64 keys, with reference rows.
- `str_ab_v3.sh`/`str_ab_v3.csv`: strings of 10–50 bytes (`cmpstr`).
- `build_ab.sh`/`build_ab.txt`: construction, 1 and 8 threads (V1).
- `analyze_ab.py <csv>`: medians over key seeds of the query time, raw and
  relative to `ref plus` in the same process, and the difference.

See the section "64-bit collisions" in `paper/NOTES.md`.
