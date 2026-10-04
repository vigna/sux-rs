# Results on Intel Xeon E-2388G (October 2026)

Machine: Intel Xeon E-2388G (Rocket Lake, 8 cores/16 threads, AVX-512),
125 GiB RAM, Linux 6.19 (Fedora 43), Rust 1.98.1, `-C target-cpu=native`
(verified in the binaries: BMI2 and AVX-512 instructions are present).
Turbo disabled (`intel_pstate/no_turbo = 1`). Runs 1–3 used the governor
`powersave` with HWP (EPP `balance_performance`); runs 4–5 and the final
`qsplit2` measurements used the `performance` governor (fixed clock). Single-threaded runs pinned to core 2;
multithreaded construction uses 8 threads. Another user's I/O-bound job was
running throughout (CPU about 85% idle).

Keys: distinct 64-bit integers. Hash: **GxHash** for both implementations,
as in the PHast paper (`--hash gx`, the default of `cmp`; `cmp` asserts that
`lab::GxKey` and `ph::seedable_hash::BuildGxHash` produce the same hashes).
Reference: `ph` at commit `7d18454`.

## Directories

| Directory | Code | Contents |
|---|---|---|
| `nexus-run1` | original PHast-R (`origin/phast-r`), XXH3 | `run.sh env tables anatomy negative` (sequential query timing) |
| `nexus-run2`, `nexus-run3` | new PHast-R, GxHash | `run.sh tables` with interleaved query timing |
| `ab/ab2.txt` | original vs new, GxHash | interleaved A/B, 10⁷ and 10⁸ keys |
| `nexus-run4`, `nexus-run5` | new PHast-R + `next_level` fix, GxHash, `performance` governor | `run.sh tables`, interleaved query timing (final) |
| `conj/phast_grid.txt` | reference PHast, GxHash | space for S = 6..14 and a grid of λ, 10⁶ keys |

## Caveat: query timing and frequency drift

With HWP, the core clock during memory-bound query loops drops (we measured
2.67–3.04 GHz with turbo off, versus 3.1 GHz during construction), and the
drop varies between runs. Sequential timings at 10⁸ keys thus vary by about
±10%, which in `nexus-run1` made PHast-R look 6 ns slower than it is (perf
showed the same cycles and cache/TLB misses per query as the reference, at a
lower clock). `cmp --interleave R` builds all structures first and then
times batches of queries on each structure in turn for R rounds, reporting
the median: all structures see the same frequency conditions. `run.sh` now
uses `--interleave 11` with batches of 2·10⁶ queries.

## Changes to PHast-R (query speed)

1. **Seed encoding.** A seed *s* now encodes pattern *r = s mod R* and shift
   *d = ⌊s/R⌋* (the pair (0, 0) is excluded, as seed 0 marks bumped
   buckets), and the offset of pattern *r* starts at bit *64r/R* of *o*.
   Since 64-bit hardware shifts take their count modulo 64, the query
   computes `o.wrapping_shr(s << (6 - log2 R)) & mask` and `s >> log2 R`: no
   decrement, no extraction of *r*, no multiplication *r·ℓ*. As a side
   effect, there are *2^S/R* shifts per pattern (64 instead of 63 for S = 8,
   R = 4), which slightly improves space.
2. **Offset hash for `[u64; 1]` signatures.** *o = h·c* with *c* odd and
   smaller than 2³¹ (one `imul $imm32` on x86) instead of a xorshift-multiply
   with a 64-bit constant (five instructions). Bit *j* of the product depends
   only on bits 0..*j* of *h*, so offsets depend on the lower bits of *h*,
   independent of the upper bits that select bucket and slice. Bump rates
   and space are unchanged.

The two changes remove 9 instructions from a query (a benchmark iteration
with XXH3 went from 49 to 40 instructions). With GxHash, a benchmark
iteration of PHast-R has 29 instructions, against about 28 for PHast+ with
wrapping.
Making log2 R a compile-time constant (removing two live registers) was
tried and gave no measurable gain.

## Interleaved A/B (GxHash, query ns, median of 15 rounds)

| | 10⁷ base | 10⁷ new | 10⁸ base | 10⁸ new |
|---|---|---|---|---|
| ref PHast+ S=8 | 25.8 | 25.1 | 38.8 | 38.6 |
| ref PHast+ wrap δ=3 S=8 | 23.7 | 23.4 | 35.6 | 34.9 |
| PHast-R S=8 d=1 (u8) | 30.0 | **26.2** | 44.1 | **38.6** |
| PHast-R S=8 d=2 (u8) | 29.9 | **25.6** | 43.1 | **38.5** |
| ref PHast+ wrap δ=3 S=10 | 27.9 | 26.7 | 39.4 | 38.8 |
| PHast-R S=10 d=1 (bit-packed) | 33.9 | **29.6** | 46.3 | **41.4** |

Space (bits/key, base → new, 10⁷): d=0 2.0220 → 2.0166, d=1 1.9351 → 1.9305,
d=2 1.9238 → 1.9201, S=10 d=1 1.8602 → 1.8599, S=10 d=2 1.8547 → 1.8531.
Construction time is unchanged (within noise).

## Fast path and slow path (`split/`)

`qsplit2` (scratch harness with a patched `ph` exposing `is_bumped`, see
`split/`) times, for both implementations, the keys placed in the first
level (fast path) and the bumped keys (slow path), interleaved, with the
`performance` governor. Bumping works in the same way in both (seed 0,
further levels, Elias–Fano remapping of the outputs to the holes); PHast-R
bumps fewer keys than PHast+ (7.6%) and wrap δ=3 (4.35%).

**Slow path.** Each bumped key cost ~15 ns more than in the reference:
`next_level` mixed *h* and *o* with two independent `mix64`, which LLVM
vectorized with AVX-512 `vpmullq` (~15 cycles of latency, two in series).
Third change: only *h* is remixed, `(mix64(h ^ salt ^ c), o)`; `mix64` is a
bijection, so distinct pairs stay distinct, and offsets stay independent of
the new bucket and slice. Space is unchanged, and the slow paths now cost
the same (53 vs 54 ns at 10⁷, 110 vs 109 ns at 10⁸ against wrap δ=3).

**Fast path.** PHast-R's fast path is ~3 ns (10⁷) to ~5 ns (10⁸) slower
than PHast+'s, with the same cache and TLB misses: the difference is
instruction count. In this benchmark the key and the seed are both cache
misses, so every instruction of a query waits in the scheduler, and each
instruction depending on the loads costs 1.5–3 ns (fewer queries in
flight). Variants on the same structure and keys (fast path, ns):

| | 10⁷ | 10⁸ |
|---|---|---|
| PHast+ | 24.4 | 34.5 |
| PHast-R | 28.0 | 40.0 |
| PHast-R, no pattern shift (outputs wrong; = PHast+ query) | 24.5 | 32.6 |
| PHast-R with *o = h* (no `imul`; outputs wrong) | 26.4 | 37.6 |
| PHast-R, displacement = whole seed | 26.5 | 37.3 |

Selecting a pattern costs at least two instructions more than PHast+ (a
second hash word, a variable shift). Using the whole seed as displacement
(pattern *r* gets displacements *r*, *r* + R, …) saves one, but costs 0.08
bits/key and 40% more construction time (it removes the ±1 adjustments of
additive placement), so it was rejected. With *o = h* one of the four
16-bit windows always overlaps the bucket bits. A compile-time number of
patterns gave no gain.

Note: an earlier version of these measurements used a `ph` patch that
accidentally moved the `#[inline(always)]` of `Function2::get`, making the
reference ~1.5 ns slower; all numbers here use the corrected patch.

## Overall query time and bump rate

Interleaved, `performance` governor, ns (all keys):

| | bits/key | build ns/key | β | 10⁷ | 10⁸ |
|---|---|---|---|---|---|
| ref PHast+ (λ=5.25) | 2.116 | 75 | 7.60% | 30.0 | 47.3 |
| ref wrap δ=3 (λ=5) | 1.970 | 135 | 4.35% | 28.1 | 42.8 |
| ref PHast (λ=4.5) | 1.922 | 915 | 1.40% | 26.9 | 39.1 |
| PHast-R λ=5, d=1 (default) | 1.931 | 160 | 3.73% | 30.5 | 45.2 |
| PHast-R λ=4.75, d=1 | 1.941 | 150 | 2.74% | 29.1 | 44.4 |
| PHast-R λ=4.75, d=2 | 1.930 | 230 | 2.61% | 29.6 | 43.2 |
| PHast-R λ=4.5, d=1 | 1.964 | 152 | 1.87% | 29.4 | 42.8 |
| PHast-R λ=4.5, d=2 | 1.956 | 214 | 1.77% | 28.8 | 43.0 |

As a replacement for PHast+, λ = 4.5 with depth-1 repair is smaller by
0.15 bits/key, faster to query at both sizes, and faster to build than the
current default; at 10⁸ it ties wrap δ=3 while being smaller.

## Slice length (`lgrid/`)

With repair, the best L differs from PHast+'s. For S = 8, L = 1024 (PHast+
uses 512) is better at every λ and depth once the priority weights are
tuned for it (`wtune 8 10 1 5.0`, now in `default_weights`; the (8, 512)
weights were also retuned, since the hashing changed): −0.007 to −0.011
bits/key, slightly fewer bumped keys, ~5% more construction time. L = 2048
is worse even when tuned (1.950 vs 1.925 on the tuning keys), L = 256 is
worse untuned. For S = 10, L = 2048 remains the best (tuned L = 4096: 1.914
vs 1.867). L does not affect the query fast path.

10⁷ keys, S = 8, L = 1024: λ=4.5/d1 1.9533 b/k (β 1.73%), λ=4.75/d1 1.9313
(β 2.61%, 166 ns/key), λ=5/d1 1.9231, λ=5/d2 1.9102, λ=5.25/d2 1.9098.
S = 8, L = 1024, λ = 4.75, depth 1 has the space of the current default
with 30% fewer bumped keys, and queries faster than PHast+ at 10⁷ and 10⁸
(30.3 vs 30.6 ns, 43.4 vs 46.7 ns): candidate default.

## Reproduced space results

`nexus-run1/anatomy.txt` and `negative.txt` reproduce Section 2 and Table 3
exactly (space does not depend on the hardware): β = 7.6%, seed entropy
7.73 bits, 2.6% self-colliding buckets, and all rows of Table 3.

## The log₂e + O(log λ/λ) conjecture (reference PHast)

Best λ for each S (`conj/phast_grid.txt`, 10⁶ keys, λ grid S/λ ∈
{1.50, 1.58, …, 2.00}):

| S | λ | bits/key | excess over log₂e | excess/(ln λ/λ) |
|---|---|---|---|---|
| 6 | 3.61 | 2.066 | 0.623 | 1.75 |
| 7 | 4.02 | 1.985 | 0.542 | 1.57 |
| 8 | 4.60 | 1.937 | 0.494 | 1.49 |
| 9 | 5.70 | 1.900 | 0.457 | 1.50 |
| 10 | 6.02 | 1.874 | 0.431 | 1.45 |
| 11 | 6.63 | 1.853 | 0.410 | 1.44 |
| 12 | 7.23 | 1.850 | 0.407 | 1.49 |
| 13 | 7.83 | 1.842 | 0.399 | 1.52 |
| 14 | 8.43 | 1.840 | 0.397 | 1.57 |

The ratio is roughly constant up to λ ≈ 7, but then grows: the excess
flattens around 0.40. The slice length of `ph` (1024 for S < 12, 2048
otherwise) should not be the bottleneck; the priority weights of `ph` are
tuned for smaller seeds, and 10⁶ keys is small, so this is not conclusive.
