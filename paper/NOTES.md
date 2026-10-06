# PHast-R: handoff notes

State of the work on PHast-R as of 2026-10-05, written to continue it on
other hardware (possibly in a new Claude Code session, which will not have
the context of the original one: point it to this file).

## What is here

- `src/func/phast_r.rs`: the implementation (`PHastR`, `PHastRBuilder`,
  `SeedStore`/`SeedStoreBuild`), with unit tests.
- `examples/bench_phast_r.rs`: quick benchmark of PHast-R alone
  (`cargo run --release --example bench_phast_r -- 10000000`).
- `paper/phast.tex`: the paper (11 pages, `latexmk -pdf phast.tex`; references in `paper/biblio.bib`).
- `paper/lean/`: Lean 4 + Mathlib formalization of Proposition 1
  (`cd paper/lean && lake build`; see its README; `.lake/` is not tracked,
  `lake exe cache get` fetches Mathlib).
- `paper/lab/`: the experimental harness, a standalone crate depending on sux
  (by path) and on the reference implementation `ph` (currently by path on
  a local clone of the fork vigna/bsuccinct-rs, branch `sux`, commit 8d722ff,
  which adds hidden analysis accessors to `Function2`; switch to the git
  revision once it is pushed). It contains the comparison driver, the
  anatomy of PHast+ (Section 2), and the tuning tools. The lab
  re-implementation of PHast+ and the prototypes of the negative results
  have been removed (October 2026): everything now uses the reference
  implementation.
- `paper/lab/results/m1-max/`: the raw results behind the paper (see the
  README there for caveats).

## Running on a new machine

```sh
git clone git@github.com:vigna/sux-rs.git && cd sux-rs && git checkout phast-r
cd paper/lab
./run.sh                      # env + tables (10-15 minutes, ~4 GiB of RAM)
./run.sh anatomy              # Section 2 (on the reference implementation)
./run.sh tune                 # weight tuning (space only, hardware-independent)
```

Results go to `results/<hostname>-<date>/`; `env.txt` records the machine,
compiler, and revisions. Useful variables: `N1`, `N2` (key counts), `THREADS`,
`CORE` (pinning core on Linux), `OUT`, `NTUNE`.

For precise timings on Linux:

- set the frequency governor to `performance`
  (`cpupower frequency-set -g performance`) and consider disabling turbo
  (`/sys/devices/system/cpu/intel_pstate/no_turbo` or
  `/sys/devices/system/cpu/cpufreq/boost`) and SMT;
- single-threaded runs are pinned with `taskset` to `CORE` (choose a
  performance core on hybrid CPUs);
- construction times are single measurements: run `./run.sh tables` two or
  three times and take medians.

The harness compiles both sux and `ph` with `-C target-cpu=native` (see
`paper/lab/.cargo/config.toml`); this matters on x86. Both implementations
hash keys with GxHash, as in the PHast paper (`lab::GxKey` reproduces
`ph::seedable_hash::BuildGxHash`, and `cmp` asserts it); `--hash xxh3`
selects XXH3-64 instead (the `BuildX` hasher in `cmp.rs` reproduces sux's
`ToSig<[u64; 1]>` for `u64`). If `CARGO_TARGET_DIR` is set, `run.sh` uses it.

On machines with HWP (Intel) the clock drops during memory-bound query loops,
by an amount that varies between runs (±10% observed): use `cmp
--interleave <rounds>` (as `run.sh` does), which builds all structures first
and then interleaves batches of queries, reporting medians. `PHastR<K>` uses
64-bit signatures (`ToSig<[u64; 1]>`): as in PHast+, keys with the same
signature are bumped and separated at the following levels, whose signatures
depend also on a second hash of the key.

## Comparison driver

`target/release/cmp -n <keys> -v <spec>,<spec>,...` where a spec is

- `ref:<plus|w1|w2|w3|phast>:<S>:<lambda>` for PHast+, PHast+ with wrapping,
  or PHast (reference implementation);
- `r:<S>:<log2 L>:<lambda>[:<log2 R>[:<u8|u16|bfv|bfvu>]]` for PHast-R
  (`bfvu` is a `BitFieldVec` with unaligned reads).

`-t <threads>` sets the threads of the reference; PHast-R uses rayon
(`RAYON_NUM_THREADS`). Every PHast-R structure is verified to be a bijection.

`target/release/qsplit -n <keys> -v <spec>,...` (specs
`<S>:<log2 L>:<lambda>[:<log2 R>]` or `ref:<plus|w3|phast>:<S>:<lambda>`)
times separately all keys, the keys placed in the first level, and the
bumped keys, in interleaved rounds (medians are reported), and prints the
bump rate; `--set <0|1|2>` measures a single set, for use with `perf stat`.
`target/release/btime -n <keys> -v <S>:<log2 L>:<lambda>[:<log2 R>]` times
construction only (byte seeds).

## Results so far (M1 Max, 10⁷ keys, single-threaded construction)

| | bits/key | build ns/key | query ns |
|---|---|---|---|
| PHast+ (S=8, λ=5.25) | 2.116 | 40 | 22.8 |
| PHast+ wrap δ=3 (S=8, λ=5) | 1.970 | 70 | 21.3 |
| PHast (S=8, λ=4.5) | 1.923 | 562 | 19.9 |
| **PHast-R** (S=8, λ=5, depth 1, default) | 1.935 | 79 | 21.4 |
| **PHast-R** (S=8, λ=5, depth 2) | 1.924 | 127 | 21.4 |
| PHast+ wrap δ=3 (S=10, λ=6) | 1.872 | 121 | 20.1 |
| PHast (S=10, λ=6.05) | 1.854 | 1646 | 19.9 |
| **PHast-R** (S=10, L=2048, λ=6, depth 1) | 1.861 | 86 | 23.7 |
| **PHast-R** (S=10, L=2048, λ=6.25, depth 2) | 1.854 | 137 | 24.0 |

On 10⁸ keys with 10 threads: PHast-R S=8/d1 13.8 ns/key (wrap3 13.6),
S=10/d1 13.0 ns/key (wrap3 S=10 19.2).

## Results on Intel Xeon E-2388G (GxHash, after the query changes below)

See `lab/results/README-nexus.md`. `performance` governor, turbo off, 10⁷
keys, single-threaded construction, query ns as medians of interleaved
rounds (runs 4 and 5, `lab/results/nexus-run4`, `nexus-run5`):

| | bits/key | build ns/key | query ns |
|---|---|---|---|
| PHast+ (S=8, λ=5.25) | 2.116 | 74–75 | 29.2–29.5 |
| PHast+ wrap δ=3 (S=8, λ=5) | 1.970 | 135–136 | 27.6 |
| PHast (S=8, λ=4.5) | 1.922 | 915–921 | 26.0–26.3 |
| **PHast-R** (S=8, λ=5, depth 1, default) | 1.931 | 160–168 | 29.4–29.6 |
| **PHast-R** (S=8, λ=4.75, depth 2) | 1.930 | 232–233 | 28.5–29.2 |
| **PHast-R** (S=8, λ=5, depth 2) | 1.920 | 247–249 | 29.6–29.7 |
| PHast+ wrap δ=3 (S=10, λ=6) | 1.872 | 274 | 31.4 |
| **PHast-R** (S=10, λ=6, depth 1) | 1.860 | 175–176 | 33.1–33.5 |

10⁸ keys (single-threaded construction): PHast+ 44.7–46.0 ns, wrap3
41.9–42.7, PHast 37.8–38.6, PHast-R default 44.4–45.3, PHast-R λ=4.75/d2
42.7–43.4. With λ = 4.5 and depth 1 (1.964 bits/key, 152 ns/key, 1.9%
bumped keys) PHast-R queries are faster than PHast+ at both sizes (29.4 vs
30.0 ns at 10⁷, 42.8 vs 47.3 at 10⁸, `qsplit2`) and tie wrap δ=3 at 10⁸:
it is the natural candidate to replace PHast+.

## Query changes (October 2026)

- Seed *s* encodes *r = s mod R*, *d = ⌊s/R⌋* ((0, 0) excluded), and pattern
  *r* reads *o* from bit *64r/R*: the query shifts *o* right by
  `s << (6 - log2 R)` (counts are taken modulo 64 by the hardware) and needs
  no decrement, extraction of *r*, or multiplication *r·ℓ*. There are now
  *2^S/R* shifts per pattern (64 for S = 8), which slightly improves space.
- `[u64; 1]` signatures: *o = h·c* with *c* odd and < 2³¹ (a single
  `imul $imm32`) instead of a xorshift-multiply.
- Levels after the first used to remix (h, o) with `next_level`; two
  independent mixes were vectorized by LLVM with AVX-512 `vpmullq` (~15 cycles
  latency each), making each bumped key ~15 ns slower than in the reference.
  Now keys are hashed again at each level (see "64-bit collisions" below).

Before the changes, on x86 PHast-R queries were 4–8 ns slower than PHast+.
Bumping works as in the reference, and PHast-R bumps fewer keys (3.7%) than
PHast+ (7.6%) and wrap δ=3 (4.35%); after the `next_level` fix bumped keys
cost the same as in the reference. The fast path is still ~3 ns (10⁷) to
~5 ns (10⁸) slower than PHast+'s: the memory behavior is identical, but in
this benchmark (key and seed are both cache misses) every instruction
depending on the loads costs 1.5–3 ns, and selecting a pattern costs at
least two instructions more than PHast+ (second hash word, variable shift).
Using the whole seed as displacement saves one instruction but costs 0.08
bits/key; a compile-time number of patterns gave no gain. The lever left is
the bump rate (smaller λ, see above). On this machine use the `performance`
governor: with `powersave`, HWP down-clocks PHast-R's query loop more than
the reference's.

## Slice length (October 2026, `lab/results/lgrid`)

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

## Construction speed (October 2026)

Defaults are now S = 8, L = 1024, λ = 4.75, depth 1. Changes, all with
bit-identical output (same space for every configuration), measured with
`lab/src/bin/btime.rs` (construction only, alternating old and new binaries):

- sort by *h*: `voracious_radix_sort` (`voracious_mt_sort` with `rayon`, as
  in `ph`) instead of rdst on all 8 bytes, which scaled poorly (1.68 s →
  ~0.6 s at 10⁸ with 8 threads). A custom parallel MSD radix sort was within
  ±3% of voracious (faster with 8 threads, slower with one), so voracious was
  kept for simplicity;
- repair: blockers are resolved lazily, in candidate order (a failed
  eviction restores the state, so the result is the same), candidates are
  packed in a `u128`, the bases of all patterns are cached, and at the last
  repair level eviction trials only flip occupancy bits (owners are written
  on success only);
- the cyclic window of a sweep is sized from the geometry (4(L + D) + 2 ·
  256 · slots per bucket, rounded to a power of two: 8192 slots for the
  default instead of 65536), so the owner array stays in L2;
- the final occupancy pass fills per-thread segments instead of using
  atomic `fetch_or`; `mark` reuses the bases of the search.

Tried without gain: branchless `get64`, `extend` instead of `push`, visiting
patterns in order of base sum (slower), fewer repair candidates (costs space:
8 candidates → 1.948 bits/key), ordering candidates by blocker size.

Result (taskset 0-7, `performance` governor), depth 1: 10⁷ keys 144 → 122
ns/key single-threaded, 28 → 25 with 8 threads; 10⁸ keys 139 → 127 and 27 → 25.
Against the reference at 10⁸ (same conditions): PHast+ 75.5 / 18.0,
wrap δ=3 131.0 / 25.9, PHast-R 129.9 / 24.9 (1 / 8 threads): construction is
now on par with PHast+ with wrapping, and 1.4–1.7 times PHast+.

Profile of what remains (single thread): search over four patterns (the
inherent cost of patterns; PHast+ searches one), eviction trials (1.37M at
10⁷ keys, 87% failing), the priority queue, and the sort.

## 64-bit collisions (October 2026, `lab/results/rehash`)

PHast+ needs only 64-bit hashes because keys with the same hash self-collide,
are bumped, and are separated at the next level, where the key is hashed
again with a new seed. The previous PHast-R remixed (h, o) at each level, so
equal 64-bit hashes stayed equal and construction failed. Now:

- the first level is unchanged (*h* = 64-bit hash, *o = h·c*);
- at each following level the key is hashed again with the seed of the level,
  and the new hash *h'* gives bucket and slice; offsets come from *o ⊕ h'*.
  The query passes *o* to the slow path, so it is still computed in parallel
  with the seed load. Reusing *o* unchanged instead fails for *R = 1* at the
  last level: keys of a bucket of a small level have nearly the same slice,
  and two of them with the same `o & (L - 1)` collide at every attempt;
- duplicate keys are detected before building: keys whose first-level hashes
  coincide are hashed again with a different seed.

Measurements (Xeon E-2388G, single thread, pinned, 11 interleaved rounds,
reference rows in each process; base = d2a9cb35):

- space and construction are unchanged (within 1%, 1 and 8 threads);
- strings of 10–50 bytes (`cmpstr`): queries 0.6–2.2 ns *faster*;
- u64 keys: queries 0.3–1.1 ns slower. `qsplit` shows that the slow path is
  2–4 ns faster (37.6 vs 40.1 ns at 10⁷), whereas the fast path is ~0.5 ns
  slower because the slow path needs the key: LLVM keeps the u64 key in a
  general-purpose register and moves it to a vector register for GxHash
  (`mov` + `vpbroadcastq xmm, r64`) instead of broadcasting it from memory.
  A base build whose slow path merely takes the key has the same fast path
  (`qsplit-basekey`: 24.4 vs 23.9 ns at 10⁷). Any scheme that hashes the key
  again (PHast+ included) pays this; it does not occur with string keys.
- Discarded routes: reusing *o* unchanged (fails at small last levels, see
  above); deriving the bucket from *x·c* (V2, +0.5–1 ns, multiplication on
  the address path); rehashing only at level 1 (fatal collisions at level 1);
  128-bit signatures (no gain, as `vpextrq` costs as much as `imul`).

## String keys and seed reads (October 2026)

- String keys (`cmpstr`, `results/rehash/str_ab_inl.*`): `PHastR::get` was
  `#[inline]`, and with the GxHash of a string inlined LLVM did not inline it
  in the query loop, whereas `ph` marks `Function2::get` `#[inline(always)]`.
  With `#[inline(always)]` queries are 0.9–1.7 ns faster up to 10⁷ keys (more
  at 10⁸), and PHast-R goes from +2.1–2.6 ns to −2.6–+0.9 ns vs PHast+.
- Seeds in a `BitFieldVec` (`results/unaligned`): the store used a private
  32-bit unaligned read, relying on the padding added by `from_seeds`. Now a
  `BitFieldVec` uses aligned reads, and `PHastR` implements
  `TryIntoUnaligned` (seeds become a `BitFieldVecU`, and the low bits of the
  Elias–Fano remapping are converted too). Unaligned reads are 2.1–4.4 ns
  faster than aligned ones for 10 and 12 bits, and as fast as the previous
  32-bit read. PHast-R with S=10 is still 2.7–4.6 ns slower than PHast+ with
  wrapping at S=10 (to be investigated).

## Wrapping (October 2026, `lab/results/wrap`)

- Fast-path anatomy (`qsplit`, diagnostic builds at 10⁷, S=8): PHast-R's
  first-level path was 4.1 ns slower than PHast+'s; removing the variable
  pattern shift `o >> (s·64/R)` recovers ~3 ns, removing the multiplication
  deriving *o* ~1 ns. PHast-R won overall only because it bumps 2.6% of the
  keys instead of 7.6% (the slow path of PHast+ costs ~54 ns).
- `PHastRBuilder::wrap(M)`: shifts with wrapping as in PHast+ with wrapping
  (`slice + ((h + s·M) mod L)`, no patterns, no *o*), with our repair, levels
  and last level. Search and repair scan *segments* of shifts in which no key
  wraps (as in `ph`); segments play the role of patterns in repair. Default
  weights are those of `ph` for wrapping.
- Without repair, the first level and the following levels reproduce `ph`
  exactly (same bumped keys at each level when built single-threaded); the
  residual space difference was the remapping: sux 0.10.3 (used by `ph`) has
  `EfSeq` with a `SelectAdaptConst<_, _, 12, 3>` inventory, current sux 11.
  PHast-R now uses its own `Remap` type with 12 (−0.0065 b/k at 10⁷,
  +4 ns on the slow path, i.e., +0.1–0.15 ns on average).
- 10⁸ keys (`pareto.txt`): PHast-R W3 d1 λ=4.75 1.9309 b/k, 33.9 ns, build
  165 ns/key; λ=5 1.9183 b/k, 35.0 ns, 186; PHast+ w3 λ=5 1.9681 b/k,
  36.5 ns, 128; PHast-R R=4 d1 1.9259 b/k, 38.5 ns, 122. Wrapping dominates
  in space and query time; construction is 30–45% slower than PHast+ w3, as
  89% of eviction trials fail after scanning the whole shift range of the
  evicted bucket (`grid.txt`: fewer candidates trade space for time).
- With 8 threads (`ph` also on 8 threads) the construction gap shrinks:
  10⁸ keys, PHast+ w3 λ=5 24.4 ns/key; PHast-R W3 d1 λ=4.75 28.8, λ=5 31.4.
  Scanning shifts by words instead of segments was tried and is slower
  (145 vs 109 ns/key at d0): segments prune much better, as the sum of
  positions grows by k at each shift within a segment.
- 10-bit seeds (`s10.txt`, unaligned reads, 10⁸ keys): PHast-R W1 d1 λ=6
  1.8526 b/k, 140 ns/key, 38.2 ns; PHast+ w3 λ=6 1.8693 b/k, 253 ns/key,
  38.9 ns; PHast+ w1 1.9073 b/k, 118 ns/key, 37.6 ns; PHast-R R=4 d1
  1.8536 b/k, 131 ns/key, 43.2 ns. W1 with repair beats PHast+ w3 in space,
  construction, and query time.

## Brainstorm from scratch (October 5, 2026)

Accounting (10⁸ keys, W3 d1 λ=5, 1.918 b/k, 3.66% bumped): the ideal cost of
placing 96.34% of the keys injectively into *n* slots is 1.215 b/k, and of
the rest 0.228, so the first level wastes 0.385 b/k (8-bit seeds where 6.1
would be ideal) and remapping plus further levels ~0.09. Seeds are nearly
uniform (empirical entropy 7.91 bits; PHast+ w3 7.89), so entropy coding
would save ≤ 0.02 b/k: the waste is the redundancy of choosing among several
feasible shifts. Bump rate by bucket size is nearly flat for sizes 2–10
(1.3–5%), so bumping is driven by congestion (deferred small buckets), not by
hard large buckets.

Negative results (3·10⁶ keys unless noted):
- Partial bumping (seeds bumping only the keys of one of C hash classes):
  a probe said 47–59% of bumped keys could stay, but the real thing is worse
  (P=64: 1.993 vs 1.924 b/k; bumped keys 4.6% vs 3.7%).
- Evicting for good a smaller blocker to place a failing bucket: +0.027 b/k;
  a larger blocker: +0.06–0.08 b/k. Holes are set by the packing density of
  the sweep: slots left by a bumped bucket are filled by later (small)
  buckets, so bumping fewer keys at a failure does not reduce holes.
- Repair candidates where two keys are blocked by the same bucket: no gain,
  +15–25% construction; 32 candidates instead of 16: −0.0014 b/k.
- Skipping eviction trials of buckets that had no alternative placement:
  their trials succeed as often (11% vs 12%).
- Weights retuned for wrapping with repair (`wtune`, `results/wtune_wrap`):
  −0.0034 b/k. Geometry grid (`results/wrap_grid`): M=3, L=1024, λ=5 is
  optimal; even multipliers are bad (a single residue class mod M).

Remapping: optimal coding of the holes and of the unused outputs of the
further levels would take 0.270 instead of 0.297 b/k at 10⁷ (≤ 0.027 b/k for
both PHast+ and PHast-R).

Batched queries (`get_batch`, hidden; `lab/src/bin/batch.rs`): hashing a
batch of keys and prefetching their first-level seeds before computing the
outputs. At 10⁷ keys, 24.4 → 16.2 ns (W3) and 28.5 → 15.8 ns (R=4); at 10⁸,
35.6 → 27.9 and 41.6 → 27.6; 10-bit seeds (unaligned) 28.8–29.9 ns at 10⁸.
In batch mode the extra operations after the seed load (patterns, unaligned
reads) cost almost nothing, so the configurations differ in space only.

9-bit seeds (8 threads, 3·10⁶): R=4, L=2048, λ=5.5 1.908 b/k at 22.8 ns/key
(untuned), vs 1.928 b/k at 35.2 ns/key for S=8 W3 and 1.981 for PHast+ w3.

## Second brainstorm: cost model, the floor, ring patterns (October 5–6, 2026)

Tools: `lab/src/bin/overload.rs` (single-level statistics through the hidden
`PHastRBuilder::level_stats`: bumped keys, holes, bump rate and feasible
shifts by bucket size, self-colliding buckets), with hidden experimental
builder options (`skew`, `log2_band`, `window`, `ring`) and environment
variables (`PHAST_FEAS`, `PHAST_OBJ`).

**Cost model** (S = 8, checked against real sizes): bits/key ≈ S/λ + 1.9 ·
bumped + (2.06 + log₂(1/holes)) · 1.045 · holes. A hole costs a remapping
entry (~6.8 bits), a bumped key its share of the next level (~1.9 bits).

**Diagnosis.**
- Single-key buckets never fail and pairs fail 1–2%: holes are plentiful for
  small buckets, but Poisson(5) has few of them (0.7% of the keys in buckets
  of size 1, 3.4% in size 2).
- When placed, a bucket has ~7 feasible shifts on average (all sizes 2–8,
  equalized by the weights), but 4% have none and 16% one or two: the count
  is overdispersed, as adjacent shifts are strongly correlated (the keys of a
  bucket are spread over a slice that spans the whole density gradient).
  The choice among feasible shifts costs ~2 bits per bucket (0.4 b/key).
- **Self-collisions**: with wrapping, two keys of a bucket with the same base
  position collide for all seeds (probability C(k,2)/L per bucket): 1.70% of
  the keys are in such buckets, all bumped — 39% of the bumped keys without
  repair, 46–47% with repair, which cannot fix them. Patterns (R = 4) bump
  only 0.13% for this reason, but pack worse (64 shifts per pattern, no
  wrapping).

**The floor.** At S = 8, λ = 5 everything converges to ~3.6% bumped keys:
W3 + repair 3.68% (depth 1) and 3.59% (depth 2), patterns + repair 3.64%,
ring patterns 3.72% without repair and 3.63%/3.61% with repair of depth 1/2.
Removing self-collisions does not add up: the freed capacity is taken by
other failures. Ideas that do not move the floor (3·10⁶ keys):
- overloading (first level with n/(1+δ) slots, later levels in direct
  ranges, only the final keys remapped onto all holes): each 1% of overload
  removes 0.19–0.28% of holes, break-even is 0.25%;
- skewed periodic bucket sizes (PHOBIC-like profiles): worse the stronger the
  skew (4.4% → 6.3–9.6% with W3, 3.7% → 3.8–6.3% with ring patterns); large
  buckets fail, as they do not find empty space in a sliding window. With a
  realistic allowance of ~4 feasible shifts even the ideal profile gives
  ~1.92 b/key;
- narrow band of offsets moved along the slice without wrapping: 10% bumped
  (self-collisions C(k,2)/w; small buckets starve);
- strict largest-first order (larger window, scaled weights): much worse;
- concave objectives instead of the sum of positions (log, cube root):
  3.72% → 3.65%; convex ones are worse;
- larger seeds with ring patterns (estimates): S = 10 1.864, S = 12 1.840,
  S = 14 1.885, S = 16 1.911 (untuned weights): no byte-aligned win.

**Ring patterns** (`PHastRBuilder::ring(R)`, hidden; cmp spec `g<R>` in the
pattern field). Seed *s* selects pattern *r = s mod R* and the offset
((lo >> 64r/R) + s·2^a) mod L, with L = 2^(S+a) and lo the lower half of the
product h·B whose upper half is the bucket (free, and uniform within a
bucket; B is forced to be odd): the seeds of a pattern cycle exactly once
around the slots of the slice in a residue class modulo R·2^a.
- No self-collisions, so the floor is reached without repair.
- Query: shift, variable shift, scaled add, mask, add after the seed load
  (one operation more than PHast+ with wrapping).
- Construction: the set of used slots is stored by residue classes, so the
  feasibility of the 64 seeds of a pattern for a key is a 64-bit read and a
  rotation; only free indices are evaluated.
- R = 4, L = 1024 is the best configuration found; R = 2 is slightly worse,
  more patterns need more hash bits than the 64 of lo, and L = 4096 would
  need new weights.

Results (`lab/results/ring/run.txt`; 10⁸ keys, GxHash, single thread unless
noted; construction with 8 threads in parentheses):

| | bits/key | build ns/key | query ns | bumped |
|---|---|---|---|---|
| PHast+ w3 (λ=5) | 1.9681 | 128.4 (24.3) | 35.7 | 4.35% |
| W3 + repair d1 (λ=5) | 1.9183 | 186.6 (32.8) | 34.3 | 3.65% |
| G4, no repair (λ=5) | 1.9243 | 104.7 (20.8) | 37.1 | 3.74% |
| G4, no repair (λ=4.75) | 1.9194 | 105.4 (20.7) | 35.6 | 2.54% |
| G4 + repair d1 (λ=5) | 1.9145 | 169.0 (30.3) | 37.0 | 3.60% |

So G4 at λ = 4.75 has the space of wrapping with repair, builds 18% faster
than PHast+ w3 (15% with 8 threads), and its queries take the same time as
those of PHast+ w3: the additional operation (~1 ns, see λ = 5) is paid by
the smaller number of bumped keys. Wrapping with repair remains the fastest
at query time. Using the seed as shift count (64 patterns of 4 seeds,
`g64`) would have the same operations as PHast+ w3, but it bumps 4.7% of the
keys at λ = 5 (64 bits do not provide enough independent offsets).

Open issues: for fewer than ~10⁵ keys both wrapping and ring patterns use
more space than PHast+ w3 (e.g., 2.79–2.84 vs 2.70 b/key at 10⁴ keys): the
geometry of small levels was tuned only for patterns without wrapping. The
experimental options and the diagnostics should be removed or moved out of
`phast_r.rs` once a design is chosen.

## Rings only: cleanup and engineering (October 5, 2026)

The design chosen is rings of patterns without repair (R = 4 and L = 1024
for 8-bit seeds; the default λ was 4.75, and it is 4.5 since the evening of
October 5, as query speed comes first: see the end of this section). `phast_r.rs` was rewritten around it (commit
`59f22b47`): wrapping, repair, the experimental options and the diagnostics
are gone, together with the lab tools that used them (`overload`, `bumps`).
Everything is still available at commit `e22eef68`, which is what the
sections above describe.

**Construction** (commit `a4d948c1`; 10⁷ keys, one thread: 111 → 58 ns/key,
with identical seeds at each step, which makes the space a regression check).
`perf` top-down on the first version showed 25% of the cycles in bad
speculation (about 10 mispredicted branches per bucket) and a scan loop of
54 instructions per key and pattern, mostly stack spills. In order of
effect:

- constant parameters for the default configuration (`Rings::DEFAULT`):
  the sweep is compiled twice, and in one version shifts and masks are
  immediates and the loop over patterns is unrolled (96 → 83 ns/key);
- one pass over the keys of a bucket, with patterns in the inner loop, and a
  single loop over all free seeds whose only branch is the exit; the sum of
  the slots of a seed is computed on packed ring indices with a population
  count (branch mispredictions per key: 2.1 → 0.75; 78 → 69);
- grouping by bucket with a two-level distribution instead of sorting with
  voracious (the sort was 27% of the single-threaded time and 41% of the CPU
  time with 8 threads; 69 → 64);
- queues by bucket size instead of a binary heap (78 → 75), unchecked reads
  of the set of used slots (62 → 60), contiguous search state (−2.5%),
  occupancy and bumped keys collected by the sweep (−2.5%).

Things that did *not* work: scanning patterns in the outer loop (more
instructions), removing bounds checks from that version (LLVM vectorizes
the loop with gathers, which are slower), a `match` on several constant
scales in the sweep or in queries (it becomes a jump table inside the loop,
or is folded back into variable shifts: use a boolean on a dedicated field),
zeroing a prefix of the search state of constant size (no gain, and the
cold paths read the stale suffix).

What is left in the sweep (62% of the time at 10⁷ keys, one thread): the scan
is about 30 instructions per key and pattern, close to what the formulation
requires; grouping is 21%, and it is limited by memory bandwidth and page
faults when run in parallel.

**Queries.** `qsplit` now interleaves rounds and separates first-level keys
from bumped keys. Findings (10⁷ keys; ns for first-level keys / bumped keys /
all keys, before the fix):

| | first level | bumped | all | bumped keys |
|---|---|---|---|---|
| PHast+ λ=5.25 | 20.8 | 58.0 | 25.5 | 7.60% |
| PHast+ wrap δ=3 λ=5 | 21.3 | 44.8 | 24.0 | 4.35% |
| PHast λ=4.5 | 22.3 | 37.2 | 23.0 | 1.40% |
| PHast-R λ=4.75 | 22.5 | 41.4 | 23.9 | 2.54% |

- Time for first-level keys grows by about 1 ns for each executed operation
  that depends on the cache-missing loads (probes replacing the position
  formula in the same structure: PHast+ formula 19.5, wrapping formula 21.2,
  rings 22.2, PHast's multiplications 23.3); register moves that are
  eliminated at renaming do not count, so sharing the shift count between
  the two shifts of the seed (L = 4096) gains nothing.
- PHast-R and PHast+ with wrapping execute the same number of operations
  after reading the seed (five). The deficit of PHast-R came from the
  handling of bumped keys: `get_slow` took the key, so the compiler kept the
  key in a general-purpose register and GxHash had to move it to a vector
  register instead of loading it directly (one more dependent operation per
  query). With `get_slow(h, h')`, where h' is a second hash computed in the
  cold branch, first-level keys take 21.3 ns. Inlining the slow path, as
  `ph` does, gains much less (the loop runs out of registers).
- The average is the first-level time plus the fraction of bumped keys
  times about 40 ns (10⁷ keys) or 110 ns (10⁸ keys), which includes the
  branch misprediction: λ trades space for query time.

Results at that point (build ns/key with one thread, query ns; the final
numbers are in `lab/results/paper/run.csv`, see the next section):

| | 10⁷: bits/key | build | query | 10⁸: bits/key | build | 8 threads | query |
|---|---|---|---|---|---|---|---|
| PHast+ λ=5.25 | 2.116 | 66 | 23.7 | 2.115 | 69 | 15.8 | 35.8 |
| PHast+ wrap δ=3 λ=5 | 1.970 | 120 | 22.1 | 1.968 | 126 | 23.2 | 32.9 |
| PHast λ=4.5 | 1.922 | 819 | 21.0 | 1.921 | 825 | – | 30.5 |
| PHast-R λ=4.5 | 1.929 | 60 | 20.5 | 1.927 | 63 | 13.1 | 29.9 |
| PHast-R λ=4.75 | 1.921 | 58 | 21.0 | 1.919 | 62 | 13.5 | 31.0 |
| PHast-R λ=5 | 1.925 | 59 | 21.7 | 1.924 | 62 | 14.3 | 32.3 |
| PHast+ wrap δ=3 S=10 λ=6 | 1.872 | 242 | 25.3 | 1.869 | 248 | 41.3 | 35.8 |
| PHast-R S=10 L=2048 λ=6 | 1.852 | 123 | 25.0 | 1.849 | 128 | 21.4 | 35.5 |

Expected bucket size (`cmp`, one thread; bits/key, build ns/key, query ns):

| λ | 10⁷ | | | 10⁸ | | |
|---|---|---|---|---|---|---|
| 4.0 | 2.042 | 73 | 20.1 | 2.041 | 77 | 28.8 |
| 4.25 | 1.961 | 63 | 20.2 | 1.960 | 67 | 29.1 |
| 4.5 | 1.929 | 60 | 20.5 | 1.927 | 63 | 29.8 |
| 4.75 | 1.921 | 58 | 21.0 | 1.919 | 62 | 30.9 |
| 5.0 | 1.925 | 59 | 21.7 | 1.924 | 62 | 32.3 |

Below 4.5 space grows quickly for a small gain in query time; 4.5 is the
default (decided by Sebastiano: "We need query speed").

## Large key sets (October 6, 2026)

The machine became free, and the 10⁸ limit was lifted. All experiments were
redone, now with 10⁹ keys too: `lab/results/paper/redo.sh` builds the lab
and runs `run.sh` (→ `<host>/run.csv`) and `large.sh` (→ `<host>/large/`:
query split, thread scaling, peak memory; `large_tables.py` formats
Tables 4–5 of the paper); the results of this
machine are in `lab/results/paper/nexus/`. Peak memory with 10⁹ keys is
about 19 GB (8 GB of keys included).

Calibration for large key sets (all with identical output):

- `group` with 2¹¹ parts of 488K keys at 10⁹ was DRAM-bound in its second
  phase (43% of the CPU time with 8 threads): now when more than 2¹¹
  cache-sized parts would be needed, parts are larger (√ of that number)
  and each one is distributed into cache-sized parts first (16.5 → 13.4
  ns/key with 8 threads at 10⁹; neutral at 10⁸).
- Signatures no longer carry key indices (8 bytes instead of 16), and
  then (October 6, see the implementation notes) they are stored just once,
  in regions that are never permuted: parts are loaded lazily by sweeps.
  Memory during construction: 33 → 18.6 → 10.7 bytes/key (ph: 9.9). Time
  (ns/key, 1 thread / 8 threads): 10⁷: 52 / 9.2 → 55 / 9.2; 10⁸: 56 / 9.9 →
  59 / 9.4; 10⁹: 60 / 11.0 → 66 / 10.3. Tried and rejected: a coarse bit
  vector in front of the fine one (unpredictable branch, slower at
  10⁷–10⁸), prefetching in the scan (±2%), rehashing instead of storing
  (three hashes per key).
- With 10⁹ keys the walk finding bumped keys is a sizable part of the CPU
  time (random DRAM accesses into a 28 MB bit vector); the sweep is about
  half.

Findings (Section 4.1 of the paper):

- A query for a first-level key costs the same in all structures (44–45 ns
  at 10⁹); a bumped key costs ~55/110/150–200 ns extra at 10⁷/10⁸/10⁹
  (including the misprediction), so each percent of bumped keys costs
  0.55–1.5 ns: λ = 4.5 is the right default for large sets.
- Transparent huge pages (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, THP in
  `madvise` mode on this machine) make queries 18% faster at 10⁸ and 40%
  faster at 10⁹ for every structure (PHast-R 24.6/28.3 ns, PHast 25.2/28.9,
  PHast+ w3 26.9/32.1 at 10⁸/10⁹), and construction 10% faster. Removed
  from the paper and from `large.sh` on October 6 (Sebastiano: "this
  dilutes a bit too much the tests"); the ordering of the structures is
  the same with and without them.
- Thread scaling at 10⁹: 64.4/34.2/18.4/10.5/9.6 ns/key with 1/2/4/8/16
  threads (6.1× with 8); PHast+ w3: 129.8/70.3/39.1/24.9/20.9.
- Final numbers (1 thread; bits/key, build ns/key, query ns):

| | 10⁷ | | | 10⁸ | | | 10⁹ | | |
|---|---|---|---|---|---|---|---|---|---|
| PHast+ w3 λ=5 | 1.970 | 119 | 22.0 | 1.968 | 129 | 33.1 | 1.968 | 130 | 52.1 |
| PHast λ=4.5 | 1.922 | 808 | 20.9 | 1.921 | 817 | 30.5 | 1.920 | 818 | 47.6 |
| PHast-R λ=4.5 | 1.929 | 55 | 20.5 | 1.927 | 59 | 30.0 | 1.927 | 66 | 47.3 |
| PHast-R λ=4.75 | 1.921 | 54 | 21.1 | 1.919 | 56 | 31.1 | 1.919 | 64 | 48.9 |
| PHast+ w3 S=10 | 1.872 | 242 | 25.2 | 1.869 | 254 | 35.8 | 1.869 | 252 | 54.2 |
| PHast-R S=10 | 1.852 | 119 | 24.7 | 1.849 | 121 | 35.3 | 1.848 | 131 | 54.2 |

  With 8 threads (build ns/key): PHast-R 9.2/9.4/10.3, PHast+ w3
  22.6/23.1/24.9; S=10: 19.2/17.3/19.0 vs 40.7/41.2/43.0.

## Older hardware (October 6, 2026, `lab/results/paper/sexus/`)

`redo.sh` on a 40-core Xeon E7-4870 (Westmere-EX, 2011: no AVX/BMI2, 256 KB
L2 per core, 4 sockets, 1 TB; cores 0–7 of `lscpu`, glibc 2.42). Same space to the bit. Queries (ns, 10⁷/10⁸/10⁹): PHast-R λ=4.5
93.1/146.7/231.3, PHast+ w3 93.1/151.5/249.4, PHast 130.5/183.6/273.3 (its
multiplications are slow there), plain PHast+ 98.0/162.1/312.9; S=10:
PHast-R 135.2/192.5/284.1 vs w3 132.8/187.1/280.2 (1–3% slower: unaligned
bit-field reads cost more on that generation). First-level path: PHast-R
90.2 vs w3 85.4 at 10⁷ (the extra shift costs ~5 ns without BMI2), made up
by fewer bumped keys. Construction (ns/key, 1 thread): PHast-R
160/183/223 vs w3 291/317/353 (55–63%); 8 threads 35/28/35 vs 56/48/62;
40 threads at 10⁹: 11.4 vs 36.8 (17.8× vs 8.7× scaling). Memory 10.6 vs
9.9 bytes/key. Huge pages (before their removal from the paper): queries
−14% at 10⁸, −30% at 10⁹ (PHast-R 125.5/161.6, w3 129.3/172.6, PHast
159.6/175.7 at 10⁸/10⁹), construction −5%.

## AWS metal instance (October 6, 2026, `lab/results/paper/c7i/`)

`redo.sh` on a `c7i.metal-24xl` (one Xeon Platinum 8488C, Sapphire Rapids:
48 cores/96 threads, 192 GB, 105 MB L3, 2 MB L2 per core; Amazon Linux
2023, rustc 1.99). Same space to the bit. Queries (ns, 10⁷/10⁸/10⁹):
PHast-R λ=4.5 11.0/29.8/40.8, PHast+ w3 12.4/32.3/47.8 (−11/−8/−15%),
PHast 11.3/30.4/41.4, plain PHast+ 16.1/36.0/54.5; at 10⁷ everything (the
structure and the 4M query keys) sits in the 105 MB L3, so that row is
L3-bound (first-level and all-keys times coincide within noise). S=10:
PHast-R λ=6 13.7/35.9/50.1 vs w3 13.5/35.3/49.2 (1.5–2% slower, as on the
Westmere; equal on nexus). Bumped key: ~140/250 ns extra at 10⁸/10⁹.
Construction (ns/key, 1 thread): PHast-R 45/51/54 vs w3 88/93/100
(51–54%), PHast 640–652; 8 threads 7.7/8.7 vs 16.9/17.8 (10⁸/10⁹); 96
threads at 10⁹: 2.8 vs 7.7 (19.6× vs 13.3×). Memory 10.7 vs 9.9 bytes/key.

The paper now reports the c7i data only (Sebastiano: "we are only
interested in the AWS data"): Tables 1–5 and Figure 3 (two panels sharing
the space axis: query time and construction time, generated by
`plot_pareto.py`; the 3-D Pareto set is listed in its comments rather than
plotted). A rerun with `r:10:11:5.75` (S=10), `pareto.sh` (bucket-size
sweeps for the figure) and 10¹⁰ keys on an m7i.metal-24xl is in progress.

## Key findings (see Section 2 of the paper; reference implementation)

- With output range m = n, holes = bumped keys; each hole costs about
  log₂(1/β) + 2 bits in Elias–Fano, close to the entropy of the hole set, so
  the only lever is the bump rate β.
- 38.5% of PHast+ bumped buckets are self-collisions (2.6% ≈ λ²/2L of the
  buckets); seeds are incompressible (7.71 bits of entropy).
- Independent offset patterns fix the rigidity; repair (half adder +
  eviction) recovers PHast's space.

## Implementation notes

- `group` computes the signatures of a level (plain `u64`, without key
  indices) and distributes them into at most 256 parts (2^k consecutive
  buckets; parts of 2¹⁴ keys when possible): each thread writes the
  signatures of its chunk of the keys at the end of the region of their
  part, in key order. Regions are sized beforehand (mean + 6 standard
  deviations + 32; `vec![0; …]` is lazily zeroed, so unused slack costs
  nothing); signatures that do not fit go to a per-chunk `extra` list
  sorted by part. `Signatures::load(p, range)` distributes a part (or the
  buckets of a range inside it, for gap sweeps) into buckets in a buffer of
  the sweep (`Parts`, which keeps the two or three parts spanned by the
  window), first into cache-sized smaller parts (2¹³ keys, also with slack)
  and then into buckets. `Signatures::walk(chunk)` visits the keys of a
  chunk in order, reading each signature from the next position of the
  region of its part (one byte per key records the part); `bumped` uses it
  with a bit vector of the buckets without a seed, and so does the
  duplicate check of the following levels. No unsafe code, about 10.7
  bytes per key (ph: 9.9), one hash per key.
- Rejected on the way (October 6): rehashing instead of keeping key-order
  signatures (10.3 bytes/key but three hashes per key: too slow for
  strings); the earlier 18.6-bytes/key design with two copies of the
  signatures and a global `bucket_begin`; a coarse pre-filter and
  prefetching in the bumped scan. Lazy loading was initially 2× slower
  with 8 threads at 10⁷ because gap sweeps loaded whole parts: loads are
  now restricted to the bucket range of the sweep.
- `sweep_level` splits buckets into chunks separated by gaps of
  ⌊L·B/num_slices⌋ + 1 buckets (it must be `num_slices`, not `m`: using `m`
  was a real bug, covered by `test_many_chunks`), sweeps chunks in parallel,
  then gaps in parallel with the slots of the neighboring buckets marked as
  used. Each `Sweep` returns the slots it used.
- `Sweep::search` scans the keys of a bucket once: for each pattern it
  rotates the ring of the key and accumulates the blocked seeds, the sum of
  the slots for the first seed, and the ring indices packed into 8/16/32-bit
  fields. Free seeds are then evaluated in a single loop (`Sweep::sum`: one
  addition, one mask and one population count per group of keys); with more
  than `MAX_FREE` = 32 free seeds `search_many` considers only the first
  free seed after each wrap point. `place` marks the slots and detects two
  keys of the bucket on the same slot (then `search_distinct` tries the
  other candidates in order).
- The set of used slots is stored by rows (residues modulo the stride) and
  columns (64 strides): word c·T + ρ contains the bits of slots (64c + i)·T
  + ρ. Columns are recycled cyclically; `retire_column` copies the bits of a
  column to the occupancy bitmap of the sweep before clearing it.
- `Window` replaces the priority queue: one FIFO queue per bucket size
  (sizes ≥ 64 go to a heap). The order is exactly that of the previous
  binary heap.
- `Sweep::run` calls `sweep` with either the actual `Rings` or the constant
  `Rings::DEFAULT` (8-bit seeds, R = 4, L = 1024): in the second case shifts
  and masks are constants and the loop over the patterns is unrolled.
- Levels after the first: the signature is `level_sig(h, h', salt)`, where
  h is the first-level signature, h' = `to_sig(key, seed ^ SECOND_HASH)` and
  `salt` is `LevelParams::salt`. The last level (≤ 4096 keys) does not bump
  and retries with a different salt. Keys with the same first-level
  signature are bumped (they collide for every seed) and separated at the
  next level; duplicate keys always reach the second level, where keys of a
  bucket with the same signature are hashed with a third seed.
- Queries: `get` inlines the first level; `get_slow(h, h')` is cold and out
  of line. `fast_scale`/`default_shifts` select constant shifts (see
  `pos0`).
- Priority weights are those of PHast+ with wrapping (δ = 3) in Beling's
  implementation (`default_weights`); they have not been retuned for rings.

## Limits of repair (Section 5 of the paper)

All numbers in Section 5 are space-only or counts, so they do not depend on
the hardware and need not be rerun. Commands (from `paper/lab`):

- Table 3 (sux, bump rate and bits/key by repair depth and breadth):
  `RAYON_NUM_THREADS=1 target/release/bumps 10000000 8:10:<d>:<cand>:<λ>...`
  (spec `<S>:<log2 L>:<depth>:<candidates>:<lambda>`).
- Proposition 1 check (mean range of the bridge vs √(πn/2)):
  `target/release/bridge 200 4096 65536 1048576`.
- Unbounded (PtrHash-style) eviction: `target/release/walk 65536 10000 2
  8:9:2:5.0:1.0:prio 8:9:2:5.0:0.95:prio` (`capped@` counts episodes over
  the cap; options `a<age cost>`, `o<owner cost>`, `z<zone>`, `single`).
- CONSENSUS-style search with chained seeds (`lab::dfs`):
  `target/release/dfs 100000 2000 8:9:2:4.0:8`; it never places more than
  ~160 buckets in any configuration.

## Open issues and next steps

1. **Queries**: at 10⁸ keys PHast (λ = 4.5) is 2% faster than PHast-R with
   λ = 4.75, because it bumps fewer keys; λ = 4.5 reverses the result with
   0.4% more space. Possible further steps: with λ = 5 the slice of a key
   could be 5 times its bucket (one `lea` instead of a multiplication, about
   1 ns), but the larger bump rate cancels the gain; bit-packed seeds
   (S > 8) cost about 4 ns per query on x86 whatever the structure.
2. **Seeds of 11–12 bits**: retune λ and weights with L = 4096/8192
   (`tune <keys> 12:12:1:7.5,8,8.5 12:13:1:8,8.5,9`, then `wtune`).
3. **Scale**: construction keeps 10.7 bytes per key (ph: 9.9); the walk
   finding bumped keys is DRAM-bound at 10⁹ keys (random accesses to a
   28 MB bit vector). Nothing was run beyond 10⁹ keys. `redo.sh` reruns
   everything on another machine (results in `results/paper/<host>/`).
4. **Repair cost**: depth-2 repair costs ~50 ns/key more than depth 1, mostly
   in nested repairs of evicted buckets that fail.
5. **String keys / other MPHFs**: add PHast-R to Beling's `mphf_benchmark`
   or to Lehmann's MPHF-Experiments to compare with PtrHash, PHOBIC, etc. on
   the standard workload (random strings of 10–50 bytes).
6. **Paper**: author line is empty; the tables are generated by
   `lab/results/paper/tables.py` from the output of `run.sh`; the new entry
   for PHOBIC in `biblio.bib` should be checked; `lean/README.md` cites line
   numbers of `phast.tex`, which change with every edit above Section 5.
   The priority weights have not been retuned for rings (`wtune`), and for
   fewer than about 10⁵ keys PHast-R uses more space than PHast+ with
   wrapping (fixed overhead and geometry of small levels).
7. **The log₂e + O(log λ/λ) conjecture** of the PHast paper: for reference
   PHast, the excess over log₂e divided by ln λ/λ is ≈ 1.5 for λ in
   4.6–7.2, but grows for λ ≥ 7.8 (excess flattening at 0.40; see
   `lab/results/conj`). A proof would reduce to showing that the greedy sweep
   has bump rate O(1/λ) with S = λ log₂e + O(log λ) (then seeds and holes
   both cost O(log λ/λ)); that step is open.
