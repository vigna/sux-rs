# PHast-R: handoff notes

State of the work on PHast-R as of 2026-10-03, written to continue it on
other hardware (possibly in a new Claude Code session, which will not have
the context of the original one: point it to this file).

## What is here

- `src/func/phast_r.rs`: the implementation (`PHastR`, `PHastRBuilder`,
  `SeedStore`/`SeedStoreBuild`), with unit tests.
- `examples/bench_phast_r.rs`: quick benchmark of PHast-R alone
  (`cargo run --release --example bench_phast_r -- 10000000`).
- `paper/phast.tex`: the paper (6 pages, `latexmk -pdf phast.tex`).
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
and then interleaves batches of queries, reporting medians. Note that
`PHastR<K>` defaults to `[u64; 2]` signatures (XXH3-128), which are safer for
huge key sets but slower to compute; the benchmarks use `[u64; 1]`.

## Comparison driver

`target/release/cmp -n <keys> -v <spec>,<spec>,...` where a spec is

- `ref:<plus|w1|w2|w3|phast>:<S>:<lambda>` for PHast+, PHast+ with wrapping,
  or PHast (reference implementation);
- `r:<S>:<log2 L>:<depth>:<lambda>[:<log2 R>[:<u8|u16|bfv>]]` for PHast-R.

`-t <threads>` sets the threads of the reference; PHast-R uses rayon
(`RAYON_NUM_THREADS`). Every PHast-R structure is verified to be a bijection.

`target/release/qsplit -n <keys> -v <S>:<log2 L>:<depth>:<lambda>,...` times
separately the keys placed in the first level and the bumped keys, and
prints the bump rate.

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
- `next_level` remixes only *h* (`(mix64(h ^ salt ^ c), o)`; `mix64` is a
  bijection). The previous two independent mixes were vectorized by LLVM with
  AVX-512 `vpmullq` (~15 cycles latency each), making each bumped key ~15 ns
  slower than in the reference; now the slow paths cost the same.

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

## Key findings (see Section 2 of the paper; reference implementation)

- With output range m = n, holes = bumped keys; each hole costs about
  log₂(1/β) + 2 bits in Elias–Fano, close to the entropy of the hole set, so
  the only lever is the bump rate β.
- 38.5% of PHast+ bumped buckets are self-collisions (2.6% ≈ λ²/2L of the
  buckets); seeds are incompressible (7.71 bits of entropy).
- Independent offset patterns fix the rigidity; repair (half adder +
  eviction) recovers PHast's space.

## Implementation notes

- Level 0 sweep: `sweep_level` splits buckets into chunks separated by gaps
  of ⌈(L + D)B/num_slices⌉ + 2 buckets (it must be `num_slices`, not `m`:
  using `m` was a real bug, now covered by `test_many_chunks`), sweeps chunks
  in parallel, then gaps with neighbor slots pre-marked as non-evictable.
- `Sweep::place` = `search` (min-sum over patterns, bit-parallel shifts) +
  repair (exactly-one masks, candidates sorted by sum, one per distinct
  blocker, nested repairs try a quarter of the candidates).
- Levels after the first derive hashes by `next_level(h, o, salt)`, so keys
  are never rehashed; the last level (< 4096 keys) does not bump and retries
  with a salt.
- Priority weights for (S=8, L=512) and (S=10, L=2048) were retuned with
  `wtune`; the others come from Beling's PHast+ (`ShiftOnly`) tables.

## Open issues and next steps

1. **Query fast path**: ~2 instructions more than PHast+ are inherent in
   selecting a pattern; bit-packed seeds (S > 8) add ~2 ns more on x86.
   Consider λ = 4.5 / depth 1 as the default (lower bump rate, faster
   queries and construction, 1.964 bits/key), and retune its priority
   weights with `wtune`.
2. **Seeds of 11–12 bits**: retune λ and weights with L = 4096/8192
   (`tune <keys> 12:12:1:7.5,8,8.5 12:13:1:8,8.5,9`, then `wtune`).
3. **Scale**: 10⁹ keys (construction keeps 16 bytes per key plus bucket
   arrays), thread scaling beyond 10 threads.
4. **Repair cost**: depth-2 repair costs ~50 ns/key more than depth 1, mostly
   in nested repairs of evicted buckets that fail.
5. **String keys / other MPHFs**: add PHast-R to Beling's `mphf_benchmark`
   or to Lehmann's MPHF-Experiments to compare with PtrHash, PHOBIC, etc. on
   the standard workload (random strings of 10–50 bytes).
6. **Paper**: author line is empty; Section 3 describes the new encoding, but
   Tables 1–2 and Figure 1 still contain the M1 numbers of the old encoding:
   regenerate them (the new numbers are in `lab/results/README-nexus.md`).
7. **The log₂e + O(log λ/λ) conjecture** of the PHast paper: for reference
   PHast, the excess over log₂e divided by ln λ/λ is ≈ 1.5 for λ in
   4.6–7.2, but grows for λ ≥ 7.8 (excess flattening at 0.40; see
   `lab/results/conj`). A proof would reduce to showing that the greedy sweep
   has bump rate O(1/λ) with S = λ log₂e + O(log λ) (then seeds and holes
   both cost O(log λ/λ)); that step is open.
