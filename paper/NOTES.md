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
  (by path) and on the reference implementation `ph` (pinned git revision).
  It contains the comparison driver, the tuning tools, and the prototypes of
  all the ideas in the "What did not work" section.
- `paper/lab/results/m1-max/`: the raw results behind the paper (see the
  README there for caveats).

## Running on a new machine

```sh
git clone git@github.com:vigna/sux-rs.git && cd sux-rs && git checkout phast-r
cd paper/lab
./run.sh                      # env + tables (10-15 minutes, ~4 GiB of RAM)
./run.sh anatomy negative     # Section 2 and Table 3
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
hash keys with XXH3-64 (the `BuildX` hasher in `cmp.rs` reproduces sux's
`ToSig<[u64; 1]>` for `u64`), so query times are comparable. Note that
`PHastR<K>` defaults to `[u64; 2]` signatures (XXH3-128), which are safer for
huge key sets but slower to compute; the benchmarks use `[u64; 1]`.

## Comparison driver

`target/release/cmp -n <keys> -v <spec>,<spec>,...` where a spec is

- `ref:<plus|w1|w2|w3|phast>:<S>:<lambda>` for PHast+, PHast+ with wrapping,
  or PHast (reference implementation);
- `r:<S>:<log2 L>:<depth>:<lambda>[:<log2 R>[:<u8|u16|bfv>]]` for PHast-R.

`-t <threads>` sets the threads of the reference; PHast-R uses rayon
(`RAYON_NUM_THREADS`). Every PHast-R structure is verified to be a bijection.

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

## Key findings (see Section 2 and 5 of the paper)

- With output range m = n, holes = bumped keys; each hole costs about
  log₂(1/β) + 2 bits in Elias–Fano, close to the entropy of the hole set, so
  the only lever is the bump rate β.
- Bumping is the slack a greedy sweep needs: removing it (wide secondary
  seeds) makes the sweep jam (escapes 6.8% → 46%).
- ~38% of PHast+ bumped buckets are self-collisions (≈ λ²/2L of the buckets);
  seeds are incompressible (7.73 bits of entropy).
- Independent offset patterns fix the rigidity; repair (half adder +
  eviction) recovers PHast's space. Repair does nothing for regular PHast
  (pseudorandom placement is already flexible).

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

1. **Bit-packed seed queries** (S > 8) are 2.5–4 ns slower than the reference
   on the M1. Not cache-line crossings (a 32-bit read did not help) nor a
   runtime width (a const-width version did not help): it looks like
   latency on the critical path of `get_by_ho`. `qiso` and `seedread`
   isolate it; check whether it reproduces on x86.
2. **Seeds of 11–12 bits**: retune λ and weights with L = 4096/8192
   (`tune <keys> 12:12:1:7.5,8,8.5 12:13:1:8,8.5,9`, then `wtune`).
3. **Scale**: 10⁹ keys (construction keeps 16 bytes per key plus bucket
   arrays), thread scaling beyond 10 threads.
4. **Repair cost**: depth-2 repair costs ~50 ns/key more than depth 1, mostly
   in nested repairs of evicted buckets that fail.
5. **String keys / other MPHFs**: add PHast-R to Beling's `mphf_benchmark`
   or to Lehmann's MPHF-Experiments to compare with PtrHash, PHOBIC, etc. on
   the standard workload (random strings of 10–50 bytes).
6. **Paper**: author line is empty; update the experimental section with
   the new hardware.
