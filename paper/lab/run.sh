#!/usr/bin/env bash
#
# Reproduces the experiments of ../phast.tex on the current machine.
#
# Usage: ./run.sh [env|tables|anatomy|negative|tune|all]...   (default: env tables)
#
# Environment variables:
#   N1       number of keys for the small experiments  (default 10000000)
#   N2       number of keys for the large experiments  (default 100000000)
#   THREADS  threads for multithreaded construction    (default: all cores)
#   CORE     on Linux, core used to pin single-threaded runs (default 2)
#   NTUNE    number of keys (per key set, two key sets) for weight tuning
#            (default 4000000)
#   OUT      output directory (default results/<hostname>-<date>)
#
# Each step writes <step>.txt (human-readable output) and, for the
# comparisons, <step>.csv (lines CSV,<name>,<n>,<bits/key>,<build ns/key>,<query ns>)
# and <step>.err (raw standard error).

set -euo pipefail
cd "$(dirname "$0")"

N1=${N1:-10000000}
N2=${N2:-100000000}
if command -v nproc > /dev/null; then
	THREADS=${THREADS:-$(nproc)}
else
	THREADS=${THREADS:-$(sysctl -n hw.ncpu)}
fi
CORE=${CORE:-2}
NTUNE=${NTUNE:-4000000}
OUT=${OUT:-results/$(hostname -s)-$(date +%Y%m%d)}
mkdir -p "$OUT"

BIN=${CARGO_TARGET_DIR:-target}/release

# Pins single-threaded runs to a core on Linux; no-op elsewhere.
pin() {
	if command -v taskset > /dev/null; then
		taskset -c "$CORE" "$@"
	else
		"$@"
	fi
}

# Runs the comparison driver: cmp_run <1|mt> <name> <cmp arguments>...
# Single-threaded runs (1) use a single rayon thread and are pinned.
cmp_run() {
	local mode=$1 name=$2
	shift 2
	echo "### $name"
	if [ "$mode" = 1 ]; then
		RAYON_NUM_THREADS=1 pin "$BIN/cmp" "$@" 2> "$OUT/$name.err" | tee "$OUT/$name.txt"
	else
		RAYON_NUM_THREADS=$THREADS "$BIN/cmp" "$@" 2> "$OUT/$name.err" | tee "$OUT/$name.txt"
	fi
	grep '^CSV' "$OUT/$name.err" > "$OUT/$name.csv" || true
}

build() {
	cargo build --release
}

step_env() {
	{
		echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "host: $(hostname)"
		uname -a
		rustc -Vv
		cargo -V
		echo "sux: $(git -C ../.. rev-parse HEAD) $(git -C ../.. status --porcelain | wc -l | tr -d ' ') dirty files"
		echo "ph: $(grep -A2 'name = "ph"' Cargo.lock | grep source)"
		echo "rustflags: $(grep rustflags .cargo/config.toml)"
		echo "THREADS=$THREADS N1=$N1 N2=$N2 CORE=$CORE NTUNE=$NTUNE"
		if command -v lscpu > /dev/null; then
			lscpu
			echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null || echo n/a)"
			echo "boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2> /dev/null || cat /sys/devices/system/cpu/intel_pstate/no_turbo 2> /dev/null || echo n/a)"
			free -g
		else
			sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize hw.l1dcachesize hw.l2cachesize 2> /dev/null || true
		fi
	} | tee "$OUT/env.txt"
}

# Tables 1 and 2 of the paper. Query times are medians of 11 interleaved
# rounds of 2M queries per structure (see --interleave in cmp.rs), which makes
# them immune to frequency drift (e.g., HWP lowering the clock of
# memory-bound phases); construction times are single measurements.
Q="-q 2000000 --interleave 11"

step_tables() {
	local ref1=ref:plus:8:5.25,ref:w1:8:5.25,ref:w2:8:5.0,ref:w3:8:5.0,ref:phast:8:4.5
	local ref2=ref:plus:10:5.15,ref:w1:10:6.2,ref:w2:10:5.9,ref:w3:10:6.0,ref:phast:10:6.05
	local r=r:8:9:0:5.0,r:8:9:1:5.0,r:8:9:2:5.0,r:8:9:2:4.75,r:9:10:1:5.75,r:10:11:0:6.0,r:10:11:1:6.0,r:10:11:2:6.25,r:11:12:1:6.75
	cmp_run 1 table1 -n "$N1" $Q -v "$ref1,$ref2,$r"

	local large=ref:plus:8:5.25,ref:w3:8:5.0,ref:w3:10:6.0,r:8:9:1:5.0,r:8:9:2:5.0,r:8:9:2:4.75,r:10:11:1:6.0,r:10:11:2:6.25
	cmp_run 1 table2-1thread -n "$N2" $Q -v "ref:phast:8:4.5,$large"
	cmp_run mt table2-mt -n "$N2" $Q -t "$THREADS" -v "$large"
}

# Section 2: anatomy of PHast+ (bump rates by size, self-collisions, seed
# entropy), with the reference size for comparison.
step_anatomy() {
	pin "$BIN/baseline" -n "$N1" -s 8 -l 5.25 --reference --hist | tee "$OUT/anatomy.txt"
}

# Table 3 of the paper (negative results), plus the repair parameter sweep.
step_negative() {
	{
		echo "### overloaded / underloaded first level"
		pin "$BIN/overload" -n "$N1" -s 8 -l 5.25 --gamma=0,0.04,-0.05
		echo "### block reseeding"
		pin "$BIN/blocks" -n "$N1" -g 256 -t 1,16 -l 5.25
		echo "### mixed bucket sizes"
		pin "$BIN/mix" -n "$N1" -q 3 -b 6 -m 2
		echo "### filler sweep"
		pin "$BIN/two" -n "$N1" -p 0,0.1 --l0 5.25 --l1 1 --s1 6
		echo "### two-tier seeds"
		pin "$BIN/esc" -n "$N1" --s2 0,12 --r2 16 -l 5.25
		echo "### chained seeds"
		pin "$BIN/chain" -n "$N1" -L 512 -t 0,64 -l 5
		echo "### fingerprint thresholds: oracle (free encoding)"
		pin "$BIN/two" -n "$N1" -p 0 --l0 5.25 --thr 7
		echo "### fingerprint thresholds: encoded in the seed"
		pin "$BIN/thr" -n "$N1" -a 239 -t 4 -l 5.25
		echo "### lattice-steered holes"
		pin "$BIN/lattice" -n "$N1" -q 8 -p 0,4,256
		echo "### row-permutation placement"
		pin "$BIN/rows" -n "$N1" --rowmode 1 -r 8 -c 0 -s 8 -l 5.5
		echo "### k-perfect PHast (reference)"
		pin "$BIN/kperf" $((N1 / 2))
		echo "### cuckoo repair on regular PHast"
		pin "$BIN/evict" -n "$N1" -r 0 -L 1024 -c 0,8 -d 1 -l 4.5
		echo "### patterns x shifts"
		pin "$BIN/multi" -n "$N1" -r 1,2,4,8 -L 512 -l 4.75,5.25
		echo "### repair sweep (lab version)"
		pin "$BIN/evict" -n "$N1" -r 2,4 -c 16,64 -d 2,3 -l 4.75,5.25,5.75
	} 2>&1 | tee "$OUT/negative.txt"
}

# Coordinate-descent tuning of the bucket-priority weights.
step_tune() {
	pin "$BIN/wtune" 8 9 1 5.0 "$NTUNE" | tee "$OUT/wtune_8_9_1.txt"
	pin "$BIN/wtune" 10 11 1 6.0 "$NTUNE" | tee "$OUT/wtune_10_11_1.txt"
}

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
	steps=(env tables)
fi
if [ "${steps[0]}" = all ]; then
	steps=(env tables anatomy negative tune)
fi

build
for s in "${steps[@]}"; do
	"step_$s"
done
echo "Results in $OUT"
