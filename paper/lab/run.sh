#!/usr/bin/env bash
#
# Reproduces the experiments of ../phast.tex on the current machine.
#
# Usage: ./run.sh [env|tables|anatomy|tune|all]...   (default: env tables)
#
# Environment variables:
#   N1       number of keys for the small experiments  (default 10000000)
#   N2       number of keys for the large experiments  (default 100000000)
#   THREADS  threads for multithreaded construction    (default: all cores)
#   CORE     on Linux, core used to pin single-threaded runs (default 2)
#   NTUNE    number of keys (per key set, eight key sets) for weight tuning
#            (default 10000000)
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
NTUNE=${NTUNE:-10000000}
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

# Section 2: anatomy of PHast+ (space breakdown, bump rates by size,
# self-collisions, seed entropy), measured on the reference implementation.
step_anatomy() {
	pin "$BIN/anatomy" -n "$N1" -s 8 -l 5.25 | tee "$OUT/anatomy.txt"
}

# Tuning of the bucket-priority weights of the default configurations (the
# key sets are built concurrently, so the tuner is not pinned).
step_tune() {
	"$BIN/wtune" 8 10 4.25 -n "$NTUNE" | tee "$OUT/wtune_8_10.txt"
	"$BIN/wtune" 10 11 5.75 -n "$NTUNE" | tee "$OUT/wtune_10_11.txt"
}

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
	steps=(env tables)
fi
if [ "${steps[0]}" = all ]; then
	steps=(env tables anatomy tune)
fi

build
for s in "${steps[@]}"; do
	"step_$s"
done
echo "Results in $OUT"
