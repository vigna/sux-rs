#!/bin/bash
#
# Full CompVFunc solvability sweep.
#
# Builds the stress binary, then runs:
#   Phase 1: n=100M, all ShardEdges, all distributions, 5 trials
#   MWHC:    n=10..100M, both MWHC variants, all distributions, 5 trials
#   Phase 2: n=1G, production ShardEdges, 2-3 trials
#   Phase 3: n=1..1M (dense), Fuse3 variants, 3 distributions, 5 trials
#
# Output: one CSV per (shard_edge, distribution) cell in $OUTDIR.
# Summary: run  python3 scripts/stress_summary.py $OUTDIR/*.csv
#
# Usage:
#   ./scripts/full_comp_vfunc_sweep.sh [OUTDIR]
#
# Default OUTDIR: /tmp/comp_vfunc_sweep_$(date +%Y%m%d)
#
set -u
cd "$(dirname "$0")/.."

OUTDIR="${1:-/tmp/comp_vfunc_sweep_$(date +%Y%m%d)}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"

log() { echo "[$(date +%Y-%m-%d' '%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── Build (single binary with all features including MWHC) ──────────
log "Building stress binary..."
cargo build --release --features "cli epserde deko rayon mwhc" \
    --example comp_vfunc_stress 2>&1 | tee -a "$LOG"
BIN=${CARGO_TARGET_DIR:-target}/release/examples/comp_vfunc_stress

run_cell() {
    local bin="$1" shard="$2" dist="$3" zipf_s="$4" nlist="$5"
    local trials="$6" queries="${7:-1024}" tag="${8:-}"
    local name="${shard}_${dist}${tag}"
    local out="$OUTDIR/${name}.csv"
    local logf="$OUTDIR/${name}.log"
    log "  $name (n-list=$(echo $nlist | cut -c1-40)... trials=$trials)"
    local t0=$SECONDS
    "$bin" --shard-edge "$shard" --distribution "$dist" \
        --zipf-s "$zipf_s" --zipf-n 1000 --uniform-max 1024 \
        --n-list "$nlist" --trials "$trials" --queries "$queries" \
        > "$out" 2> "$logf" || log "    ERROR: $name"
    local dt=$((SECONDS - t0))
    local rows=$(wc -l < "$out")
    log "    done ${dt}s ($rows rows)"
}

NLIST_100M="100000000"
NLIST_MULTI="10,100,1000,10000,100000,1000000,10000000,100000000"
NLIST_1G="1000000000"

# ── Phase 1: n=100M, all non-MWHC ShardEdges ───────────────────────
log "=== PHASE 1: n=100M, all ShardEdges ==="
for shard in fuse-lge3-shards fuse-lge3-no-shards1 fuse-lge3-no-shards2 \
             fuse-lge3-full-sigs fuse3-shards fuse3-no-shards1 fuse3-no-shards2; do
    for dist in constant uniform geom zipf; do
        run_cell "$BIN" "$shard" "$dist" 1.0 "$NLIST_100M" 5
    done
    # zipf s=2
    run_cell "$BIN" "$shard" zipf 2.0 "$NLIST_100M" 5 1024 _s2
done

# ── MWHC: n=10..100M ───────────────────────────────────────────────
log "=== MWHC: n=10..100M ==="
for shard in mwhc3-shards mwhc3-no-shards; do
    for dist in constant uniform geom zipf; do
        run_cell "$BIN" "$shard" "$dist" 1.0 "$NLIST_MULTI" 5
    done
    run_cell "$BIN" "$shard" zipf 2.0 "$NLIST_MULTI" 5 1024 _s2
done

# ── Phase 2: n=1G, production ShardEdges ────────────────────────────
log "=== PHASE 2: n=1G ==="
for dist in constant uniform geom zipf; do
    run_cell "$BIN" fuse3-shards "$dist" 1.0 "$NLIST_1G" 3
done
run_cell "$BIN" fuse3-shards zipf 2.0 "$NLIST_1G" 3 1024 _s2

for dist in constant geom uniform zipf; do
    run_cell "$BIN" fuse3-no-shards1 "$dist" 1.0 "$NLIST_1G" 2
done

for dist in constant geom zipf; do
    run_cell "$BIN" fuse3-no-shards2 "$dist" 1.0 "$NLIST_1G" 2
done

for dist in constant geom zipf; do
    run_cell "$BIN" mwhc3-shards "$dist" 1.0 "$NLIST_1G" 2
done

# ── Phase 3: n=1..1M dense, FuseLge3 variants ──────────────────────
log "=== PHASE 3: n=1..1M dense ==="
# Build dense n-list: step 1 in [1,1K], step 10 in [1K,10K],
# step 100 in [10K,100K], step 1000 in [100K,1M].
NLIST_DENSE=$(python3 -c '
ns = list(range(1, 1001))
ns += list(range(1010, 10001, 10))
ns += list(range(10100, 100001, 100))
ns += list(range(101000, 1000001, 1000))
print(",".join(str(n) for n in ns))
')
log "Dense n-list: $(echo "$NLIST_DENSE" | tr ',' '\n' | wc -l | tr -d ' ') values"

for shard in fuse-lge3-shards fuse-lge3-no-shards2 mwhc3-shards; do
    run_cell "$BIN" "$shard" zipf    1.0 "$NLIST_DENSE" 5 64
    run_cell "$BIN" "$shard" zipf    2.0 "$NLIST_DENSE" 5 64 _s2
    run_cell "$BIN" "$shard" uniform 1.0 "$NLIST_DENSE" 5 64
done

log "=== ALL PHASES DONE ==="
log "Results in $OUTDIR"
log "Summary: python3 scripts/stress_summary.py $OUTDIR/*.csv"
