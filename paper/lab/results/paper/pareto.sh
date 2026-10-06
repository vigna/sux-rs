#!/usr/bin/env bash
# Space/time trade-offs (the plots of the paper): sweeps of the expected
# bucket size for PHast, PHast+ with and without wrapping, and PHast-R, with
# 8-bit and 10-bit seeds, single-threaded, up to 10^9 keys (about an hour and
# a half, mostly spent building PHast).
# Usage: pareto.sh <cmp binary> <output prefix>
# Environment: CPU1, SIZES; see redo.sh.
B=${1:-../../target-nexus/release/cmp}
O=${2:-pareto}
CPU1=${CPU1:-2}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
V=ref:plus:8:5.0,ref:plus:8:5.25,ref:plus:8:5.5
V=$V,ref:w3:8:4.5,ref:w3:8:4.75,ref:w3:8:5.0,ref:w3:8:5.25,ref:w3:8:5.5
V=$V,ref:phast:8:4.25,ref:phast:8:4.5,ref:phast:8:4.75
V=$V,r:8:10:4.25,r:8:10:4.5,r:8:10:4.75,r:8:10:5.0,r:8:10:5.25
V=$V,ref:w3:10:5.5,ref:w3:10:5.75,ref:w3:10:6.0,ref:w3:10:6.25,ref:w3:10:6.5
V=$V,r:10:11:5.5:2:bfvu,r:10:11:5.75:2:bfvu,r:10:11:6.0:2:bfvu,r:10:11:6.25:2:bfvu,r:10:11:6.5:2:bfvu
for n in $SIZES; do
  [ $n -gt 1000000000 ] && continue
  RAYON_NUM_THREADS=1 taskset -c $CPU1 $B -n $n -q 2000000 --interleave 9 -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
done > $O.csv
echo done >> $O.csv
