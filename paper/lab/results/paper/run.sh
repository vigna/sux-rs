#!/usr/bin/env bash
# Experiments of the paper: PHast, PHast+ (with and without wrapping) and
# PHast-R on u64 keys with GxHash. Single thread pinned to a core, queries
# as medians of 9 interleaved rounds; then construction with several
# threads. With 10^9 keys the peak memory usage is about 30 GB, with 10^10
# keys about 220 GB.
# Usage: run.sh <cmp binary> <output prefix>
# Environment: CPU1 (the core for single-threaded runs), CPUS (the cores
# for multithreaded runs, as a taskset list), THREADS (their number), SIZES
# (the numbers of keys); see redo.sh.
B=${1:-../../target-nexus/release/cmp}
O=${2:-run}
CPU1=${CPU1:-2}
CPUS=${CPUS:-0-7}
THREADS=${THREADS:-8}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
V8=ref:plus:8:5.25,ref:w1:8:5.25,ref:w2:8:5.0,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:4.5,r:8:10:4.75,r:8:10:5.0
V10=ref:plus:10:5.15,ref:w1:10:6.2,ref:w2:10:5.9,ref:w3:10:6.0,ref:phast:10:6.05,r:10:11:5.75:2:bfvu,r:10:11:6.0:2:bfvu,r:10:11:6.25:2:bfvu
W8=ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:4.5,r:8:10:4.75,r:8:10:5.0
W10=ref:w3:10:6.0,r:10:11:5.75:2:bfvu,r:10:11:6.0:2:bfvu
{
for n in $SIZES; do
  if [ $n = 10000000 ]; then V=$V8,$V10; else V=$W8,ref:phast:8:4.5,$W10; fi
  RAYON_NUM_THREADS=1 taskset -c $CPU1 $B -n $n -q 2000000 --interleave 9 -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
done
for n in $SIZES; do
  RAYON_NUM_THREADS=$THREADS taskset -c $CPUS $B -n $n -t $THREADS -q 1000 --interleave 1 -v $W8,$W10 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$THREADS,/"
done
} > $O.csv
echo done >> $O.csv
