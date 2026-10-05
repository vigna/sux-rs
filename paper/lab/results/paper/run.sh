#!/usr/bin/env bash
# Experiments of the paper: PHast, PHast+ (with and without wrapping) and
# PHast-R on u64 keys with GxHash. Single thread pinned to a core, queries
# as medians of 9 interleaved rounds; then construction with 8 threads.
# With 10^9 keys the peak memory usage is about 45 GB.
# Usage: run.sh <cmp binary> <output prefix>
B=${1:-../../target-nexus/release/cmp}
O=${2:-run}
V8=ref:plus:8:5.25,ref:w1:8:5.25,ref:w2:8:5.0,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:4.5,r:8:10:4.75,r:8:10:5.0
V10=ref:plus:10:5.15,ref:w1:10:6.2,ref:w2:10:5.9,ref:w3:10:6.0,ref:phast:10:6.05,r:10:11:6.0:2:bfvu,r:10:11:6.25:2:bfvu
W8=ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:4.5,r:8:10:4.75,r:8:10:5.0
W10=ref:w3:10:6.0,r:10:11:6.0:2:bfvu
{
for n in 10000000 100000000 1000000000; do
  if [ $n = 10000000 ]; then V=$V8,$V10; else V=$W8,ref:phast:8:4.5,$W10; fi
  RAYON_NUM_THREADS=1 taskset -c 2 $B -n $n -q 2000000 --interleave 9 -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
done
for n in 10000000 100000000 1000000000; do
  RAYON_NUM_THREADS=8 taskset -c 0-7 $B -n $n -t 8 -q 1000 --interleave 1 -v $W8,$W10 2>&1 >/dev/null | grep '^CSV' | sed "s/^/8,/"
done
} > $O.csv
echo done >> $O.csv
