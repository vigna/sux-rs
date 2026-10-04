#!/usr/bin/env bash
# Space and construction time (8 threads) of PHast-R with wrapping and
# depth-1 repair, 3e6 keys: multiplier x slice length x bucket size
B=../../target-nexus/release/cmp
for ll in 9 10 11; do
  V=ref:w3:8:5.0
  for m in 2 3 4 5 6; do
    for l in 4.5 4.75 5.0 5.25 5.5; do
      V=$V,r:8:$ll:1:$l:w$m
    done
  done
  RAYON_NUM_THREADS=8 taskset -c 0-7 $B -n 3000000 -t 8 -q 1000 --interleave 1 -v $V 2>&1 | grep -v CSV
done
