#!/usr/bin/env bash
# Exhaustive comparison over sizes (GxHash, `performance` governor).
# raw.csv: single thread, pinned to core 2, 5 key sets, interleaved queries.
# raw_mt.csv: 8 threads (taskset 0-7), construction and space only.
# Lines: <key set>,CSV,<name>,<n>,<bits/key>,<build ns/key>,<query ns>
B=${CARGO_TARGET_DIR:-target}/release/cmp
V=ref:plus:8:5.25,ref:w1:8:5.25,ref:w2:8:5.0,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:0:4.75,r:8:10:1:4.75,r:8:10:2:4.75
for n in 1000 1500 2000 3000 5000 7000 10000 15000 20000 30000 50000 70000 100000 200000 500000 1000000 2000000 5000000 10000000; do
  for ks in 0 1 2 3 4; do
    RAYON_NUM_THREADS=1 taskset -c 2 $B -n $n -q 1000000 --interleave 7 --key-seed $ks -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done > results/sizes/raw.csv
for n in 10000 30000 100000 300000 1000000 3000000 10000000 30000000 100000000; do
  K="0 1 2"; [ $n -ge 30000000 ] && K="0 1"
  for ks in $K; do
    RAYON_NUM_THREADS=8 taskset -c 0-7 $B -n $n -q 1000 -r 1 -t 8 --key-seed $ks -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done > results/sizes/raw_mt.csv
