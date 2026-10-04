#!/usr/bin/env bash
# 8-thread part of run.sh only
B=${CARGO_TARGET_DIR:-target}/release/cmp
V=ref:plus:8:5.25,ref:w1:8:5.25,ref:w2:8:5.0,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:0:4.75,r:8:10:1:4.75,r:8:10:2:4.75
for n in 10000 30000 100000 300000 1000000 3000000 10000000 30000000 100000000; do
  K="0 1 2"; [ $n -ge 30000000 ] && K="0 1"
  for ks in $K; do
    RAYON_NUM_THREADS=8 taskset -c 0-7 $B -n $n -q 1000 -r 1 -t 8 --key-seed $ks -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done > results/sizes/raw_mt.csv
