#!/usr/bin/env bash
# 512 vs 1024 crossover, 3 key sets per size
for n in 700000 1000000 1500000 2000000 3000000 5000000; do
  for ks in 0 1 2; do
    RAYON_NUM_THREADS=1 taskset -c 2 ${CARGO_TARGET_DIR:-target}/release/cmp -n $n -q 1000 -r 1 --key-seed $ks -v r:8:9:1:4.75,r:8:10:1:4.75 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done
