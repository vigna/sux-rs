#!/usr/bin/env bash
for n in 1000 2000 3000 5000 7000 10000 20000 50000 200000 1000000 10000000; do
  for ks in 0 1 2; do
    RAYON_NUM_THREADS=1 taskset -c 2 ${CARGO_TARGET_DIR:-target}/release/cmp -n $n -q 1000 -r 1 --key-seed $ks -v ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:1:4.75 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done
