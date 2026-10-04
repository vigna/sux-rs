#!/usr/bin/env bash
# Small key sets: 5 runs per size, single thread, pinned; GxHash
V=ref:plus:8:5.25,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:1:4.75,r:8:10:2:4.75
for n in 1000 2000 3000 5000 7000 10000 20000 50000; do
  for run in 1 2 3 4 5; do
    RAYON_NUM_THREADS=1 taskset -c 2 ${CARGO_TARGET_DIR:-target}/release/cmp -n $n -q 2000000 --interleave 5 -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$run,/"
  done
done
