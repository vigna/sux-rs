#!/usr/bin/env bash
# Space of PHast-R (S=8, depth 1, lambda=4.75) vs slice length on small sets,
# 5 key sets per size
v=""; for ll in 5 6 7 8 9 10; do v="$v,r:8:$ll:1:4.75"; done
for n in 1000 1500 2000 3000 5000 7000 10000 15000 20000 30000 50000 100000 200000 500000; do
  for ks in 0 1 2 3 4; do
    RAYON_NUM_THREADS=1 taskset -c 2 ${CARGO_TARGET_DIR:-target}/release/cmp -n $n -q 1000 -r 1 --key-seed $ks -v ${v#,} 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done
