#!/usr/bin/env bash
# 64-bit (o = h * c) vs 128-bit (o = upper half of GxHash) signatures, single
# thread, pinned, interleaved queries
B=${CARGO_TARGET_DIR:-target}/release/cmp
V=ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:1:4.75:2:u8:s64,r:8:10:1:4.75:2:u8:s128
for n in 10000 100000 1000000 10000000 100000000; do
  K="0 1 2"; [ $n -ge 100000000 ] && K="0"
  for ks in $K; do
    RAYON_NUM_THREADS=1 taskset -c 2 $B -n $n -q 2000000 --interleave 11 --key-seed $ks -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$ks,/"
  done
done
