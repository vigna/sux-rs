#!/usr/bin/env bash
# 10-bit seeds: PHast-R with wrapping (unaligned seed reads) vs PHast+ with
# wrapping (u64 keys, GxHash, single thread, pinned, 7 interleaved rounds)
S=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad
V=ref:w1:10:6.0,ref:w3:10:6.0,r:10:11:1:6.0:2:bfvu,r:10:11:0:6.0:w1:bfvu,r:10:11:1:6.0:w1:bfvu,r:10:11:1:6.0:w2:bfvu,r:10:11:1:6.0:w3:bfvu,r:10:11:1:6.25:w3:bfvu
for n in 10000000 100000000; do
  RAYON_NUM_THREADS=1 taskset -c 2 $S/cmp-div -n $n -q 2000000 --interleave 7 -v $V 2>&1 | grep -v CSV
done
