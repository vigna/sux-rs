#!/usr/bin/env bash
# Ring patterns (G4, no repair) vs wrapping with repair vs PHast+ (u64 keys,
# GxHash); single thread pinned with 9 interleaved rounds of queries, then
# construction with 8 threads
S=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad
V=ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:1:5.0:w3,r:8:10:0:5.0:g4,r:8:10:0:4.75:g4,r:8:10:1:5.0:g4
for n in 1000000 10000000 100000000; do
  RAYON_NUM_THREADS=1 taskset -c 2 $S/cmp-ring5 -n $n -q 2000000 --interleave 9 -v $V 2>&1 | grep -v CSV
done
echo "== construction with 8 threads"
for n in 10000000 100000000; do
  RAYON_NUM_THREADS=8 taskset -c 0-7 $S/cmp-ring5 -n $n -t 8 -q 1000 --interleave 1 -v $V 2>&1 | grep -v CSV | cut -c1-90
done
