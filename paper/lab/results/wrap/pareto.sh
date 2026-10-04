#!/usr/bin/env bash
# PHast-R with wrapping vs patterns vs PHast+ (u64 keys, GxHash, single
# thread, pinned, 7 interleaved rounds); remap with sparse inventory (12, 3)
S=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad
V=ref:plus:8:5.25,ref:w3:8:4.75,ref:w3:8:5.0,ref:w3:8:5.25,r:8:10:1:4.75,r:8:10:2:4.75
V=$V,r:8:10:0:5.0:w3,r:8:10:1:4.75:w3,r:8:10:1:5.0:w3:u8:4,r:8:10:1:5.0:w3:u8:8,r:8:10:1:5.0:w3,r:8:10:2:5.0:w3:u8:4,r:8:10:2:5.0:w3
for n in 10000000 100000000; do
  RAYON_NUM_THREADS=1 taskset -c 2 $S/cmp-remap -n $n -q 2000000 --interleave 7 -v $V 2>&1 | grep -v CSV
done
