#!/usr/bin/env bash
# PHast-R with wrapping (W<M>) vs patterns and PHast+ (u64 keys, GxHash,
# single thread, pinned, 7 interleaved rounds)
S=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad
for n in 1000000 10000000; do
  RAYON_NUM_THREADS=1 taskset -c 2 $S/cmp-wrap -n $n -q 2000000 --interleave 7 \
    -v ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:1:4.75,r:8:10:0:5.0:w3,r:8:10:1:4.75:w3,r:8:10:1:5.0:w3,r:8:10:1:5.25:w3,r:8:10:2:5.0:w3,r:8:10:2:5.25:w3,r:8:10:1:5.0:w2,r:8:10:1:5.0:w1 2>&1 | grep -v CSV
done
